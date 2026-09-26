"""Prospective, paired intervention on the v1.1 agree write target.

Protocol: docs/agree_probe_plan_v11.md. Labels never enter the writing rule.
Start from a shared normal state; intervene during a fixed window; restore
normal dynamics and score a fixed future horizon. No outcome-selected events.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import train
from ckpt_npz import load_lt
from probe_phase_feedback import Blocks, verify_runner


MODES = ("current", "past_mean", "shuffle_0", "shuffle_1", "shuffle_2")


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class AgreeBlocks(Blocks):
    def __init__(self, model, batch, seeds=(0, 1, 2)):
        super().__init__(model, batch)
        self.n = len(batch["inputs"])
        self.eye = torch.eye(81, dtype=torch.bool, device="cuda")[None, None]
        self.off = ~self.eye
        self.permutations = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            p = np.stack([rng.permutation(81) for _ in range(self.n * self.inner.H)])
            self.permutations[f"shuffle_{seed}"] = torch.from_numpy(
                p.reshape(self.n, self.inner.H, 81)).cuda()
        self.diagnostics = {}
        self.norm_max_relative_error = 0.0
        self.diagonal_max_error = 0.0

    def match_target(self, candidate, target):
        target_off = target * self.off
        candidate_off = candidate * self.off
        tn = target_off.square().sum((-1, -2), keepdim=True).sqrt()
        cn = candidate_off.square().sum((-1, -2), keepdim=True).sqrt()
        if bool(((cn < 1e-12) & (tn > 1e-8)).any()):
            raise RuntimeError("Nonzero target cannot be matched by a zero candidate")
        matched = candidate_off * (tn / cn.clamp_min(1e-12)) + target * self.eye
        mn = (matched * self.off).square().sum((-1, -2), keepdim=True).sqrt()
        error = ((mn - tn).abs() / tn.clamp_min(1e-8)).max().item()
        diagonal_error = ((matched - target) * self.eye).abs().max().item()
        self.norm_max_relative_error = max(self.norm_max_relative_error, error)
        self.diagonal_max_error = max(self.diagonal_max_error, diagonal_error)
        if error > 2e-5 or diagonal_error != 0:
            raise RuntimeError(f"Target invariants failed: norm={error}, diagonal={diagonal_error}")
        return matched

    def block_with_mean(self, h, w, mean, intervene=False):
        q = self.prepare(h)
        address = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(address, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
        window = self.inner.attn_xy(address, self.kcb)
        target = self.gain * (window * agree)
        if intervene:
            assert h.shape[0] == self.n * len(MODES) and mean is not None
            for arm, mode in enumerate(MODES[1:], 1):
                sl = slice(arm * self.n, (arm + 1) * self.n)
                original = target[sl]
                if mode == "past_mean":
                    alternative = mean[sl]
                else:
                    p = self.permutations[mode]
                    alternative = agree[sl].gather(-2, p[..., :, None].expand(-1, -1, -1, 81))
                    alternative = alternative.gather(-1, p[..., None, :].expand(-1, -1, 81, -1))
                candidate = self.gain * (window[sl] * alternative)
                candidate = self.match_target(candidate, original)
                original_off, candidate_off = original * self.off, candidate * self.off
                norm2 = original_off.square().sum((-1, -2)).clamp_min(1e-20)
                relative = ((candidate_off - original_off).square().sum((-1, -2)) / norm2).sqrt()
                cosine = (candidate_off * original_off).sum((-1, -2)) / norm2
                values = torch.stack((relative, cosine), -1)
                if mode not in self.diagnostics:
                    self.diagnostics[mode] = [values.clone(), 1]
                else:
                    self.diagnostics[mode][0].add_(values)
                    self.diagnostics[mode][1] += 1
                target[sl] = candidate
        wn = target if w is None else (1 - self.eta) * w + self.eta * target
        mean_next = agree if mean is None else (1 - self.eta) * mean + self.eta * agree
        effective = (1 - self.lam) * a + self.lam * wn
        values = torch.einsum("bhtn,bnhc->bthc", effective, v)
        update = torch.einsum("bthc,hcd->btd", values, self.layer.w_sh)
        return self.inner.phi(q + update), wn, mean_next


def paired_summary(a, b, mask, samples=10000):
    x = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    if len(x) == 0:
        return {"n": 0, "interpretation": "no eligible puzzles"}
    rng = np.random.default_rng(20260926)
    resampled = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(1)
    lo, hi = np.quantile(resampled, [0.025, 0.975])
    verdict = "first_better" if lo > 0 else "second_better" if hi < 0 else "inconclusive"
    return {"n": len(x), "difference_pp": float(100 * x.mean()),
            "paired_bootstrap_ci95_pp": [float(100 * lo), float(100 * hi)],
            "first_wins": int((x > 0).sum()), "second_wins": int((x < 0).sum()),
            "ties": int((x == 0).sum()), "interpretation": verdict,
            "all_observed_differences_zero": bool(np.all(x == 0))}


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed run: {out}")
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_path = Path(__file__).resolve().parents[1] / "data/sudoku_lt_1k.npz"
    with np.load(data_path) as data:
        x = data["test_inputs"].reshape(-1, 81)[args.offset:args.offset + args.n].astype(np.int32)
        y = data["test_labels"].reshape(-1, 81)[args.offset:args.offset + args.n].astype(np.int64)
    assert len(x) == args.n
    batch = {"inputs": torch.from_numpy(x + 1).cuda(),
             "labels": torch.from_numpy(y + 1).cuda(),
             "puzzle_identifiers": torch.zeros(args.n, dtype=torch.int32, device="cuda")}
    model, cfg, step = load_lt(args.checkpoint, mod=train, batch_size=args.n,
                               loops=args.segs + 1, amp=False)
    model.requires_grad_(False)
    b = AgreeBlocks(model, batch)
    verification = verify_runner(b, model, batch)
    h = model.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = mean = None
    rh, rw = h, None
    for _ in range(8):
        h, w, mean = b.block_with_mean(h, w, mean)
        rh, rw = b.block(rh, rw)
    torch.testing.assert_close(h, rh, atol=0, rtol=0)
    torch.testing.assert_close(w, rw, atol=0, rtol=0)
    verification["normal_probe_8_blocks_exact"] = True
    print("verification", json.dumps(verification), flush=True)
    del rh, rw, h, w, mean
    metadata = {"args": vars(args), "checkpoint_step": step,
                "checkpoint_sha256": sha256(args.checkpoint), "data_sha256": sha256(data_path),
                "protocol_sha256": sha256("docs/agree_probe_plan_v11.md"),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "precision": "FP32; autocast/TF32 disabled; fp16-compressed source checkpoint",
                "modes": list(MODES), "verification": verification,
                "eta_by_head": b.eta.flatten().cpu().tolist(),
                "lambda_by_head": b.lam.flatten().cpu().tolist(),
                "gain_by_head": b.gain.flatten().cpu().tolist(),
                "primary": "seg128 exact on puzzles unsolved at shared seg16; current minus past_mean",
                "diagnostic_columns": ["target_offdiag_relative_change", "target_offdiag_cosine"]}
    write_json(out / "metadata.json", metadata)
    np.savez_compressed(out / "permutations.npz", **{k: v.cpu().numpy() for k, v in b.permutations.items()})
    preds = np.empty((args.segs, len(MODES), args.n, 81), dtype=np.uint8)
    h = model.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = mean = None
    started = time.monotonic()
    divergence = {}
    final_logits = None
    for k in range(1, args.segs * 8 + 1):
        active = args.start_seg * 8 < k <= args.stop_seg * 8
        h, w, mean = b.block_with_mean(h, w, mean, intervene=active)
        if k % 8 == 0:
            seg = k // 8
            logits = model.inner.w_cls(h)
            p = logits.argmax(-1).cpu().numpy().astype(np.uint8)
            if seg <= args.start_seg:
                preds[seg - 1] = p[None]
            else:
                preds[seg - 1] = p.reshape(len(MODES), args.n, 81)
            if seg == args.segs:
                final_logits = logits.reshape(len(MODES), args.n, 81, -1)
            if seg % 8 == 0 or seg == args.segs:
                row = {"segment": seg, "exact": dict(zip(MODES, ((preds[seg-1] == y[None] + 1).all(-1).sum(-1)).tolist())),
                       "elapsed_seconds": round(time.monotonic() - started, 2),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                print(json.dumps(row), flush=True)
                write_json(out / "progress.json", row)
                np.save(out / "predictions_partial.npy", preds[:seg])
        if k == args.start_seg * 8:
            h, w, mean = (z.repeat(len(MODES), 1, 1) if z.ndim == 3 else z.repeat(len(MODES), 1, 1, 1)
                          for z in (h, w, mean))
            for z in (h, w, mean):
                groups = z.reshape(len(MODES), args.n, *z.shape[1:])
                assert torch.equal(groups, groups[:1].expand_as(groups))
            verification["branch_initial_states_exact"] = True
        if k == args.stop_seg * 8:
            for arm, mode in enumerate(MODES[1:], 1):
                sl = slice(arm * args.n, (arm + 1) * args.n)
                divergence[mode] = {}
                for name, z in (("hidden", h), ("memory", w)):
                    rel = (z[sl] - z[:args.n]).flatten(1).norm(dim=-1) / z[:args.n].flatten(1).norm(dim=-1).clamp_min(1e-12)
                    divergence[mode][name + "_relative_difference_per_puzzle"] = rel.cpu().tolist()
    correct = (preds == y[None, None] + 1).all(-1)
    initial = correct[args.start_seg - 1, 0]
    final = correct[-1]
    summary = {"metadata": metadata, "initial_exact": int(initial.sum()),
               "primary_cohort_n": int((~initial).sum()), "results": {},
               "primary_current_minus_past_mean": paired_summary(final[0], final[1], ~initial),
               "secondary_current_minus_shuffle_mean": paired_summary(final[0], final[2:].mean(0), ~initial),
               "secondary_current_minus_each_shuffle": {}, "end_of_intervention": divergence,
               "normalization_verification": {"max_relative_norm_error": b.norm_max_relative_error,
                                              "max_diagonal_error": b.diagonal_max_error},
               "elapsed_seconds": time.monotonic() - started,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    per_puzzle = {"initial_correct": initial, "final_correct": final}
    for arm, mode in enumerate(MODES):
        ce = train.stablemax_cross_entropy(final_logits[arm], batch["labels"])
        blanks = batch["inputs"] == 1
        loss = ((ce * blanks).sum(-1) / blanks.sum(-1).clamp_min(1)).cpu().numpy()
        per_puzzle[mode + "_blank_loss"] = loss
        summary["results"][mode] = {
            "final_exact": int(final[arm].sum()), "new_exact": int((final[arm] & ~initial).sum()),
            "lost_initial_exact": int((~final[arm] & initial).sum()),
            "final_blank_loss_mean": float(loss.mean()),
            "final_blank_loss_initially_unsolved": float(loss[~initial].mean()) if (~initial).any() else None,
            "exact_by_segment": correct[:, arm].sum(-1).tolist(),
            "newly_solved_indices": (np.flatnonzero(final[arm] & ~initial) + args.offset).tolist(),
            "lost_initial_indices": (np.flatnonzero(~final[arm] & initial) + args.offset).tolist()}
        if mode in b.diagnostics:
            values, count = b.diagnostics[mode]
            values = (values / count).cpu().numpy()
            per_puzzle[mode + "_write_diagnostics"] = values
            summary["results"][mode]["write_diagnostics_mean_by_head"] = values.mean(0).tolist()
        if mode.startswith("shuffle"):
            summary["secondary_current_minus_each_shuffle"][mode] = paired_summary(final[0], final[arm], ~initial)
    np.savez_compressed(out / "trajectories.npz", X=x, Y=y, indices=np.arange(args.offset, args.offset+args.n), predictions=preds)
    np.savez_compressed(out / "per_puzzle.npz", **per_puzzle)
    write_json(out / "summary.json", summary)
    write_json(out / "metadata.json", metadata)
    print("PRIMARY", json.dumps(summary["primary_current_minus_past_mean"]), flush=True)
    for mode in MODES:
        print(mode, json.dumps({k: v for k, v in summary["results"][mode].items()
                               if k in ("final_exact", "new_exact", "lost_initial_exact")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--offset", type=int, default=512)
    p.add_argument("--start-seg", type=int, default=16)
    p.add_argument("--stop-seg", type=int, default=32)
    p.add_argument("--segs", type=int, default=128)
    p.add_argument("--out", default="runs/agree_target_v11")
    args = p.parse_args()
    assert 0 < args.start_seg < args.stop_seg < args.segs
    assert 0 <= args.offset and args.offset + args.n <= 2048
    run(args)


if __name__ == "__main__":
    main()
