"""v1.1 inference-memory interventions and paired phase perturbation probes.

No training or Sudoku constraints are added. Uses EMA weights in float32 with
TF32/autocast disabled so small perturbations are not rounded away by BF16.

python lt/probe_phase_feedback.py --n 128 --out runs/phase_feedback_v11

Phase probes rotate the entire complex address of ONE head in two randomly
chosen blank cells by +/-epsilon/2. The rotation is lifted to hidden space via
the row-orthogonal address projection. Other heads can consequently change.
"replay" supplies the unperturbed trajectory's updated memory at EVERY block;
it removes feedback through memory, without removing the baseline memory.
These are finite perturbation responses around moving trajectories, not a
fixed-point stability proof. Ground-truth labels are used for evaluation only.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train
from ckpt_npz import load_data, load_lt


class Blocks:
    def __init__(self, model, batch):
        self.inner = model.inner
        assert not model.config.legacy_gauge
        assert model.config.block_order == "pre" and not model.config.use_trace
        assert model.config.num_layers == 1 and model.config.stdp
        self.layer = self.inner.layers[0]
        self.ab = self.inner.W_C(self.layer)
        self.kc = self.inner.kernel(self.layer)
        self.kcb = self.inner.kernel(self.layer, self.layer.beta)
        self.inj = self.inner.embed_scale * self.inner.injection(batch)
        self.eta = torch.sigmoid(self.layer.eta_raw)
        self.lam = torch.sigmoid(self.layer.lam_raw)
        self.gain = F.softplus(self.layer.gain_raw)

    def prepare(self, h):
        repeats = h.shape[0] // self.inj.shape[0]
        return self.inner.boundary(self.layer, h) + self.inj.repeat(repeats, 1, 1)

    def transport(self, q, w, mode="normal", replay_n=None):
        address = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(address, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        if mode == "freeze":
            wn = w
        else:
            window = self.inner.attn_xy(address, self.kcb)
            unit_v = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
            agree = torch.einsum("bthc,bnhc->bhtn", unit_v, unit_v)
            target = self.gain * (window * agree)
            wn = target if w is None or mode == "instant" else (1 - self.eta) * w + self.eta * target
        if replay_n is not None:
            # Groups: baseline, full+, full-, replay+, replay-.
            wn = torch.cat((wn[:3 * replay_n], wn[:replay_n], wn[:replay_n]))
        effective = (1 - self.lam) * a + self.lam * wn
        values = torch.einsum("bhtn,bnhc->bthc", effective, v)
        update = torch.einsum("bthc,hcd->btd", values, self.layer.w_sh)
        return self.inner.phi(q + update), wn

    def block(self, h, w, mode="normal"):
        return self.transport(self.prepare(h), w, mode)


def predictions(blocks, h):
    return blocks.inner.w_cls(h).argmax(-1).cpu().numpy().astype(np.int8)


def verify_runner(blocks, model, batch):
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    h = blocks.inner.init_hidden.expand(batch["inputs"].shape[0], 81, -1).clone()
    w = None
    for _ in range(model.config.blocks_per_seg):
        h, w = blocks.block(h, w)
    errors = {
        "hidden_max_abs": float((h - carry.current_hidden).abs().max()),
        "memory_max_abs": float((w - carry.coupling).abs().max()),
        "logits_max_abs": float((blocks.inner.w_cls(h) - outputs["logits"]).abs().max()),
    }
    torch.testing.assert_close(h, carry.current_hidden, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(w, carry.coupling, atol=2e-5, rtol=2e-5)
    return errors


def rollout(blocks, n, segs, save_segs):
    h = blocks.inner.init_hidden.expand(n, 81, -1).clone()
    w = None
    saved, pred = {}, []
    for seg in range(1, segs + 1):
        for _ in range(8):
            h, w = blocks.block(h, w)
        pred.append(predictions(blocks, h))
        if seg in save_segs:
            saved[seg] = (h.clone(), w.clone())
        if seg % 32 == 0:
            print(f"baseline seg={seg}", flush=True)
    return np.stack(pred), saved


def continue_intervention(blocks, snapshot, start_seg, segs, mode):
    h, w = (v.clone() for v in snapshot)
    pred = []
    for seg in range(start_seg + 1, segs + 1):
        for _ in range(8):
            h, w = blocks.block(h, w, mode)
        pred.append(predictions(blocks, h))
        if seg % 32 == 0:
            print(f"{mode} seg={seg}", flush=True)
    return np.stack(pred)


def select_pairs(inputs, heads, seed):
    rng = np.random.default_rng(seed)
    pairs = np.stack([rng.choice(np.flatnonzero(row == 1), 2, replace=False) for row in inputs])
    chosen_heads = np.arange(len(inputs)) % heads
    rng.shuffle(chosen_heads)
    return pairs, chosen_heads


def phase_rotate(blocks, q, pairs, heads, epsilon):
    """Minimum-norm hidden lift; exact rotation in the selected head."""
    x, y = blocks.inner.addr_raw(q, blocks.ab)
    result = q.clone()
    idx = torch.arange(q.shape[0], device=q.device)
    a, b = (matrix[heads] for matrix in blocks.ab)
    for col, sign in ((0, 1), (1, -1)):
        cells = pairs[:, col]
        xr, yr = x[idx, cells, heads], y[idx, cells, heads]
        angle = sign * epsilon / 2
        dx = (math.cos(angle) - 1) * xr - math.sin(angle) * yr
        dy = math.sin(angle) * xr + (math.cos(angle) - 1) * yr
        result[idx, cells] += torch.einsum("bj,bjd->bd", dx, a) + torch.einsum("bj,bjd->bd", dy, b)
    return result


def pair_phase_response(blocks, q, baseline, pairs, heads):
    """Amplitude-weighted common address phase shift, cell 1 minus cell 2."""
    x, y = blocks.inner.addr_raw(q, blocks.ab)
    x0, y0 = blocks.inner.addr_raw(baseline, blocks.ab)
    idx = torch.arange(q.shape[0], device=q.device)
    angles = []
    for col in (0, 1):
        cell = pairs[:, col]
        xr, yr = x[idx, cell, heads], y[idx, cell, heads]
        xb, yb = x0[idx, cell, heads], y0[idx, cell, heads]
        angles.append(torch.atan2((xb * yr - yb * xr).sum(-1), (xb * xr + yb * yr).sum(-1)))
    return angles[0] - angles[1]


def perturb_probe(blocks, snapshot, pairs_np, heads_np, epsilon, horizon):
    n = len(pairs_np)
    pairs = torch.as_tensor(pairs_np, device="cuda")
    heads = torch.as_tensor(heads_np, device="cuda")
    h, w = snapshot
    q0 = blocks.prepare(h)
    qp = phase_rotate(blocks, q0, pairs, heads, epsilon)
    qm = phase_rotate(blocks, q0, pairs, heads, -epsilon)
    initial_phase = (pair_phase_response(blocks, qp, q0, pairs, heads) -
                     pair_phase_response(blocks, qm, q0, pairs, heads)) / (2 * epsilon)
    torch.testing.assert_close(initial_phase, torch.ones_like(initial_phase), atol=2e-4, rtol=2e-4)
    initial_norm = ((qp - qm) / (2 * epsilon)).flatten(1).norm(dim=-1)
    q = torch.cat((q0, qp, qm, qp, qm))
    ws = w.repeat(5, 1, 1, 1)
    times = sorted({0, 1, 2, 4, 8, 16, 32, horizon} & set(range(horizon + 1)))
    rows = {"full_phase": [], "replay_phase": [], "full_state_gain": [], "replay_state_gain": []}
    for k in range(horizon + 1):
        if k in times:
            groups = q.split(n)
            for mode, pos, neg in (("full", 1, 2), ("replay", 3, 4)):
                phase = (pair_phase_response(blocks, groups[pos], groups[0], pairs, heads) -
                         pair_phase_response(blocks, groups[neg], groups[0], pairs, heads)) / (2 * epsilon)
                state_gain = ((groups[pos] - groups[neg]) / (2 * epsilon)).flatten(1).norm(dim=-1) / initial_norm
                rows[f"{mode}_phase"].append(phase.cpu().numpy())
                rows[f"{mode}_state_gain"].append(state_gain.cpu().numpy())
        if k < horizon:
            hn, ws = blocks.transport(q, ws, replay_n=n)
            q = blocks.prepare(hn)
    return {"times": np.array(times), **{k: np.stack(v) for k, v in rows.items()}}


def probe_summary(result):
    rows = []
    for i, k in enumerate(result["times"]):
        f, r = result["full_phase"][i], result["replay_phase"][i]
        fs, rs = result["full_state_gain"][i], result["replay_state_gain"][i]
        rows.append({
            "blocks": int(k),
            "full_phase_rms": float(np.sqrt(np.mean(f ** 2))),
            "replay_phase_rms": float(np.sqrt(np.mean(r ** 2))),
            "full_phase_median_abs": float(np.median(abs(f))),
            "replay_phase_median_abs": float(np.median(abs(r))),
            "memory_reduces_phase_fraction": float(np.mean(abs(f) < abs(r))),
            "full_state_gain_median": float(np.median(fs)),
            "replay_state_gain_median": float(np.median(rs)),
            "memory_reduces_state_fraction": float(np.mean(fs < rs)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--segs", type=int, default=128)
    parser.add_argument("--probe-segs", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.005])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/phase_feedback_v11")
    args = parser.parse_args()
    assert args.segs > 16 and all(1 <= s <= args.segs for s in args.probe_segs)
    assert 1 <= args.n <= 2048
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg, step = load_lt(args.checkpoint, mod=train, batch_size=args.n, loops=args.segs + 1, amp=False)
    x, y, batch = load_data(n=args.n)
    blocks = Blocks(model, batch)
    verification = verify_runner(blocks, model, batch)
    print("runner check", verification, flush=True)
    t0 = time.time()
    pred, saved = rollout(blocks, args.n, args.segs, set(args.probe_segs) | {16})
    all_pred = {"normal": pred}
    for mode in ("freeze", "instant"):
        after = continue_intervention(blocks, saved[16], 16, args.segs, mode)
        all_pred[mode] = np.concatenate((pred[:16], after))
    np.savez_compressed(out / "trajectories.npz", X=x, Y=y, **all_pred)
    correct16 = (pred[15] == y + 1).all(-1)
    summaries = {}
    for mode, p in all_pred.items():
        correct = (p == y[None] + 1).all(-1)
        summaries[mode] = {
            "exact_by_seg": correct.sum(-1).tolist(),
            "final_exact": int(correct[-1].sum()),
            "best_exact": int(correct.sum(-1).max()),
            "new_correct_since16_final": int((correct[-1] & ~correct16).sum()),
            "lost_correct_since16_final": int((~correct[-1] & correct16).sum()),
            "correct_to_wrong_transitions_after16": int((correct[15:-1] & ~correct[16:]).sum()),
            "wrong_to_correct_transitions_after16": int((~correct[15:-1] & correct[16:]).sum()),
            "final_clue_changed_puzzles": int((((p[-1] - 1) != x) & (x != 0)).any(-1).sum()),
        }
        print(mode, {k: v for k, v in summaries[mode].items() if k != "exact_by_seg"}, flush=True)
    pairs, heads = select_pairs(batch["inputs"].cpu().numpy(), blocks.inner.H, args.seed)
    probe_results = {}
    for seg in args.probe_segs:
        for epsilon in args.epsilons:
            result = perturb_probe(blocks, saved[seg], pairs, heads, epsilon, args.horizon)
            key = f"seg{seg}_eps{epsilon:g}"
            np.savez_compressed(out / f"{key}.npz", pairs=pairs, heads=heads, **result)
            probe_results[key] = probe_summary(result)
            print(key, probe_results[key][-1], flush=True)
    report = {
        "args": vars(args), "step": step,
        "precision": "float32, autocast off, TF32 off; checkpoint has fp16-compressed large tensors",
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "runner_verification": verification,
        "eta": blocks.eta.flatten().cpu().tolist(),
        "lambda": blocks.lam.flatten().cpu().tolist(),
        "gain": blocks.gain.flatten().cpu().tolist(),
        "trajectory_summaries": summaries, "probes": probe_results,
        "elapsed_seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out / 'summary.json'} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
