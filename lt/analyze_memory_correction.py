"""Separate the symmetric/skew parts of v1.1's HISTORY correction.

At each block, target G is computed from the current state, memory updates to
w_next, and D = w_next - G is precisely the correction due to history. The
four read conditions are w_next, G, G + sym(D), and G + skew(D). The instantaneous
read kernel and the instantaneous write target remain unchanged. In particular,
skew-only does NOT remove every symmetric interaction in the model.

Uses no new task constraints. Labels only score outcomes and local counterfactuals.
FP32, autocast/TF32 off. Long interventions start at an identical seg16 state;
local counterfactuals always use the unmodified baseline's current state.

python lt/analyze_memory_correction.py --out runs/memory_correction_v11
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train
from ckpt_npz import load_data, load_lt
from probe_phase_feedback import Blocks, verify_runner


MODES = ("normal", "instant", "symmetric_history", "skew_history")


class CorrectionBlocks(Blocks):
    def components(self, h, w):
        q = self.prepare(h)
        u = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(u, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
        window = self.inner.attn_xy(u, self.kcb)
        target = self.gain * (window * agree)
        wn = target if w is None else (1 - self.eta) * w + self.eta * target
        return q, a, v, target, wn

    def read(self, parts, mode):
        q, a, v, target, wn = parts
        if mode == "normal":
            read_w = wn
        elif mode == "instant":
            read_w = target
        else:
            d = wn - target
            dt = d.transpose(-1, -2)
            if mode == "symmetric_history":
                correction = (d + dt) / 2
            elif mode == "skew_history":
                correction = (d - dt) / 2
            elif mode == "diagonal_history":
                correction = torch.diag_embed(d.diagonal(dim1=-2, dim2=-1))
            elif mode == "offdiag_symmetric_history":
                correction = (d + dt) / 2 - torch.diag_embed(d.diagonal(dim1=-2, dim2=-1))
            else:
                raise ValueError(mode)
            read_w = target + correction
        effective = (1 - self.lam) * a + self.lam * read_w
        o = torch.einsum("bhtn,bnhc->bthc", effective, v)
        update = torch.einsum("bthc,hcd->btd", o, self.layer.w_sh)
        return self.inner.phi(q + update)


def score(logits, labels, blanks):
    count = blanks.sum(-1).clamp_min(1)
    ce = train.stablemax_cross_entropy(logits, labels)
    gold = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    competitors = logits.clone().scatter_(-1, labels.unsqueeze(-1), -torch.inf).amax(-1)
    preds = logits.argmax(-1)
    return {
        "blank_loss": ((ce * blanks).sum(-1) / count).float(),
        "blank_margin": (((gold - competitors) * blanks).sum(-1) / count).float(),
        "blank_errors": ((preds != labels) & blanks).sum(-1),
        "correct": (preds == labels).all(-1),
    }


def baseline(blocks, batch, segs, start_seg):
    n = batch["inputs"].shape[0]
    h = blocks.inner.init_hidden.expand(n, 81, -1).clone()
    w, snapshot = None, None
    labels, blanks = batch["labels"], batch["inputs"] == 1
    pred_segments, block_correct, local = [], [], {}
    correction_stats = []
    for k in range(1, 8 * segs + 1):
        parts = blocks.components(h, w)
        hn = blocks.read(parts, "normal")
        logits = blocks.inner.w_cls(hn)
        preds = logits.argmax(-1)
        block_correct.append((preds == labels).all(-1).cpu().numpy())
        if k > start_seg * 8:
            for mode in MODES:
                scores = score(logits if mode == "normal" else blocks.inner.w_cls(blocks.read(parts, mode)), labels, blanks)
                for name, value in scores.items():
                    local.setdefault(f"{mode}_{name}", []).append(value.cpu().numpy())
            _, _, v, target, wn = parts
            d = wn - target
            sym = (d + d.transpose(-1, -2)) / 2
            skew = (d - d.transpose(-1, -2)) / 2
            # Norms per puzzle/head; diagonal only belongs to symmetric history.
            scales = []
            for mat in (target, d, sym, skew):
                weighted = blocks.lam * mat
                scales.append(weighted.square().sum((-1, -2)).sqrt())
                transported = torch.einsum("bhtn,bnhc->bthc", weighted, v)
                scales.append(transported.square().sum((1, 3)).sqrt())
            scales.append((blocks.lam * d).diagonal(dim1=-2, dim2=-1).square().sum(-1).sqrt())
            correction_stats.append(torch.stack(scales, dim=-1).cpu().numpy())
        h, w = hn, parts[-1]
        if k == 8 * start_seg:
            snapshot = (h.clone(), w.clone())
        if k % 8 == 0:
            pred_segments.append(preds.cpu().numpy().astype(np.int8))
        if k % 256 == 0:
            print(f"local baseline seg={k // 8}", flush=True)
    return (np.stack(pred_segments), np.stack(block_correct), snapshot,
            {name: np.stack(rows) for name, rows in local.items()}, np.stack(correction_stats))


def intervention(blocks, snapshot, mode, segs, start_seg):
    h, w = (value.clone() for value in snapshot)
    preds = []
    for k in range(8 * start_seg + 1, 8 * segs + 1):
        parts = blocks.components(h, w)
        h, w = blocks.read(parts, mode), parts[-1]
        if k % 8 == 0:
            preds.append(blocks.inner.w_cls(h).argmax(-1).cpu().numpy().astype(np.int8))
        if k % 256 == 0:
            print(f"{mode} seg={k // 8}", flush=True)
    return np.stack(preds)


def finite_summary(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return None if x.size == 0 else {"mean": float(x.mean()), "median": float(np.median(x)), "n": int(x.size)}


def local_summaries(local, block_correct, start_seg, segs):
    start = start_seg * 8
    was_correct = block_correct[start - 1]
    final_correct = block_correct[-1]
    groups = {
        "correct_at_start": was_correct,
        "new_final_correct": ~was_correct & final_correct,
        "still_wrong_final": ~was_correct & ~final_correct,
    }
    out = {"group_sizes": {k: int(v.sum()) for k, v in groups.items()}, "groups": {}, "before_first_correct": {}}
    for name, mask in groups.items():
        result = {}
        for mode in MODES[1:]:
            result[mode] = {}
            for metric in ("blank_loss", "blank_margin", "blank_errors"):
                # Positive loss/errors means normal memory is better; positive margin means worse.
                diff = (local[f"{mode}_{metric}"] - local[f"normal_{metric}"])[:, mask]
                per_puzzle = diff.mean(0)
                result[mode][f"counterfactual_minus_normal_{metric}"] = finite_summary(per_puzzle)
        out["groups"][name] = result
    # Event-aligned summaries: one average per puzzle in the 8 blocks BEFORE
    # first block-level exact solution, excluding the first correct block itself.
    ids = np.flatnonzero(groups["new_final_correct"])
    events = []
    for p in ids:
        after = np.flatnonzero(block_correct[start:, p])
        if len(after) and after[0] >= 8:
            events.append((int(p), int(after[0])))
    out["event_count_with_8_preceding_blocks"] = len(events)
    out["first_correct_blocks"] = {str(p): t + start + 1 for p, t in events}
    for mode in MODES[1:]:
        out["before_first_correct"][mode] = {}
        for metric in ("blank_loss", "blank_margin", "blank_errors"):
            diff = local[f"{mode}_{metric}"] - local[f"normal_{metric}"]
            vals = [diff[t-8:t, p].mean() for p, t in events]
            out["before_first_correct"][mode][f"counterfactual_minus_normal_{metric}"] = finite_summary(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--start-seg", type=int, default=16)
    ap.add_argument("--extra-modes", nargs="*", default=[],
                    choices=["diagonal_history", "offdiag_symmetric_history"])
    ap.add_argument("--out", default="runs/memory_correction_v11")
    args = ap.parse_args()
    assert 0 < args.start_seg < args.segs and 0 < args.n <= 2048
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m, cfg, step = load_lt(args.checkpoint, mod=train, batch_size=args.n, loops=args.segs + 1, amp=False)
    x, y, batch = load_data(n=args.n)
    blocks = CorrectionBlocks(m, batch)
    print("runner", verify_runner(blocks, m, batch), flush=True)
    # Match the decomposed runner's normal path to the existing block runner.
    h = m.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = None
    for _ in range(8):
        refh, refw = blocks.block(h, w)
        parts = blocks.components(h, w)
        h, w = blocks.read(parts, "normal"), parts[-1]
        torch.testing.assert_close(h, refh, atol=0, rtol=0)
        torch.testing.assert_close(w, refw, atol=0, rtol=0)
    print("decomposed runner exactly matches 8 original blocks", flush=True)
    t0 = time.time()
    normal, block_correct, snapshot, local, norms = baseline(blocks, batch, args.segs, args.start_seg)
    trajectories = {"normal": normal}
    for mode in (*MODES[1:], *args.extra_modes):
        after = intervention(blocks, snapshot, mode, args.segs, args.start_seg)
        trajectories[mode] = np.concatenate((normal[:args.start_seg], after))
    np.savez_compressed(out / "trajectories.npz", X=x, Y=y, **trajectories)
    np.savez_compressed(out / "local_counterfactuals.npz", block_correct=block_correct, correction_norms=norms, **local)
    initial = (normal[args.start_seg - 1] == y + 1).all(-1)
    normal_final = (normal[-1] == y + 1).all(-1)
    summary = {}
    for mode, p in trajectories.items():
        correct = (p == y[None] + 1).all(-1)
        summary[mode] = {
            "exact_by_seg": correct.sum(-1).tolist(), "final_exact": int(correct[-1].sum()),
            "new_correct_final": int((correct[-1] & ~initial).sum()),
            "lost_initial_correct_final": int((~correct[-1] & initial).sum()),
            "normal_final_wins": int((normal_final & ~correct[-1]).sum()),
            "intervention_final_wins": int((~normal_final & correct[-1]).sum()),
            "normal_new_solutions_retained": int((normal_final & ~initial & correct[-1]).sum()),
            "final_clue_changed_puzzles": int(((((p[-1] - 1) != x) & (x != 0)).any(-1)).sum()),
        }
        print(mode, {k: v for k, v in summary[mode].items() if k != "exact_by_seg"}, flush=True)
    report = {
        "args": vars(args), "step": step, "precision": "FP32; no autocast/TF32; fp16-compressed checkpoint",
        "definitions": {"G": "current write target including gain, beta-window and value agreement",
                        "D": "updated_memory - G; history correction before lambda",
                        "symmetric_history": "read G + (D + D^T)/2",
                        "skew_history": "read G + (D - D^T)/2"},
        "interventions": summary,
        "local": local_summaries(local, block_correct, args.start_seg, args.segs),
        "norm_columns": ["target_kernel", "target_value_transport", "history_kernel", "history_value_transport",
                         "symmetric_history_kernel", "symmetric_history_value_transport",
                         "skew_history_kernel", "skew_history_value_transport", "history_diagonal_kernel"],
        "norm_mean_by_head": norms.mean((0, 1)).tolist(),
        "elapsed_seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("pre-transition local effects", json.dumps(report["local"]["before_first_correct"]), flush=True)
    print(f"saved {out} ({report['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
