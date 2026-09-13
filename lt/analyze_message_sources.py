"""Where does the history correction to CURRENT values come from?

Runs the normal v1.1 trajectory. At each block after seg16, locally removes
history corrections from selected source columns, without changing the
instantaneous read/write terms. Source groups: observed clue, currently correctly
decoded blank, currently incorrectly decoded blank, and self. Correctness refers
to the PREVIOUS block's decoded output, not the semantic quality of its hidden
message. Labels only define diagnostic groups and evaluate counterfactuals;
they never enter the normal inference trajectory. There are no oracle rollouts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
from analyze_memory_correction import CorrectionBlocks
from ckpt_npz import load_data, load_lt


CONDITIONS = ("all_history", "other_clue", "other_correct_blank", "other_wrong_blank", "self")
METRICS = ("loss_increase_on_removal", "gold_margin_decrease_on_removal",
           "previous_choice_margin_decrease_on_removal", "errors_increase_on_removal")


def read(blocks, parts, read_w):
    q, a, v, _, _ = parts
    eff = (1 - blocks.lam) * a + blocks.lam * read_w
    o = torch.einsum("bhtn,bnhc->bthc", eff, v)
    update = torch.einsum("bthc,hcd->btd", o, blocks.layer.w_sh)
    return blocks.inner.phi(q + update)


def token_metrics(logits, labels, previous_pred):
    loss = train.stablemax_cross_entropy(logits, labels).float()
    margins = []
    for selected in (labels, previous_pred):
        chosen = logits.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        competitor = logits.clone().scatter_(-1, selected.unsqueeze(-1), -torch.inf).amax(-1)
        margins.append(chosen - competitor)
    errors = (logits.argmax(-1) != labels).float()
    return torch.stack((loss, *margins, errors), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    ap.add_argument("--out", default="runs/message_sources_v11")
    args = ap.parse_args()
    assert 1 <= args.n <= 2048 and args.segs > 16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m, _, _ = load_lt(args.checkpoint, mod=train, batch_size=args.n, loops=args.segs + 1, amp=False)
    x, y, batch = load_data(n=args.n)
    blocks = CorrectionBlocks(m, batch)
    h = m.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = None
    labels = batch["labels"]
    blank = batch["inputs"] == 1
    clue = ~blank
    rows, counts, correctness = [], [], []
    for k in range(1, args.segs * 8 + 1):
        previous_pred = m.inner.w_cls(h).argmax(-1)
        parts = blocks.components(h, w)
        hn = blocks.read(parts, "normal")
        logits = m.inner.w_cls(hn)
        correctness.append((logits.argmax(-1) == labels).all(-1).cpu().numpy())
        if k > 128:
            previous_correct = previous_pred == labels
            target_groups = (blank & ~previous_correct, blank & previous_correct)
            target_count = torch.stack([mask.sum(-1) for mask in target_groups], -1)
            counts.append(target_count.cpu().numpy())
            current = token_metrics(logits, labels, previous_pred)
            wn, target = parts[-1], parts[-2]
            d = wn - target
            diag = torch.diag_embed(d.diagonal(dim1=-2, dim2=-1))
            offdiag = d - diag
            corrections = (d, offdiag * clue[:, None, None, :],
                           offdiag * (blank & previous_correct)[:, None, None, :],
                           offdiag * (blank & ~previous_correct)[:, None, None, :], diag)
            if k == 129:
                torch.testing.assert_close(sum(corrections[1:]), d, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(read(blocks, parts, wn), hn, atol=0, rtol=0)
            mode_rows = []
            for correction in corrections:
                cf = token_metrics(m.inner.w_cls(read(blocks, parts, wn - correction)), labels, previous_pred)
                diff = cf - current
                diff[..., 1:3] *= -1  # Positive: history raises margins / lowers loss and errors.
                averaged = []
                for mask in target_groups:
                    n = mask.sum(-1)
                    v = (diff * mask[..., None]).sum(1) / n.clamp_min(1)[:, None]
                    v[n == 0] = torch.nan
                    averaged.append(v)
                mode_rows.append(torch.stack(averaged, 1))
            rows.append(torch.stack(mode_rows, 1).cpu().numpy())
        h, w = hn, parts[-1]
        if k % 256 == 0:
            print("source analysis seg", k // 8, flush=True)
    values = np.stack(rows)  # [time, puzzle, source intervention, target group, metric]
    count = np.stack(counts)
    correct = np.stack(correctness)
    np.savez_compressed(out / "effects.npz", values=values, target_counts=count, block_correct=correct)
    start, final = correct[127], correct[-1]
    events = []
    for p in np.flatnonzero(~start & final):
        t = np.flatnonzero(correct[128:, p])[0]
        if t >= 8:
            events.append((int(p), int(t)))
    report = {"args": vars(args), "conditions": CONDITIONS, "metrics": METRICS,
              "target_groups": ["previously_wrong_blank", "previously_correct_blank"],
              "precision": "FP32; autocast/TF32 off", "event_count": len(events),
              "initial_final_exact": [int(start.sum()), int(final.sum())]}
    for name, ids in (("new_final_correct", np.flatnonzero(~start & final)),
                      ("still_wrong_final", np.flatnonzero(~start & ~final))):
        # Target-cell-weighted average within each puzzle, then equal puzzle weights.
        ws = count[:, ids, None, :, None]
        numerator = (np.nan_to_num(values[:, ids]) * ws).sum(0)
        denominator = ws.sum(0)
        per_puzzle = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        report[name] = np.nanmean(per_puzzle, axis=0).tolist()
    aligned = []
    for p, t in events:
        ws = count[t-8:t, p, None, :, None]
        numerator = (np.nan_to_num(values[t-8:t, p]) * ws).sum(0)
        denominator = ws.sum(0)
        aligned.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    report["before_first_correct"] = np.nanmean(np.stack(aligned), 0).tolist() if aligned else None
    if aligned:
        # Exploratory uncertainty across selected puzzles, not independent blocks.
        a = np.stack(aligned)
        rng = np.random.default_rng(0)
        sampled = rng.integers(0, len(a), size=(5000, len(a)))
        report["pretransition_wrong_target_error_bootstrap"] = {}
        for i, mode in enumerate(CONDITIONS):
            effects = a[:, i, 0, 3]
            means = np.nanmean(effects[sampled], axis=1)
            report["pretransition_wrong_target_error_bootstrap"][mode] = {
                "mean": float(np.nanmean(effects)),
                "interval_95": np.nanquantile(means, [0.025, 0.975]).tolist(),
                "positive_puzzles": int((effects > 0).sum()),
                "zero_puzzles": int((effects == 0).sum()),
                "negative_puzzles": int((effects < 0).sum()),
                "samples": 5000, "seed": 0,
            }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for group in ("new_final_correct", "still_wrong_final", "before_first_correct"):
        print(group, flush=True)
        if report[group] is not None:
            for mode, row in zip(CONDITIONS, report[group]):
                print(mode, "wrong-target [loss, gold, previous choice, error]", np.round(row[0], 6).tolist(), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
