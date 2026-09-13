"""Relate exact phase-history statistics to local message usefulness in v1.1.

Fixed random directed pairs and heads are chosen without labels. Target cells
are input blanks. Auxiliary EMAs track complex relations but NEVER affect the
model. Their beta-weighted real projections must reconstruct sampled symmetric
memory entries. Every eight blocks, remove only one head's symmetric history
correction on an edge and score its immediate target-cell effect. Removing the
reverse edge simultaneously would not change this same-block target output.

Labels score local counterfactuals and define reporting subsets only. A lower
one-block loss is not a guarantee of better long-horizon problem solving.
First 128 puzzles are exploratory; puzzles 128..255 are a separate comparison
group. Correlated edges/blocks are not treated as independent statistical trials.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
from analyze_memory_correction import CorrectionBlocks
from ckpt_npz import load_data, load_lt


FEATURES = (
    "phase_coherence_unweighted", "phase_coherence_amplitude_weighted",
    "phase_agree_coherence", "symmetric_component_sign_coherence",
    "symmetric_edge_sign_coherence", "agree_sign_coherence",
    "current_agree", "mean_agree", "current_vs_past_phase_alignment",
    "current_symmetric_phase_kernel", "mean_symmetric_phase_kernel",
    "source_confidence", "target_confidence", "history_relative_size",
)


def update(old, value, eta):
    return value.clone() if old is None else (1 - eta) * old + eta * value


class PairHistory:
    def __init__(self, blocks, inputs, pairs_per_head, seed):
        self.blocks = blocks
        n, heads = len(inputs), blocks.inner.H
        rng = np.random.default_rng(seed)
        targets = np.empty((n, heads, pairs_per_head), dtype=np.int64)
        sources = np.empty_like(targets)
        for b in range(n):
            targets[b] = rng.choice(np.flatnonzero(inputs[b] == 1), (heads, pairs_per_head))
            candidate = rng.integers(0, 80, (heads, pairs_per_head))
            sources[b] = candidate + (candidate >= targets[b])
        self.targets = torch.as_tensor(targets, device="cuda")
        self.sources = torch.as_tensor(sources, device="cuda")
        self.bi = torch.arange(n, device="cuda")[:, None, None]
        self.hi = torch.arange(heads, device="cuda")[None, :, None]
        du = blocks.inner.pos_u[self.targets] - blocks.inner.pos_u[self.sources]
        dv = blocks.inner.pos_w[self.targets] - blocks.inner.pos_w[self.sources]
        theta = blocks.layer.theta
        angle = du[..., None] * theta[None, :, None, :, 0] + dv[..., None] * theta[None, :, None, :, 1]
        self.rotation = torch.complex(torch.cos(angle), torch.sin(angle))
        self.cos_beta = torch.cos(blocks.layer.beta)[None, :, None, :]
        self.scale = (blocks.gain[None, :, 0, 0, None] *
                      blocks.kc[0][self.hi, self.targets, self.sources])
        self.eta = blocks.eta[None, :, 0, 0, None]
        self.state = {}
        self.max_reconstruction_error = 0.0

    def step(self, parts):
        q, _, v, target, wn = parts
        b = self.blocks
        x, y = b.inner.addr(q, b.ab)
        ut = torch.complex(x[self.bi, self.targets, self.hi], y[self.bi, self.targets, self.hi])
        un = torch.complex(x[self.bi, self.sources, self.hi], y[self.bi, self.sources, self.hi])
        z = ut * un.conj() * self.rotation
        amp = z.abs()
        unit = z / amp.clamp_min(1e-12)
        vt = v[self.bi, self.targets, self.hi]
        vn = v[self.bi, self.sources, self.hi]
        agree = ((vt / (vt.norm(dim=-1, keepdim=True) + b.inner.config.eps)) *
                 (vn / (vn.norm(dim=-1, keepdim=True) + b.inner.config.eps))).sum(-1)
        weighted = agree[..., None] * z
        sym_components = self.cos_beta * weighted.real
        sym_edge = sym_components.sum(-1)
        entries = {"unit": unit, "z": z, "amp": amp, "weighted": weighted,
                   "weighted_abs": weighted.abs(), "sym_abs": sym_components.abs(),
                   "sym_edge_abs": sym_edge.abs(), "agree": agree,
                   "agree_abs": agree.abs()}
        for key, value in entries.items():
            eta = self.eta[..., None] if value.ndim == 4 else self.eta
            self.state[key] = update(self.state.get(key), value, eta)
        s = self.state
        reconstructed = self.scale * (self.cos_beta * s["weighted"].real).sum(-1)
        actual = (wn[self.bi, self.hi, self.targets, self.sources] +
                  wn[self.bi, self.hi, self.sources, self.targets]) / 2
        error = (reconstructed - actual).abs().max()
        self.max_reconstruction_error = max(self.max_reconstruction_error, float(error))
        torch.testing.assert_close(reconstructed, actual, atol=3e-6, rtol=2e-4)
        coh = lambda num, den: (num / den.clamp_min(1e-12)).clamp(0, 1)
        past_norm = s["z"].abs().square().sum(-1).sqrt()
        now_norm = amp.square().sum(-1).sqrt()
        alignment = ((s["z"] * z.conj()).real.sum(-1) /
                     (past_norm * now_norm).clamp_min(1e-12)).clamp(-1, 1)
        current_kernel = (self.cos_beta * z.real).sum(-1)
        mean_kernel = (self.cos_beta * s["z"].real).sum(-1)
        current_g = (target[self.bi, self.hi, self.targets, self.sources] +
                     target[self.bi, self.hi, self.sources, self.targets]) / 2
        d = actual - current_g
        features = [s["unit"].abs().mean(-1),
                    coh(s["z"].abs().sum(-1), s["amp"].sum(-1)),
                    coh(s["weighted"].abs().sum(-1), s["weighted_abs"].sum(-1)),
                    coh((self.cos_beta * s["weighted"].real).abs().sum(-1), s["sym_abs"].sum(-1)),
                    coh((self.cos_beta * s["weighted"].real).sum(-1).abs(), s["sym_edge_abs"]),
                    coh(s["agree"].abs(), s["agree_abs"]), agree, s["agree"], alignment,
                    current_kernel, mean_kernel]
        return features, d, vn, actual, current_g


def rank_auc(x, y):
    """ROC AUC with tied ranks; no optional analysis dependencies."""
    y = np.asarray(y, bool)
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    _, inv, count = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(count)
    ranks = (ends - count + 1 + ends) / 2
    return float((ranks[inv][y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def summarize(arrays, split_at):
    features, effect = arrays["features"], arrays["loss_effect"]
    # One-step effects too small relative to FP32 arithmetic are excluded.
    usable = ((arrays["relative_transport"] > 1e-5) & (abs(effect) > 1e-6) &
              ~arrays["target_previous_correct"])
    result = {"selection": "previously wrong target; relative transport >1e-5; |loss change| >1e-6",
              "positive_effect": "removing the history correction raises next-token Stablemax loss",
              "features": {}}
    n = effect.shape[1]
    cohorts = {"exploratory": (0, min(split_at, n)), "comparison": (min(split_at, n), n)}
    for name, (left, right) in cohorts.items():
        mask = usable[:, left:right]
        result[name] = {"puzzles": right-left, "selected_edges_times": int(mask.sum()),
                        "positive_effect_fraction": float((effect[:, left:right][mask] > 0).mean()) if mask.any() else None}
    for fi, feature in enumerate(FEATURES):
        row = {}
        for name, (left, right) in cohorts.items():
            mask = usable[:, left:right]
            x = features[:, left:right, ..., fi][mask]
            y = effect[:, left:right][mask] > 0
            if not len(x):
                row[name] = None
                continue
            aucs = []
            for p in range(left, right):
                pmask = usable[:, p]
                py = effect[:, p][pmask] > 0
                if py.sum() >= 10 and (~py).sum() >= 10:
                    aucs.append(rank_auc(features[:, p, ..., fi][pmask], py))
            by_head = []
            for head in range(effect.shape[2]):
                hm = usable[:, left:right, head]
                by_head.append(rank_auc(features[:, left:right, head, :, fi][hm], effect[:, left:right, head][hm] > 0))
            row[name] = {"pooled_auc": rank_auc(x, y),
                         "puzzle_mean_auc": float(np.mean(aucs)) if aucs else None,
                         "puzzles_with_both_signs": len(aucs), "head_auc": by_head,
                         "helpful_median": float(np.median(x[y])) if y.any() else None,
                         "harmful_median": float(np.median(x[~y])) if (~y).any() else None}
        result["features"][feature] = row
    result["robustness"] = {}
    base = (arrays["relative_transport"] > 1e-5) & ~arrays["target_previous_correct"]
    selected_features = (0, 1, 2, 4)
    for name, (left, right) in cohorts.items():
        def describe(mask):
            y = effect[:, left:right][mask] > 0
            return {"edges_times": int(mask.sum()), "pooled_auc": {
                FEATURES[fi]: rank_auc(features[:, left:right, ..., fi][mask], y)
                if y.any() and (~y).any() else None for fi in selected_features}}

        threshold_rows = {}
        for threshold in (1e-6, 1e-5, 1e-4, 1e-3):
            mask = base[:, left:right] & (abs(effect[:, left:right]) > threshold)
            threshold_rows[str(threshold)] = describe(mask)
        source_rows = {}
        groups = {
            "clue": arrays["source_is_clue"],
            "correct_blank": ~arrays["source_is_clue"] & arrays["source_previous_correct"],
            "wrong_blank": ~arrays["source_is_clue"] & ~arrays["source_previous_correct"],
        }
        for group, group_mask in groups.items():
            mask = usable[:, left:right] & group_mask[:, left:right]
            source_rows[group] = describe(mask)
        flips = arrays["error_effect"][:, left:right][base[:, left:right]]
        result["robustness"][name] = {
            "loss_effect_thresholds": threshold_rows, "source_subsets": source_rows,
            "changed_target_correctness": {
                "helpful": int((flips > 0).sum()), "harmful": int((flips < 0).sum()),
                "note": "Too few events to infer final-answer discrimination."},
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--pairs-per-head", type=int, default=4)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--split-at", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    ap.add_argument("--out", default="runs/phase_coherence_v11")
    args = ap.parse_args()
    assert 0 < args.n <= 2048 and args.segs > 16 and args.stride > 0 and args.pairs_per_head > 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, _ = load_lt(args.checkpoint, mod=train, batch_size=args.n, loops=args.segs+1, amp=False)
    inputs, labels, batch = load_data(n=args.n)
    b = CorrectionBlocks(model, batch)
    pairs = PairHistory(b, batch["inputs"].cpu().numpy(), args.pairs_per_head, args.seed)
    h = b.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = None
    rows, correct, times = {}, [], []
    label = batch["labels"][pairs.bi, pairs.targets]
    for k in range(1, args.segs * 8 + 1):
        sample = k > 128 and k % args.stride == 0
        if sample:
            old_logits = b.inner.w_cls(h)
            old_pred = old_logits.argmax(-1)
            probs = train.s(old_logits)
            confidence = (probs / probs.sum(-1, keepdim=True)).amax(-1)
        parts = b.components(h, w)
        q, a, v, g, wn = parts
        eff = (1-b.lam)*a + b.lam*wn
        o = torch.einsum("bhtn,bnhc->bthc", eff, v)
        pre = q + torch.einsum("bthc,hcd->btd", o, b.layer.w_sh)
        hn = b.inner.phi(pre)
        features, d, source_v, actual, current_g = pairs.step(parts)
        if k == 1:
            torch.testing.assert_close(hn, b.read(parts, "normal"), atol=0, rtol=0)
        if sample:
            delta = b.lam[None, :, 0, 0, None, None] * d[..., None] * torch.einsum("bhlc,hcd->bhld", source_v, b.layer.w_sh)
            target_pre = pre[pairs.bi, pairs.targets]
            cf = b.inner.phi(target_pre - delta)
            pair_h = hn[pairs.bi, pairs.targets]
            current_logits, cf_logits = b.inner.w_cls(pair_h), b.inner.w_cls(cf)
            loss_effect = (train.stablemax_cross_entropy(cf_logits, label) -
                           train.stablemax_cross_entropy(current_logits, label)).float()
            relative = delta.norm(dim=-1) / target_pre.norm(dim=-1).clamp_min(1e-12)
            features += [confidence[pairs.bi, pairs.sources], confidence[pairs.bi, pairs.targets],
                         d.abs() / (actual.abs() + current_g.abs()).clamp_min(1e-12)]
            now = {
                "features": torch.stack(features, -1), "loss_effect": loss_effect,
                "relative_transport": relative,
                "error_effect": (cf_logits.argmax(-1) != label).float() - (current_logits.argmax(-1) != label).float(),
                "target_previous_correct": old_pred[pairs.bi, pairs.targets] == label,
                "source_previous_correct": old_pred[pairs.bi, pairs.sources] == batch["labels"][pairs.bi, pairs.sources],
                "source_is_clue": batch["inputs"][pairs.bi, pairs.sources] != 1,
            }
            for name, value in now.items():
                rows.setdefault(name, []).append(value.cpu().numpy())
            times.append(k)
        h, w = hn, wn
        if k % 8 == 0:
            correct.append((b.inner.w_cls(h).argmax(-1) == batch["labels"]).all(-1).cpu().numpy())
        if k % 256 == 0:
            print("phase history seg", k//8, "reconstruction max error", pairs.max_reconstruction_error, flush=True)
    arrays = {name: np.stack(values) for name, values in rows.items()}
    np.savez_compressed(out / "edges.npz", times=times, targets=pairs.targets.cpu().numpy(),
                        sources=pairs.sources.cpu().numpy(), correct=np.stack(correct), **arrays)
    report = summarize(arrays, args.split_at)
    report.update(args=vars(args), feature_order=FEATURES,
                  precision="FP32; autocast/TF32 off; packed checkpoint",
                  max_symmetric_memory_reconstruction_error=pairs.max_reconstruction_error,
                  exact_by_seg=np.stack(correct).sum(-1).tolist())
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    for name, row in report["features"].items():
        print(name, {cohort: v["puzzle_mean_auc"] if v else None for cohort, v in row.items()}, flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
