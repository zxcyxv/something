"""Inference-time interventions on the STDP term: what the order part O and the latency part L each do
(architecture.md, section 5).

    python -m lt.analysis.interventions CHECKPOINT [--n 512] [--blocks 128] [--flip-at 32] [--batch 256] [--out f.json]

Same held-out puzzles, EMA weights, torch path, no grad. Variants from block 1:
  baseline   Hebb + alpha (O + L)          alpha0     Hebb
  L-only     Hebb + alpha L                O-only     Hebb + alpha O
  omega-     omegaT -> -omegaT (= Hebb + alpha (L - O))    omega0   omegaT = 0
and after `--flip-at` blocks run as trained: flip (Hebb + alpha (L - O)), dropO (Hebb + alpha L), alpha0.
Per variant: puzzles solved (all cells right from some block on), exact at block 16, and the decision block of
each cell (argmax constant from there on) against its L0 logical depth (blocks per round, Spearman).
Reference (run A, step 50,778, 512 puzzles): baseline 264 / 1.06 blocks per round, alpha0 225, L-only 263,
O-only 222 / 1.03, omega- 78, omega0 269 / 1.14; after 32 blocks flip 94 (deepest cells undone first).
"""
import argparse
import json
import time

import numpy as np
import torch

from ..data import eval_batches, load_data
from . import depth as sd
from .common import OrderLatencyProbe, load_model, run_blocks, spearman, suffix_block

VARIANTS = {  # name: (probe mode, start block, omegaT factor)
    'baseline': (None, 1, 1.0), 'alpha0': ('none', 1, 1.0), 'L-only': ('latency', 1, 1.0),
    'O-only': ('order', 1, 1.0), 'omega-': (None, 1, -1.0), 'omega0': (None, 1, 0.0),
    'flip@': ('reversed', None, 1.0), 'dropO@': ('latency', None, 1.0), 'alpha0@': ('none', None, 1.0)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--blocks', type=int, default=128)
    ap.add_argument('--flip-at', type=int, default=32)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--out')
    opt = ap.parse_args(argv)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(opt.checkpoint, device)
    omega, lam = model.config.block_omega, model.config.trace_lambda
    _, _, x_np, y_np, _, _ = load_data(cfg.get('data_npz'))
    x_np, y_np = x_np[:opt.n], y_np[:opt.n]
    x, y = x_np.reshape(-1, 81).astype(int), y_np.reshape(-1, 81).astype(int)
    depth, _ = sd.propagate(x, y, lookahead=False)
    labels = torch.from_numpy(y + 1)
    results = {}
    for name in opt.variants:
        mode, start, factor = VARIANTS[name]
        start = opt.flip_at + 1 if start is None else start
        began = time.monotonic()
        model.config.block_omega = omega * factor
        preds = []
        for batch in eval_batches(x_np, y_np, opt.batch):
            b = {k: v.to(device) for k, v in batch.items()}
            model.probe = None if mode is None else OrderLatencyProbe(omega * factor, lam, mode, start)
            preds.append(run_blocks(model, b, opt.blocks).cpu())
        model.probe, model.config.block_omega = None, omega
        preds = torch.cat(preds, 1)                                        # [blocks, N, 81]
        right = (preds == labels[None]).all(-1)
        solved = (suffix_block(right) <= opt.blocks).numpy()
        decision = suffix_block((preds == preds[-1:]).flatten(1)).view(len(x), 81).numpy()
        mask = solved[:, None] & (depth > 0)
        slope = float(np.polyfit(depth[mask], decision[mask], 1)[0]) if mask.sum() > 2 else None
        res = dict(solved=int(solved.sum()), exact_block16=int(right[15].sum()) if opt.blocks >= 16 else None,
                   exact_last=int(right[-1].sum()), blocks_per_round=slope,
                   spearman_depth_decision=spearman(depth[mask], decision[mask]) if mask.sum() > 2 else None,
                   seconds=time.monotonic() - began)
        if name.endswith('@'):
            res['flip_at'] = opt.flip_at
            res['solved_before_flip'] = int(right[opt.flip_at - 1].sum())
        results[name] = res
        print(f"{name:9s} solved {res['solved']:4d}  block16 {res['exact_block16']}  last {res['exact_last']:4d}  "
              f"blocks/round {slope if slope is None else round(slope, 2)}  ({res['seconds']:.0f}s)", flush=True)
    out = dict(checkpoint=str(opt.checkpoint), n=len(x), blocks=opt.blocks, variants=results)
    if opt.out:
        with open(opt.out, 'w') as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == '__main__':
    main()
