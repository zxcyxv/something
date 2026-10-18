"""Segment extrapolation: run past the 16-segment training horizon with the carry kept.

    python -m lt.analysis.extrapolate CHECKPOINT [--n 512] [--segs 128] [--split test|train] [--batch 512] [--out f.json]

EMA weights, fused kernel where available. Error-reduction rate = (best exact up to SEGS - exact at seg 16) /
(n - exact at seg 16). With --split train the original (unaugmented) training puzzles are used; the overfitting gap
is measured by running both splits with the same weights and conditions.
Reference (run A, 512 held-out): 25k 178 -> 190 (3.6%), 41k 243 -> 266 (8.6%), 50.8k 267 -> 288 (8.6%).
"""
import argparse
import json
import time

import torch

from ..data import load_data, load_task_ids
from ..train import run_segments
from .common import load_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--segs', type=int, default=128)
    ap.add_argument('--split', choices=('test', 'train'), default='test')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    ap.add_argument('--out')
    opt = ap.parse_args(argv)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(opt.checkpoint, device, weights=opt.weights, kernel='auto', amp=True)
    tr_x, tr_y, x, y, path, _ = load_data(cfg.get('data_npz'))
    tr_ids, ids = load_task_ids(path)
    if opt.split == 'train':
        x, y, ids = tr_x, tr_y, tr_ids
    x, y = x[:opt.n], y[:opt.n]
    ids = None if ids is None else ids[:opt.n]
    began = time.monotonic()
    sums, count, _ = run_segments(model, x, y, opt.segs, opt.batch, 0, 1, device, ids=ids,
                                  ignore0=cfg.get('ignore_label0', False))
    n, exact = int(count), [int(e) for e in sums[1].tolist()]
    horizon = model.config.loops
    best = max(exact)
    result = dict(checkpoint=str(opt.checkpoint), split=opt.split, weights=opt.weights, n=n, segs=opt.segs,
                  train_segment=exact[horizon - 1], best=best, best_segment=exact.index(best) + 1, last=exact[-1],
                  error_reduction=(best - exact[horizon - 1]) / max(n - exact[horizon - 1], 1),
                  exact=exact, accuracy=(sums[0] / n).tolist(), seconds=time.monotonic() - began)
    print(f"seg{horizon} {result['train_segment']}/{n} -> best {best} (seg {result['best_segment']}), last {exact[-1]}; "
          f"error reduction {100 * result['error_reduction']:.1f}%  ({result['seconds']:.0f}s)")
    print('exact at segments 1,2,4,8,16,32,64,128:', [exact[i - 1] for i in (1, 2, 4, 8, 16, 32, 64, 128) if i <= opt.segs])
    if opt.out:
        with open(opt.out, 'w') as f:
            json.dump(result, f, indent=1)
    return result


if __name__ == '__main__':
    main()
