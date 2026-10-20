"""Per-group evaluation of an lt.train checkpoint on the torus-toy exports (toy_g/export_lt.py).

    python -m toy_g.eval_lt CHECKPOINT [--data NPZ] [--weights ema raw] [--split test|train] [--batch N] [--fp32]
                            [--device cuda|cpu] [--out result.json] [--pred-out pred.npz]

Protocol of the harness evaluation (lt/train.py evaluate -> run_segments): the model of
lt.analysis.common.load_model with EMA or raw weights, a fresh carry per puzzle, the checkpoint's `loops` segments
(no lane is replaced), the last segment's prediction, batches of the evaluation batch size (global_batch_size) in
file order. ID runs read test_task (train_task with --split train) as puzzle_identifiers (zeros when the run had
task_ids_as_identifier off, as the harness does); ctx runs (ignore_label0) score the query grid only: cells whose raw
label is 0 count nowhere. BF16 autocast on a BF16 GPU as in the training-time evaluation (--fp32: FP32 without
autocast, the analysis default).

Exact = all 16 scored cells right (the whole output grid; the query grid in context); cell = mean over the 16 scored
cells. Reported per test_group (train_group with --split train: 0 train task, 1 held-out (t, c) combination, 2 task of
a held-out transformation) and overall, with Wilson 95% intervals for the exact rate; also the exact count after each
segment. The overall sums are checked against lt.train.run_segments on the same batches. --pred-out saves the last
segment's predicted outputs as raw colours [N, 4, 4] (-1 for a token that is not a colour) per weights: for the test
split, rows are in task order, the input of toy_g.audit.report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from lt.analysis.common import load_model
from lt.data import eval_batches, load_data, load_task_ids
from lt.train import run_segments

GROUP_NAMES = ('train', 'heldout_combo', 'heldout_t')     # toy_g.data.GROUP_NAMES
CELLS = 16                                                 # one 4x4 output grid
COLOUR_TOKEN0 = 2                                          # token = raw colour + 2 (npz colour + 1, lt.data + 1)


def wilson(successes, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@torch.no_grad()
def predict(model, x, y, segs, batch, device, ids=None, ignore0=False):
    """lt.train.run_segments (one process) with per-puzzle results: cell accuracy and exact flags [segs, N] and the
    last segment's predicted tokens of the scored grid (the last 16 cells) [N, 16]."""
    n_all = len(x)
    acc = np.zeros((segs, n_all))
    exact = np.zeros((segs, n_all), dtype=bool)
    tokens = np.zeros((n_all, CELLS), dtype=np.int64)
    loops, training = model.config.loops, model.training
    model.config.loops = segs + 1                              # no lane is replaced during the run
    model.eval()
    try:
        start = 0
        for b in eval_batches(x, y, batch, 0, 1, ids):
            b = {k: v.to(device) for k, v in b.items()}
            labels = b['labels']
            n = len(labels)
            keep = labels != 1 if ignore0 else torch.ones_like(labels, dtype=torch.bool)   # raw label 0 -> token 1
            if not (bool((keep.sum(-1) == CELLS).all()) and bool(keep[:, -CELLS:].all())):
                raise ValueError('expected the 16 cells of the last grid to be the scored cells')
            carry = model.initial_carry(b)
            for s in range(segs):
                carry, logits = model(carry, b)
                pred = logits.argmax(-1)
                ok = pred == labels
                acc[s, start:start + n] = ((ok & keep).sum(-1).double() / keep.sum(-1).clamp_min(1)).cpu().numpy()
                exact[s, start:start + n] = (ok | ~keep).all(-1).cpu().numpy()
            tokens[start:start + n] = pred[:, -CELLS:].cpu().numpy()
            start += n
    finally:
        model.config.loops = loops
        model.train(training)
    return acc, exact, tokens


def tokens_to_colours(tokens):
    colours = tokens - COLOUR_TOKEN0
    return np.where((colours >= 0) & (colours <= 4), colours, -1).reshape(-1, 4, 4).astype(np.int8)


def summarize(acc, exact, groups):
    segs = acc.shape[0]
    rows = {}
    for name, sel in [(f'{g} {n}', groups == g) for g, n in enumerate(GROUP_NAMES)] + [('all', np.ones_like(groups, bool))]:
        n, e = int(sel.sum()), int(exact[-1, sel].sum())
        lo, hi = wilson(e, n)
        rows[name] = dict(n=n, exact=e, exact_rate=e / max(n, 1), exact_ci95=[lo, hi],
                          cell=float(acc[-1, sel].mean()) if n else float('nan'),
                          exact_per_segment=[int(exact[s, sel].sum()) for s in range(segs)])
    return rows


def evaluate_checkpoint(checkpoint, data=None, weights=('ema', 'raw'), split='test', batch=0, fp32=False,
                        device=None, verbose=True):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    step = int(torch.load(checkpoint, map_location='cpu', weights_only=False)['step'])   # trusted files only
    result = dict(checkpoint=str(checkpoint), step=step, split=split, weights={})
    preds = {}
    for w in weights:
        began = time.monotonic()
        model, cfg = load_model(checkpoint, device, weights=w, kernel='auto', amp=not fp32)
        torch.set_float32_matmul_precision(cfg.get('matmul_precision', 'highest'))
        path = data or cfg.get('data_npz')
        if not path:
            raise FileNotFoundError("the checkpoint's data_npz is missing; pass --data")
        tr_x, tr_y, te_x, te_y, path, _ = load_data(path, test_size=10 ** 9)
        fingerprint = load_data(path, cfg.get('test_size', 2048))[5]
        tr_ids, te_ids = load_task_ids(path, test_size=10 ** 9)
        with np.load(path, allow_pickle=False) as z:
            groups = z[f'{split}_group'].astype(np.int64)
        x, y, ids = (te_x, te_y, te_ids) if split == 'test' else (tr_x, tr_y, tr_ids)
        if ids is not None and not cfg.get('task_ids_as_identifier', True):
            ids = np.zeros_like(ids)
        if len(groups) != len(x):
            raise ValueError(f'{split}_group has {len(groups)} rows for {len(x)} samples')
        segs, ignore0 = int(cfg['loops']), bool(cfg.get('ignore_label0', False))
        bs = batch or cfg.get('eval_batch_size') or cfg['global_batch_size']
        acc, exact, tokens = predict(model, x, y, segs, bs, device, ids, ignore0)
        sums, count, _ = run_segments(model, x, y, segs, bs, 0, 1, device, ids=ids, ignore0=ignore0)
        check = dict(exact_run_segments=int(sums[1, -1]), exact=int(exact[-1].sum()),
                     cell_sum_run_segments=float(sums[0, -1]), cell_sum=float(acc[-1].sum()), count=int(count))
        check['ok'] = (check['exact'] == check['exact_run_segments'] and check['count'] == len(x)
                       and abs(check['cell_sum'] - check['cell_sum_run_segments']) < 1e-6)
        amp = model._amp(device)
        result.update(data=str(path), data_fingerprint_matches=fingerprint == cfg.get('data_fingerprint'),
                      segments=segs, ignore_label0=ignore0, batch=bs, bf16_autocast=amp, device=str(device),
                      identifiers='zeros' if ids is None or not ids.any() else f'{split}_task',
                      hidden_size=cfg.get('hidden_size'), puzzle_emb_ndim=cfg.get('puzzle_emb_ndim'))
        result['weights'][w] = dict(groups=summarize(acc, exact, groups), check=check,
                                    seconds=round(time.monotonic() - began, 1))
        preds[w] = tokens_to_colours(tokens)
        if verbose and not check['ok']:
            print(f'WARNING ({w}): per-puzzle results differ from lt.train.run_segments: {check}', flush=True)
    if verbose:
        print_table(result)
    return result, preds


def print_table(result):
    head = (f"{Path(result['checkpoint']).parent.name}/{Path(result['checkpoint']).name} step {result['step']} | "
            f"{result['split']} split of {Path(result['data']).name} | {result['segments']} segments, fresh carry | "
            f"{'BF16 autocast' if result['bf16_autocast'] else 'FP32'} | identifiers: {result['identifiers']}"
            + ('' if result['data_fingerprint_matches'] else ' | WARNING: data fingerprint differs from training'))
    print(head)
    print(f"{'weights':8s}{'group':18s}{'n':>6s}{'exact':>7s}{'exact %':>9s}  {'95% CI':14s}{'cell %':>8s}"
          f"  {'exact by segment':s}")
    for w, r in result['weights'].items():
        for name, g in r['groups'].items():
            if not g['n']:
                continue
            lo, hi = g['exact_ci95']
            print(f"{w:8s}{name:18s}{g['n']:6d}{g['exact']:7d}{100 * g['exact_rate']:9.2f}  "
                  f"[{100 * lo:5.1f}, {100 * hi:5.1f}] {100 * g['cell']:8.2f}  {g['exact_per_segment']}")
        print(f"{'':8s}check vs lt.train.run_segments: {'ok' if r['check']['ok'] else 'MISMATCH ' + str(r['check'])}"
              f" ({r['seconds']} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--data', help="NPZ of toy_g/export_lt.py (default: the checkpoint's data_npz)")
    ap.add_argument('--weights', nargs='+', choices=('ema', 'raw'), default=['ema', 'raw'])
    ap.add_argument('--split', choices=('test', 'train'), default='test')
    ap.add_argument('--batch', type=int, default=0, help='evaluation batch (0: the run\'s evaluation batch)')
    ap.add_argument('--fp32', action='store_true', help='FP32 without autocast (default: BF16 autocast as training)')
    ap.add_argument('--device')
    ap.add_argument('--out', help='JSON result')
    ap.add_argument('--pred-out', help='NPZ of predicted raw colours per weights (pred_ema, pred_raw) and groups')
    opt = ap.parse_args(argv)
    result, preds = evaluate_checkpoint(opt.checkpoint, opt.data, tuple(opt.weights), opt.split, opt.batch, opt.fp32,
                                        opt.device)
    if opt.out:
        Path(opt.out).write_text(json.dumps(result, indent=1) + '\n')
    if opt.pred_out:
        with np.load(result['data'], allow_pickle=False) as z:
            groups = z[f'{opt.split}_group']
        np.savez_compressed(opt.pred_out, group=groups, **{f'pred_{w}': p for w, p in preds.items()})
    return result


if __name__ == '__main__':
    main()
