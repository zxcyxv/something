"""What changes across a held-out collapse: failure modes and parameter motion, raw vs EMA.

For each checkpoint (raw and EMA weights): held-out predictions after every segment; at the
last segment each puzzle is C (all cells correct), V (every row, column and box is a
permutation of 1..9 but the grid is not the solution) or I (some unit repeats a digit),
with the clue-cell accuracy, blank-cell accuracy and the number of repeated-digit
violations; solved count per segment. Parameters: norm per group, relative raw-EMA distance,
relative change of the raw weights from the previous checkpoint, alpha.

python -m lt.probe_collapse --run runs/<replay> --checkpoints step_50000.pt raw_ema_kept/step_51000.pt ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import train as t
from .analyze_stdp_direct import make_model
from .sudoku_solver import UNITS

GROUPS = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'b_gate_up', 'b_down', 'embed', 'w_cls', 'theta', 'stdp_alpha')


def group_of(name):
    return next((g for g in GROUPS if g in name), 'other')


@torch.no_grad()
def predictions(base, x, y, cfg, device):
    preds = torch.empty(cfg['loops'], len(x), 81, dtype=torch.long)
    start = 0
    for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        carry, n = base.initial_carry(batch), batch['inputs'].shape[0]
        for s in range(cfg['loops']):
            carry, _, _, out, _ = base(carry=carry, batch=batch, return_keys={'preds'})
            preds[s, start:start + n] = out['preds'].cpu()
        start += n
    return preds


def states(pred, x, y):
    """pred: [N,81] tokens; x, y raw digits. Returns C/V/I labels and per-puzzle details."""
    digits = pred.numpy() - 1                                    # tokens 2..10 -> 1..9
    sol, inp = y.reshape(-1, 81), x.reshape(-1, 81)
    given = inp > 0
    correct = digits == sol
    units = np.array(UNITS)                                      # [27, 9]
    vals = digits[:, units]                                      # [N, 27, 9]
    counts = np.stack([(vals == d).sum(-1) for d in range(1, 10)], -1)   # [N, 27, 9]
    violations = np.clip(counts - 1, 0, None).sum((1, 2))
    valid = (counts == 1).all((1, 2))
    solved = correct.all(1)
    label = np.where(solved, 'C', np.where(valid, 'V', 'I'))
    clue_acc = (correct & given).sum(1) / given.sum(1)
    blank_acc = (correct & ~given).sum(1) / (~given).sum(1)
    return label, violations, clue_acc, blank_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    protocol = json.loads((run / 'protocol.json').read_text())
    device = torch.device('cuda')
    rows, prev_raw, base, solved_ref = [], None, None, None
    for path in opt.checkpoints:
        ck = torch.load(path if path.is_absolute() else run / path, map_location='cpu', weights_only=False)
        cfg = dict(ck['cfg'])
        if not Path(cfg['data_npz']).is_file():
            cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
        if base is None:
            torch.set_float32_matmul_precision(protocol['precision'])
            t._resolve_precision(cfg, device)
            _, _, x, y, _, _ = t.load_data(cfg)
            base = make_model(protocol, cfg, device, ck['raw_model_state_dict'])
        raw, ema = ck['raw_model_state_dict'], ck['model_state_dict']
        params = {}
        for name, value in raw.items():
            if not value.is_floating_point() or name not in dict(base.named_parameters()):
                continue
            g = group_of(name)
            r, e = value.double(), ema[name].double()
            p = params.setdefault(g, dict(raw_sq=0., ema_sq=0., diff_sq=0., step_sq=0.))
            p['raw_sq'] += float(r.pow(2).sum())
            p['ema_sq'] += float(e.pow(2).sum())
            p['diff_sq'] += float((r - e).pow(2).sum())
            if prev_raw is not None:
                p['step_sq'] += float((r - prev_raw[name].double()).pow(2).sum())
        param_rows = {g: dict(raw_norm=p['raw_sq'] ** .5, ema_norm=p['ema_sq'] ** .5,
                              raw_ema_rel=(p['diff_sq'] / max(p['raw_sq'], 1e-30)) ** .5,
                              change_from_prev_rel=(p['step_sq'] / max(p['raw_sq'], 1e-30)) ** .5 if prev_raw is not None else None)
                      for g, p in params.items()}
        alpha = dict(raw=raw['model.inner.layers.0.stdp_alpha'].tolist(), ema=ema['model.inner.layers.0.stdp_alpha'].tolist())
        row = dict(checkpoint=str(path), step=int(ck['step']), params=param_rows, alpha=alpha, weights={})
        for which, state in (('ema', ema), ('raw', raw)):     # EMA first: the first EMA is the reference set
            base.load_state_dict(state, strict=True)
            preds = predictions(base, x, y, cfg, device)
            label, viol, clue, blank = states(preds[-1], x, y)
            per_seg = [int((preds[s].numpy() - 1 == y.reshape(-1, 81)).all(1).sum()) for s in range(cfg['loops'])]
            solved = label == 'C'
            if which == 'ema' and solved_ref is None:
                solved_ref = solved
            row['weights'][which] = dict(
                C=int((label == 'C').sum()), V=int((label == 'V').sum()), I=int((label == 'I').sum()),
                V_clue_changed=int(((label == 'V') & (clue < 1)).sum()),
                clue_acc=float(clue.mean()), blank_acc=float(blank.mean()),
                violations_mean_I=float(viol[label == 'I'].mean()) if (label == 'I').any() else 0.,
                solved_per_segment=per_seg,
                solved_also_in_first_ema=int((solved & solved_ref).sum()),
                lost_vs_first_ema=int((~solved & solved_ref).sum()), new_vs_first_ema=int((solved & ~solved_ref).sum()))
            print(json.dumps(dict(step=row['step'], weights=which, **{k: row['weights'][which][k] for k in
                  ('C', 'V', 'I', 'V_clue_changed', 'clue_acc', 'blank_acc', 'violations_mean_I')})), flush=True)
        prev_raw = raw
        rows.append(row)
        opt.out.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
