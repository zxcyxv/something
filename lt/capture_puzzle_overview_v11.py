"""Capture whole-board, all-head forward states around three late solutions.

Uses the original 12-puzzle FP32 batch and validates every replayed prediction.
Archives full-precision matrices. No interventions or altered inference rules.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_late_puzzles import restore
from probe_late_puzzles import EventBlocks, setup, write_json


def topology():
    cells = np.arange(81)
    row, col = cells // 9, cells % 9
    box = (row // 3) * 3 + col // 3
    peer = ((row[:, None] == row) | (col[:, None] == col) |
            (box[:, None] == box)) & ~np.eye(81, dtype=bool)
    units = [np.flatnonzero(group == value).tolist()
             for group in (row, col, box) for value in range(9)]
    return peer, units


def board_metrics(prediction, before, gold, givens, peer, units):
    valid = (prediction >= 1) & (prediction <= 9)
    conflicts = peer & (prediction[:, None] == prediction) & valid[:, None] & valid
    count = int(np.triu(conflicts, 1).sum())
    bad_units = []
    for unit in units:
        digits = prediction[unit]
        counts = np.bincount(digits[(digits >= 1) & (digits <= 9)], minlength=10)
        bad_units.append(int(np.maximum(counts[1:] - 1, 0).sum()))
    wrong = prediction != gold
    reciprocal = np.triu(peer & wrong[:, None] & wrong &
                        (prediction[:, None] == gold[None, :]) &
                        (gold[:, None] == prediction[None, :]) &
                        (gold[:, None] != gold[None, :]), 1)
    return {'wrong_cells': int(wrong.sum()), 'conflict_pairs': count,
            'conflict_cells': int(conflicts.any(1).sum()),
            'given_violations': int(((givens != 0) & (prediction != givens)).sum()),
            'invalid_cells': int((~valid).sum()),
            'changed_cells': int((prediction != before).sum()),
            'row_duplicate_excess': sum(bad_units[:9]),
            'column_duplicate_excess': sum(bad_units[9:18]),
            'box_duplicate_excess': sum(bad_units[18:]),
            'reciprocal_wrong_pairs': np.argwhere(reciprocal).tolist(),
            'unit_duplicate_excess': bad_units}


def capture_case(root, output, event, b, batch, ids, x, y, baseline, before, after):
    puzzle, i, c = event['puzzle'], event['batch_index'], event['cell']
    t = event['stable_complete_block']
    start, stop = max(1, t - before), min(len(baseline) - 1, t + after)
    frames = stop - start + 1
    out = output / f'puzzle_{puzzle}'
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / 'metadata.json'
    archive = out / 'states_fp32.npz'
    if archive.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta['blocks'] == list(range(start, stop + 1)):
            print('REUSE', str(archive), flush=True)
            return meta
        raise RuntimeError(f'{out} already contains a different window; choose another --out.')

    h, w = restore(root, b, start, baseline)
    initial_w = w[i].cpu().numpy().copy()
    field_names = ['psi', 'beta', 'agree', 'w']
    matrices = np.empty((4, frames, b.inner.H, 81, 81), dtype=np.float32)
    signals = np.empty((frames, b.inner.H, 81, 9), dtype=np.float32)
    q_logits = np.empty((frames, 81, 9), dtype=np.float32)
    denominators = np.empty((frames, 81), dtype=np.float32)
    logits_saved = np.empty((frames, 81, batch['labels'].new_tensor(b.inner.config.vocab_size).item()), dtype=np.float32)
    predictions = np.empty((frames, 81), dtype=np.int8)
    predictions_before = np.empty_like(predictions)
    peer, units = topology()
    metrics = []
    cls_weight = b.inner.w_cls.weight[2:11]
    projection = torch.einsum('vd,hcd->hcv', cls_weight, b.layer.w_sh)
    max_error = 0.0
    started = time.monotonic()

    for f, k in enumerate(range(start, stop + 1)):
        p = b.parts(h, w)
        hn, pre, effective = b.read(p)
        logits = b.inner.w_cls(hn)
        pred = logits.argmax(-1).cpu().numpy()
        assert np.array_equal(pred, baseline[k]), f'Baseline replay mismatch at block {k}'
        p_after = pred[i].astype(np.int16) - 1
        p_before = baseline[k - 1, i].astype(np.int16) - 1
        predictions[f], predictions_before[f] = p_after, p_before
        metrics.append(board_metrics(p_after, p_before, y[i], x[i], peer, units))
        matrices[:, f] = torch.stack((p['a'][i], p['window'][i], p['agree'][i], p['w'][i])).cpu().numpy()
        signal = torch.einsum('nhc,hcv->hnv', p['v'][i], projection)
        denominator = torch.sqrt(1 + pre[i].square().sum(-1) / b.inner.d)
        ql = torch.einsum('nd,vd->nv', p['q'][i], cls_weight) / denominator[:, None]
        reconstructed = ql + torch.einsum('hij,hjd->id', effective[i], signal) / denominator[:, None] + b.inner.w_cls.bias[2:11]
        err = float((reconstructed - logits[i, :, 2:11]).abs().max())
        max_error = max(max_error, err)
        assert err < 2e-3, (k, err)
        signals[f] = signal.cpu().numpy()
        denominators[f] = denominator.cpu().numpy()
        q_logits[f] = ql.cpu().numpy()
        logits_saved[f] = logits[i].cpu().numpy()
        h, w = hn, p['w']
        if f % 64 == 0 or k == stop:
            print(json.dumps({'puzzle': puzzle, 'block': k, 'offset': k - t,
                              'wrong': metrics[-1]['wrong_cells'],
                              'conflict_pairs': metrics[-1]['conflict_pairs'],
                              'elapsed_s': round(time.monotonic() - started, 2)}), flush=True)

    references = []
    for cell in range(81):
        history = predictions[:t - start, cell]
        wrong = np.flatnonzero((history != y[i, cell]) & (history >= 1) & (history <= 9))
        if len(wrong):
            references.append(int(history[wrong[-1]]))
        else:
            at_t = logits_saved[t - start, cell, 2:11].copy()
            at_t[y[i, cell] - 1] = -np.inf
            references.append(int(at_t.argmax()) + 1)

    scalars = {name: getattr(b, attr).detach().cpu().numpy().reshape(-1).tolist()
               for name, attr in [('lambda', 'lam'), ('eta', 'eta'), ('gain', 'gain')]}
    metadata = {
        'schema_version': 1, 'event': event, 'puzzle': puzzle, 'blocks': list(range(start, stop + 1)),
        'frames': frames, 'heads': b.inner.H, 'field_names': field_names,
        'givens': x[i].tolist(), 'gold': y[i].tolist(), 'reference_digits': references,
        'head_parameters': scalars, 'classifier_bias': b.inner.w_cls.bias[2:11].cpu().tolist(),
        'psi_parameters_degrees': torch.rad2deg(b.layer.psi).cpu().tolist(),
        'beta_parameters_degrees': torch.rad2deg(b.layer.beta).cpu().tolist(),
        'metrics': metrics, 'normal_predictions_match_all_batch_all_blocks': True,
        'maximum_logit_reconstruction_error': max_error,
        'checkpoint': 'checkpoints/v1.1_step160000.npz',
        'precision': 'FP32, same 12-puzzle batch, autocast and TF32 off',
        'timing': 'Matrices use prepared state q from the incoming h at k-1; W is updated before the read producing h_k. Before/after boards decode h_{k-1}/h_k.',
        'metrics_note': 'Conflict pairs are unique Sudoku-peer pairs with equal valid predicted digits. Gold labels are used only for analysis, never for forward inference.',
        'source_baseline_sha256': hashlib.sha256((root / 'baseline.npz').read_bytes()).hexdigest(),
    }
    print('SAVING_FP32', puzzle, flush=True)
    np.savez_compressed(archive, matrices=matrices, w_initial=initial_w,
                        digit_signals=signals, q_logits=q_logits, denominators=denominators,
                        logits=logits_saved, predictions=predictions, predictions_before=predictions_before)
    write_json(metadata_path, metadata)
    print('SAVED', puzzle, archive.stat().st_size, flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/late_puzzle_probe_v11'))
    parser.add_argument('--out', type=Path, default=Path('runs/puzzle_overview_v11'))
    parser.add_argument('--before', type=int, default=128)
    parser.add_argument('--after', type=int, default=128)
    parser.add_argument('--puzzles', type=int, nargs='+', default=[58, 209, 230])
    args = parser.parse_args()
    assert args.before >= 0 and args.after >= 0
    with np.load(args.root / 'baseline.npz') as data:
        ids, x, y, baseline = (data[key] for key in ('indices', 'X', 'Y', 'predictions'))
    events = json.loads((args.root / 'events.json').read_text())['selected_events']
    model, batch, x2, y2 = setup(ids)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    b = EventBlocks(model, batch)
    args.out.mkdir(parents=True, exist_ok=True)
    for event in events:
        if event['puzzle'] in args.puzzles:
            capture_case(args.root, args.out, event, b, batch, ids, x, y, baseline,
                         args.before, args.after)


if __name__ == '__main__':
    main()
