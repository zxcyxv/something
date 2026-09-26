"""Join whole Sudoku trajectories to exact, all-head logit arithmetic.

Reads existing FP32 captures; does not rerun or intervene on the model.
All label transitions in T-16..T are analyzed, including regressions.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# Connections inside the row-3 exchange and the overlapping column/box exchanges.
# Each tuple is (source, destination, digit supported, digit contrasted).
TRACKS = {
    58: [(29, 20, 7, 9), (20, 23, 8, 7), (14, 23, 8, 7),
         (23, 26, 6, 8), (25, 26, 6, 8), (13, 26, 6, 8)],
    209: [(38, 37, 7, 3), (74, 38, 3, 7), (37, 19, 3, 7),
          (18, 19, 3, 7), (19, 18, 7, 3)],
    230: [(1, 19, 9, 4), (26, 19, 9, 4), (46, 19, 9, 4),
          (19, 46, 2, 9), (46, 28, 1, 2), (35, 28, 1, 2)]
}


def rc(i):
    return f'r{i // 9 + 1}c{i % 9 + 1}'


def geometry():
    ids = np.arange(81)
    r, c = ids // 9, ids % 9
    b = (r // 3) * 3 + c // 3
    peers = ((r[:, None] == r) | (c[:, None] == c) | (b[:, None] == b)) & ~np.eye(81, dtype=bool)
    units = [(f'{name}{k+1}', np.flatnonzero(group == k))
             for name, group in [('row', r), ('column', c), ('box', b)] for k in range(9)]
    return peers, units


def cycles(prediction, gold, units):
    found = {}
    for name, ids in units:
        owner = {int(gold[i]): int(i) for i in ids}
        mapping = {int(i): owner.get(int(prediction[i])) for i in ids}
        for start in ids:
            path = []
            cur = int(start)
            while cur is not None and cur not in path:
                path.append(cur)
                cur = mapping[cur]
            if cur is None:
                continue
            cycle = path[path.index(cur):]
            if len(cycle) < 2:
                continue
            canonical = min(tuple(cycle[j:] + cycle[:j]) for j in range(len(cycle)))
            if canonical not in found:
                found[canonical] = {'cells': list(canonical), 'cell_names': list(map(rc, canonical)),
                                    'predictions': [int(prediction[i]) for i in canonical],
                                    'gold': [int(gold[i]) for i in canonical], 'units': []}
            if name not in found[canonical]['units']:
                found[canonical]['units'].append(name)
    return sorted(found.values(), key=lambda x: (-len(x['cells']), x['cells']))


def analyze(root, puzzle):
    directory = root / f'puzzle_{puzzle}'
    meta = json.loads((directory / 'metadata.json').read_text())
    with np.load(directory / 'states_fp32.npz') as ar:
        data = {name: ar[name] for name in ar.files}
    t = meta['blocks'].index(meta['event']['stable_complete_block'])
    gold, given = np.array(meta['gold']), np.array(meta['givens'])
    pred = data['predictions']
    before = data['predictions_before']
    psi, beta, agree, w = data['matrices']
    signals = data['digit_signals']
    lam = np.array(meta['head_parameters']['lambda'], dtype=np.float64)[:, None]
    gain = np.array(meta['head_parameters']['gain'], dtype=np.float64)[:, None]
    peers, units = geometry()
    stable = []
    for c in range(81):
        wrong = np.flatnonzero(pred[:, c] != gold[c])
        stable.append(int(wrong[-1] + 1 - t) if len(wrong) else int(-t))
    results = {'puzzle': puzzle, 'T': meta['event']['stable_complete_block'],
               'gold': gold.tolist(), 'givens': given.tolist(),
               'cycles_at_Tminus16': cycles(pred[t-16], gold, units),
               'cells': [], 'blocks': [], 'events': [], 'max_reconstruction_error': 0.,
               'max_temporal_decomposition_error': 0.,
               'method': 'Exact normal-trajectory arithmetic, no causal intervention. '
                         'Temporal products split symmetrically: delta(J*S)=delta(J)*mean(S)+mean(J)*delta(S). '
                         'Gold labels describe states; they are not inputs to the forward pass.'}
    union = np.flatnonzero(np.any(pred[t-16:t+1] != gold, axis=0))
    for c in union:
        results['cells'].append({'cell': int(c), 'rc': rc(c), 'gold': int(gold[c]),
                                'at_Tminus16': int(pred[t-16, c]), 'stable_offset': stable[c],
                                'trajectory': pred[t-16:t+1, c].tolist()})

    def components(f, i, new, old):
        s = signals[f, :, :, new-1].astype(np.float64) - signals[f, :, :, old-1].astype(np.float64)
        ap = (1-lam) * psi[f, :, i, :]
        mw = lam * w[f, :, i, :]
        denom = float(data['denominators'][f, i])
        ps = ap * s / denom
        mem = mw * s / denom
        q = float(data['q_logits'][f, i, new-1] - data['q_logits'][f, i, old-1])
        bias = meta['classifier_bias'][new-1] - meta['classifier_bias'][old-1]
        actual = float(data['logits'][f, i, new+1] - data['logits'][f, i, old+1])
        reconstructed = q + ps.sum() + mem.sum() + bias
        err = abs(reconstructed - actual)
        results['max_reconstruction_error'] = max(results['max_reconstruction_error'], err)
        assert err < 1e-4, (puzzle, f, i, err)
        return {'s': s, 'ap': ap, 'mw': mw, 'j': ap+mw, 'psi': ps, 'memory': mem,
                'read': ps+mem, 'denom': denom, 'q': q, 'bias': bias, 'margin': actual}

    for f in range(t-16, t+1):
        changed = np.flatnonzero(pred[f] != before[f])
        results['blocks'].append({'offset': f-t, 'block': meta['blocks'][f],
                                  'prediction': pred[f].tolist(),
                                  'metric': meta['metrics'][f],
                                  'cycles': cycles(pred[f], gold, units),
                                  'changes': [{'cell': int(i), 'rc': rc(i), 'old': int(before[f, i]),
                                               'new': int(pred[f, i]), 'gold': int(gold[i])} for i in changed]})
        # T-16 is the starting board; analyze the 16 following transitions.
        if f == t-16:
            continue
        for i in changed:
            old, new = int(before[f, i]), int(pred[f, i])
            if min(old, new) < 1:
                continue
            cur = components(f, i, new, old)
            prev = components(f-1, i, new, old)
            sbar = (cur['s'] + prev['s']) / 2
            invdbar = (1/cur['denom'] + 1/prev['denom']) / 2
            dpsi = (cur['ap'] - prev['ap']) * sbar * invdbar
            dw = (cur['mw'] - prev['mw']) * sbar * invdbar
            dv = (cur['j'] + prev['j']) / 2 * (cur['s'] - prev['s']) * invdbar
            qcur, qprev = cur['q'] * cur['denom'], prev['q'] * prev['denom']
            pcur = float(np.sum(cur['j'] * cur['s']))
            pprev = float(np.sum(prev['j'] * prev['s']))
            dq = (qcur-qprev) * invdbar
            dnorm = (qcur+qprev+pcur+pprev)/2 * (1/cur['denom']-1/prev['denom'])
            decomposition = {'prepared_q': dq, 'source_values': float(dv.sum()),
                             'psi_coefficients': float(dpsi.sum()), 'W_coefficients': float(dw.sum()),
                             'normalization': dnorm}
            delta = cur['margin'] - prev['margin']
            error = abs(sum(decomposition.values()) - delta)
            results['max_temporal_decomposition_error'] = max(results['max_temporal_decomposition_error'], error)
            assert error < 2e-4, (puzzle, f, i, error)
            instant_g = gain * beta[f, :, i, :] * agree[f, :, i, :]
            hist = lam * (w[f, :, i, :] - instant_g) * cur['s'] / cur['denom']
            sources = []
            for j in range(81):
                category = 'self' if j == i else ('peer_' if peers[i, j] else 'nonpeer_') + (
                    'given' if given[j] else 'correct' if before[f, j] == gold[j] else 'wrong')
                sources.append({'cell': j, 'rc': rc(j), 'category': category,
                                'before': int(before[f, j]), 'after': int(pred[f, j]), 'gold': int(gold[j]),
                                'stable_offset': stable[j],
                                'psi': float(cur['psi'][:, j].sum()), 'memory': float(cur['memory'][:, j].sum()),
                                'read': float(cur['read'][:, j].sum()), 'previous_read': float(prev['read'][:, j].sum()),
                                'delta_read': float((cur['read'][:, j]-prev['read'][:, j]).sum()),
                                'delta_psi_coefficients': float(dpsi[:, j].sum()),
                                'delta_W_coefficients': float(dw[:, j].sum()), 'delta_values': float(dv[:, j].sum()),
                                'history_difference_vs_G': float(hist[:, j].sum()),
                                'heads': [{'head': h+1, 'psi': float(psi[f, h, i, j]),
                                           'beta': float(beta[f, h, i, j]), 'agree': float(agree[f, h, i, j]),
                                           'G': float(instant_g[h, j]), 'W': float(w[f, h, i, j]),
                                           'J': float(cur['j'][h, j]),
                                           'unit_new': float(signals[f, h, j, new-1]/cur['denom']),
                                           'unit_old': float(signals[f, h, j, old-1]/cur['denom']),
                                           'unit_margin': float(cur['s'][h, j]/cur['denom']),
                                           'psi_read': float(cur['psi'][h, j]),
                                           'memory_read': float(cur['memory'][h, j]),
                                           'read': float(cur['read'][h, j]),
                                           'previous_read': float(prev['read'][h, j]),
                                           'delta_values': float(dv[h, j]),
                                           'delta_psi_coefficients': float(dpsi[h, j]),
                                           'delta_W_coefficients': float(dw[h, j])} for h in range(8)]})
            category_totals = {cat: {key: float(sum(s[key] for s in sources if s['category'] == cat))
                                    for key in ['psi', 'memory', 'read', 'delta_read', 'delta_values',
                                                'delta_psi_coefficients', 'delta_W_coefficients', 'history_difference_vs_G']}
                               for cat in sorted({s['category'] for s in sources})}
            event = {'offset': f-t, 'block': meta['blocks'][f], 'cell': int(i), 'rc': rc(i),
                     'old': old, 'new': new, 'gold': int(gold[i]),
                     'kind': 'correction' if new == gold[i] else 'regression' if old == gold[i] else 'wrong_to_wrong',
                     'stable_correction': bool(new == gold[i] and stable[i] == f-t),
                     'logical_context': {
                         'old_digit_peer_cells_before': np.flatnonzero(peers[i] & (before[f] == old)).tolist(),
                         'new_digit_peer_cells_before': np.flatnonzero(peers[i] & (before[f] == new)).tolist(),
                         'new_digit_peer_cells_after': np.flatnonzero(peers[i] & (pred[f] == new)).tolist(),
                         'whole_board_conflicts_before': meta['metrics'][f-1]['conflict_pairs'],
                         'whole_board_conflicts_after': meta['metrics'][f]['conflict_pairs']},
                     'previous_margin': prev['margin'], 'margin': cur['margin'], 'margin_change': delta,
                     'current': {'prepared_q': cur['q'], 'psi_read': float(cur['psi'].sum()),
                                 'memory_read': float(cur['memory'].sum()), 'bias': cur['bias'],
                                 'history_difference_vs_G': float(hist.sum())},
                     'temporal_decomposition': decomposition, 'source_groups': category_totals,
                     'head_totals': [{'head': h+1, 'psi': float(cur['psi'][h].sum()),
                                     'memory': float(cur['memory'][h].sum()), 'read': float(cur['read'][h].sum()),
                                     'delta_values': float(dv[h].sum()), 'delta_psi_coefficients': float(dpsi[h].sum()),
                                     'delta_W_coefficients': float(dw[h].sum())} for h in range(8)],
                     'sources': sources}
            results['events'].append(event)

    track_specs = list(TRACKS[puzzle])
    for cycle in results['cycles_at_Tminus16']:
        cells = cycle['cells']
        for j, i in zip(cells, cells[1:]+cells[:1]):
            spec = (j, i, int(gold[i]), int(pred[t-16, i]))
            if spec not in track_specs:
                track_specs.append(spec)
    results['edge_tracks'] = []
    for j, i, new, old in track_specs:
        track = {'source': j, 'source_rc': rc(j), 'destination': i, 'destination_rc': rc(i),
                 'digits': [new, old], 'frames': []}
        for f in range(t-16, t+5):
            cur = components(f, i, new, old)
            track['frames'].append({'offset': f-t, 'source_before': int(before[f, j]),
                                    'source_after': int(pred[f, j]), 'target_before': int(before[f, i]),
                                    'target_after': int(pred[f, i]), 'target_margin': cur['margin'],
                                    'psi_read': float(cur['psi'][:, j].sum()),
                                    'memory_read': float(cur['memory'][:, j].sum()),
                                    'read': float(cur['read'][:, j].sum()),
                                    'head3': {'J': float(cur['j'][2, j]),
                                              'unit_margin': float(cur['s'][2, j]/cur['denom']),
                                              'read': float(cur['read'][2, j])}})
        results['edge_tracks'].append(track)
    path = root / f'puzzle_{puzzle}_reasoning.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
    corrections = [e for e in results['events'] if e['kind'] == 'correction']
    stable_events = [e for e in corrections if e['stable_correction']]
    summary = {'puzzle': puzzle, 'events': len(results['events']), 'corrections': len(corrections),
               'stable_corrections': len(stable_events),
               'regressions': sum(e['kind'] == 'regression' for e in results['events']),
               'corrections_despite_new_digit_already_at_peer': sum(bool(e['logical_context']['new_digit_peer_cells_before']) for e in corrections),
               'median_absolute_change_components': {
                   key: float(np.median([abs(e['temporal_decomposition'][key]) for e in corrections]))
                   for key in ['prepared_q', 'source_values', 'psi_coefficients', 'W_coefficients', 'normalization']},
               'max_reconstruction_error': results['max_reconstruction_error'],
               'max_temporal_decomposition_error': results['max_temporal_decomposition_error']}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/puzzle_overview_v11'))
    p.add_argument('--puzzles', type=int, nargs='+', default=[58, 209, 230])
    args = p.parse_args()
    summaries = [analyze(args.root, puzzle) for puzzle in args.puzzles]
    (args.root / 'reasoning_validation.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
