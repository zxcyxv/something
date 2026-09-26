"""Explain a selected late correction's peer edge, as for r3c6 -> r3c9.

Separate same-state read replacement from the existing 129-block G-read branch.
The mathematical decomposition observes fixed states; it does not ablate phases
and roll out a new model. All normal replay predictions must match the baseline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_late_puzzles import restore
from probe_late_puzzles import EventBlocks, setup, write_json


def kernel_terms(b, p, batch_index, cell, source, head):
    x, y = b.inner.addr(p['q'], b.ab)
    xi, yi = x[batch_index, cell, head].double(), y[batch_index, cell, head].double()
    xj, yj = x[batch_index, source, head].double(), y[batch_index, source, head].double()
    real = xi * xj + yi * yj
    imag = yi * xj - xi * yj
    delta = torch.stack((b.inner.pos_u[cell] - b.inner.pos_u[source],
                         b.inner.pos_w[cell] - b.inner.pos_w[source])).double()
    position_phase = b.layer.theta[head].double() @ delta
    s = real * position_phase.cos() - imag * position_phase.sin()
    t = real * position_phase.sin() + imag * position_phase.cos()
    decay = b.kc[0][head, cell, source].double()
    psi, beta = b.layer.psi[head].double(), b.layer.beta[head].double()
    read_even = decay * s * psi.cos()
    read_odd = -decay * t * psi.sin()
    write_even = decay * s * beta.cos()
    write_odd = -decay * t * beta.sin()
    read, write = read_even + read_odd, write_even + write_odd
    expected_read = float(p['a'][batch_index, head, cell, source])
    expected_write = float(p['window'][batch_index, head, cell, source])
    assert abs(float(read.sum()) - expected_read) < 2e-6
    assert abs(float(write.sum()) - expected_write) < 2e-6

    def degrees(angle):
        return float(torch.atan2(angle.sin(), angle.cos()) * (180 / torch.pi))

    channels = []
    for k in range(len(psi)):
        channels.append({'channel_1based': k + 1,
                         'pair_amplitude': float(torch.sqrt(real[k]**2 + imag[k]**2)),
                         'content_phase_degrees': degrees(torch.atan2(imag[k], real[k])),
                         'position_phase_degrees': degrees(position_phase[k]),
                         'psi_degrees': degrees(psi[k]), 'beta_degrees': degrees(beta[k]),
                         'read_contribution': float(read[k]), 'write_contribution': float(write[k])})
    return {
        'relative_position_row_col': delta.tolist(),
        'manhattan_distance': float(delta.abs().sum()),
        'alpha': float(b.layer.alpha[head, 0]), 'distance_decay': float(decay),
        'address_dot_no_position_no_psi_before_decay': float(real.sum()),
        'address_dot_with_position_no_psi_before_decay': float(s.sum()),
        'address_dot_with_position_and_psi_before_decay': float(read.sum() / decay),
        'kernel_no_position_no_psi': float(decay * real.sum()),
        'kernel_with_position_no_psi': float(decay * s.sum()),
        'kernel_with_psi_no_position': float(decay * (real * psi.cos() - imag * psi.sin()).sum()),
        'read_kernel': expected_read, 'write_kernel': expected_write,
        'read_even_cos_psi': float(read_even.sum()),
        'read_odd_sin_psi': float(read_odd.sum()),
        'write_even_cos_beta': float(write_even.sum()),
        'write_odd_sin_beta': float(write_odd.sum()),
        'reverse_read_kernel': float(p['a'][batch_index, head, source, cell]),
        'positive_read_channel_sum': float(read.clamp_min(0).sum()),
        'negative_read_channel_sum': float(read.clamp_max(0).sum()),
        'channels': channels,
    }


def analyze_case(root, puzzle):
    with np.load(root / 'baseline.npz') as data:
        ids, baseline_pred = data['indices'], data['predictions']
    event = next(e for e in json.loads((root / 'events.json').read_text())['selected_events']
                 if e['puzzle'] == puzzle)
    i, c, t = event['batch_index'], event['cell'], event['stable_complete_block']
    gold, old = event['gold_digit'] + 1, event['old_digit'] + 1
    case_dir = root / f'case_{puzzle}_{c}'
    previous_results = json.loads((case_dir / 'summary.json').read_text())
    # Same retrospective selection as case 58: highest-ranked Sudoku peer
    # among the already reported history-contribution edges. Do not choose
    # a different edge depending on whether its replacement flips the answer.
    peer = next(edge for edge in previous_results['local_at_T']['top_history_edges']
                if edge['sudoku_peer'])
    source, head = peer['source'], peer['head']
    with np.load(case_dir / 'traces.npz') as cached:
        previous_G_predictions = cached['instant_memory_read_predictions']
    model, batch, _, _ = setup(ids)
    b = EventBlocks(model, batch)
    h0, w0 = restore(root, b, t - 128, baseline_pred)
    report = {'schema_version': 2, 'event': event,
              'source_cell_rc': [source // 9 + 1, source % 9 + 1], 'head_1based': head + 1,
              'source_gold_digit': int(batch['labels'][i, source]) - 1,
              'source_given': bool(batch['inputs'][i, source] != 1),
              'edge_selection': 'Highest history-contribution Sudoku peer in the existing ranking; same rule as case 58.',
              'precision': 'FP32, same 12 puzzles, autocast/TF32 off',
              'same_block_psi_can_change_from_read_w_replacement': False,
              'branches': {}}
    records = {}
    lam = float(b.lam[head, 0, 0])

    def score(hidden):
        logits = b.inner.w_cls(hidden)
        return {'digit': int(logits.argmax()) - 1,
                'gold_minus_old_margin': float(logits[gold] - logits[old])}

    def forward_chain(p, w_before, hn, pre, effective):
        denominator = torch.sqrt(1 + pre[i, c].square().sum() / b.inner.d)
        message = p['v'][i, source, head] @ b.layer.w_sh[head]
        unit_logits = b.inner.w_cls.weight @ message / denominator
        unit_margin = unit_logits[gold] - unit_logits[old]
        coefficient = effective[i, head, c, source]
        psi_coefficient = (1 - lam) * p['a'][i, head, c, source]
        memory_coefficient = coefficient - psi_coefficient
        decoder = b.inner.w_cls.weight[gold] - b.inner.w_cls.weight[old]
        direction = torch.einsum('hcd,d->hc', b.layer.w_sh, decoder)
        evidence = (p['v'][i] * direction[None]).sum(-1).T / denominator
        all_psi = (1 - b.lam[:, 0, 0, None]) * p['a'][i, :, c]
        all_memory = effective[i, :, c] - all_psi
        pieces = {'prepared_hidden': float((p['q'][i, c] * decoder).sum() / denominator),
                  'classifier_bias': float(b.inner.w_cls.bias[gold] - b.inner.w_cls.bias[old]),
                  'psi_read': float((all_psi * evidence).sum()),
                  'memory_read': float((all_memory * evidence).sum())}
        observed = score(hn[i, c])
        error = sum(pieces.values()) - observed['gold_minus_old_margin']
        assert abs(error) < 2e-3
        pair = {key: float(p[name][i, head, c, source]) for key, name in
                [('agree', 'agree'), ('a_beta', 'window'), ('G', 'target'),
                 ('w_after', 'w'), ('a_psi', 'a')]}
        pair.update(gain=float(b.gain[head, 0, 0]), eta=float(b.eta[head, 0, 0]),
                    lambda_=lam, w_before=float(w_before[i, head, c, source]),
                    psi_coefficient=float(psi_coefficient),
                    memory_read_coefficient=float(memory_coefficient),
                    effective_J=float(coefficient))
        assert abs(pair['G'] - pair['gain'] * pair['a_beta'] * pair['agree']) < 2e-5
        assert abs(pair['w_after'] - ((1 - pair['eta']) * pair['w_before'] + pair['eta'] * pair['G'])) < 2e-5
        return {'pair': pair,
                'unit_value_projection_at_actual_phi_scale': {
                    'gold_logit': float(unit_logits[gold]), 'old_logit': float(unit_logits[old]),
                    'gold_minus_old': float(unit_margin)},
                'edge_logit_contributions': {
                    'gold_logit': float(coefficient * unit_logits[gold]),
                    'old_logit': float(coefficient * unit_logits[old]),
                    'gold_minus_old': float(coefficient * unit_margin)},
                'edge_margin_components': {'psi': float(psi_coefficient * unit_margin),
                                          'memory': float(memory_coefficient * unit_margin)},
                'all_other_terms_margin': observed['gold_minus_old_margin'] - float(coefficient * unit_margin),
                'whole_cell_margin_components': pieces,
                'margin_reconstruction_error': error, 'output': observed}

    for mode in ('normal', 'instant_memory_read'):
        h, w = h0.clone(), w0.clone()
        timeline = []
        for k in range(t - 128, t + 2):
            p = b.parts(h, w)
            read_w = None
            if mode == 'instant_memory_read' and k <= t:
                read_w = p['w'].clone()
                read_w[i, :, c] = p['target'][i, :, c]
            hn, pre, effective = b.read(p, read_w)
            after = b.inner.w_cls(hn).argmax(-1)
            if mode == 'normal':
                assert np.array_equal(after.cpu().numpy(), baseline_pred[k])
            else:
                assert np.array_equal(after[i].cpu().numpy(), previous_G_predictions[k - (t - 128)])
            before = b.inner.w_cls(h).argmax(-1)
            if k - t in (-128, -16, -8, -4, -2, -1, 0, 1):
                norm = torch.sqrt(1 + pre[i, c].square().sum() / b.inner.d)
                decoder = b.inner.w_cls.weight[gold] - b.inner.w_cls.weight[old]
                unit_message = p['v'][i, source, head] @ b.layer.w_sh[head]
                evidence = (unit_message * decoder).sum() / norm
                timeline.append({'offset': k - t,
                                 'target_before': int(before[i, c]) - 1,
                                 'source_before': int(before[i, source]) - 1,
                                 'target_after': int(after[i, c]) - 1,
                                 'source_after': int(after[i, source]) - 1,
                                 'a_psi': float(p['a'][i, head, c, source]),
                                 'a_beta': float(p['window'][i, head, c, source]),
                                 'agree': float(p['agree'][i, head, c, source]),
                                 'G': float(p['target'][i, head, c, source]),
                                 'stored_w': float(p['w'][i, head, c, source]),
                                 'effective_J': float(effective[i, head, c, source]),
                                 'unit_value_gold_minus_old': float(evidence),
                                 'edge_gold_minus_old_contribution': float(effective[i, head, c, source] * evidence)})
            if k == t:
                records[mode] = {'p': p, 'hn': hn, 'pre': pre, 'effective': effective}
                report['branches'][mode] = {'at_T': score(hn[i, c]),
                                           'kernel_decomposition': kernel_terms(b, p, i, c, source, head),
                                           'forward_chain': forward_chain(p, w, hn, pre, effective)}
                expected = previous_results['continuations'][mode]
                assert int(after[i, c]) - 1 == expected['prediction_digit_at_T']
                assert abs(score(hn[i, c])['gold_minus_old_margin'] - expected['gold_minus_old_margin_at_T']) < 2e-4
            h, w = hn, p['w']
        report['branches'][mode]['timeline'] = timeline

    normal = records['normal']
    p = normal['p']
    denominator = torch.sqrt(1 + normal['pre'][i, c].square().sum() / b.inner.d)
    decoder = b.inner.w_cls.weight[gold] - b.inner.w_cls.weight[old]
    message = p['v'][i, source, head] @ b.layer.w_sh[head]
    evidence = float((message * decoder).sum() / denominator)
    local = {}
    for mode in ('normal', 'only_this_edge_G', 'target_row_G'):
        rw = p['w'].clone()
        if mode == 'only_this_edge_G':
            rw[i, head, c, source] = p['target'][i, head, c, source]
        elif mode == 'target_row_G':
            rw[i, :, c] = p['target'][i, :, c]
        hn, _, effective = b.read(p, rw)
        next_parts = b.parts(hn, p['w'])
        local[mode] = {'a_psi_at_T': float(p['a'][i, head, c, source]),
                       'memory_coefficient_at_T': lam * float(rw[i, head, c, source]),
                       'effective_J_at_T': float(effective[i, head, c, source]),
                       'edge_margin_at_normal_phi_scale': float(effective[i, head, c, source]) * evidence,
                       'output_at_T': score(hn[i, c]),
                       'a_psi_next_block': float(next_parts['a'][i, head, c, source])}
    report['same_normal_state_read_replacements'] = local

    report['normal_replay_all_predictions_exact'] = True
    report['G_branch_replay_all_case_predictions_exact'] = True
    out = case_dir / 'read_kernel_decomposition.json'
    write_json(out, report)
    compact = {'event': event, 'source_cell_rc': report['source_cell_rc'],
               'head_1based': report['head_1based'], 'source_gold_digit': report['source_gold_digit'],
               'same_normal_state_read_replacements': local}
    for mode, data in report['branches'].items():
        dec = data['kernel_decomposition']
        compact[mode] = {key: value for key, value in dec.items() if key != 'channels'}
        compact[mode]['top_negative_channels'] = sorted(dec['channels'], key=lambda r: r['read_contribution'])[:5]
        compact[mode]['timeline'] = data['timeline']
        compact[mode]['forward_chain'] = data['forward_chain']
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/late_puzzle_probe_v11'))
    parser.add_argument('--puzzles', type=int, nargs='+', default=[58])
    args = parser.parse_args()
    for puzzle in args.puzzles:
        analyze_case(args.root, puzzle)


if __name__ == '__main__':
    main()
