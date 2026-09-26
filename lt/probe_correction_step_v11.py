"""One-block follow-up separating the current write from reading old memory."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_late_puzzles import restore
from probe_late_puzzles import EventBlocks, setup, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/late_puzzle_probe_v11')
    parser.add_argument('--puzzle', type=int, default=58)
    args = parser.parse_args()
    root = Path(args.root)
    with np.load(root / 'baseline.npz') as data:
        ids, pred = data['indices'], data['predictions']
    event = next(e for e in json.loads((root / 'events.json').read_text())['selected_events']
                 if e['puzzle'] == args.puzzle)
    i, c, t = event['batch_index'], event['cell'], event['stable_complete_block']
    gold, old = event['gold_digit'] + 1, event['old_digit'] + 1
    model, batch, _, _ = setup(ids)
    b = EventBlocks(model, batch)
    h0, w0 = restore(root, b, t - 1, pred)
    previous = b.parts(h0, w0)
    h, pre_previous, effective_previous = b.read(previous)
    assert np.array_equal(b.inner.w_cls(h).argmax(-1).cpu().numpy(), pred[t - 1])
    p = b.parts(h, previous['w'])
    hn, pre, effective = b.read(p)
    assert np.array_equal(b.inner.w_cls(hn).argmax(-1).cpu().numpy(), pred[t])

    def score(hidden):
        logits = b.inner.w_cls(hidden)
        return {'digit': int(logits.argmax()) - 1,
                'gold_minus_old_margin': float(logits[gold] - logits[old])}

    scores = {'normal': score(hn[i, c]), 'previous_block_output': score(h[i, c])}
    for name, read_w in [('skip_current_write', previous['w']),
                         ('read_current_G', p['target'])]:
        alt_h, _, _ = b.read(p, read_w)
        scores[name] = score(alt_h[i, c])

    for name in ('previous_all_values', 'previous_other_cell_values',
                 'previous_self_value', 'previous_read_kernel', 'previous_residual_q'):
        alt = dict(p)
        if name.startswith('previous_') and name.endswith(('values', 'value')):
            alt['v'] = p['v'].clone()
            if name == 'previous_self_value':
                alt['v'][i, c] = previous['v'][i, c]
            else:
                alt['v'][i] = previous['v'][i]
                if name == 'previous_other_cell_values':
                    alt['v'][i, c] = p['v'][i, c]
        elif name == 'previous_read_kernel':
            alt['a'] = p['a'].clone()
            alt['a'][i, :, c] = previous['a'][i, :, c]
        else:
            alt['q'] = p['q'].clone()
            alt['q'][i, c] = previous['q'][i, c]
        alt_h, _, _ = b.read(alt)
        scores[name] = score(alt_h[i, c])

    decoder = b.inner.w_cls.weight[gold] - b.inner.w_cls.weight[old]
    bias = float(b.inner.w_cls.bias[gold] - b.inner.w_cls.bias[old])
    denominator = torch.sqrt(1 + pre[i, c].square().sum() / b.inner.d)
    lam = b.lam[:, 0, 0, None]
    direction = torch.einsum('hcd,d->hc', b.layer.w_sh, decoder)
    evidence = (p['v'][i] * direction[None]).sum(-1).T / denominator
    coefficients = {
        'instant_read': (1 - lam) * p['a'][i, :, c],
        'memory_before_current_write': lam * previous['w'][i, :, c],
        'current_write_increment': lam * (p['w'][i, :, c] - previous['w'][i, :, c]),
    }
    contributions = {name: float((value * evidence).sum()) for name, value in coefficients.items()}
    contributions['prepared_hidden'] = float((p['q'][i, c] * decoder).sum() / denominator)
    contributions['classifier_bias'] = bias
    reconstruction_error = sum(contributions.values()) - scores['normal']['gold_minus_old_margin']
    assert abs(reconstruction_error) < 2e-3

    # Exact symmetric product decomposition across the last two normal blocks.
    # This is algebraic attribution, not an intervention or a unique causal split.
    avg_v = 0.5 * (p['v'][i] + previous['v'][i])
    delta_v = p['v'][i] - previous['v'][i]
    avg_j = 0.5 * (effective[i, :, c] + effective_previous[i, :, c])
    delta_a = (1 - lam) * (p['a'][i, :, c] - previous['a'][i, :, c])
    delta_w = lam * (p['w'][i, :, c] - previous['w'][i, :, c])

    def project(coeff, values):
        return torch.einsum('hn,nhc,hcd,d->', coeff, values, b.layer.w_sh, decoder)

    raw_changes = {
        'prepared_hidden_change': ((p['q'][i, c] - previous['q'][i, c]) * decoder).sum(),
        'values_change': project(avg_j, delta_v),
        'instant_read_kernel_change': project(delta_a, avg_v),
        'memory_write_change': project(delta_w, avg_v),
    }
    previous_denominator = torch.sqrt(1 + pre_previous[i, c].square().sum() / b.inner.d)
    avg_inverse_norm = 0.5 * (1 / denominator + 1 / previous_denominator)
    changes = {name: float(value * avg_inverse_norm) for name, value in raw_changes.items()}
    avg_raw = 0.5 * ((pre[i, c] + pre_previous[i, c]) * decoder).sum()
    changes['normalization_change'] = float(avg_raw * (1 / denominator - 1 / previous_denominator))
    hidden_change = ((h[i, c] - h0[i, c]) * decoder).sum() * avg_inverse_norm
    prepared_change_split = {
        'incoming_hidden_change': float(hidden_change),
        'local_MLP_output_change': changes['prepared_hidden_change'] - float(hidden_change),
        'original_input_embedding_change': 0.0,
    }
    observed_change = scores['normal']['gold_minus_old_margin'] - scores['previous_block_output']['gold_minus_old_margin']
    change_error = sum(changes.values()) - observed_change
    assert abs(change_error) < 2e-3

    head, source = 2, 23  # Previously discussed third head, r3c6 -> r3c9.
    peer = {name: float(p[key][i, head, c, source]) for name, key in
            [('agree', 'agree'), ('write_window', 'window'), ('G', 'target'),
             ('w_after', 'w'), ('read_kernel', 'a')]}
    peer['w_before'] = float(previous['w'][i, head, c, source])
    peer['eta'] = float(b.eta[head, 0, 0])
    peer['lambda'] = float(b.lam[head, 0, 0])
    peer['normal_effective_read'] = float(effective[i, head, c, source])
    peer['G_effective_read'] = float((1 - lam[head, 0]) * p['a'][i, head, c, source]
                                    + lam[head, 0] * p['target'][i, head, c, source])
    peer['write_increment_margin'] = float(coefficients['current_write_increment'][head, source]
                                           * evidence[head, source])

    report = {'event': event, 'normal_replay_predictions_exact_at_previous_and_current': True,
              'scope': 'One-block read interventions at fixed normal pre-transition state; no rollout claims.',
              'scores': scores, 'chronological_margin_components': contributions,
              'chronological_reconstruction_error': reconstruction_error,
              'last_block_margin_change_symmetric_algebra': changes,
              'prepared_hidden_change_split': prepared_change_split,
              'observed_last_block_margin_change': observed_change,
              'change_reconstruction_error': change_error,
              'previously_discussed_peer_head': peer}
    out = root / f'case_{args.puzzle}_{c}' / 'forward_step.json'
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
