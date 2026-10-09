"""Side-by-side comparison of the STDP readouts of two checkpoints (default 20k vs 100k).

python3 docs/research/2026-10-09/compare_interp_steps.py <run diagnostics dir> 20000 100000
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

D = Path(sys.argv[1])
A, B = sys.argv[2], sys.argv[3]
div = lambda a, b: a / b if b > 0 else float('nan')


def load(step):
    direct = json.loads((D / f'stdp_direct_step{step}.json').read_text())
    timing = json.loads((D / f'stdp_token_timing_step{step}.json').read_text())
    flow_dir = D / f'stdp_message_flow_step{step}'
    flow = json.loads((flow_dir / 'aggregate.json').read_text())
    syn = torch.load(flow_dir / 'synapses.pt')
    return direct, timing, flow, syn


da, ta, fa, sa = load(A)
db, tb, fb, sb = load(B)
print(f"solved: {A} {da['solved']}/{da['puzzles']}  |  {B} {db['solved']}/{db['puzzles']}")
print(f"alpha {A}: {np.round(da['alpha'], 2).tolist()}\nalpha {B}: {np.round(db['alpha'], 2).tolist()}")

print('\n[1] STDP readout (co-firing weighted): LTP | gain<0 | |aGs|/|Gh| | cos(Gs,Gh) | flip')
for g in ('solved', 'unsolved'):
    for s in ('1', '16'):
        cells = []
        for d in (da, db):
            x = d['summary'][g][s]
            cells.append(f"{x['ltp']:.3f} {x['gain_neg']:.3f} {x['ratio']:.3f} {x['cos']:+.3f} {x['flip']:.3f}")
        print(f'  {g:9s} seg{s:>2s}   {A}: {cells[0]}   |   {B}: {cells[1]}')

print('\n[2] empty-cell accuracy per segment (solved / unsolved)')
for d, name in ((da, A), (db, B)):
    ec = d['empty_correct']
    print(f'  {name}: solved ' + ' '.join(f'{v:.2f}' for v in ec['solved'][:8]) + ' ... ' + f"{ec['solved'][-1]:.2f}"
          + ' | unsolved ' + ' '.join(f'{v:.2f}' for v in ec['unsolved'][:4]) + ' ... ' + f"{ec['unsolved'][-1]:.2f}")

print('\n[3] per-cell STDP residual beyond the best channel-pair mask, |res|/|G_S| (segments 1,2,4,8,16)')
for t, name in ((ta, A), (tb, B)):
    r = t['residual_GS']['best_mask']
    print(f'  {name}: solved ' + ' '.join(f'{v:.2f}' for v in r[0]) + ' | unsolved ' + ' '.join(f'{v:.2f}' for v in r[1]))

S = fa['status']
print('\n[4] message gain kappa, peer messages to open targets, by source state (seg1 | solving | solved)')
for f, name in ((fa, A), (fb, B)):
    acc = {k: np.array(v) for k, v in f['acc'].items()}
    rows = []
    for pi in range(3):
        rows.append(' '.join(f"{div(acc['ek'][pi, 0, si, 0], acc['energy'][pi, 0, si, 0]):.3f}" for si in range(len(S))))
    print(f'  {name} ({f["puzzles"]} puzzles): ' + '  |  '.join(rows) + f"   [{' / '.join(S)}]")

print('\n[5] kappa aligned to the block where a source becomes committed')
offs = fa['offsets']
for f, name in ((fa, A), (fb, B)):
    se = {k: np.array(v) for k, v in f['source_event'].items()}
    pick = [-16, -8, -2, 0, 1, 2, 3, 4, 8, 16]
    print(f'  {name}: ' + '  '.join(f"{o:+d}:{div(se['ek'][offs.index(o)], se['energy'][offs.index(o)]):.3f}" for o in pick))

print('\n[6] elimination pressure on a target before it commits (settled peers H, S | open peers H, S), offsets -16, -1')
for f, name in ((fa, A), (fb, B)):
    te = {k: np.array(v) for k, v in f['target_event'].items()}
    for o in (-16, -1):
        i, n = offs.index(o), te['n'][offs.index(o)]
        print(f"  {name} {o:+d}: settled H {div(te['setH'][i], n):+.2f} S {div(te['setS'][i], n):+.2f} | open H {div(te['opnH'][i], n):+.2f} S {div(te['opnS'][i], n):+.2f}")

print('\n[7] lags carrying peer messages: strengthening share by source state (solving)')
for f, name in ((fa, A), (fb, B)):
    sw = np.array(f['lag_strong_weak'])
    print(f'  {name}: ' + '  '.join(f'{s} {sw[1, si, 0] / sw[1, si].sum():.3f}' for si, s in enumerate(S)))

print('\n[8] per head alpha*W by source state (solving), with message-weight share')
for syn, name in ((sa, A), (sb, B)):
    w, ww = syn['syn_w'].numpy(), syn['syn_wW'].numpy()
    print(f'  {name}:')
    for h in range(w.shape[2]):
        share = w[1, :, h].sum() / w[1].sum()
        print(f'    h{h} share {share:.3f} ' + ' '.join(f'{div(ww[1, si, h].sum(), w[1, si, h].sum()):+.3f}' for si in range(len(S))))
for syn, name in ((sa, A), (sb, B)):
    w, ww = syn['syn_w'].numpy(), syn['syn_wW'].numpy()
    f = lambda i: ww[1, i] / np.maximum(w[1, i], 1e-30)
    g, o = f(0), (ww[1, 2] + ww[1, 3]) / np.maximum(w[1, 2] + w[1, 3], 1e-30)
    share = w[1].sum(0) / w[1].sum()
    m = share >= 5e-4
    d = (g - o)[m]
    print(f'  {name}: weighted mean (given - open) alpha*W over synapses with share >= 0.05%: {(share[m] * d).sum() / share[m].sum():+.3f}; '
          f'weight with contrast > 0.3: {share[m][d > 0.3].sum() / share[m].sum():.2f}')
