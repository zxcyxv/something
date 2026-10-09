"""Print the aggregate of lt.stdp_message_flow (aggregate.json + synapses.pt)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

D = Path(sys.argv[1])
r = json.loads((D / 'aggregate.json').read_text())
A = {k: np.array(v) for k, v in r['acc'].items()}            # [phase, rel, status, target]
P, S, T = r['phases'], r['status'], r['targets']
alpha = np.array(r['alpha'])
print(f"puzzles {r['puzzles']} | read reconstruction check {r['recon_check']:.1e} | alpha {np.round(alpha, 2).tolist()}")
div = lambda a, b: a / b if b > 0 else float('nan')

print('\n[1] message gain kappa (energy weighted) and share of message energy strengthened (kappa > 1)')
print('    rows: phase / relation / target state; columns: source state')
for pi, ph in enumerate(P):
    for ri, rel in enumerate(r['relations']):
        for ti, tg in enumerate(T):
            cells = [f"{s}: {div(A['ek'][pi, ri, si, ti], A['energy'][pi, ri, si, ti]):.3f} ({div(A['estr'][pi, ri, si, ti], A['energy'][pi, ri, si, ti]):.2f})"
                     for si, s in enumerate(S)]
            print(f'  {ph:7s} {rel:5s} target {tg:9s} ' + '  '.join(cells))

print('\n[2] per pair, peer messages: elimination of the source digit at the target (negative = eliminates), H vs S')
for pi, ph in enumerate(P):
    for ti, tg in enumerate(T):
        cells = [f"{s}: H {div(A['eH'][pi, 0, si, ti], A['evalid'][pi, 0, si, ti]):+.4f} S {div(A['eS'][pi, 0, si, ti], A['evalid'][pi, 0, si, ti]):+.4f}"
                 for si, s in enumerate(S)]
        print(f'  {ph:7s} target {tg:9s} ' + '  '.join(cells))
print('    support of the target answer per pair (peer): ')
for pi, ph in enumerate(P):
    cells = [f"{s}: H {div(A['sH'][pi, 0, si, 0], A['pairs'][pi, 0, si, 0]):+.4f} S {div(A['sS'][pi, 0, si, 0], A['pairs'][pi, 0, si, 0]):+.4f}"
             for si, s in enumerate(S)]
    print(f'  {ph:7s} target open      ' + '  '.join(cells))

o = np.array(r['offsets'])
se = {k: np.array(v) for k, v in r['source_event'].items()}
te = {k: np.array(v) for k, v in r['target_event'].items()}
print('\n[3] aligned to the block where a source cell becomes committed (0 = first block reading it committed)')
print('    offset: kappa of its messages to empty peers | elimination per pair H, S')
for i, off in enumerate(o):
    if off % 2 == 0 or abs(off) <= 3:
        print(f'  {off:+3d}  kappa {div(se["ek"][i], se["energy"][i]):.3f} | elim H {div(se["eH"][i], se["evalid"][i]):+.4f} S {div(se["eS"][i], se["evalid"][i]):+.4f}')
print('\n[4] aligned to the block where a target cell becomes committed: per target, sums over its peers')
print('    offset: support of its answer H, S | elimination from settled peers H, S | from open peers H, S')
for i, off in enumerate(o):
    if off % 2 == 0 or abs(off) <= 3:
        n = te['n'][i]
        print(f'  {off:+3d}  support H {div(te["supH"][i], n):+.3f} S {div(te["supS"][i], n):+.3f} | settled H {div(te["setH"][i], n):+.3f} S {div(te["setS"][i], n):+.3f} '
              f'| open H {div(te["opnH"][i], n):+.3f} S {div(te["opnS"][i], n):+.3f}')

lag = np.array(r['lag_hist'])                                 # [phase, status, 36]
sw = np.array(r['lag_strong_weak'])
print('\n[5] lags of the spike pairs carrying peer messages (weight = share in the messages), per 30 degrees, delta = phiV - phiK')
print('    bins: ' + ' '.join(f'{a:+d}' for a in range(-180, 180, 30)) + '   | strengthening share')
for pi, ph in enumerate(P):
    for si, s in enumerate(S):
        h = lag[pi, si]
        if h.sum() <= 0:
            continue
        c = (h / h.sum()).reshape(12, 3).sum(1)
        print(f'  {ph:7s} {s:12s} ' + ' '.join(f'{x:.2f}' for x in c) + f'   | {sw[pi, si, 0] / sw[pi, si].sum():.3f}')

syn = torch.load(D / 'synapses.pt')
w, ww = syn['syn_w'].numpy(), syn['syn_wW'].numpy()           # [phase, status, H, A, B]
tot = w[1].sum(0)                                              # solving phase, all source states
flat = np.argsort(tot.ravel())[::-1][:8]
print('\n[6] synapses (head, post A <- pre B) carrying the most peer-message weight during solving;')
print('    alpha*W averaged over their message weight, by source state (solving | solved)')
for f in flat:
    h, a, b = np.unravel_index(f, tot.shape)
    cells = [f"{s} {div(ww[1, si, h, a, b], w[1, si, h, a, b]):+.2f}|{div(ww[2, si, h, a, b], w[2, si, h, a, b]):+.2f}" for si, s in enumerate(S)]
    print(f'  h{h} A{a:2d}<-B{b:2d} share {tot[h, a, b] / tot.sum():.4f}  ' + '  '.join(cells))
ph_w = w[1].sum((1, 2, 3)); ph_ww = ww[1].sum((1, 2, 3))
print('  all synapses (solving): ' + '  '.join(f'{s} {ph_ww[si] / ph_w[si]:+.3f}' for si, s in enumerate(S)))
per_head = [[div(ww[1, si, h].sum(), w[1, si, h].sum()) for si in range(len(S))] for h in range(w.shape[2])]
print('  per head (solving), alpha*W by source state given/committed/open_ok/open_wrong:')
for h, row in enumerate(per_head):
    print(f'    h{h} share {w[1, :, h].sum() / w[1].sum():.3f} ' + ' '.join(f'{x:+.3f}' for x in row))
