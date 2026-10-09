import json, math, sys
import numpy as np

r = json.load(open(sys.argv[1]))
alpha = np.array(r['alpha'])
c = np.array([0.9857296165734425, 0.08535327631449735])
d = np.linspace(-math.pi, math.pi, 200001)[:-1]
W = c[0] * np.sin(d) + c[1] * np.sin(2 * d)
uni = dict(abs_alphaW=float(np.mean([np.abs(a * W).mean() for a in alpha])),
           gain_neg=float(np.mean([(1 + a * W < 0).mean() for a in alpha])),
           gain_lt_05=float(np.mean([(1 + a * W < 0.5).mean() for a in alpha])),
           gain_gt_15=float(np.mean([(1 + a * W > 1.5).mean() for a in alpha])))
print(f"puzzles {r['puzzles']} solved {r['solved']} | official {r['official']} | mismatch {r['pass2_pred_mismatch']} "
      f"| checks {r['checks']} | readout agreement {r['readout_agreement']:.5f}")
print('alpha', np.round(alpha, 3).tolist(), '| uniform-phase reference', {k: round(v, 3) for k, v in uni.items()})
S = r['summary']
f = lambda x: f'{x:.3f}'
print('\n[1] timing / LTP-LTD (co-firing weighted)            seg  LTP (null)      |aW| (null)   g<0    g<.5   g>1.5  R (null)')
for g in ('solved', 'unsolved'):
    for s in ('1', '16'):
        x = S[g][s]
        print(f"  {g:9s} {s:>3s}  {f(x['ltp'])} ({f(x['ltp_null'])})  {f(x['abs_alphaW'])} ({f(x['abs_alphaW_null'])})  "
              f"{f(x['gain_neg'])}  {f(x['gain_lt_05'])}  {f(x['gain_gt_15'])}  {f(x['R'])} ({f(x['R_null'])})")
for s in ('1', '16'):
    x = r['init']['summary'][s]
    print(f"  {'init':9s} {s:>3s}  {f(x['ltp'])} ({f(x['ltp_null'])})  {f(x['abs_alphaW'])} ({f(x['abs_alphaW_null'])})  "
          f"{f(x['gain_neg'])}  {f(x['gain_lt_05'])}  {f(x['gain_gt_15'])}  {f(x['R'])} ({f(x['R_null'])})")

print('\n[2] matrices                 seg  |aGs|/|Gh|  |a read_S|/|read_H|  cos(Gs,Gh)  a*pot   flip   dGs')
for g in ('solved', 'unsolved'):
    for s in ('1', '16'):
        x = S[g][s]
        print(f"  {g:9s} {s:>3s}  {f(x['ratio'])}  {f(x['read_ratio'])}  {f(x['cos'])}  {f(x['pot'])}  {f(x['flip'])}  {f(x['gs_change'])}")
for s in ('1', '16'):
    x = r['init']['summary'][s]
    print(f"  {'init':9s} {s:>3s}  {f(x['ratio'])}  {f(x['read_ratio'])}  {f(x['cos'])}  {f(x['pot'])}  {f(x['flip'])}  {f(x['gs_change'])}")

print('\n[3] dynamics per segment: empty-correct | LTP | g<0 | cos | read ratio | flip | dGs   (solved / unsolved)')
for s in range(1, 17):
    row = []
    for g in ('solved', 'unsolved'):
        x = S[g][str(s)]
        row.append(f"{r['empty_correct'][g][s - 1]:.3f} {f(x['ltp'])} {f(x['gain_neg'])} {f(x['cos'])} {f(x['read_ratio'])} {f(x['flip'])} {f(x['gs_change'])}")
    print(f'  {s:2d}  ' + '  |  '.join(row))

print('\n[4] per head, seg 16 solved: alpha  LTP (null)  g<0  R (null)  |aGs|/|Gh|  cos')
for h, x in enumerate(r['per_head_seg16']['solved']):
    print(f"  h{h} {alpha[h]:.3f}  {f(x['ltp'])} ({f(x['ltp_null'])})  {f(x['gain_neg'])}  {f(x['R'])} ({f(x['R_null'])})  {f(x['ratio'])}  {f(x['cos'])}")

cls = r['sources']['classes']
print('\n[5] read share by source class (targets = empty cells), Hebbian vs STDP; enrichment = share / source fraction')
for gi, g in enumerate(('solved', 'unsolved')):
    for s in (1, 16):
        sh = np.array(r['sources']['share'][gi][s - 1]); fr = np.array(r['sources']['frac'][gi][s - 1])
        print(f'  {g:9s} seg{s:2d} ' + '  '.join(f'{k}: H {sh[0, i]:+.3f} S {sh[1, i]:+.3f} (frac {fr[i]:.3f}, enr H {sh[0, i] / fr[i]:+.2f} S {sh[1, i] / fr[i]:+.2f})' for i, k in enumerate(cls)))

el = r['elimination']
print('\n[6] last block, first-order logit effect of source n on cell t (empty targets)')
print('    e = dlogit[digit of n] - mean dlogit[digits]   (negative = eliminates the source digit)')
for gi, g in enumerate(('solved', 'unsolved')):
    e = np.array(el['e_per_pair'][gi]); p = np.array(el['peer_pressure_per_target'][gi]); sp = np.array(el['support_per_target'][gi])
    print(f'  {g:9s} per pair  ' + '  '.join(f'{k}: H {e[0, i]:+.4f} S {e[1, i]:+.4f}' for i, k in enumerate(el['classes'])))
    print(f'  {g:9s} peer pressure per target  given: H {p[0, 0]:+.3f} S {p[1, 0]:+.3f}   empty: H {p[0, 1]:+.3f} S {p[1, 1]:+.3f}')
    print(f'  {g:9s} support of the correct digit per target  ' + '  '.join(f'{k}: H {sp[0, i]:+.3f} S {sp[1, i]:+.3f}' for i, k in enumerate(cls)))

sf = r['shared_fraction']
print('\n[7] shared fraction across puzzles |mean|^2 / mean|.|^2 (head mean), last block of segment')
for gi, g in enumerate(('solved', 'unsolved')):
    for j, s in enumerate(sf['segments']):
        print(f"  {g:9s} seg{s:2d}  G_S {np.mean(sf['GS'][gi][j]):.3f}  G_H {np.mean(sf['GH'][gi][j]):.3f}  polarity {np.mean(sf['pol'][gi][j]):.3f}")
print(f"  init      seg 1..16 G_S {np.round(np.mean(r['init']['shared_fraction']['GS'], 1), 3).tolist()}  G_H {np.round(np.mean(r['init']['shared_fraction']['GH'], 1), 3).tolist()}")

h = np.array(r['hist']['delta']['values'])          # [G, S, H, 36]
edges = np.linspace(-180, 180, 37)
print('\n[8] co-firing weighted delta distribution (degrees, all heads), share per 30 deg')
for gi, g in enumerate(('solved', 'unsolved')):
    for s in (1, 16):
        v = h[gi, s - 1].sum(0); v = v / v.sum()
        coarse = v.reshape(12, 3).sum(1)
        print(f'  {g:9s} seg{s:2d} ' + ' '.join(f'{x:.3f}' for x in coarse))
print('  bins: ' + ' '.join(f'{int(a)}..{int(a + 30)}' for a in edges[::3][:-1]))
