"""Direct STDP structure in the trained Rule 110 run-3 model: does the firing order of a synapse's pre and post
neurons (hence LTP vs LTD) switch with the situation, systematically?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_stdp_structure.py [--seed 0] [--n 1024]

Neurons are channel pairs: 16 pre (K) and 16 post (V) per cell; their tags are the firing phases.
Synapse (post a, pre i) at a cell undergoes LTP when phi_V,a > phi_K,i in the window (sin > 0), LTD otherwise.
On held-out rows unsolved after block 1, at blocks 2, 4 and 8:
  - per synapse: P(LTP) over all cells, and P(LTP | the cell's answer = 1) vs P(LTP | answer = 0);
    'switching' synapses = LTP for one answer and LTD for the other in >= 90% / <= 10% of cells
  - firing order per answer: circular mean phase of each neuron for answer-0 and answer-1 cells
  - a raster of one cell: phases of all neurons over blocks 1..8 and its LTP/LTD signs on the top switching synapses
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_dissect as rd                                  # noqa: E402
from rule110_ltp_ltd import T, BLOCKS, DEVICE, OUT            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=1024)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    name = f'rule110_k{r110.K_STEPS}_run3_seed{opt.seed}'
    ck = torch.load(OUT / 'checkpoints' / f'{name}.pt', map_location=DEVICE)
    base, inner = r110.build('run3', opt.seed)
    base.load_state_dict(ck['state'])
    base.eval()
    ctl = {}
    rd.install(inner, ctl)
    xt, yt = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    lg = rd.run(inner, ctl, xt)
    unsolved = ~((lg[0].argmax(-1) - 1) == yt).all(-1)
    x, y = xt[unsolved][:opt.n], yt[unsolved][:opt.n]
    ctl['record'] = []
    rd.run(inner, ctl, x)
    recs = ctl.pop('record')
    report = {}
    ans = y.reshape(-1).bool()                                                    # [B*T]
    for b in (1, 3, 7):
        pk, pv = recs[b]['pk'].reshape(-1, 16), recs[b]['pv'].reshape(-1, 16)     # [B*T, P]
        ltp = torch.sin(pv[:, :, None] - pk[:, None, :]) > 0                      # [B*T, post a, pre i]
        p_all = ltp.float().mean(0)
        p1, p0 = ltp[ans].float().mean(0), ltp[~ans].float().mean(0)
        switching = ((p1 >= 0.9) & (p0 <= 0.1)) | ((p1 <= 0.1) & (p0 >= 0.9))
        fixed = ((p1 >= 0.9) & (p0 >= 0.9)) | ((p1 <= 0.1) & (p0 <= 0.1))
        gap = (p1 - p0).abs()
        top = torch.argsort(gap.flatten(), descending=True)[:8]
        print(f'\n=== block {b + 1}: 256 synapses (16 post x 16 pre), cells {ans.numel()}')
        print(f'  switching with the answer (LTP for one answer, LTD for the other, >=90%/<=10%): {int(switching.sum())}')
        print(f'  same sign for both answers (>=90% or <=10% in both):                          {int(fixed.sum())}')
        print(f'  |P(LTP|1) - P(LTP|0)| quantiles 25/50/75/90%: '
              + ' '.join(f'{v:.2f}' for v in torch.quantile(gap.flatten(), torch.tensor([0.25, 0.5, 0.75, 0.9], device=DEVICE)).tolist()))
        print('  top synapses by |P(LTP|answer 1) - P(LTP|answer 0)|:')
        for f in top.tolist():
            a, i = divmod(f, 16)
            print(f'    V{a + 1:<2d} <- K{i + 1:<2d}  P(LTP | answer 1) {float(p1[a, i]):.3f}   P(LTP | answer 0) {float(p0[a, i]):.3f}')
        # firing order per answer: circular mean phase
        order = {}
        for label, m in (('answer 0', ~ans), ('answer 1', ans)):
            mk = torch.atan2(pk[m].sin().mean(0), pk[m].cos().mean(0))
            mv = torch.atan2(pv[m].sin().mean(0), pv[m].cos().mean(0))
            rk = torch.sqrt(pk[m].sin().mean(0) ** 2 + pk[m].cos().mean(0) ** 2)
            rv = torch.sqrt(pv[m].sin().mean(0) ** 2 + pv[m].cos().mean(0) ** 2)
            neurons = [(f'K{i + 1}', float(mk[i]), float(rk[i])) for i in range(16)] + \
                      [(f'V{a + 1}', float(mv[a]), float(rv[a])) for a in range(16)]
            neurons.sort(key=lambda z: z[1])
            order[label] = neurons
            print(f'  firing order, {label} (circular mean phase, concentration R): '
                  + ', '.join(f'{n_}({p:+.2f},R{r:.2f})' for n_, p, r in neurons))
        report[f'block{b + 1}'] = dict(switching=int(switching.sum()), fixed=int(fixed.sum()),
                                       p_ltp_answer1=p1.tolist(), p_ltp_answer0=p0.tolist(), p_ltp=p_all.tolist(),
                                       order=order)
    # raster: one cell of row 0 with answer 1 and one with answer 0; top switching synapses at block 8
    b = 7
    pk, pv = recs[b]['pk'].reshape(-1, 16), recs[b]['pv'].reshape(-1, 16)
    ltp = torch.sin(pv[:, :, None] - pk[:, None, :]) > 0
    p1, p0 = ltp[ans].float().mean(0), ltp[~ans].float().mean(0)
    top = torch.argsort((p1 - p0).abs().flatten(), descending=True)[:4].tolist()
    print('\n=== raster: row 0, phases (rad) of the neurons in the top switching synapses, blocks 1..8')
    raster = []
    for cell in (int((y[0] == 1).nonzero()[0]), int((y[0] == 0).nonzero()[0])):
        print(f'  cell {cell} (input {int(x[0, cell])}, answer {int(y[0, cell])})')
        for f in top:
            a, i = divmod(f, 16)
            k_line = [float(recs[bb]['pk'][0, cell, i]) for bb in range(BLOCKS)]
            v_line = [float(recs[bb]['pv'][0, cell, a]) for bb in range(BLOCKS)]
            signs = ''.join('+' if math.sin(vv - kk) > 0 else ('-' if math.sin(vv - kk) < 0 else '0') for kk, vv in zip(k_line, v_line))
            print(f'    K{i + 1:<2d} ' + ' '.join(f'{v_:+.2f}' for v_ in k_line))
            print(f'    V{a + 1:<2d} ' + ' '.join(f'{v_:+.2f}' for v_ in v_line) + f'   LTP(+)/LTD(-) by block: {signs}')
            raster.append(dict(cell=cell, answer=int(y[0, cell]), pre=f'K{i + 1}', post=f'V{a + 1}', phi_k=k_line, phi_v=v_line))
    report['raster'] = raster
    (OUT / 'results' / f'{name}_stdp_structure.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
