"""Outer integral (accumulated synapse) at random initialization, bounded hybrid model, Rule 110 inputs.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_outer_init.py [--seeds 0 1 2] [--blocks 16] [--rows 2048] [--rhos 1]

Bounded hybrid clock (tags c sin(phi) + b omegaT; c 0.14, omegaT 0.34, lambda 0.5), random init, no training, alpha 1.
The complex pair sum of block b, S_b = sum_n [ (v e^{i phi_V}) (x) RoPE(E_K after this block) + (lam E_V before) (x) (k~ e^{-i phi_K}) ],
contains every (post, pre) pair whose later event is in block b (Im S_b is the run-3 timing increment; checked in-script).
Variants (the model itself reads the matrix, so the trajectories differ):
  V0 increment only      read = q~ (Hebb_b + alpha Im S_b)^T                       (run 3 / bounded hybrid as is)
  V1 STDP accumulated    W_b = W_{b-1} + Im S_b,  read = q~ (Hebb_b + alpha W_b)^T  (order-signed outer integral)
  V2 order-blind accum.  W_b = W_{b-1} + Re S_b,  read = q~ (Hebb_b + alpha W_b)^T  (same pairs, no order sign)
--rhos r1 r2 ...: leaky versions W_b = rho W_{b-1} + (Im|Re) S_b for each rho (rho = 1: the pure integral above).
Per block: ||W_b|| (V0: ||Im S_b||), and logistic probe accuracy (rows split 70/30, per cell) from
  (a) the part of the read that comes from the timing matrix, q~ (alpha M)^T, M = Im S_b / W_b
  (b) the hidden state h_b
for: own input, input at -1 / +1 (mean of both), the 1-step state, the answer after 4 steps.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_hybrid as hy                                   # noqa: E402
from rule110_unsup_init import probe                          # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402

ACC = dict(mode='V0', rho=1.0, W=None, mats=None, err=0.0)


def step(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = hy._STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    ekr_new, eki_new = new[0], new[1]
    evr, evi = traces[2], traces[3]
    lv2 = lv.repeat_interleave(2, -1)
    post_now = v * torch.exp(1j * pv.repeat_interleave(2, -1))
    pre_all = rope(ekr_new) + 1j * rope(eki_new)
    post_past = lv2 * evr + 1j * lv2 * evi
    pre_now = kr * torch.exp(-1j * pk.repeat_interleave(2, -1))
    s = (post_now.transpose(-1, -2) @ pre_all) + (post_past.transpose(-1, -2) @ pre_now)       # [B,H,D,D] complex
    ACC['err'] = max(ACC['err'], float((s.imag - im_s).norm() / im_s.norm().clamp_min(1e-12)))
    if ACC['mode'] == 'V0':
        m = im_s
    else:
        inc = s.imag if ACC['mode'] == 'V1' else s.real
        ACC['W'] = inc if ACC['W'] is None else ACC['rho'] * ACC['W'] + inc
        m = ACC['W']
    ACC['mats'].append(m.detach())
    return hebb, m, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--rhos', type=float, nargs='+', default=[1.0])
    opt = ap.parse_args()
    configs = [('V0', 1.0)] + [(m, r) for r in opt.rhos for m in ('V1', 'V2')]
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14, record=None)
    rp.advance_clock, rp.phase_clock_step = hy.advance, step
    x = torch.randint(0, 2, (opt.rows, T), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    shift = lambda z, o: torch.roll(F.pad(z, (2, 2)), -o, 1)[:, 2:-2]
    targets = {'own input': [x], 'input -1/+1': [shift(x, -1), shift(x, 1)], '1-step state': [r110.evolve(x, 1)],
               'answer (4 steps)': [r110.evolve(x, 4)]}
    split = int(0.7 * opt.rows)
    show = [b for b in (2, 4, 8, 12, 16) if b <= opt.blocks]
    agg = {}
    for mode, rho in configs:
        for seed in opt.seeds:
            base, inner = r110.build('run3', seed)
            base.eval()
            layer = inner.layers[0]
            with torch.no_grad():
                layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(0.5)))
            ACC.update(mode=mode, rho=rho, W=None, mats=[])
            hy.CTL['block'] = 0
            batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(opt.rows, dtype=torch.int32, device=DEVICE))
            qs, hs = [], []
            with torch.no_grad():
                h = inner.init_hidden.float().expand(opt.rows, T, HIDDEN).clone()
                inj = inner.injection(batch)
                mem = ek = ev = None
                tables = inner.rope_tables(layer)
                for _ in range(opt.blocks):
                    q = layer.q_proj(h + inner.embed_scale * inj).reshape(opt.rows, T, 1, -1).transpose(1, 2).float()
                    q = q / (q.norm(dim=-1, keepdim=True) + inner.config.eps)
                    qs.append(inner.apply_rope(q, layer, tables)[:, 0])
                    h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                    hs.append(h.float())
            a = float(layer.stdp_alpha)
            for b in range(1, opt.blocks):
                m = ACC['mats'][b][:, 0]
                feats = {'timing read': torch.einsum('btd,bed->bte', qs[b], a * m), 'state h': hs[b]}
                key = (f'{mode}|{rho:g}', b + 1)
                agg.setdefault((key, 'norm'), []).append(float(m.norm(dim=(-1, -2)).mean()))
                for fname, feat in feats.items():
                    for tname, tgts in targets.items():
                        acc = sum(probe(feat, tg, split) for tg in tgts) / len(tgts)
                        agg.setdefault((key, f'{fname} | {tname}'), []).append(acc)
    print(f'max |Im S - im_s| / |im_s| = {ACC["err"]:.1e};  seeds {opt.seeds}, rows {opt.rows}')
    mean = lambda mode, b, k_: sum(agg[((mode, b), k_)]) / len(agg[((mode, b), k_)])
    names = {'V0|1': 'V0 increment only'}
    for m, r in configs[1:]:
        names[f'{m}|{r:g}'] = ('STDP' if m == 'V1' else 'order-blind') + (' accumulated' if r == 1 else f' leaky rho {r:g}')
    print('\n||timing matrix|| (V0: increment, V1/V2: accumulated W), block ' + ', '.join(map(str, show)))
    for mode in names:
        print(f'  {names[mode]:24s} ' + ' '.join(f'{mean(mode, b, "norm"):9.2f}' for b in show))
    for fname in ('timing read', 'state h'):
        for tname in targets:
            print(f'\n[{fname}] {tname}  (probe accuracy at block ' + ', '.join(map(str, show)) + ')')
            for mode in names:
                print(f'  {names[mode]:24s} ' + ' '.join(f'{mean(mode, b, f"{fname} | {tname}"):9.3f}' for b in show))
    (OUT / 'results' / ('rule110_outer_init.json' if opt.rhos == [1.0] else 'rule110_outer_init_leaky.json')).write_text(
        json.dumps({f'{m}|{b}|{k_}': v_ for ((m, b), k_), v_ in agg.items()}, indent=1) + '\n')


if __name__ == '__main__':
    main()
