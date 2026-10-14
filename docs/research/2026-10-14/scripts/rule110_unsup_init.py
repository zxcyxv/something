"""Unsupervised effect of the order-signed (STDP) cross-block write at random initialization, bounded hybrid model.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_unsup_init.py [--seeds 0 1 2] [--blocks 8]

Model: the run-3 class at the Rule 110 size with the bounded hybrid clock (tags c sin(phi) + b omegaT; c 0.14,
omegaT 0.34, lambda 0.5), random initialization, no training, alpha = 1. Inputs: random 36-cell rows (4,096).
Per block b the cross-block part of the complex pair sum is split exactly:
  S_cross = sum_n [ (v e^{i phi_V})_now (x) RoPE(lam E_K) + (lam E_V) (x) (k~ e^{-i phi_K})_now ]
  odd  = Im S_cross: order-signed pairs (pre earlier -> +, post earlier -> -), the STDP difference
  even = Re S_cross: the same pairs added regardless of order (decaying Hebbian)
and each is read with the cell's query: read = q~ M^T. The Hebbian read q~ (sum_n v k~^T)^T is the reference.
Reported per block (seed means):
  cancel  ||sum_n X_n|| / sum_n ||X_n|| over cells, for the odd and the even cross matrices
  corr    across rows, Pearson correlation of ||odd cross matrix|| (resp. even) with the mean state change ||h_b - h_{b-1}||
  probes  logistic accuracy from each read (32-d per cell; rows split 70/30) of: own input bit, the input bits at
          offsets -2, -1, +1, +2, the bit after one Rule 110 step, the answer after 4 steps
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
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402


def probe(feat, tgt, split):
    k = int(tgt.max()) + 1
    clf = torch.nn.Linear(feat.shape[-1], k).to(DEVICE)
    o = torch.optim.Adam(clf.parameters(), lr=0.05)
    a, b = feat[:split].reshape(-1, feat.shape[-1]), tgt[:split].reshape(-1)
    mu, sd = a.mean(0), a.std(0) + 1e-6
    for _ in range(300):
        o.zero_grad()
        F.cross_entropy(clf((a - mu) / sd), b).backward()
        o.step()
    with torch.no_grad():
        return float((clf((feat[split:].reshape(-1, feat.shape[-1]) - mu) / sd).argmax(-1) == tgt[split:].reshape(-1)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--blocks', type=int, default=8)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14)
    rp.advance_clock, rp.phase_clock_step = hy.advance, hy.step
    x = torch.randint(0, 2, (4096, T), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    shift = lambda z, o: torch.roll(F.pad(z, (2, 2)), -o, 1)[:, 2:-2]               # z[t + o], 0 outside the row
    targets = {'own input': x, 'input -2': shift(x, -2), 'input -1': shift(x, -1), 'input +1': shift(x, 1),
               'input +2': shift(x, 2), '1-step state': r110.evolve(x, 1), 'answer (4 steps)': r110.evolve(x, 4)}
    split = int(0.7 * x.shape[0])
    show = [b for b in (2, 4, 8) if b <= opt.blocks]
    agg = {}
    for seed in opt.seeds:
        base, inner = r110.build('run3', seed)
        base.eval()
        layer = inner.layers[0]
        with torch.no_grad():
            layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(0.5)))
        hy.CTL['record'] = []
        hy.CTL['block'] = 0
        batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=DEVICE))
        qs, hs = [], []
        with torch.no_grad():
            h = inner.init_hidden.float().expand(x.shape[0], T, HIDDEN).clone()
            inj = inner.injection(batch)
            mem = ek = ev = None
            tables = inner.rope_tables(layer)
            for _ in range(opt.blocks):
                a_in = h + inner.embed_scale * inj
                q = layer.q_proj(a_in).reshape(x.shape[0], T, 1, -1).transpose(1, 2).float()
                q = q / (q.norm(dim=-1, keepdim=True) + inner.config.eps)
                qs.append(inner.apply_rope(q, layer, tables)[:, 0])
                h_new, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                hs.append((h_new - h).norm(dim=-1).mean(-1))
                h = h_new
        recs = hy.CTL['record']
        hy.CTL['record'] = None
        for b in range(1, opt.blocks):
            r, qr = recs[b], qs[b]
            ekr, eki, evr, evi = r['traces']
            lk2, lv2 = hy.rep2(r['lk']), hy.rep2(r['lv'])
            post_now = r['v'] * torch.exp(1j * hy.rep2(r['pv']))
            pre_past = r['rope']((lk2 * ekr)[:, None])[:, 0] + 1j * r['rope']((lk2 * eki)[:, None])[:, 0]
            post_past = lv2 * evr + 1j * lv2 * evi
            pre_now = r['kr'] * torch.exp(-1j * hy.rep2(r['pk']))
            xn = post_now[..., :, None] * pre_past[..., None, :] + post_past[..., :, None] * pre_now[..., None, :]   # [B,T,D,D]
            out = {}
            for name, m in (('odd', xn.imag), ('even', xn.real)):
                s = m.sum(1)
                out[f'cancel {name}'] = float((s.norm(dim=(-1, -2)) / m.norm(dim=(-1, -2)).sum(1)).mean())
                nrm = s.norm(dim=(-1, -2))
                dh = hs[b]
                out[f'corr {name}~dh'] = float(torch.corrcoef(torch.stack([nrm, dh]))[0, 1])
            reads = {'Hebb': torch.einsum('btd,bnd,bne->bte', qr, r['kr'], r['v']),
                     'odd (STDP)': torch.einsum('btd,bed->bte', qr, xn.imag.sum(1)),
                     'even (order-blind)': torch.einsum('btd,bed->bte', qr, xn.real.sum(1))}
            for rname, feat in reads.items():
                for tname, tgt in targets.items():
                    out[f'{rname} | {tname}'] = probe(feat, tgt, split)
            for k_, v_ in out.items():
                agg.setdefault((b + 1, k_), []).append(v_)
    mean = lambda b, k_: sum(agg[(b, k_)]) / len(agg[(b, k_)])
    print(f'seeds {opt.seeds}; random init, bounded hybrid (omegaT 0.34, c 0.14, lambda 0.5), alpha 1\n')
    print('                         ' + ' '.join(f'   block {b}' for b in show))
    for k_ in ('cancel odd', 'cancel even', 'corr odd~dh', 'corr even~dh'):
        print(f'  {k_:22s} ' + ' '.join(f'{mean(b, k_):10.3f}' for b in show))
    print('\nprobe accuracy from the read (per cell), block ' + ', '.join(map(str, show)))
    for tname in targets:
        print(f'  {tname}')
        for rname in ('Hebb', 'odd (STDP)', 'even (order-blind)'):
            print(f'    {rname:20s} ' + ' '.join(f'{mean(b, f"{rname} | {tname}"):10.3f}' for b in show))
    (OUT / 'results' / 'rule110_unsup_init.json').write_text(json.dumps({f'{b}|{k_}': v_ for (b, k_), v_ in agg.items()}, indent=1) + '\n')


if __name__ == '__main__':
    main()
