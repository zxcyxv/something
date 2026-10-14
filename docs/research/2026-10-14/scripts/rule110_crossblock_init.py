"""Cross-block pairs of the run-3 write at random initialization, with a larger trace retention lambda.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_crossblock_init.py [--seeds 0 1 2] [--blocks 16] [--n 256]

Model: the run-3 class at the Rule 110 size (hidden 32, one head, 16 K and 16 V pairs), random initialization,
no training; lambda_K = lambda_V set to 0.1 / 0.5 / 0.9 (a pair from k blocks ago enters with weight (1-lam) lam^k
on the earlier event). Inputs: random 36-cell rows (tokens bit + 1).
Per block, the timing term alpha Im S is split exactly into the same-block part and the cross-block part (pairs whose
earlier event is in an earlier block), each as a sum over cells n of per-cell matrices X_n. Reported:
  size      ||sum_n X_n||_F for Hebb, same-block, cross-block (alpha = 1)
  cancel    ||sum_n X_n||_F / sum_n ||X_n||_F  (1: all cells add up; ~1/sqrt(36) = 0.17: random signs cancel)
  order     lag-1 pairs at the same cell: P(LTP) when the pre event (K) is in block b-1 and the post (V) in block b,
            versus when the post is in b-1 and the pre in b (STDP by block order would give > 0.5 vs < 0.5)
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
from rule110_ltp_ltd import T, DEVICE, OUT, rp                # noqa: E402
from lt import train as t                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--n', type=int, default=256)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    x = torch.randint(0, 2, (opt.n, T), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    report = {}
    original = rp.phase_clock_step
    for lam in (0.1, 0.5, 0.9):
        for seed in opt.seeds:
            base, inner = r110.build('run3', seed)
            base.eval()
            layer = inner.layers[0]
            with torch.no_grad():
                layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(lam)))
            ctl = {}
            rd.install(inner, ctl)
            stats = []

            def hooked(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
                hebb, im_s, new = original(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
                ekr, eki, evr, evi = traces
                ck, sk, cv, sv = (rd.rep2(z) for z in (pk.cos(), pk.sin(), pv.cos(), pv.sin()))
                lk2, lv2 = rd.rep2(lk), rd.rep2(lv)
                a = layer.stdp_alpha.detach()
                outer = lambda p, q: p[..., :, None] * q[..., None, :]         # per cell [B,H,T,D_v,D_k]
                same = a * outer(v, kr) * (1 - lk2)[..., None, :] * torch.sin(rd.rep2(pv)[..., :, None] - rd.rep2(pk)[..., None, :])
                pre_re, pre_im = rope(lk2 * ekr), rope(lk2 * eki)                 # past K events, decayed to now
                dvr, dvi = lv2 * evr, lv2 * evi                                   # past V events, decayed to now
                cross = a * (outer(v * cv, pre_im) + outer(v * sv, pre_re) + outer(dvi, kr * ck) - outer(dvr, kr * sk))
                heb = outer(v, kr)
                total = (same + cross).sum(2)
                err = float((total - a * im_s).norm() / (a * im_s).norm().clamp_min(1e-12))
                row = dict(err=err)
                for name, xn in (('hebb', heb), ('same', same), ('cross', cross)):
                    s = xn.sum(2)
                    row[name] = float(s.norm(dim=(-1, -2)).mean())
                    row[name + '_cancel'] = float((s.norm(dim=(-1, -2)) / xn.norm(dim=(-1, -2)).sum(2).clamp_min(1e-12)).mean())
                stats.append(row)
                return hebb, im_s, new
            rp.phase_clock_step = hooked
            ctl['record'], ctl['block'] = [], 0
            try:
                with torch.no_grad():
                    r110.forward(inner, x, opt.blocks)
            finally:
                rp.phase_clock_step = original
            recs = ctl.pop('record')
            for b in range(1, opt.blocks):
                pk0, pv0, pk1, pv1 = recs[b - 1]['pk'], recs[b - 1]['pv'], recs[b]['pk'], recs[b]['pv']
                pre_first = torch.sin(pv1[..., :, None] - pk0[..., None, :]) > 0           # K at b-1, V at b
                post_first = torch.sin(pv0[..., :, None] - pk1[..., None, :]) > 0          # V at b-1, K at b
                stats[b].update(p_ltp_pre_first=float(pre_first.float().mean()), p_ltp_post_first=float(post_first.float().mean()),
                                phase=float(pk1.abs().mean() + pv1.abs().mean()) / 2)
            report[f'lam{lam}_seed{seed}'] = stats
    # print seed means
    show = [2, 3, 4, 8, 16]
    for lam in (0.1, 0.5, 0.9):
        runs = [report[f'lam{lam}_seed{s}'] for s in opt.seeds]
        mean = lambda key, b: sum(r[b - 1][key] for r in runs) / len(runs)
        print(f'\n=== lambda {lam}  (seeds {opt.seeds}; max split error {max(r[b - 1]["err"] for r in runs for b in range(2, opt.blocks + 1)):.1e})')
        print('  block                     ' + ' '.join(f'{b:>7d}' for b in show))
        for key, label in (('hebb', '||Hebb||'), ('same', '||same-block||'), ('cross', '||cross-block||'),
                           ('hebb_cancel', 'cancel Hebb'), ('same_cancel', 'cancel same'), ('cross_cancel', 'cancel cross'),
                           ('p_ltp_pre_first', 'P(LTP) pre first'), ('p_ltp_post_first', 'P(LTP) post first'), ('phase', 'mean |phase|')):
            print(f'  {label:25s} ' + ' '.join(f'{mean(key, b):7.3f}' for b in show))
    (OUT / 'results' / 'rule110_crossblock_init.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
