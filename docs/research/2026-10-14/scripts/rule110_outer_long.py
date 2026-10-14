"""Long-run norm of the accumulated STDP synapse (V1 of rule110_outer_init.py) at random initialization.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_outer_long.py [--seed 0] [--blocks 512] [--rows 512] [--rho 1]

Bounded hybrid clock (c 0.14, omegaT 0.34, lambda 0.5), random init, alpha 1, W_b = rho W_{b-1} + Im S_b read by the model
(rho = 1: pure outer integral; rho < 1: leaky integral).
Per block (mean over rows): ||W_b||_F, ||Im S_b||_F (the increment), cos(W_{b-1}, Im S_b), ||h_b - h_{b-1}|| / ||h_b||.
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
import rule110_hybrid as hy                                   # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402

STATE = dict(W=None, rows=[], rho=1.0)


def step(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = hy._STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    prev = STATE['W']
    W = im_s if prev is None else STATE['rho'] * prev + im_s
    row = dict(W=float(W.norm(dim=(-1, -2)).mean()), inc=float(im_s.norm(dim=(-1, -2)).mean()))
    if prev is not None:
        row['cos'] = float(((prev * im_s).sum((-1, -2)) / (prev.norm(dim=(-1, -2)) * im_s.norm(dim=(-1, -2))).clamp_min(1e-12)).mean())
    STATE['W'] = W
    STATE['rows'].append(row)
    return hebb, W, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=512)
    ap.add_argument('--rows', type=int, default=512)
    ap.add_argument('--rho', type=float, default=1.0)
    opt = ap.parse_args()
    STATE['rho'] = opt.rho
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14, record=None, block=0)
    rp.advance_clock, rp.phase_clock_step = hy.advance, step
    x = torch.randint(0, 2, (opt.rows, T), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    base, inner = r110.build('run3', opt.seed)
    base.eval()
    layer = inner.layers[0]
    with torch.no_grad():
        layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(0.5)))
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(opt.rows, dtype=torch.int32, device=DEVICE))
    drift = []
    with torch.no_grad():
        h = inner.init_hidden.float().expand(opt.rows, T, HIDDEN).clone()
        inj = inner.injection(batch)
        mem = ek = ev = None
        for _ in range(opt.blocks):
            h_new, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
            drift.append(float(((h_new - h).norm(dim=-1) / h_new.norm(dim=-1)).mean()))
            h = h_new
    rows = STATE['rows']
    for b, r in enumerate(rows):
        r['drift'] = drift[b]
    finite = bool(torch.isfinite(h).all()) and all(math.isfinite(r['W']) for r in rows)
    print(f'seed {opt.seed}, rows {opt.rows}, blocks {opt.blocks}, rho {opt.rho}, all finite: {finite}')
    print(' block |   ||W||   ||inc||  cos(W_prev, inc)  state change')
    for b in [2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]:
        if b <= len(rows):
            r = rows[b - 1]
            print(f'{b:6d} | {r["W"]:8.2f} {r["inc"]:8.3f} {r.get("cos", float("nan")):10.3f} {r["drift"]:12.4f}')
    for lo, hi in [(16, 64), (64, 128), (128, 256), (256, 512)]:
        if hi <= len(rows):
            print(f'  growth of ||W|| per block, blocks {lo}-{hi}: {(rows[hi - 1]["W"] - rows[lo - 1]["W"]) / (hi - lo):.3f}')
    (OUT / 'results' / (f'rule110_outer_long_seed{opt.seed}.json' if opt.rho == 1 else f'rule110_outer_long_rho{opt.rho:g}_seed{opt.seed}.json')).write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
