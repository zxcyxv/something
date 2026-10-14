"""Inference interventions on the bounded hybrid Rule 110 checkpoint: is the block order (omegaT offset) used?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_hybrid_interventions.py [--ckpt NAME]

Held-out 4,096 rows, 16 blocks (trained with 8). Conditions (only the tags fed to the write change):
  as trained          omegaT = +0.34, bounded tags
  reverse order       omegaT = -0.34: pairs k blocks apart get -k omegaT (pre earlier -> LTD); |sin| about the same
  no order            omegaT = 0: cross-block pairs keep only the bounded within-block difference (smaller |sin|)
  flip within-block   bounded tags negated, offset kept: same-block LTP <-> LTD, cross-block order unchanged
  alpha = 0           no timing term (scale-changing reference)
"""
import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_hybrid as hy                                   # noqa: E402
from rule110_ltp_ltd import DEVICE, OUT, rp                   # noqa: E402

MODE = dict(omega=0.34, sign_within=1.0)


def advance(*a, **kw):
    pk, pv, lk, lv, st = hy._ADVANCE(*a, **kw)
    off = hy.CTL['block'] * MODE['omega']
    hy.CTL['block'] += 1
    c = hy.CTL['bound']
    return MODE['sign_within'] * c * torch.sin(pk) + off, MODE['sign_within'] * c * torch.sin(pv) + off, lk, lv, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='rule110_k4_hybrid_wT0.34_lam0.5_bound0.14_seed0')
    opt = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(OUT / 'checkpoints' / f'{opt.ckpt}.pt', map_location=DEVICE)
    hy.CTL['bound'] = ck['bound']
    rp.advance_clock = advance
    base, inner = r110.build('run3', ck['seed'])
    base.load_state_dict(ck['state'])
    base.eval()
    layer = inner.layers[0]
    x, y = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    w = ck['omega_t']
    conds = [('as trained', w, 1.0, False), ('reverse order', -w, 1.0, False), ('no order (omegaT=0)', 0.0, 1.0, False),
             ('flip within-block', w, -1.0, False), ('alpha = 0', w, 1.0, True)]
    show = (2, 3, 4, 8, 12, 16)
    print('exact (cell) at blocks ' + ', '.join(map(str, show)))
    report = {}
    for label, omega, sw, a0 in conds:
        MODE.update(omega=omega, sign_within=sw)
        saved = layer.stdp_alpha.data.clone()
        if a0:
            layer.stdp_alpha.data.zero_()
        with torch.no_grad():
            lg = hy.forward(inner, x, 16)
        layer.stdp_alpha.data.copy_(saved)
        cell, exact = r110.accuracy(lg, y)
        report[label] = dict(exact=exact.tolist(), cell=cell.tolist())
        print(f'  {label:22s} ' + '  '.join(f'{float(exact[b - 1]):.3f} ({float(cell[b - 1]):.3f})' for b in show))
    (OUT / 'results' / f'{opt.ckpt}_interventions.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
