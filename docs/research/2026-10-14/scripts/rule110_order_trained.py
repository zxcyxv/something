"""Block order vs LTP/LTD in the trained Rule 110 run-3 checkpoint (lag-1 pairs at the same cell).

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_order_trained.py [--seed 0] [--blocks 16]

Held-out rows unsolved after block 1. For blocks b >= 2 and every cell, post V pair a, pre K pair i:
  pre first   K at block b-1, V at block b:  LTP if sin(phi_V,a(b) - phi_K,i(b-1)) > 0
  post first  V at block b-1, K at block b:  LTP if sin(phi_V,a(b-1) - phi_K,i(b)) > 0
Shares over all pairs, and weighted by the product of the two pairs' amplitudes (|v pair| |k~ pair|).
Block 1 tags are 0 by definition. Trained with 8 blocks; blocks 9.. are beyond the training horizon.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_dissect as rd                                  # noqa: E402
from rule110_ltp_ltd import DEVICE, OUT                       # noqa: E402


def amp(x):
    return torch.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)     # pair amplitudes [B, T, P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=16)
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
    x = xt[~((lg[0].argmax(-1) - 1) == yt).all(-1)][:1024]
    ctl['record'], ctl['block'] = [], 0
    with torch.no_grad():
        r110.forward(inner, x, opt.blocks)
    recs = ctl.pop('record')
    rows = []
    print('block | P(LTP) pre first  post first | amplitude-weighted: pre first  post first')
    for b in range(1, opt.blocks):
        r0, r1 = recs[b - 1], recs[b]
        s_pre = torch.sin(r1['pv'][..., :, None] - r0['pk'][..., None, :])           # [B, T, post a, pre i]
        s_post = torch.sin(r0['pv'][..., :, None] - r1['pk'][..., None, :])
        w_pre = amp(r1['v'])[..., :, None] * amp(r0['kr'])[..., None, :]
        w_post = amp(r0['v'])[..., :, None] * amp(r1['kr'])[..., None, :]
        row = dict(block=b + 1, p_pre=float((s_pre > 0).float().mean()), p_post=float((s_post > 0).float().mean()),
                   w_pre=float((w_pre * (s_pre > 0)).sum() / w_pre.sum()), w_post=float((w_post * (s_post > 0)).sum() / w_post.sum()))
        rows.append(row)
        print(f"{b + 1:5d} | {row['p_pre']:.3f}  {row['p_post']:.3f} | {row['w_pre']:.3f}  {row['w_post']:.3f}")
    (OUT / 'results' / f'{name}_order.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
