"""Segment training (2 blocks per segment, 4 segments, carry detached, loss and update per segment) on Rule 110,
then iteration to 128 blocks. Bounded hybrid clock (c 0.14, omegaT 0.34, lambda 0.5 fixed).

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_segments.py --model {plain,leaky} [--batches 2000] [--rho 0.5]
                          [--optim adamw|adamatan2] [--lr 1e-3] [--wd 0.01]

plain  read = q~ (Hebb_b + alpha Im S_b)^T                    (bounded hybrid, no outer integral)
leaky  W_b = rho W_{b-1} + Im S_b, read = q~ (Hebb_b + alpha W_b)^T
Per batch of 512 fresh rows: 4 segments x 2 blocks; at each segment end a cross-entropy loss on that block's readout,
backward, gradient clip 1.0, optimizer step (AdamW: wd except alpha/lambda/theta/biases; AdamATan2: the harness class and
its no-decay rule lt.train._is_no_decay, decoupled decay p <- p (1 - lr wd)), then the carry (hidden state,
clock state, traces, W, block counter) is kept detached for the next segment. 2,000 batches = 8,000 optimizer steps.
"""
import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_hybrid as hy                                   # noqa: E402
import rule110_leaky_train as lk                              # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402

SEG_BLOCKS, SEGMENTS = 2, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('plain', 'leaky'), required=True)
    ap.add_argument('--batches', type=int, default=2000)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iterate', type=int, default=128)
    ap.add_argument('--optim', choices=('adamw', 'adamatan2'), default='adamw')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-blocks', type=int, default=16)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14, record=None)
    rp.advance_clock, rp.phase_clock_step = hy.advance, lk.step
    lk.ACC.update(on=opt.model == 'leaky', rho=opt.rho)
    name = f'rule110_k{r110.K_STEPS}_seg2x4_{opt.model}' + (f'{opt.rho:g}' if opt.model == 'leaky' else '')
    if opt.optim != 'adamw' or opt.lr != 1e-3 or opt.wd != 0.01:
        name += f'_{opt.optim}_lr{opt.lr:g}_wd{opt.wd:g}'
    if opt.batches != 2000:
        name += f'_b{opt.batches}'
    name += f'_seed{opt.seed}'
    base, inner = lk.build(opt.seed)
    layer = inner.layers[0]
    params = [(n, p) for n, p in base.named_parameters() if p.requires_grad]
    if opt.optim == 'adamw':
        no_decay = lambda n, p: n.endswith('bias') or any(k in n for k in ('stdp_alpha', 'clock_lam_raw', 'theta'))
        make = lambda groups: torch.optim.AdamW(groups, lr=opt.lr)
    else:
        no_decay = lambda n, p: t._is_no_decay(n, p)
        make = lambda groups: t.AdamATan2(groups, lr=opt.lr)
    optim = make([dict(params=[p for n, p in params if not no_decay(n, p)], weight_decay=opt.wd),
                  dict(params=[p for n, p in params if no_decay(n, p)], weight_decay=0.0)])
    gen = torch.Generator(device=DEVICE).manual_seed(1000 + opt.seed)
    xt, yt = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    log, began = [], time.monotonic()
    for bi in range(1, opt.batches + 1):
        x, y = r110.sample(512, gen)
        batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(512, dtype=torch.int32, device=DEVICE))
        hy.CTL['block'], lk.ACC['W'] = 0, None
        h = inner.init_hidden.float().expand(512, T, HIDDEN).clone()
        mem = ek = ev = None
        for seg in range(SEGMENTS):
            inj = inner.injection(batch)
            for _ in range(SEG_BLOCKS):
                h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
            loss = F.cross_entropy(inner.w_cls(h).reshape(-1, 3), (y + 1).reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
            optim.step()
            h, mem, ek, ev = h.detach(), mem.detach(), ek.detach(), ev.detach()
            if lk.ACC['W'] is not None:
                lk.ACC['W'] = lk.ACC['W'].detach()
        if bi % opt.eval_every == 0 or bi == opt.batches:
            r = lk.iterate(inner, xt, yt, opt.eval_blocks)
            row = dict(batch=bi, loss=float(loss.detach()), seconds=time.monotonic() - began, alpha=float(layer.stdp_alpha),
                       exact=[round(v_, 4) for v_ in r['exact']])
            log.append(row)
            print(json.dumps(row), flush=True)
    torch.save(dict(state=base.state_dict(), model=opt.model, rho=opt.rho, batches=opt.batches, seed=opt.seed,
                    optim=opt.optim, lr=opt.lr, wd=opt.wd),
               OUT / 'checkpoints' / f'{name}.pt')
    base.eval()
    r = lk.iterate(inner, xt, yt, opt.iterate)
    show = [b for b in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if b <= opt.iterate]
    best = max(range(len(r['exact'])), key=lambda i: r['exact'][i])
    print(f'\n[{name}] finite: {r["finite"]}; best exact {r["exact"][best]:.4f} at block {best + 1}')
    print('  block      ' + ' '.join(f'{b:>7d}' for b in show))
    print('  exact      ' + ' '.join(f'{r["exact"][b - 1]:7.4f}' for b in show))
    print('  cell       ' + ' '.join(f'{r["cell"][b - 1]:7.4f}' for b in show))
    print('  ||matrix|| ' + ' '.join(f'{r["norm"][b - 1]:7.2f}' for b in show))
    print('  drift      ' + ' '.join(f'{r["drift"][b - 1]:7.4f}' for b in show))
    (OUT / 'results' / f'{name}.json').write_text(json.dumps(dict(train=log, iterate=r), indent=1) + '\n')


if __name__ == '__main__':
    main()
