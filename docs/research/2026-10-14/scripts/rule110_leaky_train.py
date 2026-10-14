"""Bounded hybrid + leaky outer integral (accumulated STDP synapse), trained on Rule 110, then iterated past the horizon.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_leaky_train.py [--steps 2000] [--rho 0.9] [--seed 0]
                          [--optim adamw|adamatan2] [--lr 1e-3] [--wd 0.01] [--eval-every 500] [--eval-blocks 8]

Model: rule110_hybrid.py with --omega-t 0.34 --bound 0.14 --lam 0.5 (tags c sin(phi) + b omegaT, lambda fixed), and
the read matrix W_b = rho W_{b-1} + Im S_b (W reset to 0 for a new input; gradients flow through the accumulation),
read = q~ (Hebb_b + alpha W_b)^T. Training as rule110_ltp_ltd.py (8 blocks, loss after every block, AdamW 1e-3).
After training, held-out 4,096 rows are iterated for 128 blocks: exact / cell accuracy, mean ||W_b||, state change.
The bounded hybrid checkpoint without the outer integral is evaluated the same way for reference (no training).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_hybrid as hy                                   # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, BLOCKS, DEVICE, OUT, rp  # noqa: E402
from lt import train as t                                     # noqa: E402

ACC = dict(on=True, rho=0.9, W=None, norms=None)


def step(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = hy._STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    if not ACC['on']:
        m = im_s
    else:
        ACC['W'] = im_s if ACC['W'] is None else ACC['rho'] * ACC['W'] + im_s
        m = ACC['W']
    if ACC['norms'] is not None:
        ACC['norms'].append(float(m.detach().norm(dim=(-1, -2)).mean()))
    return hebb, m, new


def forward(inner, x, blocks=BLOCKS):
    ACC['W'] = None
    return hy.forward(inner, x, blocks)


def iterate(inner, x, y, blocks):
    """Per-block exact and cell accuracy, ||matrix||, state change, without keeping all logits."""
    ACC['W'], ACC['norms'] = None, []
    hy.CTL['block'] = 0
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=DEVICE))
    exact, cell, drift = [], [], []
    with torch.no_grad():
        h = inner.init_hidden.float().expand(x.shape[0], T, HIDDEN).clone()
        inj = inner.injection(batch)
        mem = ek = ev = None
        layer = inner.layers[0]
        for _ in range(blocks):
            h_new, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
            drift.append(float(((h_new - h).norm(dim=-1) / h_new.norm(dim=-1)).mean()))
            h = h_new
            pred = inner.w_cls(h).argmax(-1) - 1
            exact.append(float((pred == y).all(-1).float().mean()))
            cell.append(float((pred == y).float().mean()))
    norms, ACC['norms'] = ACC['norms'], None
    return dict(exact=exact, cell=cell, norm=norms, drift=drift, finite=bool(torch.isfinite(h).all()))


def build(seed):
    base, inner = r110.build('run3', seed)
    layer = inner.layers[0]
    with torch.no_grad():
        layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(0.5)))
    layer.clock_lam_raw.requires_grad_(False)
    return base, inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--rho', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iterate', type=int, default=128)
    ap.add_argument('--optim', choices=('adamw', 'adamatan2'), default='adamw')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-blocks', type=int, default=8)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14, record=None)
    rp.advance_clock, rp.phase_clock_step = hy.advance, step
    ACC.update(on=True, rho=opt.rho)
    name = f'rule110_k{r110.K_STEPS}_hybrid_wT0.34_lam0.5_bound0.14_leaky{opt.rho:g}'
    if opt.optim != 'adamw' or opt.lr != 1e-3 or opt.wd != 0.01:
        name += f'_{opt.optim}_lr{opt.lr:g}_wd{opt.wd:g}'
    if opt.steps != 2000:
        name += f'_s{opt.steps}'
    name += f'_seed{opt.seed}'
    base, inner = build(opt.seed)
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
    for s in range(1, opt.steps + 1):
        x, y = r110.sample(512, gen)
        logits = forward(inner, x)
        loss = F.cross_entropy(logits.reshape(-1, 3), (y + 1).repeat(BLOCKS, 1).reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        optim.step()
        if s % opt.eval_every == 0 or s == opt.steps:
            r = iterate(inner, xt, yt, opt.eval_blocks)
            row = dict(step=s, loss=float(loss.detach()), seconds=time.monotonic() - began, alpha=float(layer.stdp_alpha),
                       exact=[round(v_, 4) for v_ in r['exact']])
            log.append(row)
            print(json.dumps(row), flush=True)
    torch.save(dict(state=base.state_dict(), steps=opt.steps, omega_t=0.34, lam=0.5, bound=0.14, rho=opt.rho, seed=opt.seed,
                    optim=opt.optim, lr=opt.lr, wd=opt.wd),
               OUT / 'checkpoints' / f'{name}.pt')
    base.eval()
    runs = {'leaky outer integral (trained now)': iterate(inner, xt, yt, opt.iterate)}
    ref = torch.load(OUT / 'checkpoints' / 'rule110_k4_hybrid_wT0.34_lam0.5_bound0.14_seed0.pt', map_location=DEVICE)
    base_r, inner_r = build(0)
    base_r.load_state_dict(ref['state'])
    base_r.eval()
    ACC['on'] = False
    runs['no outer integral (bounded hybrid, reference)'] = iterate(inner_r, xt, yt, opt.iterate)
    ACC['on'] = True
    show = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    show = [b for b in show if b <= opt.iterate]
    for label, r in runs.items():
        best = max(range(len(r['exact'])), key=lambda i: r['exact'][i])
        print(f'\n[{label}]  finite: {r["finite"]}; best exact {r["exact"][best]:.4f} at block {best + 1}')
        print('  block      ' + ' '.join(f'{b:>7d}' for b in show))
        print('  exact      ' + ' '.join(f'{r["exact"][b - 1]:7.4f}' for b in show))
        print('  cell       ' + ' '.join(f'{r["cell"][b - 1]:7.4f}' for b in show))
        print('  ||matrix|| ' + ' '.join(f'{r["norm"][b - 1]:7.2f}' for b in show))
        print('  drift      ' + ' '.join(f'{r["drift"][b - 1]:7.4f}' for b in show))
    (OUT / 'results' / f'{name}.json').write_text(json.dumps(dict(train=log, iterate=runs), indent=1) + '\n')


if __name__ == '__main__':
    main()
