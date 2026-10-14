"""Run-3 model with a fixed common phase advance per block (damped-oscillator traces), trained on Rule 110.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_hybrid.py [--steps 2000] [--omega-t 0.5] [--lam 0.5] [--bound 0] [--seed 0]

Change from run 3: the K and V tags of block b (b = 0 for the first block of a puzzle) both get + b * omegaT.
Same-block pairs are unchanged (the common advance cancels); a pair whose events are k blocks apart gets + k*omegaT on
the post-minus-pre phase, i.e. the trace is carried as lam * e^{-i omegaT} per block. lam_K = lam_V = lam fixed (not
trained); omegaT fixed.
--bound c > 0 first maps each tag to c * sin(tag), so within-block phase differences lie in [-2c, 2c]; with
omegaT > 2c and 8 omegaT + 2c < pi every pair up to 8 blocks apart takes its LTP/LTD sign from the block order. Everything else as rule110_ltp_ltd.py (hidden 32, one head, 16 K/V pairs, 8 blocks, loss after
every block, AdamW lr 1e-3).
After training, on held-out rows unsolved after block 1 (blocks up to 16; 9.. beyond the training horizon):
  per-block exact accuracy;
  ||same-block|| and ||cross-block|| parts of alpha Im S (sum over cells), as in rule110_crossblock_init.py;
  for lag k = 1..8 at the same cell: P(LTP) when the pre (K) event is k blocks earlier vs when the post (V) is,
  using the tags actually fed to the write (offset included), plain and amplitude-weighted.
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
from rule110_ltp_ltd import T, BLOCKS, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402

CTL = dict(block=0, omega_t=0.5, bound=0.0, record=None)
_ADVANCE = rp.advance_clock
_STEP = rp.phase_clock_step


def advance(*a, **kw):
    pk, pv, lk, lv, st = _ADVANCE(*a, **kw)
    offset = CTL['block'] * CTL['omega_t']
    CTL['block'] += 1
    if CTL['bound'] > 0:
        pk, pv = CTL['bound'] * torch.sin(pk), CTL['bound'] * torch.sin(pv)
    return pk + offset, pv + offset, lk, lv, st


def step(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = _STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    if CTL['record'] is not None:
        CTL['record'].append(dict(k=k_unit[:, 0].detach(), kr=kr[:, 0].detach(), v=v[:, 0].detach(), pk=pk[:, 0].detach(),
                                  pv=pv[:, 0].detach(), lk=lk[:, 0].detach(), lv=lv[:, 0].detach(),
                                  traces=tuple(z[:, 0].detach() for z in traces), rope=rope))
    return hebb, im_s, new


def forward(inner, x, blocks=BLOCKS):
    CTL['block'] = 0
    return r110.forward(inner, x, blocks)


def rep2(z):
    return z.repeat_interleave(2, -1)


def amp(z):
    return torch.sqrt(z[..., 0::2] ** 2 + z[..., 1::2] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--omega-t', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--bound', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    CTL['omega_t'], CTL['bound'] = opt.omega_t, opt.bound
    rp.advance_clock, rp.phase_clock_step = advance, step
    name = f'rule110_k{r110.K_STEPS}_hybrid_wT{opt.omega_t:g}_lam{opt.lam:g}' + (f'_bound{opt.bound:g}' if opt.bound > 0 else '') + f'_seed{opt.seed}'
    base, inner = r110.build('run3', opt.seed)
    layer = inner.layers[0]
    with torch.no_grad():
        layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(opt.lam)))
    layer.clock_lam_raw.requires_grad_(False)

    # ---- training (as rule110_ltp_ltd.train)
    no_decay = lambda n: n.endswith('bias') or any(k in n for k in ('stdp_alpha', 'clock_lam_raw', 'theta'))
    params = [(n, p) for n, p in base.named_parameters() if p.requires_grad]
    optim = torch.optim.AdamW([dict(params=[p for n, p in params if not no_decay(n)], weight_decay=0.01),
                               dict(params=[p for n, p in params if no_decay(n)], weight_decay=0.0)], lr=1e-3)
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
        if s % 500 == 0 or s == opt.steps:
            with torch.no_grad():
                cell, exact = r110.accuracy(forward(inner, xt), yt)
            row = dict(step=s, loss=float(loss.detach()), seconds=time.monotonic() - began, alpha=float(layer.stdp_alpha),
                       test_cell=[round(float(v_), 4) for v_ in cell], test_exact=[round(float(v_), 4) for v_ in exact])
            log.append(row)
            print(json.dumps(row), flush=True)
    (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=base.state_dict(), steps=opt.steps, omega_t=opt.omega_t, lam=opt.lam, bound=opt.bound, seed=opt.seed),
               OUT / 'checkpoints' / f'{name}.pt')

    # ---- observation
    base.eval()
    nb = 16
    with torch.no_grad():
        lg = forward(inner, xt, nb)
    pred = lg.argmax(-1) - 1
    uns = ~(pred[0] == yt).all(-1)
    print('\nheld-out 4096: exact per block 1..16: ' + ' '.join(f'{float(v_):.3f}' for v_ in (pred == yt).all(-1).float().mean(-1)))
    x = xt[uns][:512]
    CTL['record'] = []
    with torch.no_grad():
        forward(inner, x, nb)
    recs = CTL['record']
    CTL['record'] = None
    a = layer.stdp_alpha.detach()
    print(f'alpha {float(a):.3f}, lambda {opt.lam}, omegaT {opt.omega_t}; rows unsolved after block 1: {int(uns.sum())} (using 512)')
    print('block | ||Hebb|| ||same|| ||cross|| | lag k: P(LTP) pre-first / post-first  (amplitude-weighted)')
    rows = []
    outer = lambda p, q: p[..., :, None] * q[..., None, :]
    for b in range(1, nb):
        r = recs[b]
        ekr, eki, evr, evi = r['traces']
        ck, sk, cv, sv = (rep2(z) for z in (r['pk'].cos(), r['pk'].sin(), r['pv'].cos(), r['pv'].sin()))
        lk2, lv2 = rep2(r['lk']), rep2(r['lv'])
        same = a * outer(r['v'], r['kr']) * (1 - lk2)[..., None, :] * torch.sin(rep2(r['pv'])[..., :, None] - rep2(r['pk'])[..., None, :])
        rope = r['rope']
        pre_re, pre_im = rope((lk2 * ekr)[:, None])[:, 0], rope((lk2 * eki)[:, None])[:, 0]
        dvr, dvi = lv2 * evr, lv2 * evi
        cross = a * (outer(r['v'] * cv, pre_im) + outer(r['v'] * sv, pre_re) + outer(dvi, r['kr'] * ck) - outer(dvr, r['kr'] * sk))
        row = dict(block=b + 1, hebb=float(outer(r['v'], r['kr']).sum(1).norm(dim=(-1, -2)).mean()),
                   same=float(same.sum(1).norm(dim=(-1, -2)).mean()), cross=float(cross.sum(1).norm(dim=(-1, -2)).mean()), lags={})
        parts = []
        for k in range(1, 9):
            if b - k < 0:
                continue
            r0 = recs[b - k]
            s_pre = torch.sin(r['pv'][..., :, None] - r0['pk'][..., None, :])            # K k blocks earlier
            s_post = torch.sin(r0['pv'][..., :, None] - r['pk'][..., None, :])           # V k blocks earlier
            w_pre = amp(r['v'])[..., :, None] * amp(r0['kr'])[..., None, :]
            w_post = amp(r0['v'])[..., :, None] * amp(r['kr'])[..., None, :]
            vals = (float((s_pre > 0).float().mean()), float((s_post > 0).float().mean()),
                    float((w_pre * (s_pre > 0)).sum() / w_pre.sum()), float((w_post * (s_post > 0)).sum() / w_post.sum()))
            row['lags'][k] = vals
            parts.append(f'k{k} {vals[0]:.2f}/{vals[1]:.2f}')
        rows.append(row)
        print(f"{b + 1:5d} | {row['hebb']:7.2f} {row['same']:7.2f} {row['cross']:7.2f} | " + '  '.join(parts))
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    (OUT / 'results' / f'{name}.json').write_text(json.dumps(dict(train=log, observe=rows, alpha=float(a)), indent=1) + '\n')


if __name__ == '__main__':
    main()
