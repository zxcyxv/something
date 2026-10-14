"""Do hidden states cluster by local context at random initialization when STDP increments are accumulated?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_cluster_init.py [--seed 0] [--blocks 16] [--rows 2048] [--rhos 0.5 0.9]

Same models as rule110_outer_init.py (bounded hybrid clock, random init, alpha 1): V0 increment only; STDP leaky
W_b = rho W_{b-1} + Im S_b; order-blind leaky W_b = rho W_{b-1} + Re S_b. Inputs: random 36-cell rows.
A cell's own input bit is encoded perfectly in its state, so clustering is measured WITHIN each own-bit group, with the
class = (left input bit, right input bit) (4 classes), i.e. information that must come from other cells:
  within/total   tr(within-class scatter) / tr(total scatter) of the hidden states (lower = tighter classes)
  k-means NMI    k-means (k = 4, 8 restarts, 25 iterations) on the states, normalized mutual information with the class
Both averaged over the two own-bit groups; 20,000 cells per group.
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
import rule110_outer_init as oi                               # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402
from lt import train as t                                     # noqa: E402


def nmi(a, b):
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    joint = torch.zeros(ka, kb, device=a.device)
    joint.index_put_((a, b), torch.ones_like(a, dtype=torch.float), accumulate=True)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    nz = p > 0
    mi = (p[nz] * (p[nz] / (pa @ pb)[nz]).log()).sum()
    ent = lambda q: -(q[q > 0] * q[q > 0].log()).sum()
    return float(2 * mi / (ent(pa) + ent(pb)))


def kmeans(xs, k, gen, restarts=8, iters=25):
    best, best_inertia = None, float('inf')
    for _ in range(restarts):
        c = xs[torch.randperm(xs.shape[0], generator=gen, device=xs.device)[:k]].clone()
        for _ in range(iters):
            lab = torch.cdist(xs, c).argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = xs[m].mean(0)
        inertia = float(((xs - c[lab]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def scores(h, x, gen, n=20000):
    left, right = F.pad(x, (1, 0))[:, :-1], F.pad(x, (0, 1))[:, 1:]
    cls = (2 * left + right).reshape(-1)
    own = x.reshape(-1)
    hs = h.reshape(-1, h.shape[-1])
    wt, nm = [], []
    for bit in (0, 1):
        idx = (own == bit).nonzero()[:, 0]
        idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)[:n]]
        z, c = hs[idx], cls[idx]
        total = ((z - z.mean(0)) ** 2).sum()
        within = sum(((z[c == j] - z[c == j].mean(0)) ** 2).sum() for j in range(4) if (c == j).any())
        wt.append(float(within / total))
        nm.append(nmi(kmeans(z, 4, gen), c))
    return sum(wt) / 2, sum(nm) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--rhos', type=float, nargs='+', default=[0.5, 0.9])
    opt = ap.parse_args()
    torch.set_num_threads(4)
    hy.CTL.update(omega_t=0.34, bound=0.14, record=None)
    rp.advance_clock, rp.phase_clock_step = hy.advance, oi.step
    x = torch.randint(0, 2, (opt.rows, T), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    configs = [('V0', 1.0, 'increment only')] + [(m, r, ('STDP' if m == 'V1' else 'order-blind') + f' leaky rho {r:g}')
                                                 for r in opt.rhos for m in ('V1', 'V2')]
    show = [b for b in (1, 2, 4, 8, 12, 16) if b <= opt.blocks]
    report = {}
    print('within-own-bit clustering by (left, right) input; within/total (lower = tighter) | k-means NMI (higher = better)')
    print('  blocks ' + ', '.join(map(str, show)))
    for mode, rho, label in configs:
        base, inner = r110.build('run3', opt.seed)
        base.eval()
        layer = inner.layers[0]
        with torch.no_grad():
            layer.clock_lam_raw.fill_(t.inv_softplus(-math.log(0.5)))
        oi.ACC.update(mode=mode, rho=rho, W=None, mats=[])
        hy.CTL['block'] = 0
        batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(opt.rows, dtype=torch.int32, device=DEVICE))
        rows = []
        gen = torch.Generator(device=DEVICE).manual_seed(5)
        with torch.no_grad():
            h = inner.init_hidden.float().expand(opt.rows, T, HIDDEN).clone()
            inj = inner.injection(batch)
            mem = ek = ev = None
            for b in range(1, opt.blocks + 1):
                h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                if b in show:
                    rows.append((b,) + scores(h.float(), x, gen))
        report[label] = rows
        print(f'  {label:26s} ' + '  '.join(f'{w:.3f}|{n:.3f}' for _, w, n in rows))
    (OUT / 'results' / 'rule110_cluster_init.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
