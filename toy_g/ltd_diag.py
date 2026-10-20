"""Diagnostics of a trained LTDRuleModel (toy_g/ltd_rule.py): what the learned basis is and what the accuracy rests on.

    python -m toy_g.ltd_diag runs/ltd/learned_Q

1. how far Q moved from its start, and R = F^H Q (the learned channels in Fourier modes): mass of each channel on its
   main D4 orbit of frequencies (orbits: {0}, {(2,2)}, {(0,2),(2,0)}, three of size 4)
2. energy sparsity of operators in the basis: entries holding 90% / 99% of the energy, for the 128 lawful
   transformations and 1,000 random permutations (a monomial needs 16 for both)
3. accuracy on the fixed instances (k = 3) as trained, with the sparsity LTD off (theta_w = 0), and with Q replaced by
   its start Q0 (same scalars)
4. the in-context W after the demonstrations against the true operator of the task in the basis (relative error)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import group as G
from .ltd_rule import LTDRuleModel, fourier, images, infer, read


def energy_count(M, frac):
    e = (M.abs() ** 2).flatten(1).sort(1, descending=True).values
    c = e.cumsum(1) / e.sum(1, keepdim=True)
    return (c < frac).sum(1).float() + 1


def operators(src, Q):
    P = torch.zeros(len(src), 16, 16, dtype=Q.dtype, device=Q.device)
    P[torch.arange(len(src))[:, None], torch.arange(16)[None], src] = 1
    return Q.T @ P @ Q.conj()                                                 # W u = v for code images (column convention)


@torch.no_grad()
def accuracy(model, inst, dev, theta_w=None, Q=None):
    n = len(inst['task'])
    din = torch.as_tensor(inst['demo_in'].reshape(n, 3, 16).astype(np.int64), device=dev)
    dout = torch.as_tensor(inst['demo_out'].reshape(n, 3, 16).astype(np.int64), device=dev)
    qin = torch.as_tensor(inst['test_in'].reshape(n, 16).astype(np.int64), device=dev)
    target = torch.as_tensor(inst['test_out'].reshape(n, 16).astype(np.int64), device=dev)
    Q = model.codes() if Q is None else Q
    th = model.log_theta_w.exp() if theta_w is None else torch.tensor(theta_w, device=dev)
    U, V = images(din, Q), images(dout, Q)
    W, C = infer(U, V, torch.ones(n, 3, dtype=torch.bool, device=dev), model.log_s_w.exp().clamp(max=1.0), th,
                 model.log_s_c.exp().clamp(max=1.0), model.log_theta_c.exp(), model.T)
    s = read(images(qin, Q), W, C, Q)
    logits = torch.cat([model.bg.expand(n, 1, 16), s], 1)
    ok = (logits.argmax(1) == target).all(1).cpu().numpy()
    g = inst['group'][inst['task']]
    return {f'g{k}': round(float(ok[g == k].mean()), 4) for k in range(3)}, W


@torch.no_grad()
def main(run):
    dev = torch.device('cuda')
    cfg = json.loads((Path(run) / 'config.json').read_text())
    model = LTDRuleModel(cfg['basis'], cfg['seed'], cfg['T'], cfg.get('T_grad', 50)).to(dev)
    model.load_state_dict(torch.load(Path(run) / 'final.pt', map_location=dev))
    model.eval()
    Q, Q0, F = model.codes(), model.Q0, fourier(dev)
    print(f'|Q - Q0|_F / |Q0|_F = {float((Q - Q0).norm() / Q0.norm()):.3f}')
    freqs = [(a, b) for a in range(4) for b in range(4)]
    orbits = [{(0, 0)}, {(2, 2)}, {(0, 2), (2, 0)}, {(0, 1), (0, 3), (1, 0), (3, 0)}, {(1, 1), (1, 3), (3, 1), (3, 3)},
              {(1, 2), (2, 1), (2, 3), (3, 2)}]
    orbit_of = torch.tensor([next(i for i, o in enumerate(orbits) if f in o) for f in freqs], device=dev)
    for name, B in (('start Q0', Q0), ('learned Q', Q)):
        R = F.conj().T @ B                                                    # [16 freq, 16 channels]
        mass = torch.zeros(6, 16, device=dev).index_add_(0, orbit_of, (R.abs() ** 2).real)
        top_mode = (R.abs() ** 2).max(0).values
        print(f'{name}: channel mass on its main orbit: median {mass.max(0).values.median():.3f} min {mass.max(0).values.min():.3f};'
              f' on its main single mode: median {top_mode.median():.3f}')
    rng = np.random.default_rng(0)
    for name, B in (('fourier', F), ('start Q0', Q0), ('learned Q', Q)):
        lawful = operators(torch.as_tensor(G.SRC, device=dev), B)
        rand = operators(torch.as_tensor(np.stack([rng.permutation(16) for _ in range(1000)]), device=dev), B)
        print(f'{name:9s}: entries for 90% / 99% energy: lawful {energy_count(lawful, .9).median():.0f} / '
              f'{energy_count(lawful, .99).median():.0f}, random permutations {energy_count(rand, .9).median():.0f} / '
              f'{energy_count(rand, .99).median():.0f}')
    inst = D.load_instances()
    acc, W = accuracy(model, inst, dev)
    print('accuracy as trained:', acc, '| theta_w =', round(model.log_theta_w.exp().item(), 4))
    print('sparsity LTD off   :', accuracy(model, inst, dev, theta_w=0.0)[0])
    print('Q replaced by Q0   :', accuracy(model, inst, dev, Q=Q0)[0])
    print('Q replaced by F    :', accuracy(model, inst, dev, Q=F)[0])
    t = inst['task'] // G.NUM_C
    P = operators(torch.as_tensor(G.SRC[t], device=dev), Q)
    rel = ((W - P).norm(dim=(1, 2)) / P.norm(dim=(1, 2))).cpu().numpy()
    print(f'in-context W vs true operator (relative error): median {np.median(rel):.3f}, 90th pct {np.percentile(rel, 90):.3f}; '
          f'W entries for 90% energy: median {energy_count(W, .9).median():.0f}')


if __name__ == '__main__':
    main(sys.argv[1])
