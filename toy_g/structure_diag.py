"""Exactness diagnostics of a trained FreeStructureModel (version C), for the identifiability argument of
docs/claude_2026-10-20_arc_logic_foundations.md section 8.

r_w = realised read map of word w (output o reads cell argmax_x cos(U_w g_o, g_x)); pi_j = r_(j) for the letters.
- valid(L): fraction of length-L words whose r_w is one of the 128 toy transformations
- compositional(L): fraction of (w, j) with |w| = L - 1 and r_{wj} = pi_j o r_w   (L-exactness, letter by letter)
- letter margins: min over cells of cos(A_j g_x, g_pi(x)) - max_{y != pi(x)} cos(A_j g_x, g_y)
- Cayley ball: the set {pi_w : |w| <= L} under exact composition of the letters vs the realised maps of words <= L

    python -m toy_g.structure_diag runs/rs/C_diag_lam1 runs/rs/C_diag_lam0
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import group as G
from .learned_structure import FreeStructureModel


def load(run):
    cfg = json.loads((Path(run) / 'config.json').read_text())
    m = FreeStructureModel(d=cfg['d'], K=cfg['K'], N=cfg['N'], lam=cfg.get('lam', 0.0), read=cfg.get('read', 'cos'))
    m.load_state_dict(torch.load(Path(run) / 'final.pt', map_location='cpu'), strict=False)
    return m.eval()


@torch.no_grad()
def main(runs):
    rows = {tuple(r): t for t, r in enumerate(G.SRC.tolist())}
    heldout = D.make_split(seed=0)['heldout_t_elements'].tolist()
    for run in runs:
        m = load(run)
        maps = m.read_weights().argmax(-1).numpy()                              # [W, 16]
        index = {w: i for i, w in enumerate(m.words)}
        lens = np.array([len(w) for w in m.words])
        t_of = np.array([rows.get(tuple(r), -1) for r in maps.tolist()])
        pi = {j: maps[index[(j,)]] for j in range(m.K)}
        print(f'== {run}: letters realise ' + ', '.join(
            f'{j}:{"t" + str(t_of[index[(j,)]]) if t_of[index[(j,)]] >= 0 else ("perm" if len(set(pi[j])) == 16 else "non-perm")}'
            for j in range(m.K)))
        line_v, line_c = [], []
        for L in range(1, m.N + 1):
            sel = lens == L
            line_v.append(f'{L}:{(t_of[sel] >= 0).mean():.2f}')
            ok = [np.array_equal(maps[index[w + (j,)]], pi[j][maps[index[w]]]) for w in m.words if len(w) == L - 1
                  for j in range(m.K)]
            line_c.append(f'{L}:{np.mean(ok):.2f}')
        print('  valid fraction by length        ' + ' '.join(line_v))
        print('  compositional fraction by length ' + ' '.join(line_c))
        A = m.primitives()
        g = F.normalize(m.G, dim=-1)
        cos = torch.einsum('jxd,yd->jxy', torch.einsum('jde,xe->jxd', A, g), g)  # [K, 16(x), 16(y)]
        margins = []
        for j in range(m.K):
            c = cos[j].clone()
            top = c[torch.arange(16), torch.as_tensor(pi[j])]
            c[torch.arange(16), torch.as_tensor(pi[j])] = -9
            margins.append(float((top - c.max(-1).values).min()))
        off = (g @ g.T - torch.eye(16)).abs().max().item()
        print('  letter margins ' + ' '.join(f'{x:+.2f}' for x in margins) + f' | code max |cos| off-diagonal {off:.2f}')
        # exact Cayley balls under the letters' permutations vs realised maps
        ball, frontier, realised_any = {tuple(range(16))}, {tuple(range(16))}, set()
        line_b = []
        for L in range(1, m.N + 1):
            frontier = {tuple(pi[j][list(p)]) for p in frontier for j in range(m.K)} - ball
            ball |= frontier
            real = {tuple(r) for r, l in zip(maps.tolist(), lens) if l <= L}
            line_b.append(f'{L}:{len(ball)}/{len(real & ball)}/{len(real)}')
        print('  Cayley ball size / realised maps inside it / realised maps, by max length ' + ' '.join(line_b))
        G_ball = {rows[p] for p in ball if p in rows}
        short = {}
        for t, l in zip(t_of, lens):
            if t >= 0:
                short[t] = min(short.get(t, 99), int(l))
        print(f'  exact ball contains {len(G_ball)} of the 128 transformations (held-out {sum(h in G_ball for h in heldout)}/16); '
              f'held-out shortest realised lengths {sorted(short.get(h, -1) for h in heldout)}')


if __name__ == '__main__':
    main(sys.argv[1:])
