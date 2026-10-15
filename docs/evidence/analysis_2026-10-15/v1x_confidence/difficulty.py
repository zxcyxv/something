"""Difficulty of the first 512 held-out puzzles: clues, L0 rounds (depth without lookahead), cells needing L1, L1 rounds."""
import sys, numpy as np, torch, time
sys.path.insert(0, '/workspace/something')
from lt.analysis.depth import propagate
from lt.data import load_data
_, _, x, y, _, _ = load_data(None, 512)
# load_data encoding: 0 = blank, 1..9 digits, shape [N,9,9] (as lt/analysis/interventions.py)
xd = np.asarray(x[:512]).reshape(-1, 81).astype(np.int64); yd = np.asarray(y[:512]).reshape(-1, 81).astype(np.int64)
t = time.time(); d0, st0 = propagate(xd, yd, lookahead=False); print('L0 done %.0fs' % (time.time() - t), flush=True)
t = time.time(); d1, st1 = propagate(xd, yd, lookahead=True, inner=0); print('L1 done %.0fs' % (time.time() - t), flush=True)
clues = (xd > 0).sum(1)
l0_rounds = np.where((d0 >= 0).all(1), d0.max(1), -1)                 # -1: not L0-solvable
l1_needed = ((d0 < 0) & (xd == 0)).sum(1)                             # blanks unresolved by L0
l1_rounds = np.where((d1 >= 0).all(1), d1.max(1), -1)
torch.save(dict(clues=torch.tensor(clues), l0_rounds=torch.tensor(l0_rounds), l1_needed=torch.tensor(l1_needed),
                l1_rounds=torch.tensor(l1_rounds), depth0=torch.tensor(d0), depth1=torch.tensor(d1)), sys.argv[1])
print('clues mean %.1f; L0-solvable %d/512; L1-solvable %d/512; L1-needed cells mean %.1f' % (
    clues.mean(), (l0_rounds >= 0).sum(), (l1_rounds >= 0).sum(), l1_needed.mean()))
