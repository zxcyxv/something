"""seg16 vs best up to seg128 for every extrapolation json in a diagnostics dir.

error reduction = (best exact - seg16 exact) / (n - seg16 exact)
"""
import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
rows = []
for p in D.glob('extrap_step*_n*_seg*.json'):
    m = re.match(r'extrap_step(\d+)_(ema|raw)_n(\d+)_seg(\d+)\.json', p.name)
    if not m:
        continue
    d = json.loads(p.read_text())
    seg = {s['segment']: s for s in d['segments']}
    tr, best, fin = d['train'], d['best_exact'], d['final']
    n = d['n']
    red = (best['exact'] - tr['exact']) / max(n - tr['exact'], 1)
    rows.append((m[2], int(m[3]), int(m[1]), n, tr, best, fin, red, seg))
for w, n_, step, n, tr, best, fin, red, seg in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
    pick = ' '.join(f"{s}:{seg[s]['exact']}" for s in (16, 32, 64, 128) if s in seg)
    print(f"{w:3s} n={n:4d} step {step:6d}: seg16 {tr['exact']:4d} ({tr['exact']/n:.1%}, acc {tr['acc']:.4f})"
          f" -> best {best['exact']:4d} @seg{best['segment']:3d} ({best['exact']/n:.1%}, acc {seg[best['segment']]['acc']:.4f})"
          f" | final {fin['exact']:4d} | err.red {red:.1%} | {pick}")
