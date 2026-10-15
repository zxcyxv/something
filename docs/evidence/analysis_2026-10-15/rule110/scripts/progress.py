"""Grokking-style progress measures over the rerun's checkpoints (every 250 steps)."""
import sys, glob, re, numpy as np, torch
sys.path.insert(0, sys.argv[1]); from r110lib import *
torch.set_grad_enabled(False)
R = sys.argv[1]; BLK = [1, 2, 4, 8, 16, 32, 64]; NROW = 8192
H = hadamard(); f = truth9().double(); Ft = (2 * f - 1).float(); st = (H @ Ft) / 512; sup = st.abs() > 1e-6
deg = torch.tensor([bin(s).count('1') for s in range(512)], device=dev)
gen0 = torch.Generator(device=dev).manual_seed(321); x, y = sample(NROW, gen0)
w4, w3 = windows(x, 4).flatten(), windows(x, 3).flatten()
cnt4 = torch.bincount(w4, minlength=512).double(); cnt3 = torch.bincount(w3, minlength=128).double()
rows = []
files = sorted(glob.glob(f'{R}/ckpt/step_*.pt'))
for fp in files:
    step = int(re.findall(r'step_(\d+)', fp)[0])
    model, cfg = build(fp)
    rec = {}
    def each(b, h, logits, box, state):
        if b not in BLK: return
        z = (logits[..., 2] - logits[..., 1])[:, 8:28].double().flatten()
        acc = float(((logits.argmax(-1) - 1) == y).all(-1).double().mean())
        g = torch.zeros(512, device=dev, dtype=torch.float64).index_add_(0, w4, z) / cnt4.clamp_min(1)
        within = lambda w, c: float(((torch.zeros_like(c).index_add_(0, w, z * z)) - (torch.zeros_like(c).index_add_(0, w, z)) ** 2 / c.clamp_min(1)).sum())
        tot = float(((z - z.mean()) ** 2).sum())
        sm = (H.double() @ g) / 512
        e = sm ** 2
        rec[b] = dict(acc=acc, cos=float(sm @ st.double() / (sm.norm() * st.norm() + 1e-12)), off=float(e[~sup].sum() / e.sum()),
                      deg=[float(e[deg == k].sum() / e.sum()) for k in range(10)], r4=1 - within(w4, cnt4) / tot, r3=1 - within(w3, cnt3) / tot,
                      table=float(((g > 0).double() == f).double().mean()))
    run(model, x, max(BLK), each)
    pn = float(sum((p.double() ** 2).sum() for n, p in model.named_parameters() if p.ndim >= 2) ** 0.5)
    rows.append(dict(step=step, pnorm=pn, alpha=float(model.block.alpha), blocks={str(b): rec[b] for b in BLK}))
    print(step, f"acc8 {rec[8]['acc']:.3f} acc32 {rec[32]['acc']:.3f} cos8 {rec[8]['cos']:.3f} table8 {rec[8]['table']:.3f} pnorm {pn:.2f}", flush=True)
import json; json.dump(rows, open(f'{R}/progress.json', 'w'))
