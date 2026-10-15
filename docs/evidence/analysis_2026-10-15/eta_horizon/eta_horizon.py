"""Is the optimal memory rate tied to the run length?  v1.1 160k, eta scaled from seg17 on by factors f (eta' = f * eta),
run for T segments; report solved at seg T and late solves (unsolved at 16 -> solved by T).  Run from the archive root."""
import sys, torch
sys.path.insert(0, 'lt')
from ckpt_npz import load_lt, load_data
torch.set_grad_enabled(False)
ckpt = sys.argv[1]; T_list = [int(t) for t in sys.argv[2].split(',')]; factors = [float(f) for f in sys.argv[3].split(',')]
Tmax = max(T_list)
m, cfg, step = load_lt(ckpt, batch_size=512, loops=Tmax + 1)
X, Y, batch = load_data('.', 512); y = batch['labels']
layers = m.inner.layers; orig = {id(L): L.eta_raw.detach().clone() for L in layers}
def set_factor(f):
    for L in layers:
        eta = torch.sigmoid(orig[id(L)]) * f; eta = eta.clamp(1e-5, 0.99); L.eta_raw.data.copy_(torch.log(eta / (1 - eta)))
print(f'{ckpt} step {step}: eta factor applied from seg17; solved@T (late solves) for T in {T_list}')
header = 'factor ' + ' '.join(f'{"T=" + str(t):>14}' for t in T_list); print(header, flush=True)
for f in factors:
    for L in layers: L.eta_raw.data.copy_(orig[id(L)])
    carry = m.initial_carry(batch); solved = []
    for s in range(Tmax):
        if s == 16: set_factor(f)
        carry, out = m(carry, batch); solved.append((out['logits'].argmax(-1) == y).all(-1).cpu())
    solved = torch.stack(solved); s16 = solved[15]
    cells = []
    for t in T_list:
        st = solved[t - 1]; late = (st & ~s16).sum() if t > 16 else torch.tensor(0)
        cells.append(f'{int(st.sum()):4d} ({int(late):3d})')
    print(f'{f:6.3f} ' + ' '.join(f'{c:>14}' for c in cells), flush=True)
for L in layers: L.eta_raw.data.copy_(orig[id(L)])
