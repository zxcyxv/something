"""Same sweep for the EMA-trained B (outer_eta 0.125): from seg17 on the EMA rate is scaled by f (scale preserved:
W' = (1 - f eta) W' + f eta Im S, injected through the probe so the model's W equals W'). Solved@T (late solves)."""
import sys, torch
from lt.analysis.common import load_model, run_blocks
from lt.data import load_data, eval_batches
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
ckpt = sys.argv[1]; T_list = [int(t) for t in sys.argv[2].split(',')]; factors = [float(f) for f in sys.argv[3].split(',')]
model, cfg = load_model(ckpt, dev); eta0 = float(model.config.outer_eta); bps = model.config.blocks_per_seg
_, _, x, y, _, _ = load_data(None, 512)
batch = next(eval_batches(x[:512], y[:512], 512)); b = {k: v.to(dev) for k, v in batch.items()}; labels = b['labels']
Tmax = max(T_list)
print(f'{ckpt}: eta {eta0}, factor applied from seg17; solved@T (late solves) for T in {T_list}')
print('factor ' + ' '.join(f'{"T=" + str(t):>14}' for t in T_list), flush=True)
for f in factors:
    state = {'W': None}
    def probe(rec, f=f, state=state):
        n = rec.count[:, :1, :1, :1]
        fresh = (n == 0)
        if state['W'] is None or bool(fresh.any()): state['W'] = torch.zeros_like(rec.im_s)
        eta = torch.where(n >= 16 * bps, torch.full_like(rec.im_s[..., :1, :1], f * eta0), torch.full_like(rec.im_s[..., :1, :1], eta0))
        Wp = state['W']; Wn = (1 - eta) * Wp + eta * rec.im_s; state['W'] = Wn
        return (Wn - (1 - eta0) * Wp) / eta0   # the model computes W = (1 - eta0) W + eta0 x  ->  x makes it equal Wn
    model.probe = probe
    preds = run_blocks(model, b, Tmax * bps)
    solved = (preds == labels[None]).all(-1)[bps - 1::bps]
    s16 = solved[15]
    cells = [f'{int(solved[t-1].sum()):4d} ({int((solved[t-1] & ~s16).sum()) if t > 16 else 0:3d})' for t in T_list]
    print(f'{f:6.3f} ' + ' '.join(f'{c:>14}' for c in cells), flush=True)
