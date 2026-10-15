"""EMA form of model B's STDP memory at inference, scale-preserving: B trained with W = 0.5 W + Im S = 2 EMA_{0.5}(Im S).
Here W' = 2 EMA_eta(Im S) for several eta (eta = 0.5 reproduces B exactly); the fixed point 2 Im S* is unchanged, only the
averaging window (1/eta blocks) changes. Implemented through Model.probe: the model keeps W = 0.5 W + x, so returning
x = W'_b - 0.5 W'_{b-1} makes the model's W equal W'.  Reports seg16 / late / never / lost on 512 held-out puzzles, 128 segments."""
import sys, torch
from lt.analysis.common import load_model, run_blocks
from lt.data import load_data, eval_batches
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
ckpt = sys.argv[1]; etas = [float(e) for e in sys.argv[2].split(',')]
model, cfg = load_model(ckpt, dev)
_, _, x, y, _, _ = load_data(None, 512)
batch = next(eval_batches(x[:512], y[:512], 512)); b = {k: v.to(dev) for k, v in batch.items()}; labels = b['labels']
rho = model.config.outer_rho; assert rho == 0.5
for eta in etas:
    state = {'W': None}
    def probe(rec, eta=eta, state=state):
        if eta == 0.5: return None                                    # B as trained
        fresh = (rec.count[:, :1, :1, :1] == 0)
        if state['W'] is None or bool(fresh.any()): state['W'] = torch.zeros_like(rec.im_s)
        Wp = state['W']; Wn = (1 - eta) * Wp + 2 * eta * rec.im_s
        state['W'] = Wn
        return Wn - rho * Wp
    model.probe = probe
    preds = run_blocks(model, b, 128 * 8)                              # [blocks, N, 81]
    solved = (preds == labels[None]).all(-1)[7::8]                     # per segment [128, N]
    s16 = solved[15]; late = solved[16:].any(0) & ~s16; never = ~solved.any(0); lost = s16 & ~solved[-1]
    print(f"eta {eta:<6} (window {1/eta:5.1f} blocks): seg16 {int(s16.sum())}, late-solved {int(late.sum())}, never {int(never.sum())}, seg128 {int(solved[-1].sum())}, lost after 16 {int(lost.sum())}", flush=True)
