"""Share of the order term O in Im S and in Im S's fluctuation (seg17..64), per class. Same subset as alt.py."""
import sys, torch
from lt.analysis.common import load_model, run_blocks, OrderLatencyProbe
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
from lt.data import load_data, eval_batches
dev = torch.device('cuda'); R = torch.load(sys.argv[2])
model, cfg = load_model(sys.argv[1], dev); bps = model.config.blocks_per_seg
_, _, x, y, _, _ = load_data(None, 512)
batch = next(eval_batches(x[:512], y[:512], 512)); b = {k: v.to(dev)[R['sub'].to(dev)] for k, v in batch.items()}
olp = OrderLatencyProbe(model.config.block_omega, model.config.trace_lambda); keep = {'S': [], 'O': []}; n = [0]
def probe(rec):
    olp(rec); t = n[0]; n[0] += 1
    if 16 * bps <= t < 64 * bps:
        keep['S'].append(rec.im_s.half().cpu()); keep['O'].append(olp.last['O'].half().cpu())
model.probe = probe; run_blocks(model, b, 64 * bps)
X = torch.stack(keep['S']).float(); O = torch.stack(keep['O']).float(); Lt = X - O        # [t, B, H, D, D]
f = lambda a: a.pow(2).sum((2, 3, 4))                                                       # [t, B]
dev_ = lambda a: a - a.mean(0, keepdim=True)
cls = R['cls']
for name, val in [('|O|^2/|ImS|^2', f(O).mean(0) / f(X).mean(0)),
                  ('|O fluct|^2/|ImS fluct|^2', f(dev_(O)).mean(0) / f(dev_(X)).mean(0)),
                  ('|ImS fluct|^2/|ImS|^2', f(dev_(X)).mean(0) / f(X).mean(0))]:
    print(f'{name:28s} ' + '  '.join(f'{c} {float(val[[k for k, cc in enumerate(cls) if cc == c]].mean()):.3f}' for c in 'SLN'))
