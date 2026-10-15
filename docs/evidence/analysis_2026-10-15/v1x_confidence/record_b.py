"""Same recording for a lt/ (main) checkpoint: B 40k. usage: python record_b.py <ckpt> <out.pt> <name>"""
import sys, torch
from lt.analysis.common import load_model
from lt.data import load_data, eval_batches
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
ckpt, out_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
model, cfg = load_model(ckpt, dev)
_, _, x, y, _, _ = load_data(None, 512)
batch = next(eval_batches(x[:512], y[:512], 512)); b = {k: v.to(dev) for k, v in batch.items()}
mc = model.config; saved = mc.loops; mc.loops = 129
carry = model.initial_carry(b)
A, H, Pm = [], [], []
for s in range(128):
    carry, logits = model(carry, b)
    lg = logits.float(); dig = lg[..., 2:11]; p = torch.softmax(dig, -1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    A.append(lg.argmax(-1).to(torch.int8).cpu()); H.append(ent.half().cpu()); Pm.append(p.max(-1).values.half().cpu())
mc.loops = saved
torch.save(dict(name=name, step=cfg.get('step'), P=torch.stack(A), H=torch.stack(H), Pmax=torch.stack(Pm),
                y=b['labels'].cpu(), given=(b['inputs'] != 1).cpu()), out_path)
solved = (torch.stack(A) == b['labels'].cpu().to(torch.int8)[None]).all(-1)
print(f'{name}: solved seg16 {int(solved[15].sum())}, best {int(solved.any(0).sum())}, seg128 {int(solved[-1].sum())}', flush=True)
