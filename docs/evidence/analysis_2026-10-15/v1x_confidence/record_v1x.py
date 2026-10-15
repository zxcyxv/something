"""Record per-segment argmax, digit-entropy and max-prob of a v1.x npz checkpoint on the first 512 held-out puzzles,
128 segments (run from the archive worktree root; usage: python record_v1x.py <ckpt.npz> <out.pt> <name>)."""
import sys, torch, math
sys.path.insert(0, 'lt')
from ckpt_npz import load_lt, load_data
torch.set_grad_enabled(False)
ckpt, out_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
m, cfg, step = load_lt(ckpt, batch_size=512, loops=129)
X, Y, batch = load_data('.', 512)
carry = m.initial_carry(batch)
A, H, Pm = [], [], []
for s in range(128):
    carry, out = m(carry, batch)
    lg = out['logits'].float()                       # [N,81,V]
    dig = lg[..., 2:11]                              # digit classes 1..9
    p = torch.softmax(dig, -1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    A.append(lg.argmax(-1).to(torch.int8).cpu()); H.append(ent.half().cpu()); Pm.append(p.max(-1).values.half().cpu())
torch.save(dict(name=name, step=step, P=torch.stack(A), H=torch.stack(H), Pmax=torch.stack(Pm),
                y=batch['labels'].cpu(), given=(batch['inputs'] != 1).cpu()), out_path)
solved = (torch.stack(A) == batch['labels'].cpu().to(torch.int8)[None]).all(-1)
print(f'{name} step {step}: solved seg16 {int(solved[15].sum())}, best {int(solved.any(0).sum())}, seg128 {int(solved[-1].sum())}', flush=True)
