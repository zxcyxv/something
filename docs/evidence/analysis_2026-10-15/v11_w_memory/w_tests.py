"""Is the continued evolution of the slow coupling memory w necessary for late solves in v1.x?
Run from the archive worktree root: python w_tests.py <ckpt.npz> <name> <out.pt>
Variants applied from segment 17 on (parameters changed in place between segments; state continues):
  base      as trained
  freeze_w  eta -> 0        (w frozen at its seg16 value, still read)
  no_read_w lam -> 0        (w keeps updating but is not read; read = instantaneous coupling only)
  eta_x4    eta -> min(0.99, 4 eta)   (memory 4x shorter, as in the late v1.71 drift)
  eta_d4    eta -> eta / 4            (memory 4x longer)
Baseline also records per puzzle and segment: |dw|/|w|, and the alignment of w with the solution's same-digit relation.
"""
import sys, math, torch
sys.path.insert(0, 'lt')
from ckpt_npz import load_lt, load_data
torch.set_grad_enabled(False)
ckpt, name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
m, cfg, step = load_lt(ckpt, batch_size=512, loops=129)
X, Y, batch = load_data('.', 512)
y = batch['labels']; given = (batch['inputs'] != 1)
layers = m.inner.layers
orig = {id(L): (L.eta_raw.detach().clone(), L.lam_raw.detach().clone()) for L in layers}
def restore():
    for L in layers:
        e, l = orig[id(L)]; L.eta_raw.data.copy_(e); L.lam_raw.data.copy_(l)
def apply(variant):
    for L in layers:
        e, l = orig[id(L)]
        if variant == 'freeze_w': L.eta_raw.data.fill_(-30.0)
        elif variant == 'no_read_w': L.lam_raw.data.fill_(-30.0)
        elif variant == 'eta_x4':
            eta = torch.sigmoid(e).clamp(max=0.99 / 4) * 4; L.eta_raw.data.copy_(torch.log(eta / (1 - eta)))
        elif variant == 'eta_d4':
            eta = torch.sigmoid(e) / 4; L.eta_raw.data.copy_(torch.log(eta / (1 - eta)))
idx = torch.arange(81, device=y.device); r, c = idx // 9, idx % 9
NB = (((r[:, None] == r[None]) | (c[:, None] == c[None]) | (((r // 3)[:, None] == (r // 3)[None]) & ((c // 3)[:, None] == (c // 3)[None]))) & ~torch.eye(81, dtype=torch.bool, device=y.device))
same = (y[:, :, None] == y[:, None, :]) & ~torch.eye(81, dtype=torch.bool, device=y.device)[None]   # [N,81,81] same solution digit
diff = ~same & ~torch.eye(81, dtype=torch.bool, device=y.device)[None]
results = {}
for variant in ('base', 'freeze_w', 'no_read_w', 'eta_x4', 'eta_d4'):
    restore()
    carry = m.initial_carry(batch); solved = []; dw_rel = []; w_norm = []; align = []; align_nb = []
    w_prev = None
    for s in range(128):
        if s == 16: apply(variant)
        carry, out = m(carry, batch)
        pred = out['logits'].argmax(-1); solved.append((pred == y).all(-1).cpu())
        w = carry.coupling.float()                                                 # [N,H,81,81]
        if variant == 'base':
            wn = w.flatten(1).norm(dim=-1)
            w_norm.append(wn.cpu())
            dw_rel.append(((w - w_prev).flatten(1).norm(dim=-1) / (wn + 1e-12)).cpu() if w_prev is not None else torch.zeros(512))
            # alignment: per head, (mean w over same-digit pairs - mean over different-digit pairs) / std of w; averaged over heads
            wm = w.mean(1)                                                        # [N,81,81] head-averaged coupling
            sd = wm.flatten(1).std(dim=-1) + 1e-12
            a_same = (wm * same).sum((1, 2)) / same.sum((1, 2)); a_diff = (wm * diff).sum((1, 2)) / diff.sum((1, 2))
            align.append(((a_same - a_diff) / sd).cpu())
            nb = NB[None].expand_as(same)
            a_nb = (wm * nb).sum((1, 2)) / nb.sum((1, 2)); a_nnb = (wm * (~nb & diff)).sum((1, 2)) / (~nb & diff).sum((1, 2))
            align_nb.append(((a_nb - a_nnb) / sd).cpu())
            w_prev = w.clone()
    solved = torch.stack(solved); s16 = solved[15]; late = solved[16:].any(0) & ~s16; never = ~solved.any(0)
    results[variant] = dict(solved=solved, late=int(late.sum()), never=int(never.sum()), s16=int(s16.sum()), s128=int(solved[-1].sum()),
                            lost=int((s16 & ~solved[-1]).sum()))
    if variant == 'base':
        results['base'].update(dw_rel=torch.stack(dw_rel), w_norm=torch.stack(w_norm), align=torch.stack(align), align_nb=torch.stack(align_nb))
    print(f"{name} {variant:>10}: seg16 {results[variant]['s16']}, late-solved {results[variant]['late']}, never {results[variant]['never']}, seg128 {results[variant]['s128']}, lost after 16 {results[variant]['lost']}", flush=True)
restore()
torch.save(dict(name=name, step=step, results=results, y=y.cpu(), given=given.cpu()), out_path)
# baseline dynamics summary
R = results['base']; solved = R['solved']; s16 = solved[15]; late = solved[16:].any(0) & ~s16; never = ~solved.any(0)
dw, al, aln = R['dw_rel'], R['align'], R['align_nb']
print(f"\n{name} baseline: |dw|/|w| per segment (median), alignment(same-digit) and alignment(neighbour) by group")
print(f"{'seg':>4} {'dw late':>8} {'dw never':>9} {'dw solved@16':>12} {'align late':>10} {'align never':>11} {'align s16':>9} {'alignNB late':>12} {'alignNB never':>13}")
for seg in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
    i = seg - 1
    f = lambda t, m: float(t[i][m].median()) if m.any() else float('nan')
    print(f"{seg:4d} {f(dw, late):8.4f} {f(dw, never):9.4f} {f(dw, s16):12.4f} {f(al, late):10.3f} {f(al, never):11.3f} {f(al, s16):9.3f} {f(aln, late):12.3f} {f(aln, never):13.3f}")
# does w's motion at seg16 predict a late solve? quartiles of dw_rel at seg16 among unsolved puzzles
uns = ~s16; d16 = dw[15][uns]; l = late[uns]
q = torch.quantile(d16, torch.tensor([0.25, 0.5, 0.75]))
print(f"unsolved@16: late-solve rate by quartile of |dw|/|w| at seg16: " + ", ".join(
    f"Q{k+1} {float(l[(d16 > (q[k-1] if k > 0 else -1)) & (d16 <= (q[k] if k < 3 else 1e9))].float().mean())*100:.0f}%" for k in range(4)) + f"  (edges {q.tolist()})")
