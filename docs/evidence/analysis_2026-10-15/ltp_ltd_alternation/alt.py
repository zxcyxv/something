"""Do LTP/LTD alternate at the same synapse entry (i, j) of Im S in puzzles that do not settle? B-EMA 50k.
Pass 0: 512 held-out puzzles, 128 segments -> classes S (solved by seg16), L (solved 17..128, stays), N (never).
Pass 1 (subset): final W* = model W at the last block (eta 1/8 EMA of Im S, reconstructed exactly).
Pass 2 (same subset, same batch, deterministic): per block, for X = Im S and X = O (order part):
  * cancellation C(L) = sum_ij |sum_{t in w} X_t| / sum_ij sum_{t in w} |X_t| over aligned windows of L blocks
  * lag-1 correlation of increments corr(X_t - X_{t-1}, X_{t-1} - X_{t-2})  (negative = alternation)
  * cos(EMA_L(t), W*) for EMA windows L (eta = 1/L), Im S only, pre-solve blocks."""
import sys, json, torch
from lt.analysis.common import load_model, run_blocks, suffix_block, OrderLatencyProbe
from lt.data import load_data, eval_batches
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
ckpt = sys.argv[1]; SEGS = 128
model, cfg = load_model(ckpt, dev); bps = model.config.blocks_per_seg; eta0 = float(model.config.outer_eta)
NB = SEGS * bps
_, _, x, y, _, _ = load_data(None, 512)
batch = next(eval_batches(x[:512], y[:512], 512)); b = {k: v.to(dev) for k, v in batch.items()}
preds = run_blocks(model, b, NB)
ok = (preds == b['labels'][None]).all(-1)                          # [NB, B]
solve_blk = suffix_block(ok)                                       # first block of the final solved run (NB+1 = never)
s16 = ok[16 * bps - 1]; s128 = ok[-1]
S = torch.where(s16 & s128)[0]; L = torch.where(~s16 & s128 & (solve_blk > 16 * bps))[0]; N = torch.where(~s128)[0]
print(f'classes: S {len(S)}  L {len(L)}  N {len(N)}', flush=True)
g = torch.Generator().manual_seed(0)
pick = lambda idx, n: idx[torch.randperm(len(idx), generator=g)[:n]].to(idx.device)
sub = torch.cat([pick(S, 24), L, pick(N, 24)]); cls = ['S'] * min(24, len(S)) + ['L'] * len(L) + ['N'] * min(24, len(N))
sb = {k: v[sub] for k, v in b.items()}; sblk = solve_blk[sub]
print('L solve segments:', sorted(((sblk[[c == 'L' for c in cls]] - 1) // bps + 1).tolist()), flush=True)

def ema_run(Ls, Wstar=None):
    """Run the subset; returns per-block stats."""
    olp = OrderLatencyProbe(model.config.block_omega, model.config.trace_lambda)
    st = dict(ema={l: None for l in Ls}, prev={}, prevd={}, win={}, n=0)
    out = dict(cos={l: [] for l in Ls}, inc={'S': [], 'O': []}, canc={})
    CL = [1, 2, 4, 8, 16, 32, 64, 128]
    acc = {(k, l): [torch.zeros(len(sub), device=dev), torch.zeros(len(sub), device=dev)] for k in 'SO' for l in CL}
    def probe(rec):
        olp(rec)
        t = st['n']; st['n'] += 1
        X = {'S': rec.im_s, 'O': olp.last['O']}
        for l in Ls:
            e = st['ema'][l]; eta = 1.0 / l
            st['ema'][l] = rec.im_s.clone() * eta if e is None else (1 - eta) * e + eta * rec.im_s
        if Wstar is not None:
            for l in Ls:
                e = st['ema'][l]
                c = (e * Wstar).sum((-1, -2)) / (e.flatten(-2).norm(dim=-1) * Wstar.flatten(-2).norm(dim=-1) + 1e-12)
                out['cos'][l].append(c.mean(1).cpu())               # mean over heads [B]
            for k in 'SO':
                if k in st['prev'] and k in st['prevd']:
                    d1 = X[k] - st['prev'][k]; d0 = st['prevd'][k]
                    c = (d1 * d0).sum((1, 2, 3)) / (d1.flatten(1).norm(dim=1) * d0.flatten(1).norm(dim=1) + 1e-12)
                    out['inc'][k].append(c.cpu())
                if k in st['prev']:
                    st['prevd'][k] = X[k] - st['prev'][k]
                st['prev'][k] = X[k].clone()
                if 16 * bps <= t < 64 * bps:                        # seg17..64
                    for l in CL:
                        key = (k, l)
                        if (t - 16 * bps) % l == 0: st['win'][key] = [torch.zeros_like(X[k]), torch.zeros_like(X[k])]
                        w = st['win'][key]; w[0] += X[k]; w[1] += X[k].abs()
                        if (t - 16 * bps) % l == l - 1:
                            acc[key][0] += w[0].abs().sum((1, 2, 3)); acc[key][1] += w[1].sum((1, 2, 3))
        return None
    model.probe = probe
    p = run_blocks(model, sb, NB)
    model.probe = None
    out['canc'] = {f'{k}{l}': (acc[(k, l)][0] / acc[(k, l)][1].clamp_min(1e-12)).cpu() for k in 'SO' for l in CL}
    return st, out, p

Ls = [1, 2, 4, 8, 16, 32, 64, 128]
st1, _, p1 = ema_run([8])
Wstar = st1['ema'][8].clone()
st2, out, p2 = ema_run(Ls, Wstar)
print('pass1 == pass2 predictions:', bool((p1 == p2).all()), flush=True)
cls_t = torch.tensor([{'S': 0, 'L': 1, 'N': 2}[c] for c in cls])
COS = {l: torch.stack(out['cos'][l]) for l in Ls}                 # [NB, B]
INC = {k: torch.stack(out['inc'][k]) for k in 'SO'}                # [NB-2, B]
torch.save(dict(COS=COS, INC=INC, CANC=out['canc'], cls=cls, sblk=sblk.cpu(), sub=sub.cpu()),
           __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'alt_res.pt'))
names = ['S', 'L', 'N']
print('\n== cancellation C(L) over seg17..64 (1 = no cancellation), mean per class')
for k in 'SO':
    print(('Im S' if k == 'S' else 'O   ') + ' ' + '  '.join(f'L={l:<3}' for l in [1, 2, 4, 8, 16, 32, 64, 128]))
    for i, n in enumerate(names):
        m = cls_t == i
        print(f'  {n}  ' + '  '.join(f'{float(out["canc"][f"{k}{l}"][m].mean()):.3f}' for l in [1, 2, 4, 8, 16, 32, 64, 128]))
print('\n== lag-1 correlation of increments (negative = alternation), blocks in seg17..64, mean per class')
for k in 'SO':
    v = INC[k][16 * bps - 2:64 * bps - 2]
    print(('Im S' if k == 'S' else 'O   ') + '  ' + '  '.join(f'{n} {float(v[:, cls_t == i].mean()):+.3f}' for i, n in enumerate(names)))
print('\n== cos(EMA_L(t), W*) on pre-solve blocks (W* = model W at block', NB, ')')
sb_c = sblk.cpu()
tt = torch.arange(1, NB + 1)[:, None]
for name, rng in [('S early: blocks 9..solve', (9, None, 0)), ('L late: seg17..solve', (16 * bps + 1, None, 1))]:
    lo, _, ci = rng
    mask = (tt >= lo) & (tt < sb_c[None]) & (cls_t[None] == ci)
    print(f'  {name:28s} n={int(mask.sum())}: ' + '  '.join(f'L={l} {float(COS[l][mask].mean()):.3f}' for l in Ls))
