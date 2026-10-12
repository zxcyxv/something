"""Usage: python docs/research/2026-10-12/scripts/probe_block_flips.py [N=512] [out.json]

How often does a (token, synapse) observation switch LTD <-> LTP between consecutive blocks, and which
phase moves? Trained alpha=1 SwiGLU BF16 fused 100k (EMA). Theory review 2026-10-12, read-only."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.analyze_stdp_direct import make_model, predict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ck = torch.load(REPO / 'checkpoints/alpha1_swiglu_bf16fused_step100000.pt', map_location='cpu', weights_only=False)
protocol, cfg = ck['protocol'], dict(ck['cfg'])
cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
device = torch.device('cuda')
t._resolve_precision(cfg, device)
_, _, x, y, _, fp = t.load_data(cfg)
assert fp == cfg['data_fingerprint']
x, y = x[:N], y[:N]
base = make_model(protocol, cfg, device, ck['model_state_dict'])
preds, official = predict(base, x, y, cfg, device)
labels = torch.from_numpy(y.reshape(-1, 81).astype(np.int64) + 1)
solved = (preds[-1].long() == labels).all(-1)
print('solved', int(solved.sum()), '/', N, 'official exact', official['exact'], flush=True)

inner = base.model.inner
S, B8 = cfg['loops'], inner.config.blocks_per_seg
NB = S * B8
names = ('m', 'up', 'down', 'up_post', 'down_post', 'zero', 'dV', 'wV', 'dK', 'wK', 'ltp', 'all_post', 'm_prev')
acc = {k: torch.zeros(2, NB, dtype=torch.float64, device=device) for k in names}
alpha = inner.layers[0].stdp_alpha.detach().float().to(device)
c1, c2 = inner.window_coefficients.tolist()
wrap = lambda a: torch.remainder(a + math.pi, 2 * math.pi) - math.pi
state = {}

orig = inner.memory_step
def memory_step(L, q, k, v, *a, **kw):
    out = orig(L, q, k, v, *a, **kw)
    blk = state['blk']; g = state['g']
    with torch.autocast('cuda', enabled=False):
        k = k.float(); v = v.float()
        b, H, T, D = v.shape; P = D // 2
        kU = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + inner.config.eps)
        vp, kp = v.reshape(b, H, T, P, 2), kU.reshape(b, H, T, P, 2)
        pV, pK = torch.atan2(vp[..., 1], vp[..., 0]), torch.atan2(kp[..., 1], kp[..., 0])
        aV, aK = vp.norm(dim=-1), kp.norm(dim=-1)
        prev = state.get('prev')
        for s0 in range(0, b, 16):
            sl = slice(s0, s0 + 16); gg = g[sl]
            d = pV[sl][..., :, None] - pK[sl][..., None, :]
            m = aV[sl][..., :, None] * aK[sl][..., None, :]
            W = c1 * torch.sin(d) + c2 * torch.sin(2 * d)
            ltp = W > 0
            red = lambda z: z.sum(dim=(1, 2, 3, 4)).double()
            acc['m'][:, blk].index_add_(0, gg, red(m))
            acc['ltp'][:, blk].index_add_(0, gg, red(m * ltp))
            if prev is not None:
                ppV, ppK = prev['pV'][sl], prev['pK'][sl]
                dpV, dpK = wrap(pV[sl] - ppV).abs(), wrap(pK[sl] - ppK).abs()
                dprev = ppV[..., :, None] - ppK[..., None, :]
                Wp = c1 * torch.sin(dprev) + c2 * torch.sin(2 * dprev)
                ltp_p = Wp > 0
                up, down = ltp & ~ltp_p, ~ltp & ltp_p
                post = dpV[..., :, None] > dpK[..., None, :]
                gain, gain_p = 1 + alpha.view(1, H, 1, 1, 1) * W, 1 + alpha.view(1, H, 1, 1, 1) * Wp
                acc['up'][:, blk].index_add_(0, gg, red(m * up))
                acc['down'][:, blk].index_add_(0, gg, red(m * down))
                acc['up_post'][:, blk].index_add_(0, gg, red(m * (up & post)))
                acc['down_post'][:, blk].index_add_(0, gg, red(m * (down & post)))
                acc['zero'][:, blk].index_add_(0, gg, red(m * ((gain > 0) != (gain_p > 0))))
                acc['all_post'][:, blk].index_add_(0, gg, red(m * post))
                acc['m_prev'][:, blk].index_add_(0, gg, red(m))
                r2 = lambda z: z.sum(dim=(1, 2, 3)).double()
                acc['dV'][:, blk].index_add_(0, gg, r2(aV[sl] * dpV)); acc['wV'][:, blk].index_add_(0, gg, r2(aV[sl]))
                acc['dK'][:, blk].index_add_(0, gg, r2(aK[sl] * dpK)); acc['wK'][:, blk].index_add_(0, gg, r2(aK[sl]))
            del d, m, W, ltp
        state['prev'] = dict(pV=pV, pK=pK)
    state['blk'] += 1
    return out
inner.memory_step = memory_step

groups = (~solved).long()
began, start = time.monotonic(), 0
with torch.no_grad():
    for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = batch['inputs'].shape[0]
        state.update(blk=0, prev=None, g=groups[start:start + n].to(device))
        carry = base.initial_carry(batch)
        for s in range(S):
            carry, _, _, out, _ = base(carry=carry, batch=batch, return_keys={'preds'})
        start += n
        print('batch done', start, f'{time.monotonic() - began:.0f}s', flush=True)
del inner.memory_step

res = {}
for gi, gname in enumerate(('solved', 'unsolved')):
    if int((groups == gi).sum()) == 0:
        continue
    rows = {}
    for seg in (1, 2, 4, 8, 16):
        sl = slice((seg - 1) * B8 + (1 if seg == 1 else 0), seg * B8)
        f = lambda k: float(acc[k][gi, sl].sum())
        m = f('m')
        rows[seg] = dict(ltp=f('ltp') / m, up=f('up') / m, down=f('down') / m,
                         up_post=f('up_post') / max(f('up'), 1e-30), down_post=f('down_post') / max(f('down'), 1e-30),
                         gain_sign_change=f('zero') / m, all_post=f('all_post') / max(f('m_prev'), 1e-30), mean_abs_dphiV=f('dV') / f('wV'), mean_abs_dphiK=f('dK') / f('wK'))
    res[gname] = dict(puzzles=int((groups == gi).sum()), per_segment=rows)
res['note'] = ('per segment: transitions between consecutive blocks inside the segment (and across the boundary from the previous '
               'segment), weighted by co-firing |v_A||k_B|; up = LTD->LTP, up_post = share of up-flips where |dphiV_A| > |dphiK_B|')
print(json.dumps(res, indent=1))
Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / 'results/block_flips_step100000.json').write_text(json.dumps(res, indent=1))
