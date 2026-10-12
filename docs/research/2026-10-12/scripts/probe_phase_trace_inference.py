"""Inference-only phase traces on the trained 100k weights (no retraining).

Usage: python docs/research/2026-10-12/scripts/probe_phase_trace_inference.py [N=512] [SEGS=128] [out.json] [etas, e.g. none,0.2,0.1]

The timing term reads block-EMA traces of the K/V unit phasors (rate eta); the Hebbian carrier and
q, k, v stay the current block's (lt/experiment_free_phase_windows.py, phase_trace_eta, torch path).
eta = 1 is the trained model on the torch path. Also measures, over the last segment, how often the
effective timing sign and the gain sign of an observation flip between consecutive blocks.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.analyze_stdp_direct import make_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEGS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / f'results/phase_trace_inference_n{N}_seg{SEGS}.json'
ETAS = (None, 1.0, 0.5, 0.2, 0.1)          # None = fused kernel (official path)
if len(sys.argv) > 4:                       # e.g. none,0.2,0.1,0.05
    ETAS = tuple(None if e == 'none' else float(e) for e in sys.argv[4].split(','))

ck = torch.load(REPO / 'checkpoints/alpha1_swiglu_bf16fused_step100000.pt', map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
torch.set_float32_matmul_precision(ck['protocol'].get('precision', 'highest'))
device = torch.device('cuda')
t._resolve_precision(cfg, device)
_, _, x, y, _, fp = t.load_data(cfg)
assert fp == cfg['data_fingerprint']
x, y = x[:N], y[:N]
labels = torch.from_numpy(y.reshape(-1, 81).astype(np.int64) + 1)


def run(eta):
    protocol = dict(ck['protocol'])
    if eta is not None:
        protocol.update(kernel='torch', phase_trace_eta=eta)
    base = make_model(protocol, cfg, device, ck['model_state_dict'])
    lt = base.model
    inner = lt.inner
    coef = inner.window_coefficients.float()
    alpha = inner.layers[0].stdp_alpha.detach().float()
    orig = inner.memory_step
    flips = dict(seg=None)
    acc = torch.zeros(N, 3, dtype=torch.float64)       # co-firing mass, timing-sign flips, gain-sign flips
    state = {}

    def memory_step(L, q, k, v, *a, **kw):
        out = orig(L, q, k, v, *a, **kw)
        if flips['seg'] != SEGS:
            return out
        with torch.autocast('cuda', enabled=False):
            kf, vf = k.float(), v.float()
            kU = kf / (torch.linalg.vector_norm(kf, dim=-1, keepdim=True) + inner.config.eps)
            zk = out[2] if out[2] is not None else None
            zv = out[3] if out[3] is not None else None
            def polar(z):
                p = z.reshape(*z.shape[:-1], z.shape[-1] // 2, 2)
                return p.norm(dim=-1), torch.atan2(p[..., 1], p[..., 0])
            ak, pk = polar(kU)
            av, pv = polar(vf)
            if zk is not None:
                rk, pk = polar(zk.float())
                rv, pv = polar(zv.float())
            else:
                rk, rv = torch.ones_like(ak), torch.ones_like(av)
            b = vf.shape[0]
            for s0 in range(0, b, 16):
                sl = slice(s0, s0 + 16)
                d = pv[sl][..., :, None] - pk[sl][..., None, :]
                W = sum(coef[r] * torch.sin((r + 1) * d) for r in range(coef.numel()))
                W = W * rv[sl][..., :, None] * rk[sl][..., None, :]
                m = av[sl][..., :, None] * ak[sl][..., None, :]
                ts, gs = W > 0, (1 + alpha.view(1, -1, 1, 1, 1) * W) > 0
                key = (id(L), s0)
                prev = state.get(key)
                idx = slice(state['start'] + s0, state['start'] + s0 + ts.shape[0])
                acc[idx, 0] += m.sum(dim=(1, 2, 3, 4)).double().cpu()
                if prev is not None:
                    acc[idx, 1] += (m * (ts != prev[0])).sum(dim=(1, 2, 3, 4)).double().cpu()
                    acc[idx, 2] += (m * (gs != prev[1])).sum(dim=(1, 2, 3, 4)).double().cpu()
                state[key] = (ts, gs)
        return out

    inner.memory_step = memory_step
    solved = torch.zeros(SEGS, N, dtype=torch.bool)
    churn = torch.zeros(SEGS, N)
    loops0, lt.config.loops = lt.config.loops, SEGS + 1
    start = 0
    with torch.no_grad():
        for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
            b = {kk: vv.to(device) for kk, vv in batch.items()}
            n = b['inputs'].shape[0]
            carry, prev = lt.initial_carry(b), None
            for key in [k for k in state if k != 'start']:
                del state[key]
            state['start'] = start
            for s in range(SEGS):
                flips['seg'] = s + 1
                carry, out = lt(carry, b)
                pred = out['logits'].argmax(-1)
                solved[s, start:start + n] = (pred == b['labels']).all(-1).cpu()
                if prev is not None:
                    churn[s, start:start + n] = (pred != prev).float().mean(-1).cpu()
                prev = pred
            start += n
    lt.config.loops = loops0
    del inner.memory_step, base
    torch.cuda.empty_cache()
    ex = solved.sum(1)
    s16, best = int(ex[15]), int(ex.max())
    last = solved[-1]
    blocks = cfg['blocks_per_seg'] - 1                      # flips counted within the last segment
    def rate(mask, col):
        return float(acc[mask, col].sum() / acc[mask, 0].sum().clamp_min(1e-30) * cfg['blocks_per_seg'] / max(blocks, 1))
    return dict(eta=eta, seg16=s16, best=best, best_segment=int(ex.argmax()) + 1, last=int(ex[-1]),
                error_reduction=(best - s16) / max(N - s16, 1), exact_by_segment=ex.tolist(),
                late_churn=float(churn[-16:].mean()),
                last_segment_flip_rate=dict(
                    unsolved=dict(timing=rate(~last, 1), gain=rate(~last, 2), puzzles=int((~last).sum())),
                    solved=dict(timing=rate(last, 1), gain=rate(last, 2), puzzles=int(last.sum()))),
                solved16=solved[15].tolist(), solved_last=last.tolist())


results, began = [], time.monotonic()
for eta in ETAS:
    r = run(eta)
    results.append(r)
    f = r['last_segment_flip_rate']
    print(f"eta {eta}: seg16 {r['seg16']}  best {r['best']} (seg {r['best_segment']})  last {r['last']}  "
          f"reduction {100 * r['error_reduction']:.1f}%  late churn {r['late_churn']:.4f}  "
          f"seg{SEGS} flips/block unsolved timing {f['unsolved']['timing']:.4f} gain {f['unsolved']['gain']:.4f}  "
          f"solved timing {f['solved']['timing']:.4f}  ({time.monotonic() - began:.0f}s)", flush=True)
ref16 = torch.tensor(results[0]['solved16'])
ref_last = torch.tensor(results[0]['solved_last'])
for r in results:
    s16, sl = torch.tensor(r.pop('solved16')), torch.tensor(r.pop('solved_last'))
    r.update(kept16=int((s16 & ref16).sum()), new16=int((s16 & ~ref16).sum()),
             kept_last=int((sl & ref_last).sum()), new_last=int((sl & ~ref_last).sum()))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dict(checkpoint='checkpoints/alpha1_swiglu_bf16fused_step100000.pt (EMA, R=2)', n=N, segments=SEGS,
                               note='eta None = fused kernel; flip rates are per block, co-firing weighted, over the last segment',
                               results=results), indent=1))
print('WROTE', OUT)
for r in results:
    print(f"eta {r['eta']}: seg16 {r['seg16']} (kept {r['kept16']}/{results[0]['seg16']}, new {r['new16']})  "
          f"last {r['last']} (kept {r['kept_last']}/{results[0]['last']}, new {r['new_last']})")
