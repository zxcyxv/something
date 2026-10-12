"""Inference-only cross-block KV memory on the trained 100k weights (no retraining).

Usage: python docs/research/2026-10-12/scripts/probe_inference_memory.py [N=512] [SEGS=128] [out.json]

Each block's write G_k (the model's own, from the fused kernel) feeds a normalized, bias-corrected EMA
    c_k = (1 - eta) c_{k-1} + eta,   M_k = M_{k-1} + (eta / c_k) (G_k - M_{k-1})      (M_1 = G_1)
and the read becomes
    read_k = qr_k [(1 - lam) G_k + lam M_k]^T .
lam = 0 is the trained model. At a fixed point M = G, so the fixed points are those of the trained model;
only transients (and stability) change. M is kept across segments and reset for each new batch of puzzles.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.analyze_stdp_direct import make_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEGS = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / f'results/inference_memory_n{N}_seg{SEGS}.json'
GRID = [(0.0, 1.0)] + [(lam, eta) for lam in (0.25, 0.5, 0.75, 1.0) for eta in (0.5, 0.2, 0.05)]

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
lt = base.model
inner = lt.inner
orig = inner.memory_step
state = {}


def memory_step(L, q, k, v, *a, **kw):
    read, current, e_k, e_v = orig(L, q, k, v, *a, **kw)
    lam, eta = state['lam'], state['eta']
    if lam == 0.0:
        return read, current, e_k, e_v
    with torch.autocast('cuda', enabled=False):
        key = id(L)
        c_prev, M_prev = state['c'].get(key, 0.0), state['M'].get(key)
        c = (1 - eta) * c_prev + eta
        G = current.float()
        M = G if M_prev is None else M_prev + (eta / c) * (G - M_prev)
        state['c'][key], state['M'][key] = c, M
        qf = q.float()
        qU = qf / (torch.linalg.vector_norm(qf, dim=-1, keepdim=True) + inner.config.eps)
        qr = inner.apply_rope(qU, L, inner.rope_tables(L))
        mixed = (1 - lam) * read.float() + lam * (qr @ M.transpose(-1, -2))
    return mixed, current, e_k, e_v


inner.memory_step = memory_step
labels = torch.from_numpy(y.reshape(-1, 81).astype(np.int64) + 1)
results = []
began = time.monotonic()
for lam, eta in GRID:
    state.update(lam=lam, eta=eta)
    solved = torch.zeros(SEGS, N, dtype=torch.bool)
    churn = torch.zeros(SEGS, N)
    loops0 = lt.config.loops
    lt.config.loops = SEGS + 1
    start = 0
    with torch.no_grad():
        for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
            b = {kk: vv.to(device) for kk, vv in batch.items()}
            n = b['inputs'].shape[0]
            state.update(c={}, M={})
            carry, prev = lt.initial_carry(b), None
            for s in range(SEGS):
                carry, out = lt(carry, b)
                pred = out['logits'].argmax(-1)
                solved[s, start:start + n] = (pred == b['labels']).all(-1).cpu()
                if prev is not None:
                    churn[s, start:start + n] = (pred != prev).float().mean(-1).cpu()
                prev = pred
            start += n
    lt.config.loops = loops0
    ex = solved.sum(1)
    s16 = int(ex[15])
    best = int(ex.max())
    results.append(dict(lam=lam, eta=eta, seg16=s16, best=best, best_segment=int(ex.argmax()) + 1, last=int(ex[-1]),
                        error_reduction=(best - s16) / max(N - s16, 1), exact_by_segment=ex.tolist(),
                        solved16=solved[15].tolist(), solved_last=solved[-1].tolist(),
                        late_churn=float(churn[-16:].mean())))
    print(f'lam {lam:.2f} eta {eta:.2f}: seg16 {s16}  best {best} (seg {int(ex.argmax()) + 1})  last {int(ex[-1])}  '
          f'reduction {100 * (best - s16) / max(N - s16, 1):.1f}%  late churn {float(churn[-16:].mean()):.4f}  '
          f'({time.monotonic() - began:.0f}s)', flush=True)
inner.memory_step = orig

base0 = torch.tensor(results[0]['solved16'])
base_last = torch.tensor(results[0]['solved_last'])
for r in results:
    s16, sl = torch.tensor(r['solved16']), torch.tensor(r['solved_last'])
    r['kept16_of_base16'] = int((s16 & base0).sum())
    r['new16_vs_base16'] = int((s16 & ~base0).sum())
    r['kept_last_of_base_last'] = int((sl & base_last).sum())
    r['new_last_vs_base_last'] = int((sl & ~base_last).sum())
    del r['solved16'], r['solved_last']
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dict(checkpoint='checkpoints/alpha1_swiglu_bf16fused_step100000.pt (EMA)', n=N, segments=SEGS,
                               results=results), indent=1))
print('WROTE', OUT)
for r in results:
    print(f"lam {r['lam']:.2f} eta {r['eta']:.2f}: seg16 {r['seg16']} (kept {r['kept16_of_base16']}/{results[0]['seg16']}, "
          f"new {r['new16_vs_base16']})  last {r['last']} (kept {r['kept_last_of_base_last']}/{results[0]['last']}, "
          f"new {r['new_last_vs_base_last']})  best {r['best']}")
