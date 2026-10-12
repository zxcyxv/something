"""Segment extrapolation of a trace-run checkpoint (EMA), with the trace rate overridable at inference.

Usage: python docs/research/2026-10-12/scripts/extrapolate_trace_run.py RUN_DIR CHECKPOINT [N=512] [SEGS=128] [BATCH=512] [etas=0.2,1 | learned,1]

eta = the run's own value reproduces the trained model; eta = 1 switches the trace off (current phases only)
with the same weights. Carry kept across segments (loops raised to SEGS + 1), as in lt.train.extrapolate.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.analyze_stdp_direct import make_model

run, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
SEGS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 512
ETAS = tuple(e if e == 'learned' else float(e) for e in (sys.argv[6] if len(sys.argv) > 6 else '0.2,1').split(','))

protocol = json.loads((run / 'protocol.json').read_text())
ck = torch.load(ckpt, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
device = torch.device('cuda')
t._resolve_precision(cfg, device)
_, _, x, y, _, fp = t.load_data(cfg)
assert fp == cfg['data_fingerprint']
x, y = x[:N], y[:N]
results, began = [], time.monotonic()
for eta in ETAS:
    state = ck['model_state_dict']
    if eta == 'learned':          # the run's own learned per-head, per-pair rates
        proto = dict(protocol)
    else:                         # a fixed rate for every neuron (learned rates dropped)
        proto = dict(protocol, phase_trace_eta=eta, phase_trace_learn=False)
        state = {k: v for k, v in state.items() if '_eta_raw' not in k}
    base = make_model(proto, cfg, device, state)
    lt = base.model
    loops0, lt.config.loops = lt.config.loops, SEGS + 1
    solved = torch.zeros(SEGS, N, dtype=torch.bool)
    start = 0
    with torch.no_grad():
        for batch in t.eval_batches(x, y, BATCH, 0, 1):
            b = {k: v.to(device) for k, v in batch.items()}
            n = b['inputs'].shape[0]
            carry = lt.initial_carry(b)
            for s in range(SEGS):
                carry, out = lt(carry, b)
                solved[s, start:start + n] = (out['logits'].argmax(-1) == b['labels']).all(-1).cpu()
            start += n
    lt.config.loops = loops0
    ex = solved.sum(1)
    s16, best = int(ex[15]), int(ex.max())
    r = dict(eta=eta, seg16=s16, best=best, best_segment=int(ex.argmax()) + 1, last=int(ex[-1]),
             error_reduction=(best - s16) / max(N - s16, 1), exact_by_segment=ex.tolist())
    results.append(r)
    print(f"eta {eta}: seg16 {s16}  best {best} (seg {r['best_segment']})  last {r['last']}  "
          f"reduction {100 * r['error_reduction']:.1f}%  ({time.monotonic() - began:.0f}s)", flush=True)
    del base, lt
    torch.cuda.empty_cache()
out = Path(__file__).resolve().parents[1] / f"results/extrap_{run.name}_step{int(ck['step'])}_n{N}_seg{SEGS}.json"
out.write_text(json.dumps(dict(run=run.name, step=int(ck['step']), weights='EMA', n=N, segments=SEGS, results=results), indent=1))
print('WROTE', out)
