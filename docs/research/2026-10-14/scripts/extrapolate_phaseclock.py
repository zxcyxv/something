"""Segment extrapolation of a phase-clock run checkpoint (EMA weights, fused kernel).

Usage (repository root): python docs/research/2026-10-14/scripts/extrapolate_phaseclock.py RUN_DIR STEP [N=512] [SEGS=128] [BATCH=512]

Carry (hidden, phases, both traces) kept across segments with loops raised to SEGS + 1, as in lt.train.extrapolate.
Writes docs/research/2026-10-14/results/extrap_<run>_step<STEP>_n<N>_seg<SEGS>.json.
"""
import json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt import research_phase_clock_stdp as rp
from lt.kv_stability import ORIGINAL_MODEL_ID

run, step = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
SEGS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 512
SPLIT = sys.argv[6] if len(sys.argv) > 6 else 'held_out'      # 'train': original unaugmented training puzzles
protocol = json.loads((run / 'protocol.json').read_text())
ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'], global_batch_size=BATCH)
cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
device = torch.device('cuda')
t._resolve_precision(cfg, device)
t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
tr_x, tr_y, x, y, _, _ = t.load_data(cfg)
if SPLIT == 'train':
    x, y = tr_x, tr_y
x, y = x[:N], y[:N]
began = time.monotonic()
base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'])
lt = base.model
lt.config.loops = SEGS + 1
solved = torch.zeros(SEGS, N, dtype=torch.bool)
cells = torch.zeros(SEGS, dtype=torch.float64)
start = 0
with torch.no_grad():
    for batch in t.eval_batches(x, y, BATCH, 0, 1):
        b = {k: v.to(device) for k, v in batch.items()}
        n = b['inputs'].shape[0]
        carry = lt.initial_carry(b)
        for s in range(SEGS):
            carry, out = lt(carry, b)
            pred = out['logits'].argmax(-1)
            solved[s, start:start + n] = (pred == b['labels']).all(-1).cpu()
            cells[s] += (pred == b['labels']).float().mean(-1).sum().cpu().double()
        start += n
ex = solved.sum(1)
s16, best = int(ex[15]), int(ex.max())
ever = int(solved.any(0).sum())
churn = int((solved[15:-1] & ~solved[16:]).sum())
r = dict(run=run.name, step=step, n=N, segs=SEGS, batch=BATCH, split=SPLIT, seg16=s16, best=best, best_segment=int(ex.argmax()) + 1,
         last=int(ex[-1]), ever_solved=ever, unsolve_events_after_seg16=churn,
         error_reduction=(best - s16) / max(N - s16, 1), exact_by_segment=ex.tolist(),
         cell_accuracy_by_segment=(cells / N).tolist(), seconds=time.monotonic() - began)
out = REPO / 'docs/research/2026-10-14/results' / (f'extrap_{run.name}_step{step}' + ('_train' if SPLIT == 'train' else '') + f'_n{N}_seg{SEGS}.json')
out.write_text(json.dumps(r, indent=1) + '\n')
print(f"step {step}: seg16 {s16}/{N}  best {best} (seg {r['best_segment']})  last {r['last']}  ever {ever}  "
      f"error reduction {r['error_reduction'] * 100:.1f}%  ({r['seconds']:.0f}s)")
print('exact by segment (1,2,4,8,16,24,32,48,64,96,128):', [int(ex[i - 1]) for i in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)])
print('cell acc (16,32,64,128):', [round(float(cells[i - 1] / N), 4) for i in (16, 32, 64, 128)])
print('WROTE', out)
