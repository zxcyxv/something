"""Longer memory at inference, scale-preserving: does it solve the puzzles the trained model leaves unsolved?

Usage (repository root):  python docs/research/2026-10-14/scripts/memory_sweep.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

Weights unchanged, torch path, FP32, no grad.  Variants:
  lam=x      trace retention lambda_K = lambda_V = x (trained 0.5); traces are EMAs lam E + (1-lam) x, so their scale is bounded
  rho=x      the STDP term read through an EMA over blocks, W_b = rho W_{b-1} + (1-rho) Im S_b, read = q~ (Hebb + alpha W_b)^T
             (same scale as the increment, horizon 1/(1-rho) blocks; zero at the puzzle start)
  lam+rho    both
Per variant: puzzles solved within BLOCKS blocks (all cells right from some block on), by tier (L0 / L1-only / beyond-L1),
exact at blocks 16 and 32, final cell accuracy, blocks per L0 round, and against the baseline: newly solved / lost lanes.
Writes docs/research/2026-10-14/results/memory_sweep_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sudoku_depth as sd                                     # noqa: E402
from lt import train as t                                     # noqa: E402
from lt import research_phase_clock_stdp as rp                # noqa: E402
from lt.kv_stability import ORIGINAL_MODEL_ID                 # noqa: E402

REPO = sd.REPO
run, step = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256
protocol = json.loads((run / 'protocol.json').read_text())
REC = dict(rho=None, W=None)
ORIG_STEP = rp.phase_clock_step
VARIANTS = [('baseline', 0.5, None), ('lam=0.6', 0.6, None), ('lam=0.7', 0.7, None), ('lam=0.8', 0.8, None),
            ('rho=0.5', 0.5, 0.5), ('rho=0.8', 0.5, 0.8), ('rho=0.9', 0.5, 0.9), ('lam=0.7+rho=0.8', 0.7, 0.8)]


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    if REC['rho'] is None:
        return hebb, im_s, new
    rho = REC['rho']
    REC['W'] = (1 - rho) * im_s if REC['W'] is None else rho * REC['W'] + (1 - rho) * im_s
    return hebb, REC['W'], new


rp.phase_clock_step = step_hook


def main():
    ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'], global_batch_size=BATCH)
    cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x_np, y_np, _, _ = t.load_data(cfg)
    x_np, y_np = x_np[:N], y_np[:N]
    x, y = x_np.reshape(-1, 81).astype(int), y_np.reshape(-1, 81).astype(int)
    began = time.monotonic()
    depth, _ = sd.propagate(x, y, lookahead=False)
    depth1, _ = sd.propagate(x, y, lookahead=True, inner=0)
    TIER = {1: 'L0', 2: 'L1-only', 3: 'beyond-L1'}
    puzzle_tier = np.where(~(depth == -1).any(1), 1, np.where(~(depth1 == -1).any(1), 2, 3))
    labels = y + 1
    print(f'solver {time.monotonic() - began:.0f}s')
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    inner = base.model.inner
    layer = inner.layers[0]
    raw0 = layer.clock_lam_raw.detach().clone()
    T = 81
    results, solved_vec = {}, {}
    for name, lam, rho in VARIANTS:
        t0 = time.monotonic()
        with torch.no_grad():
            layer.clock_lam_raw.copy_(torch.full_like(raw0, float(t.inv_softplus(-math.log(lam)))))
        REC['rho'] = rho
        preds_all = []
        with torch.no_grad():
            for batch in t.eval_batches(x_np, y_np, BATCH, 0, 1):
                b = {k: v.to(device) for k, v in batch.items()}
                B = b['inputs'].shape[0]
                inj = inner.injection(b)
                h = inner.init_hidden.float().expand(B, T, inner.d).clone()
                mem = ek = ev = None
                REC['W'] = None
                preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
                for blk in range(1, BLOCKS + 1):
                    h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                    preds[blk - 1] = inner.w_cls(h).argmax(-1).to(torch.int16)
                preds_all.append(preds.cpu())
        preds = torch.cat(preds_all, 1).long().numpy()
        final = preds[-1]
        same = preds == final[None]
        dec = (np.flip(np.cumprod(np.flip(same, 0), 0), 0).astype(bool)).argmax(0) + 1
        right = (preds == labels[None]).all(-1)
        S = np.flip(np.cumprod(np.flip(right, 0), 0), 0).astype(bool)[-1]
        solved_vec[name] = S
        mf = S[:, None] & (depth > 0)
        slope = float(np.polyfit(depth[mf].astype(float), dec[mf].astype(float), 1)[0]) if mf.sum() > 2 else None
        res = dict(solved=int(S.sum()), exact_block16=int(right[15].sum()), exact_block32=int(right[31].sum()),
                   cell_accuracy=float((final == labels)[depth != 0].mean()),
                   solved_by_tier={n: int((S & (puzzle_tier == k)).sum()) for k, n in TIER.items()}, blocks_per_round=slope,
                   newly_solved=int((S & ~solved_vec['baseline']).sum()), lost=int((~S & solved_vec['baseline']).sum()),
                   seconds=time.monotonic() - t0)
        results[name] = res
        print(f"{name:16s} solved {res['solved']:3d} (blk16 {res['exact_block16']:3d}, blk32 {res['exact_block32']:3d}); cell {res['cell_accuracy']:.3f}; "
              f"L0 {res['solved_by_tier']['L0']:3d} L1 {res['solved_by_tier']['L1-only']:3d} beyond {res['solved_by_tier']['beyond-L1']:3d}; "
              f"blocks/round {None if slope is None else round(slope, 2)}; vs baseline +{res['newly_solved']} / -{res['lost']}  ({res['seconds']:.0f}s)")
    with torch.no_grad():
        layer.clock_lam_raw.copy_(raw0)
    out = dict(run=run.name, step=step, n=N, blocks=BLOCKS, variants=results, seconds=time.monotonic() - began)
    path = REPO / 'docs/research/2026-10-14/results' / f'memory_sweep_{run.name}_step{step}_n{N}_b{BLOCKS}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', path)


if __name__ == '__main__':
    main()
