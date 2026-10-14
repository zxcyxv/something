"""Inference-time interventions on the trained bounded phase-clock model, measured with the logical-depth tooling.

Usage (repository root):  python docs/research/2026-10-14/scripts/stdp_interventions_depth.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

All variants run the torch path (FP32, no grad) on the same held-out puzzles:
  baseline      the model as trained
  alpha0        STDP term removed (alpha_h = 0): read = q~ Hebb^T
  L-only        order part removed: read = q~ (Hebb + alpha L)^T     (L = (Im S(+) + Im S(-)) / 2 via shadow traces)
  O-only        latency part removed: read = q~ (Hebb + alpha O)^T   (O = (Im S(+) - Im S(-)) / 2)
  omega-        block spacing reversed, omegaT -> -omegaT (later blocks get EARLIER tags)
  omega0        block spacing removed, omegaT = 0 (tags = c sin phi only)
Per variant: puzzles solved within BLOCKS blocks by puzzle tier (L0 / L1-only / beyond-L1, see sudoku_depth.py), final cell
accuracy by cell tier, decision block by L0 depth, blocks per round (slope), Spearman(depth, decision) on solved lanes.
Writes docs/research/2026-10-14/results/interventions_depth_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
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
WT = float(protocol['omega_t'])
REC = dict(n=0, shadow=None, mode='baseline')
ORIG_STEP = rp.phase_clock_step


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    if REC['mode'] not in ('L-only', 'O-only'):
        return hebb, im_s, new
    shift = 2 * REC['n'] * WT
    shadow = traces if REC['shadow'] is None else REC['shadow']
    _, im_s2, shadow_new = ORIG_STEP(k_unit, kr, v, rope, pk - shift, pv - shift, lk, lv, shadow)
    REC['shadow'] = shadow_new
    return hebb, (im_s + im_s2) / 2 if REC['mode'] == 'L-only' else (im_s - im_s2) / 2, new


rp.phase_clock_step = step_hook
VARIANTS = ['baseline', 'alpha0', 'L-only', 'O-only', 'omega-', 'omega0']


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
    tier = np.where(depth == 0, 0, np.where(depth > 0, 1, np.where(depth1 > 0, 2, 3)))
    TIER = {1: 'L0', 2: 'L1-only', 3: 'beyond-L1'}
    puzzle_tier = np.where(~(depth == -1).any(1), 1, np.where(~(depth1 == -1).any(1), 2, 3))
    labels = y + 1
    print(f'solver {time.monotonic() - began:.0f}s; puzzles by tier ' + ', '.join(f'{n} {int((puzzle_tier == k).sum())}' for k, n in TIER.items()))
    T = 81
    results = {}
    for variant in VARIANTS:
        t0 = time.monotonic()
        proto = dict(protocol)
        if variant == 'omega-':
            proto['omega_t'] = -WT
        if variant == 'omega0':
            proto['omega_t'] = 0.0
        base = rp.make_model(proto, cfg, device, state=ck['model_state_dict'], kernel='torch')
        inner = base.model.inner
        layer = inner.layers[0]
        if variant == 'alpha0':
            with torch.no_grad():
                layer.stdp_alpha.zero_()
        REC['mode'] = variant
        preds_all = []
        with torch.no_grad():
            for batch in t.eval_batches(x_np, y_np, BATCH, 0, 1):
                b = {k: v.to(device) for k, v in batch.items()}
                B = b['inputs'].shape[0]
                inj = inner.injection(b)
                h = inner.init_hidden.float().expand(B, T, inner.d).clone()
                mem = ek = ev = None
                REC.update(n=0, shadow=None)
                preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
                for blk in range(1, BLOCKS + 1):
                    REC['n'] = blk - 1
                    h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                    preds[blk - 1] = inner.w_cls(h).argmax(-1).to(torch.int16)
                preds_all.append(preds.cpu())
        preds = torch.cat(preds_all, 1).long().numpy()
        final = preds[-1]
        same = preds == final[None]
        dec = (np.flip(np.cumprod(np.flip(same, 0), 0), 0).astype(bool)).argmax(0) + 1
        right = (preds == labels[None]).all(-1)
        S = np.flip(np.cumprod(np.flip(right, 0), 0), 0).astype(bool)[-1]
        finite = depth > 0
        mf = S[:, None] & finite
        by_depth = {r: float(dec[S[:, None] & (depth == r)].mean()) for r in range(1, 13) if (S[:, None] & (depth == r)).any()}
        slope = float(np.polyfit(depth[mf].astype(float), dec[mf].astype(float), 1)[0]) if mf.sum() > 2 else None
        res = dict(solved=int(S.sum()), exact_block16=int(right[15].sum()), exact_block32=int(right[31].sum()),
                   cell_accuracy=float((final == labels)[depth != 0].mean()),
                   solved_by_tier={n: int((S & (puzzle_tier == k)).sum()) for k, n in TIER.items()},
                   cell_acc_by_tier={n: float((final == labels)[tier == k].mean()) for k, n in TIER.items()},
                   decision_by_depth=by_depth, blocks_per_round=slope, spearman=sd.spearman(depth[mf], dec[mf]) if mf.sum() > 2 else None,
                   seconds=time.monotonic() - t0)
        results[variant] = res
        print(f"{variant:9s} solved {res['solved']:3d} (blk16 {res['exact_block16']:3d}, blk32 {res['exact_block32']:3d}); cell {res['cell_accuracy']:.3f}; "
              f"tier L0 {res['solved_by_tier']['L0']:3d} L1 {res['solved_by_tier']['L1-only']:3d} beyond {res['solved_by_tier']['beyond-L1']:3d}; "
              f"cell acc L0 {res['cell_acc_by_tier']['L0']:.3f} L1 {res['cell_acc_by_tier']['L1-only']:.3f} beyond {res['cell_acc_by_tier']['beyond-L1']:.3f}; "
              f"blocks/round {slope if slope is None else round(slope, 2)}; spearman {None if res['spearman'] is None else round(res['spearman'], 3)}; "
              f"dec by depth 1..8 " + ' '.join(f'{by_depth[r]:.1f}' for r in range(1, 9) if r in by_depth) + f"  ({res['seconds']:.0f}s)")
    out = dict(run=run.name, step=step, n=N, blocks=BLOCKS, puzzles_by_tier={n: int((puzzle_tier == k).sum()) for k, n in TIER.items()}, variants=results)
    path = REPO / 'docs/research/2026-10-14/results' / f'interventions_depth_{run.name}_step{step}_n{N}_b{BLOCKS}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', path)


if __name__ == '__main__':
    main()
