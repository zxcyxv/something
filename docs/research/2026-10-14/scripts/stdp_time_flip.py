"""Reverse the order part of the STDP term mid-run: does the solution run backwards?

Usage (repository root):  python docs/research/2026-10-14/scripts/stdp_time_flip.py RUN_DIR STEP [N=512] [BLOCKS=128] [FLIP=32] [BATCH=256]

Torch path, FP32, no grad, shadow traces with reversed block spacing (Im S(-) = L - O, see stdp_order_decomp.py).
Every lane runs FLIP blocks as trained; from block FLIP + 1 the read uses, per mode,
  none     Hebb + alpha (L + O)    (unchanged)
  flip     Hebb + alpha (L - O)    (order part reversed = the time-reversed STDP term)
  dropO    Hebb + alpha L
  alpha0   Hebb
  negL     Hebb + alpha (O - L)    (latency part reversed; control)
  noise    unchanged read, Gaussian noise of 30% of the cell norm added to h after every block (control)
Measured per block after the flip (lanes solved at block FLIP unless stated):
  exact / cell accuracy (all lanes), nearest pre-flip block (argmin over blocks 1..FLIP of the hidden-state L2 distance; a
  retrace would make it fall), its distance, the model's candidate count (softmax > 0.1), |O| / |Hebb|;
  un-decision order: for cells that become wrong after the flip, Spearman between the block they first turn wrong and
  their original decision block (negative = last decided, first undone), plus the same for the cells' L0 depth.
Writes docs/research/2026-10-14/results/time_flip_<run>_step<STEP>_n<N>_b<BLOCKS>_f<FLIP>.json.
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
FLIP = int(sys.argv[5]) if len(sys.argv) > 5 else 32
BATCH = int(sys.argv[6]) if len(sys.argv) > 6 else 256
protocol = json.loads((run / 'protocol.json').read_text())
WT = float(protocol['omega_t'])
REC = dict(n=0, shadow=None, mode='none', O=None, hebb=None)
ORIG_STEP = rp.phase_clock_step
MODES = ['none', 'flip', 'dropO', 'alpha0', 'negL', 'noise']


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    shift = 2 * REC['n'] * WT
    shadow = traces if REC['shadow'] is None else REC['shadow']
    _, im_s2, shadow_new = ORIG_STEP(k_unit, kr, v, rope, pk - shift, pv - shift, lk, lv, shadow)
    REC['shadow'] = shadow_new
    REC['O'], REC['hebb'] = (im_s - im_s2) / 2, hebb
    if REC['n'] < FLIP or REC['mode'] == 'none':
        return hebb, im_s, new
    if REC['mode'] == 'flip':
        return hebb, im_s2, new
    if REC['mode'] == 'dropO':
        return hebb, (im_s + im_s2) / 2, new
    if REC['mode'] == 'negL':                                   # latency part reversed, order part kept: Hebb + alpha (O - L)
        return hebb, -im_s2, new
    if REC['mode'] == 'noise':                                  # STDP term unchanged; the state is perturbed outside the hook
        return hebb, im_s, new
    return hebb, torch.zeros_like(im_s), new


rp.phase_clock_step = step_hook


def fro(m):
    return m.flatten(-2).norm(dim=-1)


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
    depth, _ = sd.propagate(x, y, lookahead=False)
    labels = torch.from_numpy(y + 1)
    began = time.monotonic()
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    inner = base.model.inner
    layer = inner.layers[0]
    T = 81
    results = {}
    for mode in MODES:
        REC['mode'] = mode
        exact = torch.zeros(BLOCKS)
        cell = torch.zeros(BLOCKS)
        ncand = torch.zeros(BLOCKS)
        o_ratio = torch.zeros(BLOCKS)
        nearest = torch.zeros(BLOCKS)
        near_dist = torch.zeros(BLOCKS)
        n_solved = 0
        undo_pairs, undo_depth = [], []
        with torch.no_grad():
            for batch in t.eval_batches(x_np, y_np, BATCH, 0, 1):
                b = {k: v.to(device) for k, v in batch.items()}
                B = b['inputs'].shape[0]
                lab = b['labels'].long()
                inj = inner.injection(b)
                h = inner.init_hidden.float().expand(B, T, inner.d).clone()
                mem = ek = ev = None
                REC.update(n=0, shadow=None)
                pre = torch.zeros(FLIP, B, T, inner.d, device=device, dtype=torch.float16)
                preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
                for blk in range(1, BLOCKS + 1):
                    REC['n'] = blk - 1
                    h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                    if mode == 'noise' and blk > FLIP:
                        h = h + 0.3 * h.norm(dim=-1, keepdim=True) * torch.randn_like(h) / (h.shape[-1] ** 0.5)
                    logits = inner.w_cls(h).float()
                    p = logits.argmax(-1)
                    preds[blk - 1] = p.to(torch.int16)
                    i = blk - 1
                    exact[i] += (p == lab).all(-1).float().sum().cpu()
                    cell[i] += (p == lab).float().mean(-1).sum().cpu()
                    ncand[i] += (torch.softmax(logits[..., 2:11], -1) > 0.1).sum(-1).float().mean(-1).sum().cpu()
                    o_ratio[i] += (fro(REC['O']) / fro(REC['hebb'])).mean(1).sum().cpu()
                    if blk <= FLIP:
                        pre[i] = h.half()
                    if blk == FLIP:
                        solved_at_flip = (p == lab).all(-1)                                     # [B]
                        n_solved += int(solved_at_flip.sum())
                    if blk > FLIP:
                        d = (pre.float() - h[None]).flatten(2).norm(dim=-1)                       # [FLIP, B]
                        idx = d.argmin(0)
                        nearest[i] += (idx[solved_at_flip].float() + 1).sum().cpu()
                        near_dist[i] += d.min(0).values[solved_at_flip].sum().cpu()
                # un-decision order on lanes solved at the flip
                pr = preds.long()
                pre_same = pr[:FLIP] == pr[FLIP - 1][None]
                dec = (torch.flip(torch.cumprod(torch.flip(pre_same, [0]), 0), [0]).bool()).float().argmax(0) + 1   # [B,T]
                wrong_after = pr[FLIP:] != lab[None]                                               # [BLOCKS-FLIP, B, T]
                ever = wrong_after.any(0)
                first_wrong = wrong_after.float().argmax(0) + FLIP + 1
                m = solved_at_flip[:, None] & ever & (dec > 1)
                if m.any():
                    undo_pairs.append(torch.stack((first_wrong[m].float(), dec[m].float()), 1).cpu())
                undo_depth.append((m.cpu(), first_wrong.cpu()))                                  # lane order = depth order
        n_lanes = N
        res = dict(exact=exact.tolist(), cell=(cell / n_lanes).tolist(), ncand=(ncand / n_lanes).tolist(), O_hebb=(o_ratio / n_lanes).tolist(),
                   solved_at_flip=n_solved, nearest=[(v / max(n_solved, 1)) for v in nearest[FLIP:].tolist()],
                   nearest_dist=[(v / max(n_solved, 1)) for v in near_dist[FLIP:].tolist()])
        if undo_pairs:
            up = torch.cat(undo_pairs, 0).numpy()
            res['undo_cells'] = int(up.shape[0])
            res['spearman_firstwrong_vs_decision'] = sd.spearman(up[:, 0], up[:, 1])
        # depth version: assemble per-lane masks in order
        masks = torch.cat([mm for mm, _ in undo_depth], 0).numpy()
        fw = torch.cat([ff for _, ff in undo_depth], 0).numpy()
        sel = masks & (depth > 0)
        res['spearman_firstwrong_vs_depth'] = sd.spearman(fw[sel], depth[sel]) if sel.sum() > 2 else None
        res['undo_cells_with_depth'] = int(sel.sum())
        results[mode] = res
        show = [b for b in (FLIP, FLIP + 1, FLIP + 2, FLIP + 4, FLIP + 8, FLIP + 16, FLIP + 32, FLIP + 64, BLOCKS) if b <= BLOCKS]
        print(f"[{mode:6s}] solved at flip {n_solved}; exact at blocks {show}: " + ' '.join(f'{int(exact[b - 1])}' for b in show)
              + '; cell ' + ' '.join(f'{cell[b - 1] / n_lanes:.3f}' for b in show))
        print(f"          candidates/cell " + ' '.join(f'{ncand[b - 1] / n_lanes:.2f}' for b in show)
              + ';  |O|/|Hebb| ' + ' '.join(f'{o_ratio[b - 1] / n_lanes:.3f}' for b in show))
        print(f"          nearest pre-flip block (solved lanes) at +1,+2,+4,+8,+16,+32,+64,+96: "
              + ' '.join(f'{res["nearest"][j - 1]:.1f}' for j in (1, 2, 4, 8, 16, 32, 64, 96) if j <= BLOCKS - FLIP)
              + ';  distance ' + ' '.join(f'{res["nearest_dist"][j - 1]:.1f}' for j in (1, 2, 4, 8, 16, 32, 64, 96) if j <= BLOCKS - FLIP))
        print(f"          cells undone after flip {res.get('undo_cells', 0)}: Spearman(first wrong, decision block) {res.get('spearman_firstwrong_vs_decision')}, "
              f"Spearman(first wrong, L0 depth) {res['spearman_firstwrong_vs_depth']} (n={res['undo_cells_with_depth']})")
    out = dict(run=run.name, step=step, n=N, blocks=BLOCKS, flip=FLIP, seconds=time.monotonic() - began, modes=results)
    path = REPO / 'docs/research/2026-10-14/results' / f'time_flip_{run.name}_step{step}_n{N}_b{BLOCKS}_f{FLIP}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', path)


if __name__ == '__main__':
    main()
