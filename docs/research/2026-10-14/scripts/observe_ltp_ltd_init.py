"""Run-3 model (displacement clock) at random initialization on 6x6 sudoku: do LTP/LTD occur?

Usage (repository root):  python docs/research/2026-10-14/scripts/observe_ltp_ltd_init.py [--blocks 32] [--n 256]

Builds the actual run-3 class (lt/research_phase_clock_stdp.py, torch path, FP64) at a small size:
hidden 12, one head -> 12 channels per head = 6 pre (K) and 6 post (V) neurons (channel pairs).
No training. Hooks phase_clock_step and records, per block, for every cell and every (post a, pre i) synapse:
  the same-block window factor  1 + alpha (1 - lam_K) sin(phi_V,a - phi_K,i)   (> 1: LTP, < 1: LTD),
and the norms of the Hebbian term and of alpha * Im S.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lt import train as t                                   # noqa: E402
from lt import research_phase_clock_stdp as rp              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, default=32)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    data = np.load(ROOT / 'docs/research/2026-10-14/data/sudoku6_level1.npz')
    x = torch.as_tensor(data['inputs'][:opt.n].astype(np.int64))
    y = torch.as_tensor(data['labels'][:opt.n].astype(np.int64))

    t.KVSTDPInner = rp.model_class(clock='displacement', kernel='torch')
    cfg = dict(t.DEFAULT_CFG, **t.PRESETS['kv_stdp'])
    cfg.update(hidden_size=12, num_heads=1, puzzle_emb_ndim=0, global_batch_size=opt.n, batch_size=opt.n,
               seq_len=36, grid=6, vocab_size=7, num_puzzle_identifiers=1, blocks_per_seg=1, loops=opt.blocks,
               amp=False, amp_dtype='float32', forward_dtype='float64', activation_checkpoint=False,
               compile=False, kv_write_reduction='mean', nograd_blocks=0)
    torch.manual_seed(opt.seed)
    base = t.ACTLossHead(t.LT(cfg), q_weight=cfg['q_weight'])
    base.eval()
    lt = base.model
    layer = lt.inner.layers[0]

    records = []
    original = rp.phase_clock_step

    def hooked(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
        hebb, im_s, new = original(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces)
        alpha = layer.stdp_alpha.detach()[None, :, None, None]
        # same-block window factor per cell and synapse: [B, H, T, post P, pre P]
        diff = phase_v[..., :, None] - phase_k[..., None, :]
        factor = 1 + alpha[..., None] * (1 - lam_k[..., None, :]) * torch.sin(diff)
        records.append(dict(phase_k=phase_k.clone(), phase_v=phase_v.clone(), factor=factor.clone(),
                            hebb=hebb.norm(dim=(-1, -2)).mean().item(),
                            timing=(alpha * im_s).norm(dim=(-1, -2)).mean().item()))
        return hebb, im_s, new
    rp.phase_clock_step = hooked
    batch = dict(inputs=x, labels=y, puzzle_identifiers=torch.zeros(opt.n, dtype=torch.int32))
    with torch.no_grad():
        carry = lt.initial_carry(batch)
        accs = []
        for _ in range(opt.blocks):
            carry, out = lt(carry, batch)
            accs.append(float((out['logits'].argmax(-1) == y).double().mean()))
    rp.phase_clock_step = original

    rows = []
    print(f'alpha {layer.stdp_alpha.tolist()}  lambda_K {torch.exp(-torch.nn.functional.softplus(layer.clock_lam_raw[0])).mean():.3f}')
    print('block | mean|phi_K| mean|phi_V| (rad) | LTP share  LTD share  zero share | factor 5%/50%/95% | ||alpha Im S||/||Hebb||')
    for b, r in enumerate(records, 1):
        f = r['factor']
        ltp, ltd = float((f > 1 + 1e-6).double().mean()), float((f < 1 - 1e-6).double().mean())
        q = torch.quantile(f.flatten()[:2_000_000], torch.tensor([0.05, 0.5, 0.95])).tolist()
        row = dict(block=b, phase_k=float(r['phase_k'].abs().mean()), phase_v=float(r['phase_v'].abs().mean()),
                   ltp=ltp, ltd=ltd, zero=1 - ltp - ltd, factor_q=q, ratio=r['timing'] / r['hebb'], acc=accs[b - 1])
        rows.append(row)
        if b <= 4 or b in (8, 16, 24, 32) or b == len(records):
            print(f"{b:5d} | {row['phase_k']:.3f} {row['phase_v']:.3f} | {ltp:.3f} {ltd:.3f} {row['zero']:.3f} | "
                  f"{q[0]:.3f} {q[1]:.3f} {q[2]:.3f} | {row['ratio']:.3f}")
    # one example cell (puzzle 0, first blank cell): 6x6 table of same-block factors at a few blocks
    cell = int((x[0] == 0).nonzero()[0])
    examples = {}
    for b in (2, 3, 8, 32):
        if b <= len(records):
            tab = records[b - 1]['factor'][0, 0, cell]
            examples[b] = tab.tolist()
            print(f'\npuzzle 0, cell {cell} (row {cell // 6}, col {cell % 6}), block {b}: 1 + alpha(1-lam) sin(phi_V,a - phi_K,i); rows = post V1..V6, cols = pre K1..K6')
            print(f"  phi_K = {[round(v, 2) for v in records[b - 1]['phase_k'][0, 0, cell].tolist()]}")
            print(f"  phi_V = {[round(v, 2) for v in records[b - 1]['phase_v'][0, 0, cell].tolist()]}")
            for a in range(tab.shape[0]):
                print('  ' + ' '.join(f'{v:6.3f}' for v in tab[a].tolist()))
    out = ROOT / 'docs/research/2026-10-14/results'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'ltp_ltd_init_seed{opt.seed}_n{opt.n}_b{opt.blocks}.json').write_text(
        json.dumps(dict(config=dict(hidden=12, heads=1, pairs=6, blocks=opt.blocks, n=opt.n, seed=opt.seed,
                                    dtype='float64', kernel='torch', clock='displacement', trained=False),
                        per_block=rows, example_cell=cell, example_factors=examples), indent=1) + '\n')


if __name__ == '__main__':
    main()
