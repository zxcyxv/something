"""Order/latency split of the trained STDP term and the label-free convergence detector (architecture.md, 4).

    python -m lt.analysis.order_latency CHECKPOINT [--n 512] [--blocks 128] [--batch 256] [--out results.json]

Held-out puzzles, EMA weights, torch path, one block at a time from a fresh carry. Per block (mean over lanes and
heads): |O|/|Hebb|, |L|/|Hebb|, |L|/|SB| against the fixed-point prediction kappa, cos(L, SB), and |h_b - h_{b-1}|.
Per lane: solve block (all cells right from there on) and settle block (|O|/|Hebb| below 10% of its own peak from
there on); 'settled' predicts 'solved' without labels.
Reference (run A, step 46,872, 512 puzzles): |L|/|SB| 1.218 vs kappa 1.221, cos 0.999; |O|/|Hebb| -> 0.000 on solved
puzzles, 0.054 floor on unsolved; settle vs solve Spearman 0.994; detector 97%.
"""
import argparse
import json
import time

import torch

from ..data import eval_batches, load_data
from .common import OrderLatencyProbe, fro, kappa, latency_term, load_model, run_blocks, spearman, suffix_block


def cosine(a, b):
    return (a.flatten(-2) * b.flatten(-2)).sum(-1) / (fro(a) * fro(b) + 1e-12)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--blocks', type=int, default=128)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out')
    opt = ap.parse_args(argv)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(opt.checkpoint, device)
    mc = model.config
    k_pred = kappa(mc.trace_lambda, mc.block_omega)
    _, _, x, y, _, _ = load_data(cfg.get('data_npz'))
    x, y = x[:opt.n], y[:opt.n]
    began = time.monotonic()
    keys = ('O_hebb', 'L_hebb', 'L_sb', 'cos_L_sb', 'dh')
    curves = {k: torch.zeros(opt.blocks, dtype=torch.float64) for k in keys}
    o_ratio, solved_at = [], []
    for batch in eval_batches(x, y, opt.batch):
        b = {k: v.to(device) for k, v in batch.items()}
        probe = OrderLatencyProbe(mc.block_omega, mc.trace_lambda)
        model.probe = probe
        lane_o = torch.zeros(opt.blocks, len(b['labels']), device=device)
        prev_h = [None]

        def each(blk, carry, logits):
            O, L, rec = probe.last['O'], probe.last['L'], probe.last['rec']
            hebb, sb = rec.hebb, latency_term(rec, mc.block_omega)
            w = len(b['labels']) / len(x)
            i = blk - 1
            curves['O_hebb'][i] += w * float((fro(O) / fro(hebb)).mean())
            curves['L_hebb'][i] += w * float((fro(L) / fro(hebb)).mean())
            curves['L_sb'][i] += w * float((fro(L) / (fro(sb) + 1e-12)).mean())
            curves['cos_L_sb'][i] += w * float(cosine(L, sb).mean())
            if prev_h[0] is not None:
                curves['dh'][i] += w * float((carry.hidden - prev_h[0]).norm(dim=-1).mean())
            prev_h[0] = carry.hidden
            lane_o[i] = (fro(O) / fro(hebb)).mean(1)
        preds = run_blocks(model, b, opt.blocks, each)
        model.probe = None
        o_ratio.append(lane_o.cpu())
        solved_at.append(suffix_block((preds == b['labels'][None]).all(-1)).cpu())
    o_ratio, solve = torch.cat(o_ratio, 1), torch.cat(solved_at)
    solved = solve <= opt.blocks
    settle = suffix_block(o_ratio < 0.1 * o_ratio.max(0).values[None])
    settled = settle <= opt.blocks
    late = slice(max(0, opt.blocks - 32), opt.blocks)
    both = solved & settled
    summary = dict(
        checkpoint=str(opt.checkpoint), n=len(solve), blocks=opt.blocks, kappa=k_pred,
        alpha=[round(a, 4) for a in model.block.alpha.detach().tolist()], seconds=time.monotonic() - began,
        solved=int(solved.sum()), settled=int(settled.sum()),
        detector_accuracy=float((settled == solved).double().mean()),
        spearman_solve_settle=spearman(solve[both], settle[both]),
        late=dict(L_sb=float(curves['L_sb'][late].mean()), cos_L_sb=float(curves['cos_L_sb'][late].mean()),
                  O_hebb_solved=float(o_ratio[late][:, solved].mean()) if solved.any() else None,
                  O_hebb_unsolved=float(o_ratio[late][:, ~solved].mean()) if (~solved).any() else None),
        curves={k: v.tolist() for k, v in curves.items()})
    show = [b for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if b <= opt.blocks]
    print('block      ' + ' '.join(f'{b:7d}' for b in show))
    for k in keys:
        print(f'{k:10s} ' + ' '.join(f'{curves[k][b - 1]:7.3f}' for b in show))
    print(f"solved {summary['solved']}/{summary['n']}, settled {summary['settled']}, detector {summary['detector_accuracy']:.3f}, "
          f"spearman(solve, settle) {summary['spearman_solve_settle']:.3f}; late |L|/|SB| {summary['late']['L_sb']:.3f} "
          f"(kappa {k_pred:.3f}), cos {summary['late']['cos_L_sb']:.3f}")
    if opt.out:
        with open(opt.out, 'w') as f:
            json.dump(summary, f, indent=1)
    return summary


if __name__ == '__main__':
    main()
