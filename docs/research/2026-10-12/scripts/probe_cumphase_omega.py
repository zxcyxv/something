"""Angular velocity omega and offset lambda of a cumulative-phase checkpoint (EMA), with solved puzzles per segment.

Usage: python docs/research/2026-10-12/scripts/probe_cumphase_omega.py RUN_DIR STEP [STEP ...] [--n 128] [--segs 16]

Runs held-out puzzles with the carry kept across segments and records, per block, the mean |omega| (rad/block)
and the fraction of (puzzle, head, cell) entries with |omega| > 0.1 rad; at the last block the spread of lambda.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.cumulative_phase import make_model

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('steps', nargs='+', type=int)
p.add_argument('--n', type=int, default=128)
p.add_argument('--segs', type=int, default=16)
a = p.parse_args()
run = Path(a.run)
protocol = json.loads((run / 'protocol.json').read_text())
device = torch.device('cuda')
results = []
for step in a.steps:
    ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
    t._resolve_precision(cfg, device)
    sd = ck['model_state_dict']
    w = sd[next(k for k in sd if k.endswith('omega_proj'))].float()
    c = sd[next(k for k in sd if k.endswith('omega_bias'))].float()
    base = make_model(protocol, cfg, device, sd)
    inner, lt = base.model.inner, base.model
    orig, rec = inner.phase_offset, []
    def hook(layer, k, v, e_k, fresh, orig=orig, rec=rec):
        om, lam = orig(layer, k, v, e_k, fresh)
        m = om.detach().float().abs()
        rec.append((m.mean().item(), (m > 0.1).float().mean().item(), lam.detach().float()))
        return om, lam
    inner.phase_offset = hook
    _, _, x, y, _, fp = t.load_data(cfg)
    assert fp == cfg['data_fingerprint']
    solved = []
    with torch.no_grad():
        b = {k: v.to(device) for k, v in next(t.eval_batches(x[:a.n], y[:a.n], a.n, 0, 1)).items()}
        carry = lt.initial_carry(b)
        for s in range(a.segs):
            carry, out = lt(carry, b)
            solved.append(int((out['logits'].argmax(-1) == b['labels']).all(-1).sum()))
    om = np.array([r[0] for r in rec])
    big = np.array([r[1] for r in rec])
    lam = rec[-1][2]
    r = dict(step=step, blocks=len(rec), mean_abs_omega=om.round(4).tolist(), frac_omega_gt_0p1=big.round(4).tolist(),
             final_lambda_std=lam.std().item(), final_lambda_abs_mean=lam.abs().mean().item(),
             final_lambda_head_mean=lam.mean(dim=(0, 2, 3)).cpu().numpy().round(4).tolist(),
             omega_proj_norm_per_head=w.flatten(1).norm(dim=1).numpy().round(4).tolist(),
             omega_bias_per_head=c.flatten().numpy().round(4).tolist(), solved_per_segment=solved)
    results.append(r)
    pick = [0, 7, 31, len(om) - 1]
    print(f"step {step}: mean |omega| blocks {[i + 1 for i in pick]}: {' / '.join(f'{om[i]:.3f}' for i in pick)} rad; "
          f"|omega|>0.1 at last block {big[-1]:.3f}; final lambda std {r['final_lambda_std']:.3f}; "
          f"solved per segment (of {a.n}): {solved}", flush=True)
    del base, inner, lt
    torch.cuda.empty_cache()
out = Path(__file__).resolve().parents[1] / f"results/cumphase_omega_{run.name}_n{a.n}_seg{a.segs}.json"
old = json.loads(out.read_text())['results'] if out.exists() else []
merged = {r['step']: r for r in old + results}
out.write_text(json.dumps(dict(run=run.name, weights='EMA', n=a.n, segments=a.segs,
                               results=[merged[s] for s in sorted(merged)]), indent=1))
print('WROTE', out)
