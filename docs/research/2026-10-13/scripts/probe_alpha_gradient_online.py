"""Record dL/d alpha_h at every optimizer step while continuing the real training loop from a checkpoint.

Usage (repository root): python docs/research/2026-10-13/scripts/probe_alpha_gradient_online.py RUN STEP STEPS [OUT_DIR]

Resumes the run (raw weights, optimizer and EMA state, carry, RNG) in a scratch directory and trains STEPS more
steps with the trainer itself (augmented batches, 16-segment carry, AdamATan2). Hooks _check_finite_gradients
(called after backward, before the step) to log stdp_alpha.grad and the value of alpha per step. Writes
docs/research/2026-10-13/results/alpha_gradient_online_<run>_step<STEP>_<STEPS>.json with per-head sign fractions.
"""
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lt import train as t
from lt import research_phase_clock_stdp as rp

run, step, steps = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2]), int(sys.argv[3])
out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / 'runs' / f'_alpha_online_{run.name}_{step}'
ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
protocol = json.loads((run / 'protocol.json').read_text())
cfg = dict(ck['cfg'])
cfg.update(out_dir=str(out_dir), resume_from=str(run / f'step_{step}.pt'), require_resume=True, max_steps=step + steps,
           save_every_steps=0, save_at_boundary=False, keep_last=1, milestone_every=0, log_every=100,
           data_npz=str(ROOT / 'data/sudoku_lt_1k.npz'))
torch.set_float32_matmul_precision(protocol['precision'])
t.KVSTDPInner = rp.model_class(protocol['lambda_init'], protocol['alpha_init'], protocol['dt_init'], protocol['boundary_ffn'],
                               protocol.get('kernel', 'torch'), protocol.get('feature_precision', 'float32'),
                               protocol.get('clock_precision', 'float32'))
t.model_id_of = lambda c: ck['model_id']
grads, values = [], []
old = t._check_finite_gradients


def hook(base, loss, device, ws):
    old(base, loss, device, ws)
    a = base.model.inner.layers[0].stdp_alpha
    grads.append(a.grad.detach().float().cpu().clone())
    values.append(a.detach().float().cpu().clone())
t._check_finite_gradients = hook
t.main(cfg)
t._check_finite_gradients = old
g, v = torch.stack(grads), torch.stack(values)
res = dict(run=run.name, from_step=step, steps=len(grads), alpha_start=v[0].tolist(), alpha_end=v[-1].tolist(),
           grad_mean=g.mean(0).tolist(), grad_abs_mean=g.abs().mean(0).tolist(), frac_negative=(g < 0).float().mean(0).tolist(),
           grad_mean_over_abs_mean=(g.mean(0) / g.abs().mean(0)).tolist())
path = ROOT / 'docs/research/2026-10-13/results' / f'alpha_gradient_online_{run.name}_step{step}_{len(grads)}.json'
path.write_text(json.dumps(res, indent=1) + '\n')
print('head  alpha start -> end   mean grad   |grad| mean   frac<0   mean/|mean|')
for h in range(g.shape[1]):
    print(f'{h:4d}  {v[0, h]:.3f} -> {v[-1, h]:.3f}   {g[:, h].mean():+.4f}   {g[:, h].abs().mean():.4f}   {(g[:, h] < 0).float().mean():.2f}   {g[:, h].mean() / g[:, h].abs().mean():+.2f}')
print('WROTE', path)
