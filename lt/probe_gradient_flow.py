"""Gradient flow of address-angle windows against the plain Hebbian write.

For one checkpoint (or the seed-0 initialisation) the same weights are run as
several model classes that differ only in the write: pairangle with the K phase
from the address before RoPE, pairangle with the rotated K phase, and the plain
Hebbian product (window = 1, the B-only computation). One training batch, one
supervised segment after `--warm` no-grad segments, eager, no activation
checkpoint. Reports per block: hidden gradient, attention read RMS and the
attention residual relative to the hidden state; per parameter: gradient norm;
and for pairangle the share of the K/V/Q gradient that flows through the phase
(full gradient minus the gradient with detached phases).

python -m lt.probe_gradient_flow --ckpt runs/<run>/step_2000.pt --out runs/<run>/diagnostics/gradflow_step2000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .experiment_free_phase_windows import configuration, model_class

VARIANTS = {
    'pairangle_unrotated': dict(window='pairangle', phase_frame='unrotated'),
    'pairangle_rotated': dict(window='pairangle', phase_frame='rotated'),
    'hebbian': dict(window='hebbian', phase_frame='rotated'),
}


def build(variant, cfg, device, state):
    spec = VARIANTS[variant]
    t.KVSTDPInner = model_class(spec['window'], spec['window'] == 'pairangle', 2, .35, 'diagonal' if spec['window'] == 'hebbian' else 'dense',
                                'float32', 1.0, False, False, True, True, 2.0, 0.5, 'none', False, 1.0, 0.0,
                                spec['phase_frame'])
    torch.manual_seed(cfg['seed'])
    with torch.device(device):
        base = t.ACTLossHead(t.LT(dict(cfg, batch_size=state['batch'], seq_len=81, num_puzzle_identifiers=1)),
                             q_weight=cfg['q_weight'])
    if state['weights'] is not None:
        own = base.state_dict()
        missing = [k for k in own if k not in state['weights'] and 'phase_local' not in k]
        if missing:
            raise KeyError(missing)
        base.load_state_dict({k: state['weights'][k] for k in own if k in state['weights']}, strict=False)
    else:
        state['weights'] = {k: v.detach().clone() for k, v in base.state_dict().items()}
    base.train()
    return base


def run(base, batch, warm, detach_phase=False):
    inner = base.model.inner
    records = []
    original_block, original_phases = inner.block, inner.phases
    original_memory = inner.memory_step

    def memory_step(L, q, k, v, *args, **kwargs):
        out = original_memory(L, q, k, v, *args, **kwargs)
        if torch.is_grad_enabled():
            records[-1]['read_rms'] = float(out[0].detach().float().pow(2).mean().sqrt())
        return out

    def block(L, h, *args):
        if torch.is_grad_enabled():
            records.append({})
        h_in = h
        out = original_block(L, h, *args)
        if torch.is_grad_enabled():
            out[0].retain_grad()
            records[-1].update(h_out=out[0], h_in_rms=float(h_in.detach().float().pow(2).mean().sqrt()))
        return out

    def phases(*args, **kwargs):
        return tuple(p.detach() for p in original_phases(*args, **kwargs))

    inner.block, inner.memory_step = block, memory_step
    if detach_phase:
        inner.phases = phases
    try:
        carry = base.initial_carry(batch)
        with torch.no_grad():
            for _ in range(warm):
                carry, *_ = base(carry=carry, batch=batch, return_keys=set())
        base.zero_grad(set_to_none=True)
        records.clear()
        _, loss, metrics, *_ = base(carry=carry, batch=batch, return_keys=set())
        loss.backward()
    finally:
        inner.block, inner.memory_step, inner.phases = original_block, original_memory, original_phases
    blocks = [dict(read_rms=r.get('read_rms'), h_in_rms=r['h_in_rms'],
                   grad_h_out=float(r['h_out'].grad.float().norm()) if r['h_out'].grad is not None else 0.0)
              for r in records]
    grads = {n.replace('model.inner.', ''): p.grad.detach().float().clone()
             for n, p in base.named_parameters() if p.grad is not None}
    return float(loss), float(metrics['accuracy'] / max(float(metrics['count']), 1)), blocks, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=None, help='raw weights of this checkpoint; omitted: seed-0 initialisation')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--warm', type=int, nargs='+', default=[0, 7], help='no-grad segments before the supervised one')
    ap.add_argument('--out', type=Path, required=True)
    opt = ap.parse_args()
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = configuration()
    cfg.update(activation_checkpoint=False, compile=False)
    t._resolve_precision(cfg, device)
    state = dict(batch=opt.batch, weights=None)
    if opt.ckpt is not None:
        state['weights'] = torch.load(opt.ckpt, map_location='cpu', weights_only=False)['raw_model_state_dict']
    x, y, *_ = t.load_data(cfg)
    batch = {k: v.to(device) for k, v in next(t.eval_batches(x[:opt.batch], y[:opt.batch], opt.batch, 0, 1)).items()}
    report = dict(checkpoint=str(opt.ckpt) if opt.ckpt else 'seed-0 init', batch=opt.batch, results={})
    for warm in opt.warm:
        for variant in VARIANTS:
            base = build(variant, cfg, device, state)
            loss, acc, blocks, grads = run(base, batch, warm)
            row = dict(loss=loss, cell_accuracy=acc, blocks=blocks,
                       grad_norms={n: float(g.norm()) for n, g in grads.items()})
            if variant.startswith('pairangle'):
                _, _, _, detached = run(base, batch, warm, detach_phase=True)
                row['phase_path_share'] = {n: float((grads[n] - detached[n]).norm() / grads[n].norm())
                                           for n in grads if any(s in n for s in ('q_proj', 'k_proj', 'v_proj', 'theta'))}
            report['results'][f'{variant}@warm{warm}'] = row
            del base
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps(report, indent=2) + '\n')
    for key, row in report['results'].items():
        g = row['grad_norms']
        print(f"{key:28s} loss {row['loss']:.4f} acc {row['cell_accuracy']:.3f} | "
              + ' '.join(f"{n.split('.')[-2]}={g[n]:.3g}" for n in g if 'proj' in n or 'b_' in n))
        print('   grad|h| per block:', ' '.join(f"{b['grad_h_out']:.3g}" for b in row['blocks']))
        print('   read rms per block:', ' '.join(f"{b['read_rms']:.3g}" for b in row['blocks']))
        if 'phase_path_share' in row:
            print('   phase-path share:', {n.split('.')[-2] if 'proj' in n else n.split('.')[-1]: round(v, 3)
                                          for n, v in row['phase_path_share'].items()})


if __name__ == '__main__':
    main()
