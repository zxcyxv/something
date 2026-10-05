"""Measure window approximation on saved real activations, entirely on CPU.

The source model is the stopped local monotone-warp model, so its phase
distribution is contextual evidence, not the distribution of a trained free-
phase model. No labels are broken down into clues/blanks.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import types

import torch

from . import train as t
from .kv_stability import install
from .research_free_phase_windows import DEST, smooth_window, smooth_derivative, two_feature_write


def capture_source():
    """Optional recapture requires the original local-warp code and checkpoint.

    The default analysis uses the archived activation sample and needs neither.
    """
    root = Path('runs/kv_phase_local_warp_exp_fresh_20261005')
    path = max(root.glob('step_*.pt'), key=lambda p: int(p.stem[5:]))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'], batch_size=4, seq_len=81, num_puzzle_identifiers=1,
               amp=False, activation_checkpoint=False)
    install('phase_local_warp_exp_current_only')
    base = t.ACTLossHead(t.LT(cfg))
    base.load_state_dict(ck['raw_model_state_dict'], strict=True)
    base.eval()
    def take(value):
        if isinstance(value, torch.Tensor):
            return value[:4].clone()
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value
    carry = t.LTCarry(**take(ck['rank_states'][0]['carry']))
    batch = carry.current_data
    inner = base.model.inner
    original = inner.memory_step
    saved, counter = [], [0]
    def capture(self, layer, q, k, v, *args, **kwargs):
        if counter[0] in (0, 7):
            qr, kr = (self.apply_rope(x.float(), layer) for x in (q, k))
            pk, pv = self.phases(layer, kr, v.float())
            saved.append(dict(block=counter[0], **{n: x[:2].detach().clone() for n, x in
                          zip(('q', 'k', 'v', 'pk', 'pv'), (qr, kr, v.float(), pk, pv))}))
        counter[0] += 1
        return original(layer, q, k, v, *args, **kwargs)
    inner.memory_step = types.MethodType(capture, inner)
    with torch.no_grad():
        base.model(carry, batch)
    torch.save(dict(checkpoint=str(path), step=ck['step'], scope='raw source model CPU FP32 continuation',
                    samples=saved), DEST / 'activation_samples.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true', help='Recapture from the original stopped local-warp run')
    opt = ap.parse_args()
    torch.set_num_threads(2)
    if opt.capture:
        capture_source()
    data = torch.load(DEST / 'activation_samples.pt', map_location='cpu', weights_only=False)
    saved = data['samples']
    fits = json.loads((DEST / 'shape_study.json').read_text())['rows']
    report = dict(checkpoint=data['checkpoint'], step=data['step'],
                  scope='Actual source-model states replayed with CPU FP32, not a trained free-phase model.',
                  rows=[])
    for sample in saved:
        # One batch/head keeps the direct gradient reference modest in memory.
        args = tuple(sample[n][:1, :1].double().requires_grad_() for n in ('q', 'k', 'v', 'pk', 'pv'))
        q, k, v, pk, pv = args
        delta = pv[..., :, None] - pk[..., None, :]
        gaps = delta.detach().abs().flatten()
        gap_metrics = {str(threshold): float((gaps < threshold).double().mean())
                       for threshold in (.02, .05, .1, .2, .35, .5)}
        torch.manual_seed(73 + sample['block'])
        cotangent = torch.randn_like(q)
        for epsilon, modes in ((.2, 8), (.35, 4), (.35, 8), (.5, 4)):
            fit = next(r for r in fits if r['family'] == 'optimized_frequencies'
                       and r['epsilon'] == epsilon and r['modes'] == modes)
            omega = torch.tensor(fit['frequencies'], dtype=q.dtype)
            coefficient = torch.tensor(fit['coefficients'], dtype=q.dtype)
            direct = (v[..., :, None] * k[..., None, :] * smooth_window(delta, epsilon)).mean(-3)
            expected = q @ direct.transpose(-1, -2)
            grads = torch.autograd.grad((expected * cotangent).sum(), args, retain_graph=True)
            actual = q @ two_feature_write(k, v, pk, pv, omega, coefficient).transpose(-1, -2)
            approximations = torch.autograd.grad((actual * cotangent).sum(), args)
            row = dict(block=sample['block'], epsilon=epsilon, modes=modes, phase_gap_fractions=gap_metrics,
                       read_relative_l2=float((actual.detach() - expected.detach()).norm() / expected.detach().norm()),
                       gradients={name: float((a - b).norm() / b.norm()) for name, a, b in
                                  zip(('q', 'k', 'v', 'pk', 'pv'), approximations, grads)})
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
    (DEST / 'activation_approximation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
