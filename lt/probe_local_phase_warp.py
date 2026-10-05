"""CPU algebra checks only; does not install or train a model.

Preserve additive token writes by predicting phases locally. A common strictly
increasing phase-axis warp per token/head preserves the baseline K/V ordering,
which makes the signed exponential exact with two KV matrix multiplications.
"""
import json
import math
from pathlib import Path

import torch

from .probe_dynamic_phase_factorization import compare, direct_write, ordered_write, shared_write


def warp(base_phase, logits):
    """Continuous increasing piecewise-linear map [-pi/2,pi/2] -> itself.

    base_phase: [heads,channels]; logits: [batch,heads,tokens,intervals].
    Zero logits give identity. One map is shared by K and V at each token/head.
    Interval counts describe the PHASE axis; no spatial token compression.
    """
    limit = math.pi / 2
    intervals = logits.shape[-1]
    lengths = 2 * limit * logits.softmax(-1)
    lower_endpoint = torch.full_like(logits[..., :1], -limit)
    knots = torch.cat((lower_endpoint, -limit + lengths.cumsum(-1)), -1)
    coordinate = (base_phase + limit) * intervals / (2 * limit)
    index = coordinate.floor().long().clamp(0, intervals - 1)
    fraction = coordinate - index
    index = index[None, :, None].expand(*logits.shape[:-1], base_phase.shape[-1])
    lo, hi = (knots.gather(-1, index + offset) for offset in (0, 1))
    return lo + fraction[None, :, None] * (hi - lo)


def phases(k, v, weight, tk, tv):
    # A local head can use the existing contextual K/V for each token. The
    # probe uses a linear head; this is not an implemented model architecture.
    logits = torch.cat((k, v), -1) @ weight.T
    base_k, base_v = ((math.pi / 2) * x.tanh() for x in (tk, tv))
    return warp(base_k, logits), warp(base_v, logits)


def reference(k, v, weight, tk, tv):
    pk, pv = phases(k, v, weight, tk, tv)
    return direct_write(v, k, pv, pk)


def candidate(k, v, weight, tk, tv):
    pk, pv = phases(k, v, weight, tk, tv)
    # tanh and the local warp are both increasing, so raw offsets have the
    # same order as all local phase outputs. These masks do not depend on n.
    delta0 = tv[..., :, None] - tk[..., None, :]
    return ordered_write(v, k, pv, pk, delta0 > 0, delta0 < 0)


def contributions(k, v, weight, tk, tv):
    pk, pv = phases(k, v, weight, tk, tv)
    delta = pv[..., :, None] - pk[..., None, :]
    window = delta.sign() * (-delta.abs()).exp()
    return v[..., :, None] * k[..., None, :] * window


def main():
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(20261005)

    def randn(shape, scale=1):
        return (scale * torch.randn(shape, generator=gen, dtype=torch.float64)).requires_grad_()

    k, v = [randn((2, 2, 9, 8)) for _ in range(2)]
    weight = randn((8, 16), .35)
    tk, tv = [randn((2, 8), .35) for _ in range(2)]
    report = dict(seed=20261005, device='cpu', dtype='float64',
                  scope='operator proof prototype, not a training or speed experiment',
                  formulas=dict(warp='positive normalized interval lengths; common increasing map for K/V at each token',
                                phase='phi_role[n,c]=F_n((pi/2)*tanh(theta_role[c]))',
                                positive='P * ((V*exp(-phiV)).T @ (K*exp(phiK))) / N',
                                negative='-Q * ((V*exp(phiV)).T @ (K*exp(-phiK))) / N'),
                  checks={})
    checks = report['checks']
    checks['two_gemm_output_and_full_gradients'] = compare(
        reference, candidate, (k, v, weight, tk, tv), gen)
    checks['gradient_input_order'] = ['K', 'V', 'local_head_weight', 'thetaK_raw', 'thetaV_raw']

    zero_weight = torch.zeros_like(weight, requires_grad=True)
    base_k, base_v = ((math.pi / 2) * x.tanh() for x in (tk, tv))
    initial = candidate(k, v, zero_weight, tk, tv)
    fixed = shared_write(v, k, base_v, base_k)
    error = (initial - fixed).abs().max().item()
    assert error < 1e-12
    gradient = torch.autograd.grad(initial.square().sum(), zero_weight)[0]
    assert gradient.isfinite().all() and gradient.norm() > 0
    checks['baseline_recovery'] = dict(max_abs_error=error,
                                      initial_local_head_gradient_norm=gradient.norm().item())

    with torch.no_grad():
        before = contributions(k, v, weight, tk, tv)
        changed = v.clone()
        changed[:, :, 0, 0] += 1
        after = contributions(k, changed, weight, tk, tv)
        unchanged_error = (before[:, :, 1:] - after[:, :, 1:]).abs().max().item()
        changed_norm = (before[:, :, 0] - after[:, :, 0]).norm().item()
        assert unchanged_error == 0 and changed_norm > 0
        checks['other_tokens_writes_unchanged'] = dict(max_abs_error=unchanged_error,
                                                     altered_token_write_change_norm=changed_norm)
        full_sum = 9 * candidate(k, v, weight, tk, tv)
        split_sum = (4 * candidate(k[:, :, :4], v[:, :, :4], weight, tk, tv)
                     + 5 * candidate(k[:, :, 4:], v[:, :, 4:], weight, tk, tv))
        error = (full_sum - split_sum).abs().max().item()
        assert error < 1e-12
        checks['unnormalized_write_additivity'] = dict(max_abs_error=error)

        pk, pv = phases(k, v, weight, tk, tv)
        actual_order = (pv[..., :, None] - pk[..., None, :]).sign()
        baseline_order = (tv[..., :, None] - tk[..., None, :]).sign()[None, :, None]
        flip_fraction = (actual_order != baseline_order).double().mean().item()
        assert flip_fraction == 0
        phase_displacement = torch.cat((pk - base_k[None, :, None], pv - base_v[None, :, None]), -1)
        checks['order_preserved_with_large_phase_motion'] = dict(
            flip_fraction=flip_fraction,
            rms_displacement_radians=phase_displacement.square().mean().sqrt().item(),
            max_abs_displacement_radians=phase_displacement.abs().max().item())

        # Equal K/V base phases remain exactly equal under the same warp;
        # both masks are zero and the original exact-tie convention is retained.
        tied_v = tv.clone()
        tied_v[:, 0] = tk[:, 0]
        direct = reference(k, v, weight, tk, tied_v)
        fast = candidate(k, v, weight, tk, tied_v)
        assert torch.equal(direct[..., 0, 0], torch.zeros_like(direct[..., 0, 0]))
        assert (direct - fast).abs().max() < 1e-12
        checks['exact_tie_zero'] = True

    report['cuda_initialized'] = torch.cuda.is_initialized()
    path = Path('docs/research/2026-10-05/local_phase_warp_algebra.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
