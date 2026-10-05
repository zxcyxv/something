"""Independent research on smooth, freely crossing, difference-only STDP windows.

This module fits scalar windows and supplies exact finite-feature KV operators.
It does not modify any running model. A finite feature operator is exact for its
chosen surrogate window, not for the discontinuous signed exponential.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/research/2026-10-05/free_phase_window_research'


def smooth_scale(epsilon, tau=1.0):
    lo, hi = 0.0, 10 * max(epsilon, tau)
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid * mid * math.sqrt(mid * mid + epsilon * epsilon) > tau * epsilon * epsilon:
            hi = mid
        else:
            lo = mid
    peak = (lo + hi) / 2
    radius = math.sqrt(peak * peak + epsilon * epsilon)
    return radius / peak * math.exp(radius / tau), peak


def smooth_window(delta, epsilon, tau=1.0, scale=None):
    scale = smooth_scale(epsilon, tau)[0] if scale is None else scale
    radius = (delta.square() + epsilon ** 2).sqrt()
    return scale * delta / radius * (-radius / tau).exp()


def smooth_derivative(delta, epsilon, tau=1.0, scale=None):
    scale = smooth_scale(epsilon, tau)[0] if scale is None else scale
    radius = (delta.square() + epsilon ** 2).sqrt()
    return scale * (-radius / tau).exp() * (
        epsilon ** 2 / radius ** 3 - delta.square() / (tau * radius.square()))


def sine_window(delta, frequencies, coefficients):
    return (torch.sin(delta[..., None] * frequencies) * coefficients).sum(-1)


def sine_derivative(delta, frequencies, coefficients):
    return (torch.cos(delta[..., None] * frequencies) * coefficients * frequencies).sum(-1)


def direct_write(k, v, pk, pv, frequencies, coefficients):
    delta = pv[..., :, None] - pk[..., None, :]
    w = sine_window(delta, frequencies, coefficients)
    return (v[..., :, None] * k[..., None, :] * w).mean(-3)


def feature_write(k, v, pk, pv, frequencies, coefficients):
    """Pack 2R token features into one GEMM, contraction length 2R*T.

    [...,T,D] -> [...,R,T,D] -> [...,2R*T,D]. No [T,D,D] tensor.
    Frequencies and signed coefficients may be differentiated normally.
    Reference/layout audit only: this environment's compiled GPU backward
    failed the V-gradient check. Models use manual_feature_write instead.
    """
    omega = frequencies[:, None, None]
    kp, vp = pk.unsqueeze(-3) * omega, pv.unsqueeze(-3) * omega
    coeff = coefficients[:, None, None]
    va = torch.cat((v.unsqueeze(-3) * vp.sin() * coeff,
                    -v.unsqueeze(-3) * vp.cos() * coeff), dim=-3).flatten(-3, -2)
    ka = torch.cat((k.unsqueeze(-3) * kp.cos(),
                    k.unsqueeze(-3) * kp.sin()), dim=-3).flatten(-3, -2)
    return va.transpose(-1, -2) @ ka / k.shape[-2]


def loop_write(k, v, pk, pv, frequencies, coefficients):
    """Same chosen window; 2R separate GEMMs, useful as a layout control."""
    result = 0
    for omega, coefficient in zip(frequencies.unbind(), coefficients.unbind()):
        ka, va = omega * pk, omega * pv
        result = result + coefficient * (
            (v * va.sin()).transpose(-1, -2) @ (k * ka.cos())
            - (v * va.cos()).transpose(-1, -2) @ (k * ka.sin()))
    return result / k.shape[-2]


def two_feature_write(k, v, pk, pv, frequencies, coefficients):
    """Two GEMMs of contraction length R*T, without concatenated branches.

    Same mathematical work as feature_write. Kept separately because the
    installed torch.compile version produced an incorrect V gradient for both
    packed layouts with R=4. Use the independently verified explicit backward
    in manual_feature_write for compiled models.
    """
    omega = frequencies[:, None, None]
    kp, vp = pk.unsqueeze(-3) * omega, pv.unsqueeze(-3) * omega
    coefficient = coefficients[:, None, None]
    vs = (v.unsqueeze(-3) * vp.sin() * coefficient).flatten(-3, -2)
    vc = (v.unsqueeze(-3) * vp.cos() * coefficient).flatten(-3, -2)
    kc = (k.unsqueeze(-3) * kp.cos()).flatten(-3, -2)
    ks = (k.unsqueeze(-3) * kp.sin()).flatten(-3, -2)
    return (vs.transpose(-1, -2) @ kc - vc.transpose(-1, -2) @ ks) / k.shape[-2]


def _bf16_mm_fp32_output(a, b):
    """BF16 operands, FP32 output; phase evaluation and reductions stay FP32.

    Used only inside our explicit first-order autograd Function: the installed
    PyTorch has no automatic derivative for bmm(out_dtype=float32).
    """
    shape = a.shape[:-2] + (a.shape[-2], b.shape[-1])
    a = a.to(torch.bfloat16).flatten(0, -3)
    b = b.to(torch.bfloat16).flatten(0, -3)
    return torch.bmm(a, b, out_dtype=torch.float32).reshape(shape)


class _ManualFeatureWrite(torch.autograd.Function):
    """Analytic full backward, also differentiating frequencies/coefficients.

    Recompute O(R*T*D) features. This is an alternate derivation used to audit
    compiler-generated broadcast/concatenation backwards, not a surrogate grad.
    """
    @staticmethod
    def forward(ctx, k, v, pk, pv, frequencies, coefficients, bf16=False):
        ctx.bf16 = bf16
        ctx.save_for_backward(k, v, pk, pv, frequencies, coefficients)
        if bf16:
            omega, c = frequencies[:, None, None], coefficients[:, None, None]
            kp, vp = pk.unsqueeze(-3)*omega, pv.unsqueeze(-3)*omega
            vs = (v.unsqueeze(-3)*vp.sin()*c).flatten(-3,-2)
            vc = (v.unsqueeze(-3)*vp.cos()*c).flatten(-3,-2)
            kc = (k.unsqueeze(-3)*kp.cos()).flatten(-3,-2)
            ks = (k.unsqueeze(-3)*kp.sin()).flatten(-3,-2)
            return (_bf16_mm_fp32_output(vs.transpose(-1,-2),kc)
                    - _bf16_mm_fp32_output(vc.transpose(-1,-2),ks))/k.shape[-2]
        return two_feature_write(k, v, pk, pv, frequencies, coefficients)

    @staticmethod
    def backward(ctx, gradient):
        k, v, pk, pv, frequencies, coefficients = ctx.saved_tensors
        omega, c = frequencies[:, None, None], coefficients[:, None, None]
        k1, v1 = k.unsqueeze(-3), v.unsqueeze(-3)
        pk1, pv1 = pk.unsqueeze(-3), pv.unsqueeze(-3)
        sk, ck = (pk1 * omega).sin(), (pk1 * omega).cos()
        sv, cv = (pv1 * omega).sin(), (pv1 * omega).cos()
        vs, vc, kc, ks = v1 * sv * c, v1 * cv * c, k1 * ck, k1 * sk
        dg = gradient / k.shape[-2]
        mm = _bf16_mm_fp32_output if ctx.bf16 else torch.matmul
        a = mm(kc.flatten(-3, -2), dg.transpose(-1, -2)).reshape_as(vs)
        b = mm(ks.flatten(-3, -2), dg.transpose(-1, -2)).reshape_as(vc)
        u = mm(vs.flatten(-3, -2), dg).reshape_as(kc)
        z = mm(vc.flatten(-3, -2), dg).reshape_as(ks)
        # Both signs are explicit in every local derivative.
        dk = (ck * u - sk * z).sum(-3)
        dv = (c * (sv * a - cv * b)).sum(-3)
        dpk_terms = -k1 * (sk * u + ck * z)
        dpv_terms = c * v1 * (cv * a + sv * b)
        dpk, dpv = (omega * dpk_terms).sum(-3), (omega * dpv_terms).sum(-3)
        reduction = tuple(i for i in range(vs.ndim) if i != vs.ndim - 3)
        dc = (v1 * (sv * a - cv * b)).sum(reduction)
        dw = (pk1 * dpk_terms + pv1 * dpv_terms).sum(reduction)
        return dk, dv, dpk, dpv, dw, dc, None


def manual_feature_write(k, v, pk, pv, frequencies, coefficients):
    return _ManualFeatureWrite.apply(k, v, pk, pv, frequencies, coefficients, False)


def bf16_feature_write(k, v, pk, pv, frequencies, coefficients):
    """Mixed-precision numerical approximation, not the FP32-verified operator.

    Both forward and analytic backward GEMMs round operands to BF16. No sign
    surrogate or detach is introduced; report its measured numerical errors.
    """
    return _ManualFeatureWrite.apply(k, v, pk, pv, frequencies, coefficients, True)


def harmonic_coefficients(epsilon, modes, domain=math.pi):
    x = torch.linspace(0, domain, 20001, dtype=torch.float64)
    frequencies = torch.arange(1, modes + 1, dtype=x.dtype) * math.pi / domain
    coefficients = 2 / domain * torch.trapezoid(
        smooth_window(x, epsilon)[:, None] * torch.sin(x[:, None] * frequencies), x, dim=0)
    return frequencies, coefficients


def sign_certificate(frequencies, coefficients, domain=math.pi, points=65537):
    """Analytic small-lag positivity plus a Lipschitz grid lower bound.

    A numerical certificate with a reported margin, not interval arithmetic.
    Coefficients/frequencies must be positive for the analytic first interval.
    """
    w=torch.as_tensor(frequencies,dtype=torch.float64)
    c=torch.as_tensor(coefficients,dtype=torch.float64)
    if not bool((w>0).all() and (c>0).all()):
        return dict(certified=False,reason='The small-lag positive-sine argument does not apply.')
    edge=min(domain,math.pi/(2*float(w.max())))
    x=torch.linspace(edge,domain,points,dtype=torch.float64)
    sampled=float(sine_window(x,w,c).min())
    lipschitz=float((w*c).abs().sum())
    bound=sampled-lipschitz*(domain-edge)/(2*(points-1))
    return dict(certified=bound>1e-10,analytic_positive_interval=[0,edge],sampled_min=sampled,
                global_derivative_abs_bound=lipschitz,between_samples_lower_bound=bound,
                note='Float64 calculation; analytic all-positive sines near zero; bound margin exceeds 1e-10, not formal interval arithmetic.')


def fit_free_frequencies(epsilon, modes, seed=0, steps=240, derivative_weight=.025):
    """Deterministic offline nonlinear fit, with sampled sign penalties.

    The penalty is NOT a proof of sign preservation between samples.
    Optimizing frequencies is a numerical proposal, not a claim from the
    cited positive-definite-kernel quadrature theorems (our kernel is odd).
    """
    x = torch.linspace(0, math.pi, 1537, dtype=torch.float64)
    target = smooth_window(x, epsilon)
    target_grad = smooth_derivative(x, epsilon)
    starts = [torch.arange(1, modes + 1, dtype=x.dtype),
              torch.linspace(.5, max(2.0, modes * 1.6), modes, dtype=x.dtype),
              torch.logspace(math.log10(.55), math.log10(max(2.0, modes * 2.3)), modes, dtype=x.dtype)]
    best = None
    for index, initial in enumerate(starts):
        torch.manual_seed(seed + index)
        # Invert softplus to optimize positive frequencies.
        raw = torch.nn.Parameter(torch.log(torch.expm1(initial)))
        design = torch.sin(x[:, None] * initial)
        coefficient = torch.nn.Parameter(torch.linalg.lstsq(design, target).solution)
        optimizer = torch.optim.LBFGS((raw, coefficient), lr=.8, max_iter=steps,
                                     line_search_fn='strong_wolfe', tolerance_grad=1e-10,
                                     tolerance_change=1e-13)
        def closure():
            optimizer.zero_grad()
            frequency = F.softplus(raw)
            prediction = sine_window(x, frequency, coefficient)
            gradient = sine_derivative(x, frequency, coefficient)
            loss = ((prediction - target).square().mean()
                    + derivative_weight * epsilon ** 2 * (gradient - target_grad).square().mean()
                    + 8 * (-prediction).relu().square().mean()
                    + 1e-9 * coefficient.square().sum())
            loss.backward()
            return loss
        optimizer.step(closure)
        loss = float(closure().detach())
        frequency = F.softplus(raw).detach()
        order = frequency.argsort()
        candidate = (loss, frequency[order], coefficient.detach()[order])
        if best is None or loss < best[0]:
            best = candidate
    return best[1], best[2], best[0]


def measure_shape(epsilon, frequency, coefficient):
    x = torch.linspace(0, math.pi, 32769, dtype=torch.float64)
    target, dtarget = smooth_window(x, epsilon), smooth_derivative(x, epsilon)
    actual, dactual = sine_window(x, frequency, coefficient), sine_derivative(x, frequency, coefficient)
    error = actual - target
    # Difference of two independently uniform phase coordinates has triangular density.
    triangular = 1 - x / math.pi
    tail = x >= 1.5
    center = x <= epsilon
    peak_idx = int(actual.argmax())
    return dict(rmse=float(error.square().mean().sqrt()), max_abs_error=float(error.abs().max()),
                phase_pair_weighted_rmse=float((error.square() * triangular).sum().div(triangular.sum()).sqrt()),
                derivative_rmse=float((dactual - dtarget).square().mean().sqrt()),
                derivative_relative_l2=float((dactual - dtarget).norm() / dtarget.norm()),
                derivative_zero=float(dactual[0]), target_derivative_zero=float(dtarget[0]),
                central_max_error=float(error[center].abs().max()),
                tail_max_error=float(error[tail].abs().max()),
                positive_half_min=float(actual[1:-1].min()),
                wrong_sign_fraction=float((actual[1:-1] < -1e-8).double().mean()),
                peak_location=float(x[peak_idx]), peak_value=float(actual[peak_idx]),
                target_peak_location=smooth_scale(epsilon)[1],
                negative_derivative_fraction=float((dactual < 0).double().mean()),
                coefficient_abs_sum=float(coefficient.abs().sum()))


def operator_checks():
    torch.manual_seed(7201)
    shape = (2, 2, 7, 8)
    k, v, pk, pv = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    omega = torch.tensor([.72, 2.31, 4.67], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor([.61, .32, .18], dtype=torch.float64, requires_grad=True)
    inputs = (k, v, pk, pv, omega, coefficient)
    ref = direct_write(*inputs)
    cot = torch.randn_like(ref)
    grad_ref = torch.autograd.grad((ref * cot).sum(), inputs)
    checks = {}
    for name, function in [('packed', feature_write), ('two_packed', two_feature_write),
                           ('manual', manual_feature_write), ('loop', loop_write)]:
        actual = function(*inputs)
        grads = torch.autograd.grad((actual * cot).sum(), inputs)
        err = float((actual - ref).abs().max())
        ge = max(float((a - b).abs().max()) for a, b in zip(grads, grad_ref))
        assert err < 1e-12 and ge < 1e-11, (name, err, ge)
        checks[name] = dict(output_max_abs_error=err, full_gradient_max_abs_error=ge)
    with torch.no_grad():
        shift = torch.randn(shape[:-1] + (1,), dtype=k.dtype)
        shifted = feature_write(k, v, pk + shift, pv + shift, omega, coefficient)
        error = float((shifted - ref).abs().max())
        assert error < 1e-12
        checks['common_delay_shift_invariance'] = error
        # Same exact lag=0 convention, now through a continuous odd window.
        tied = sine_window(torch.zeros(11, dtype=k.dtype), omega, coefficient)
        assert tied.count_nonzero() == 0
        checks['exact_tie_zero'] = True
        split = (3 * feature_write(k[..., :3, :], v[..., :3, :], pk[..., :3, :], pv[..., :3, :], omega, coefficient)
                 + 4 * feature_write(k[..., 3:, :], v[..., 3:, :], pk[..., 3:, :], pv[..., 3:, :], omega, coefficient))
        error = float((split - 7 * ref).abs().max())
        assert error < 1e-11
        checks['unnormalized_write_additivity'] = error
    return checks


def rank_bounds(epsilon, grid=256):
    x = torch.linspace(-math.pi / 2, math.pi / 2, grid, dtype=torch.float64)
    matrix = smooth_window(x[:, None] - x[None, :], epsilon)
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    return {str(rank): float(energy[rank:].sum().sqrt() / energy.sum().sqrt())
            for rank in (2, 4, 8, 16, 32, 64)}


def shape_study(destination=DEST, epsilons=(.08, .2, .35, .5), modes=(1, 2, 4, 8)):
    torch.set_num_threads(2)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(scope='CPU scalar/kernel study, no GPU speed or learning claim',
                  phase_range=[-math.pi / 2, math.pi / 2], difference_range=[-math.pi, math.pi],
                  target='C*delta/sqrt(delta^2+epsilon^2)*exp(-sqrt(delta^2+epsilon^2)), peak magnitude 1',
                  tau=1.0, finite_feature_cost='2R ordinary KV products, packed contraction 2R*T',
                  sign_warning='Free-frequency sign penalties and dense grids are not global sign certificates.',
                  extrapolation='Finite sums of sines do not decay to zero at both infinities; fit domain is bounded.',
                  checks=operator_checks(), rank_lower_bounds={}, rows=[])
    start = time.monotonic()
    path = destination / 'shape_study.json'
    for epsilon in epsilons:
        report['rank_lower_bounds'][str(epsilon)] = rank_bounds(epsilon)
        for r in modes:
            for family in ('integer_fourier', 'optimized_frequencies'):
                if family == 'integer_fourier':
                    omega, coefficient = harmonic_coefficients(epsilon, r)
                    objective = None
                else:
                    omega, coefficient, objective = fit_free_frequencies(epsilon, r)
                row = dict(family=family, epsilon=epsilon, modes=r,
                           frequencies=omega.tolist(), coefficients=coefficient.tolist(),
                           optimization_objective=objective, **measure_shape(epsilon, omega, coefficient))
                report['rows'].append(row)
                report['elapsed_seconds'] = time.monotonic() - start
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(json.dumps({k: row[k] for k in ('family', 'epsilon', 'modes', 'rmse', 'max_abs_error',
                                                     'derivative_relative_l2', 'wrong_sign_fraction')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEST)
    arguments = parser.parse_args()
    shape_study(arguments.out)
