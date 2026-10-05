"""CPU checks for efficient dynamic phase writes, independent of live training.

K is understood to already include the existing spatial RoPE. Token/observation
index n is not a spike time. phi denotes delay on a common carrier, so a positive
phi_V - phi_K is assigned positive STDP. All writes have G[value,key] layout.

The shared-phase and fixed-order paths preserve the isolated-pair exponential
exactly under their stated restrictions. Harmonic features preserve their
specified finite window exactly, not the discontinuous isolated-pair window.
No model, optimizer, or training configuration is installed by this module.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .research_phase_stdp_math import pulse_coefficients


def exponential(delta):
    return delta.sign() * (-delta.abs()).exp()


def direct_write(v, k, pv, pk, window=exponential):
    delta = pv[..., :, None] - pk[..., None, :]
    return (v[..., :, None] * k[..., None, :] * window(delta)).mean(-3)


def shared_write(v, k, pv, pk):
    """Phases [...,channels] shared over tokens, but dependent on the example."""
    window = exponential(pv[..., :, None] - pk[..., None, :])
    return (v.transpose(-1, -2) @ k) * window / v.shape[-2]


def puzzle_state_phases(hidden, weight_k, weight_v, offset_k, offset_v):
    """One phase per example/head/channel, computed afresh at each recurrence.

    hidden is [batch,tokens,hidden] after input injection. Pooling is followed
    by affine-free RMSNorm; this is separate from the two residual RMSNorms.
    In production these reductions/projections should run outside BF16 AMP.
    Zero projection weights exactly recover the fixed-phase baseline initially.
    """
    summary = hidden.mean(-2)
    summary = summary * torch.rsqrt(summary.square().mean(-1, keepdim=True) + 1e-5)
    return tuple((math.pi / 2) * (
        (summary @ weight.T).reshape(hidden.shape[0], *offset.shape) + offset
    ).tanh() for weight, offset in ((weight_k, offset_k), (weight_v, offset_v)))


def ordered_write(v, k, pv, pk, positive, negative):
    """Local phases, but strict order masks [...,Dv,Dk] shared over tokens.

    This is an exact identity only when every token obeys the supplied masks.
    It is not valid for arbitrary independently crossing local phases.
    """
    plus = (v * (-pv).exp()).transpose(-1, -2) @ (k * pk.exp())
    minus = (v * pv.exp()).transpose(-1, -2) @ (k * (-pk).exp())
    return (positive * plus - negative * minus) / v.shape[-2]


def harmonic_window(delta, coeff):
    return sum(c * torch.sin(m * delta) for m, c in enumerate(coeff, 1))


def packed_harmonic_write(v, k, pv, pk, coeff):
    """One GEMM with contraction length 2*R*N, not baseline GEMM FLOPs.

    Only O(R*N*d) features and the final d*d write are materialized. There is
    no [N,d,d] tensor. Each mode contributes sin(phiV)cos(phiK)-cos(phiV)sin(phiK).
    """
    va, ka = [], []
    for m, c in enumerate(coeff, 1):
        va.extend((c * v * torch.sin(m * pv), -c * v * torch.cos(m * pv)))
        ka.extend((k * torch.cos(m * pk), k * torch.sin(m * pk)))
    return torch.cat(va, -2).transpose(-1, -2) @ torch.cat(ka, -2) / v.shape[-2]


def compare(reference_fn, candidate_fn, inputs, generator):
    reference = reference_fn(*inputs)
    candidate = candidate_fn(*inputs)
    cotangent = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    grad_ref = torch.autograd.grad((reference * cotangent).sum(), inputs, retain_graph=True)
    grad = torch.autograd.grad((candidate * cotangent).sum(), inputs)
    output_error = (reference - candidate).abs().max().item()
    gradient_errors = [(a - b).abs().max().item() for a, b in zip(grad_ref, grad)]
    assert output_error < 1e-11, output_error
    assert max(gradient_errors) < 1e-10, gradient_errors
    return dict(output_max_abs_error=output_error, gradient_max_abs_errors=gradient_errors)


def verify():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(20261005)
    dtype = torch.float64
    shape = (2, 2, 7, 6)

    def randn(s):
        return torch.randn(s, dtype=dtype, generator=gen).requires_grad_()

    checks = {}
    v, k = randn(shape), randn(shape)
    pv, pk = randn(shape[:-2] + (shape[-1],)), randn(shape[:-2] + (shape[-1],))
    checks['state_shared_exact'] = compare(
        lambda v, k, pv, pk: direct_write(v, k, pv.unsqueeze(-2), pk.unsqueeze(-2)),
        shared_write, (v, k, pv, pk), gen)

    # Selected follow-up: phases are functions of the current puzzle state.
    v, k = randn(shape), randn(shape)
    hidden = randn((2, 7, 12))
    wk, wv = randn((12, 12)), randn((12, 12))
    tk, tv = randn((2, 6)), randn((2, 6))

    def reference(v, k, hidden, wk, wv, tk, tv):
        pk, pv = puzzle_state_phases(hidden, wk, wv, tk, tv)
        return direct_write(v, k, pv.unsqueeze(-2), pk.unsqueeze(-2))

    def candidate(v, k, hidden, wk, wv, tk, tv):
        pk, pv = puzzle_state_phases(hidden, wk, wv, tk, tv)
        return shared_write(v, k, pv, pk)

    checks['puzzle_state_generator_exact'] = compare(
        reference, candidate, (v, k, hidden, wk, wv, tk, tv), gen)
    wk, wv = (torch.zeros((12, 12), dtype=dtype, requires_grad=True) for _ in range(2))
    initial = candidate(v, k, hidden, wk, wv, tk, tv)
    fixed = shared_write(v, k, (math.pi / 2) * tv.tanh(), (math.pi / 2) * tk.tanh())
    assert torch.equal(initial, fixed)
    grad = torch.autograd.grad(initial.square().sum(), (wk, wv))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grad)
    checks['zero_initialized_phase_projection'] = dict(
        baseline_exact_equal=True, phase_projection_gradient_norms=[g.norm().item() for g in grad])

    # State-dependent prefix gaps give each token distinct delays, preserving a
    # common interleaved K/V order. This construction has nonzero local gradients.
    v, k, raw_gaps = randn(shape), randn(shape), randn(shape[:-1] + (13,))

    def ordered_phases(raw):
        gaps = torch.nn.functional.softplus(raw) + .1
        phase = -1.4 + 2.8 * gaps.cumsum(-1) / gaps.sum(-1, keepdim=True)
        return phase[..., 1:12:2], phase[..., :12:2]

    def reference(v, k, raw):
        pv, pk = ordered_phases(raw)
        return direct_write(v, k, pv, pk)

    def candidate(v, k, raw):
        pv, pk = ordered_phases(raw)
        dv = torch.arange(6) * 2 + 1
        dk = torch.arange(6) * 2
        return ordered_write(v, k, pv, pk, dv[:, None] > dk[None], dv[:, None] < dk[None])

    checks['local_order_preserving_exact'] = compare(reference, candidate, (v, k, raw_gaps), gen)

    # Low-rank phase projection has independent K and V output rows. Gradients
    # are checked through the actual phase generator, not only supplied phases.
    for modes in (1, 2, 4, 8):
        v, k, x = randn(shape), randn(shape), randn(shape[:-1] + (9,))
        u, w = randn((9, 3)), randn((3, 12))
        coeff = pulse_coefficients(modes).tolist()

        def local_phases(x, u, w):
            pv, pk = ((x @ u @ w) * .15).chunk(2, -1)
            return 1.4 * pv.tanh(), 1.4 * pk.tanh()

        def reference(v, k, x, u, w):
            pv, pk = local_phases(x, u, w)
            return direct_write(v, k, pv, pk, lambda d: harmonic_window(d, coeff))

        def candidate(v, k, x, u, w):
            pv, pk = local_phases(x, u, w)
            return packed_harmonic_write(v, k, pv, pk, coeff)

        checks[f'local_finite_pulse_{modes}_modes'] = compare(
            reference, candidate, (v, k, x, u, w), gen)
        grid = torch.linspace(1e-5, math.pi - 1e-5, 10001, dtype=dtype)
        values = harmonic_window(grid, coeff)
        assert values.min() >= -1e-14
        checks[f'local_finite_pulse_{modes}_modes'].update(
            coefficients=coeff, positive_halfcycle_grid_min=values.min().item(),
            peak_phase=grid[values.argmax()].item(),
            zero_slope=sum(m * c for m, c in enumerate(coeff, 1)),
            note='Exact for finite positive pulse + periodized STDP; not isolated-pair exp.')

    # A full-rank family refutes an exact fixed small separation rank for all
    # independent phases. General algorithms need not be feature factorizations.
    positions = torch.linspace(-1.4, 1.4, 104, dtype=dtype)
    kernel = exponential(positions[:, None] - positions[None])
    checks['independent_phase_rank_example'] = dict(
        channels=104, numerical_rank=int(torch.linalg.matrix_rank(kernel)),
        interpretation='No universally exact fixed small feature rank for this kernel.')
    pv, pk = randn(shape), randn(shape)
    shift = randn(shape[:-1] + (1,))
    checks['common_phase_shift_cancels'] = (
        exponential(pv[..., :, None] - pk[..., None, :])
        - exponential((pv + shift)[..., :, None] - (pk + shift)[..., None, :])
    ).abs().max().item()
    assert checks['common_phase_shift_cancels'] < 1e-12

    d, heads, inter, rank = 832, 8, 2304, 16
    baseline = 4 * d * d + 3 * d * inter + 2 * d * d // heads
    costs = {}
    for modes in (2, 4, 8):
        extra = (2 * modes - 1) * d * d // heads + 3 * d * rank
        costs[str(modes)] = dict(extra_matmul_macs_fraction=extra / baseline,
            phase_projection_rank=rank, write_contraction_multiplier=2 * modes)
    return dict(
        seed=20261005, dtype='float64', device='cpu', checks=checks,
        matrix_operation_estimate=dict(baseline_macs_per_token=baseline, harmonic=costs,
            excludes='trigonometry, memory traffic, activations, optimizer, checkpoint recompute, backward; not a GPU latency estimate'),
        sources=['https://arxiv.org/html/2603.15569v1#S3.SS2',
                 'https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py',
                 'lt/research_phase_stdp_math.py:pulse_coefficients'],
        selected_design=dict(
            phase_scope='per puzzle, head and channel; shared across tokens; recomputed each recurrent block',
            summary='affine-free RMSNorm(mean_tokens(hidden + input injection)), eps=1e-5, FP32',
            phase='(pi/2)*tanh(theta_role + W_role @ summary), independent K/V projections',
            initialization='zero W_K and W_V; preserve baseline learned-offset initialization',
            window='sign(phiV-phiK)*exp(-abs(phiV-phiK)/tau), tau=1 rad; no phase wrapping',
            write='G=(V.T @ RoPE(K)/N) elementwise L; current G only',
            read='RoPE(Q) @ G.T',
            residual_norms='RMSNorm after attention residual, then after FFN residual; FP32 mean square, eps=1e-5, no affine',
            extra_parameters_at_hidden832=2*832*832,
            limitations='mean pool is a summary, not a sufficient statistic; no independent per-token phase; no GPU latency or training claim',
            implementation_status='isolated operator prototype only; not installed in the training runner'),
        cuda_initialized=torch.cuda.is_initialized())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = verify()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
