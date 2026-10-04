"""Verify and benchmark unit phase direction with a Gaussian STDP envelope.

An isolated current-G write/read benchmark, not full-model training. Signed
Gaussian controls use the same window within the principal phase-difference
range. The original exponential/fixed-phase rows are different-rule cost
references; their times are not evidence about model quality.
"""
from __future__ import annotations

import argparse
import gc
import math
import time
import warnings
from pathlib import Path

import torch

from lt.benchmark_dynamic_phase_stdp import benchmark, fixed_gemm, make_args, save, split_g
from lt.benchmark_shared_order_stdp import edge_cases
from lt.unit_phase_stdp import (
    gaussian_lag_reference, gaussian_lag_triton, unit_inner_gaussian_reference,
    unit_phase_gaussian, unit_phase_gaussian_reference, unit_phase_precomputed,
)


def compare(fn, args, atol, rtol, tau=1., gain=1.):
    ref = gaussian_lag_reference(*args, tau=tau, gain=gain)
    gen = torch.Generator(device=args[0].device).manual_seed(1709)
    cot = torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=gen)
    expected = torch.autograd.grad((ref * cot).sum(), args)
    out = fn(*args, tau=tau, gain=gain)
    actual = torch.autograd.grad((out * cot).sum(), args)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return dict(output_max_abs_error=(out - ref).abs().max().item(),
                gradient_max_abs_errors={name: (a - b).abs().max().item()
                                         for name, a, b in zip(('Q', 'K', 'V', 'phiK', 'phiV'),
                                                               actual, expected)})


def cases(device, dtype):
    values = edge_cases(device, dtype)
    values['random'] = make_args(2, 2, 11, 17, device=device, dtype=dtype, requires_grad=True)
    for sign in (-1, 1):
        anchor = torch.tensor(1., device=device, dtype=dtype)
        target = torch.tensor(math.inf * sign, device=device, dtype=dtype)
        pv = torch.nextafter(anchor, target).reshape(1, 1, 1, 1)
        args = [torch.ones_like(pv) for _ in range(3)] + [anchor.reshape_as(pv).clone(), pv]
        values['adjacent_float_' + ('positive' if sign > 0 else 'negative')] = tuple(
            x.requires_grad_() for x in args)
    vals = list(make_args(1, 1, 7, 10, device=device, dtype=dtype))
    values['noncontiguous_inputs'] = tuple(x[..., ::2].requires_grad_() for x in vals)
    one = torch.ones((1, 1, 2, 1), device=device, dtype=dtype)
    mixed = (one.clone(), one.clone(), one.clone(), torch.zeros_like(one),
             torch.tensor([.1, -1.], device=device, dtype=dtype).reshape_as(one))
    values['normalize_before_observation_sum'] = tuple(x.requires_grad_() for x in mixed)
    return values


def verify_cpu():
    checks = []
    for label, args in cases('cpu', torch.float64).items():
        for name, fn in [('literal_unit_inner', unit_inner_gaussian_reference),
                         ('relative_angle', unit_phase_gaussian_reference)]:
            r = compare(fn, args, 2e-12, 2e-12)
            checks.append(dict(case=label, implementation=name, **r))
    args = make_args(1, 1, 7, 9, requires_grad=True)
    checks.append(dict(case='different_width_and_gain', **compare(
        unit_phase_gaussian_reference, args, 2e-12, 2e-12, tau=.7, gain=.37)))
    # A common carrier rotation adds the same angle to both phase vectors.
    gen = torch.Generator().manual_seed(98)
    shift = torch.rand((1, 1, 7, 1), generator=gen, dtype=torch.float64) * .8
    shifted = (*args[:3], args[3] + shift, args[4] + shift)
    invariant_error = (unit_phase_gaussian_reference(*args)
                       - unit_phase_gaussian_reference(*shifted)).abs().max().item()
    assert invariant_error < 2e-12
    return dict(value_gradient_checks=checks, common_carrier_max_abs_error=invariant_error)


def window_probe():
    lags = torch.tensor([-1.5, -1., -.1, -1e-7, 0., 1e-7, .1, 1., 1.5], dtype=torch.float64)
    shape = (len(lags), 1, 1, 1)
    one = torch.ones(shape, dtype=torch.float64)
    pv = lags.reshape(shape).requires_grad_()
    args = (one, one, one, torch.zeros_like(one), pv)
    value = unit_phase_gaussian_reference(*args)
    gradient = torch.autograd.grad(value.sum(), pv)[0]
    result = [dict(lag=x.item(), A=x.sin().sign().item(), gaussian=y.item(),
                   gaussian_phase_gradient=g.item(), exponential=x.sign().mul(x.abs().neg().exp()).item())
              for x, y, g in zip(lags, value.flatten(), gradient.flatten())]
    assert abs(result[3]['gaussian'] + 1.) < 2e-14
    assert abs(result[5]['gaussian'] - 1.) < 2e-14
    assert result[4]['gaussian'] == result[4]['gaussian_phase_gradient'] == 0.
    return result


def verify_cuda():
    checks = []
    methods = {'unit_phase_triton': unit_phase_gaussian, 'gaussian_lag_triton': gaussian_lag_triton,
               'unit_phase_precomputed': unit_phase_precomputed}
    for label, args in cases('cuda', torch.float32).items():
        for name, fn in methods.items():
            r = compare(fn, args, 5e-5, 4e-4)
            if label.startswith(('near_zero_', 'adjacent_float_')):
                r['read_value'] = fn(*args).item()
                assert abs(r['read_value']) > .99999
                assert (r['read_value'] > 0.) == label.endswith('positive')
            checks.append(dict(case=label, implementation=name, compilation='eager wrapper / Triton', **r))
    args = make_args(1, 1, 7, 9, device='cuda', dtype=torch.float32, requires_grad=True)
    checks.append(dict(case='different_width_and_gain', implementation='unit_phase_triton', **compare(
        unit_phase_gaussian, args, 5e-5, 4e-4, tau=.7, gain=.37)))
    for tokens in (81, 900):
        args = make_args(1, 1, tokens, 104, device='cuda', dtype=torch.float32, requires_grad=True)
        for name, fn in methods.items():
            print(f'CUDA CHECK tokens={tokens} {name}', flush=True)
            r = compare(torch.compile(fn, fullgraph=True, dynamic=False), args, 5e-5, 4e-4)
            checks.append(dict(case='random', tokens=tokens, implementation=name,
                               compilation='torch.compile(default, fullgraph=True)', **r))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lengths', default='81,900')
    ap.add_argument('--variants', default='fixed_gemm,split_g,unit_phase_triton,unit_phase_precomputed,gaussian_lag_triton')
    ap.add_argument('--checks-only', action='store_true')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', default='runs/unit_phase_gaussian_20261004/compiled_b8.json')
    opt = ap.parse_args()
    warnings.filterwarnings('ignore', message="Logical operators 'and' and 'or' are deprecated")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    report = dict(protocol=dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, batch=opt.batch,
        heads=8, channels_per_head=104, dtype='float32', tf32=False,
        compilation='torch.compile(default, fullgraph=True)',
        phases='independent per token/channel in [-1.3,1.3]; every difference within (-pi,pi)',
        rule='A=sign(Im(exp(i phiV) conj(exp(i phiK))))=sign(sin(delta)); L=A exp(-delta^2/tau)',
        tau=1., gain=1., zero_lag='zero at exact ties; one-sided limits +/-1; no epsilon/surrogate',
        derivatives='ordinary sign gradient zero; phase envelope derivative -2 delta L/tau',
        scope='current G write + all-Q read with full first-order gradients; '
              'excludes QKV/phase projection, RoPE, FFN, RMSNorm, optimizer, full-model recurrence',
        kernels='fused pairwise tiles and custom backward; no saved B,H,T,D,D tensor',
        precomputed='sin/cos prepared per neuron; near-cancellation reevaluation, not sign smoothing',
        matched_control='gaussian_lag_triton: identical kernels with direct lag signs',
        baseline='fixed_gemm/split_g use the original exponential window and are cost references, not quality controls',
        memory='PyTorch allocated peak; extra peak excludes supplied inputs',
        repetitions=opt.reps, rounds=opt.rounds),
        cpu_correctness=verify_cpu(), window=window_probe(), rows=[])
    dest = Path(opt.out)
    save(report, dest)
    print('CPU float64 value/gradient/tie/carrier checks and window probe passed', flush=True)
    if opt.checks_only:
        report['cuda_correctness'] = verify_cuda()
        save(report, dest)
        print('CUDA FP32 eager/compiled value/gradient/adjacent-float checks passed', flush=True)
        print(f'SAVED {dest}', flush=True)
        return
    methods = {'fixed_gemm': fixed_gemm, 'split_g': split_g,
               'unit_phase_triton': unit_phase_gaussian, 'gaussian_lag_triton': gaussian_lag_triton,
               'unit_phase_precomputed': unit_phase_precomputed,
               'unit_phase_torch': unit_phase_gaussian_reference, 'gaussian_lag_torch': gaussian_lag_reference}
    methods = {name: torch.compile(fn, fullgraph=True, dynamic=False) for name, fn in methods.items()}
    for tokens in map(int, opt.lengths.split(',')):
        for name in opt.variants.split(','):
            gc.collect()
            torch.cuda.empty_cache()
            args = make_args(opt.batch, 8, tokens, 104, device='cuda', dtype=torch.float32,
                             requires_grad=True)
            if name == 'fixed_gemm':
                args = (*args[:3], *(x[0, :, 0].detach().clone().requires_grad_() for x in args[3:]))
            row = dict(tokens=tokens, implementation=name)
            for training in (False, True):
                mode = 'forward_backward' if training else 'forward'
                print(f'START batch={opt.batch} tokens={tokens} {name} {mode}', flush=True)
                start = time.monotonic()
                row[mode] = benchmark(methods[name], args, opt.reps, opt.rounds, training)
                print(f'DONE {name} {mode}: {row[mode]["median_ms"]:.3f} ms; '
                      f'extra peak {row[mode]["extra_peak_mib"]:.1f} MiB; '
                      f'setup+measure {time.monotonic()-start:.1f}s', flush=True)
                report['rows'] = [r for r in report['rows']
                                  if (r['tokens'], r['implementation']) != (tokens, name)] + [row]
                save(report, dest)
            del args
    print(f'SAVED {dest}', flush=True)


if __name__ == '__main__':
    main()
