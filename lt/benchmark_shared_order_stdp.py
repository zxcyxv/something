"""Verify and measure exact shared-order STDP against matched GPU controls.

This benchmarks only current G write and all-Q read, including sorting and the
custom backward. It does not run Sudoku/ARC training or change any architecture.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import torch

from lt.benchmark_dynamic_phase_stdp import benchmark, direct_g, fixed_gemm, make_args, save, split_g
from lt.shared_order_stdp import direct_triton, shared_order_reference, shared_order_triton


def compare(fn, args, atol, rtol):
    reference = direct_g(*args)
    gen = torch.Generator(device=args[0].device).manual_seed(503)
    cot = torch.randn(reference.shape, generator=gen, device=reference.device, dtype=reference.dtype)
    grad_ref = torch.autograd.grad((reference * cot).sum(), args)
    output = fn(*args)
    gradients = torch.autograd.grad((output * cot).sum(), args)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
    for actual, expected in zip(gradients, grad_ref):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return dict(output_max_abs_error=(output - reference).abs().max().item(),
                gradient_max_abs_errors={name: (a - b).abs().max().item()
                                         for name, a, b in zip(('Q', 'K', 'V', 'phiK', 'phiV'),
                                                               gradients, grad_ref)})


def edge_cases(device, dtype):
    cases = {}
    vals = list(make_args(1, 1, 3, 4, device=device, dtype=dtype))
    tie_phase = torch.tensor([-.3, 0., 0., .3], device=device, dtype=dtype)
    vals[3] = tie_phase.expand(1, 1, 3, 4).clone()
    vals[4] = tie_phase.flip(-1).expand(1, 1, 3, 4).clone()
    cases['repeated_exact_ties'] = tuple(x.requires_grad_() for x in vals)
    vals = list(make_args(1, 1, 3, 4, device=device, dtype=dtype))
    vals[3], vals[4] = torch.zeros_like(vals[3]), torch.zeros_like(vals[4])
    cases['all_phases_equal'] = tuple(x.requires_grad_() for x in vals)
    for label, lag in [('positive', 1e-7), ('negative', -1e-7)]:
        vals = [torch.ones((1, 1, 1, 1), device=device, dtype=dtype) for _ in range(3)]
        vals += [torch.zeros_like(vals[0]), torch.full_like(vals[0], lag)]
        cases['near_zero_' + label] = tuple(x.requires_grad_() for x in vals)
    vals = list(make_args(1, 1, 7, 5, device=device, dtype=dtype))
    vals[0] = vals[0][..., :3, :].clone()
    cases['queries_different_from_sources'] = tuple(x.requires_grad_() for x in vals)
    vals = list(make_args(2, 2, 11, 17, device=device, dtype=dtype))
    vals[3] = vals[3][0, :, 0][None, :, None].expand_as(vals[3]).clone()
    vals[4] = vals[4][0, :, 0][None, :, None].expand_as(vals[4]).clone()
    cases['phases_shared_across_tokens'] = tuple(x.requires_grad_() for x in vals)
    return cases


def verify_cpu():
    checks = {}
    cases = edge_cases('cpu', torch.float64)
    cases['random'] = make_args(2, 2, 11, 17, requires_grad=True)
    for name, args in cases.items():
        checks[name] = compare(shared_order_reference, args, 2e-12, 2e-12)
    for sign in ('positive', 'negative'):
        out = shared_order_reference(*cases['near_zero_' + sign]).item()
        assert abs(out) > .99999 and (out > 0) == (sign == 'positive')
        checks['near_zero_' + sign]['read_value'] = out
    assert shared_order_reference(*cases['all_phases_equal']).count_nonzero().item() == 0
    return checks


def verify_cuda():
    checks = []
    methods = {'shared_order_triton': shared_order_triton, 'direct_triton_control': direct_triton}
    for case, args in edge_cases('cuda', torch.float32).items():
        for name, fn in methods.items():
            check = compare(fn, args, 4e-5, 3e-4)
            check.update(case=case, implementation=name, compilation='eager wrapper / Triton kernels')
            if case.startswith('near_zero_'):
                value = fn(*args).item()
                assert abs(value) > .99999 and (value > 0) == case.endswith('positive')
                check['read_value'] = value
            checks.append(check)
    for tokens in (81, 900):
        args = make_args(1, 1, tokens, 104, device='cuda', dtype=torch.float32, requires_grad=True)
        for name, fn in methods.items():
            print(f'CUDA CHECK tokens={tokens} {name}', flush=True)
            check = compare(torch.compile(fn, fullgraph=True, dynamic=False), args, 4e-5, 3e-4)
            check.update(case='random', tokens=tokens, implementation=name,
                         compilation='torch.compile(default, fullgraph=True)')
            checks.append(check)
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lengths', default='81,900')
    ap.add_argument('--variants', default='fixed_gemm,split_g,shared_order_triton,direct_triton_control')
    ap.add_argument('--checks-only', action='store_true')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', default='runs/shared_order_stdp_20261004/compiled_b8.json')
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
        phases='independent per token/channel; [-1.3,1.3] radians; tau=1; no wrapping',
        rule='sign(phiV-phiK) exp(-abs(phiV-phiK)); exact ties zero; no smoothing',
        scope='current G write and all-Q read; includes sort/search/lookup and full backward; '
              'excludes QKV/phase projection, RoPE, FFN, norms, optimizer and full-model recurrence',
        ordering='K sort once per observation, V strict prefix/suffix locations, shared integer prefix table',
        kernels='dedicated forward and first-order backward; no saved token-by-channel-pair tensor',
        matched_control='direct_triton_control has identical tiles/loops/backward, direct phase comparisons',
        baseline='fixed_gemm is a fixed-phase cost reference, not an equivalent dynamic model',
        memory='PyTorch allocated peak; extra excludes supplied inputs',
        repeats=opt.reps, rounds=opt.rounds, random_input_limit='operator equivalence/cost, not learning quality'),
        cpu_correctness=verify_cpu(), rows=[])
    dest = Path(opt.out)
    save(report, dest)
    print('CPU float64 value/gradient/tie/near-zero checks passed', flush=True)
    if opt.checks_only:
        report['cuda_correctness'] = verify_cuda()
        save(report, dest)
        print('CUDA FP32 eager and compiled value/gradient checks passed', flush=True)
        print(f'SAVED {dest}', flush=True)
        return
    methods = {'fixed_gemm': fixed_gemm, 'split_g': split_g,
               'shared_order_triton': shared_order_triton, 'direct_triton_control': direct_triton}
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
