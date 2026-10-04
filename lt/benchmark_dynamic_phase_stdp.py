"""Compare exact implementations of the original state-dependent STDP window.

This is an isolated write/read benchmark, not an architecture or trainer change.
All dynamic implementations compute the same unwrapped signed exponential with
zero contribution at exact ties. No timing jitter, periodization, or smoothing.
Phases are supplied as current-state outputs; their projection cost is excluded.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


def fixed_gemm(q, k, v, pk, pv):
    """Cost reference: the trained fixed-phase operator, with shared head phases."""
    delta = pv[:, :, None] - pk[:, None, :]
    window = delta.sign() * torch.exp(-delta.abs())
    g = (v.transpose(-1, -2) @ k) / k.shape[-2]
    return q @ (g * window[None]).transpose(-1, -2)


def direct_g(q, k, v, pk, pv):
    """Build G from the direct channel-pair exponential, then reuse it for Q."""
    delta = pv[..., :, None] - pk[..., None, :]
    window = delta.sign() * torch.exp(-delta.abs())
    g = (v[..., :, None] * k[..., None, :] * window).mean(-3)
    return q @ g.transpose(-1, -2)


def split_g(q, k, v, pk, pv):
    """Exact two-branch factorization; only O(TD) exponentials, shared G read."""
    vm, vp = v * torch.exp(-pv), v * torch.exp(pv)
    kp, km = k * torch.exp(pk), k * torch.exp(-pk)
    lo = pv[..., :, None] > pk[..., None, :]
    hi = pv[..., :, None] < pk[..., None, :]
    g = torch.where(lo, vm[..., :, None] * kp[..., None, :],
                    torch.where(hi, -vp[..., :, None] * km[..., None, :], 0.)).mean(-3)
    return q @ g.transpose(-1, -2)


def split_g_tiled(q, k, v, pk, pv, tile=16):
    """Eager channel tiling; forward workspace is O(T*tile*D), not O(TD²)."""
    vm, vp = v * torch.exp(-pv), v * torch.exp(pv)
    kp, km = k * torch.exp(pk), k * torch.exp(-pk)
    parts = []
    for start in range(0, v.shape[-1], tile):
        end = start + tile
        p = pv[..., start:end, None]
        lo, hi = p > pk[..., None, :], p < pk[..., None, :]
        part = torch.where(lo, vm[..., start:end, None] * kp[..., None, :],
                           torch.where(hi, -vp[..., start:end, None] * km[..., None, :], 0.))
        parts.append(part.mean(-3))
    g = torch.cat(parts, -2)
    return q @ g.transpose(-1, -2)


def scan_preparation(k, v, pk, pv):
    """Sort K channels once per token; queries share this preparation."""
    phase, order = pk.sort(-1)
    ks = k.gather(-1, order)
    left = torch.searchsorted(phase.contiguous(), pv.contiguous(), right=False)
    right = torch.searchsorted(phase.contiguous(), pv.contiguous(), right=True)
    return (ks * torch.exp(phase), ks * torch.exp(-phase),
            v * torch.exp(-pv), v * torch.exp(pv), order, left, right)


def scan_chunk(q, kp, km, vm, vp, order, left, right):
    """Compute a subset of Q rows using strict prefix/suffix sums."""
    b, h, t, d = kp.shape
    nq = q.shape[-2]
    qs = q[:, :, None].expand(b, h, t, nq, d).gather(
        -1, order[:, :, :, None].expand(b, h, t, nq, d))
    plus = (qs * kp[..., None, :]).cumsum(-1)
    minus = (qs * km[..., None, :]).flip(-1).cumsum(-1).flip(-1)
    zero = torch.zeros(b, h, t, nq, 1, device=q.device, dtype=q.dtype)
    plus = torch.cat((zero, plus), -1)
    minus = torch.cat((minus, zero), -1)
    a = plus.gather(-1, left[..., None, :].expand(b, h, t, nq, vm.shape[-1]))
    z = minus.gather(-1, right[..., None, :].expand(b, h, t, nq, vm.shape[-1]))
    return (vm[..., None, :] * a - vp[..., None, :] * z).mean(-3)


def make_scan(chunk_fn=scan_chunk, tile=32, checkpoint_chunks=True):
    def run(q, k, v, pk, pv):
        prepared = scan_preparation(k, v, pk, pv)
        outs = []
        for start in range(0, q.shape[-2], tile):
            qi = q[..., start:start + tile, :]
            if checkpoint_chunks and torch.is_grad_enabled():
                outs.append(checkpoint(chunk_fn, qi, *prepared, use_reentrant=False,
                                       preserve_rng_state=False))
            else:
                outs.append(chunk_fn(qi, *prepared))
        return torch.cat(outs, -2)
    return run


def make_args(batch, heads, tokens, dim, device='cpu', dtype=torch.float64,
              requires_grad=False, seed=20261004):
    gen = torch.Generator(device=device).manual_seed(seed + tokens)
    shape = (batch, heads, tokens, dim)
    vals = [torch.randn(shape, generator=gen, device=device, dtype=dtype) for _ in range(3)]
    vals += [(torch.rand(shape, generator=gen, device=device, dtype=dtype) - .5) * 2.6
             for _ in range(2)]
    return tuple(x.requires_grad_(requires_grad) for x in vals)


def verify():
    args = make_args(2, 2, 11, 17, requires_grad=True)
    reference = direct_g(*args)
    cot = torch.randn_like(reference)
    grad_ref = torch.autograd.grad((reference * cot).sum(), args)
    checks = {}
    candidates = {'split_g': split_g, 'split_g_tiled': split_g_tiled,
                  'scan_full': make_scan(tile=11, checkpoint_chunks=False),
                  'scan_tiled_checkpoint': make_scan(tile=4)}
    for name, fn in candidates.items():
        out = fn(*args)
        grad = torch.autograd.grad((out * cot).sum(), args)
        err = (out - reference).abs().max().item()
        ge = max((a - b).abs().max().item() for a, b in zip(grad, grad_ref))
        assert err < 1e-12 and ge < 1e-11, (name, err, ge)
        checks[name] = dict(output_max_abs_error=err, gradient_max_abs_error=ge)
    # Equal phases must be excluded in both branches, including duplicate K ties.
    tie = list(make_args(1, 1, 3, 4))
    tie[3] = torch.tensor([[[[-.2, 0., 0., .2]]]], dtype=torch.float64).expand(1, 1, 3, 4)
    tie[4] = tie[3].clone()
    for name, fn in candidates.items():
        e = (fn(*tie) - direct_g(*tie)).abs().max().item()
        assert e < 1e-12, (name, e)
        checks[name]['exact_tie_error'] = e
    # Dynamic implementations reduce exactly to the fixed experiment when phases
    # are shared over tokens and examples.
    fixed = make_args(2, 2, 11, 17)
    pk, pv = (p[0, :, 0].clone() for p in fixed[3:])
    dynamic = (*fixed[:3], pk[None, :, None].expand_as(fixed[3]),
               pv[None, :, None].expand_as(fixed[4]))
    e = (fixed_gemm(*fixed[:3], pk, pv) - direct_g(*dynamic)).abs().max().item()
    assert e < 1e-12
    checks['fixed_phase_reduction'] = dict(output_max_abs_error=e)
    return checks


def verify_cuda():
    """Verify actual compiled FP32 values/gradients at both target lengths."""
    implementations = {
        'direct_g_compiled': torch.compile(direct_g, fullgraph=True, dynamic=False),
        'split_g_compiled': torch.compile(split_g, fullgraph=True, dynamic=False),
        'scan_q_compiled_checkpoint': make_scan(
            torch.compile(scan_chunk, fullgraph=True, dynamic=False), tile=32),
    }
    checks = []
    for tokens in (81, 900):
        args = make_args(1, 1, tokens, 104, device='cuda', dtype=torch.float32,
                         requires_grad=True)
        out_ref = direct_g(*args)
        cot = torch.randn_like(out_ref)
        grads_ref = torch.autograd.grad((out_ref * cot).sum(), args)
        for name, fn in implementations.items():
            print(f'CUDA CHECK tokens={tokens} {name}', flush=True)
            out = fn(*args)
            grads = torch.autograd.grad((out * cot).sum(), args)
            torch.testing.assert_close(out, out_ref, atol=5e-5, rtol=3e-4)
            for g, ref in zip(grads, grads_ref):
                torch.testing.assert_close(g, ref, atol=5e-5, rtol=3e-4)
            checks.append(dict(tokens=tokens, implementation=name,
                output_max_abs_error=(out - out_ref).abs().max().item(),
                gradient_max_abs_errors={key: (g - ref).abs().max().item()
                    for key, g, ref in zip(('Q', 'K', 'V', 'K_phase', 'V_phase'),
                                           grads, grads_ref)}))
            del out, grads
        del args, out_ref, grads_ref
        gc.collect()
        torch.cuda.empty_cache()
    return checks


def benchmark(fn, args, reps, rounds, training):
    def run():
        for x in args:
            x.grad = None
        if training:
            y = fn(*args)
            y.square().mean().backward()
        else:
            with torch.no_grad():
                y = fn(*args)
        return y

    # Compilation and initial CUDA/autograd setup are excluded.
    y = run()
    del y
    y = run()
    del y
    torch.cuda.synchronize()
    for x in args:
        x.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            y = run()
            del y
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / reps)
    for x in args:
        x.grad = None
    return dict(median_ms=statistics.median(samples), rounds_ms=samples,
                base_allocated_mib=base / 2**20,
                peak_allocated_mib=peak / 2**20,
                extra_peak_mib=(peak - base) / 2**20)


def save(report, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dim', type=int, default=104)
    ap.add_argument('--lengths', default='81,900')
    ap.add_argument('--variants', default='fixed_gemm,direct_g,split_g,scan_q')
    ap.add_argument('--eager', action='store_true')
    ap.add_argument('--scan-tile', type=int, default=32)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--verify-cuda-only', action='store_true')
    ap.add_argument('--out', default='runs/dynamic_phase_benchmark_20261004/compiled_b8.json')
    opt = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    report = dict(protocol=dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, batch=opt.batch,
        heads=opt.heads, channels_per_head=opt.dim, dtype='float32', tf32=False,
        compilation='eager' if opt.eager else 'torch.compile(default, fullgraph=True)',
        phases='independent per token/channel; [-1.3,1.3] radians; tau=1; no wrapping',
        zero_lag='zero, with strict comparisons; no smoothing or firing jitter',
        scope='current G write + all-Q read; excludes phase/QKV projections, RoPE, FFN, norms, optimizer',
        baseline='fixed_gemm uses shared learned head/channel phases; cost reference only',
        training='forward + scalar mean-square loss + full backward, no gradient truncation',
        scan='K sort/search once per token; Q chunks with activation checkpointing in backward',
        scan_query_tile=opt.scan_tile, reps=opt.reps, rounds=opt.rounds,
        random_input_limit='algorithm performance and equivalence only, not model accuracy'),
        correctness=verify(), rows=[])
    dest = Path(opt.out)
    save(report, dest)
    print('CPU float64 value/gradient/tie checks passed', flush=True)
    if opt.verify_cuda_only:
        report['cuda_correctness'] = verify_cuda()
        save(report, dest)
        print(json.dumps(report['cuda_correctness'], indent=2), flush=True)
        print(f'SAVED {dest}', flush=True)
        return
    methods = {'fixed_gemm': fixed_gemm, 'direct_g': direct_g, 'split_g': split_g,
               'split_g_tiled': split_g_tiled}
    if not opt.eager:
        methods = {name: torch.compile(fn, fullgraph=True, dynamic=False)
                   for name, fn in methods.items()}
        compiled_chunk = torch.compile(scan_chunk, fullgraph=True, dynamic=False)
    else:
        compiled_chunk = scan_chunk
    methods['scan_q'] = make_scan(compiled_chunk, opt.scan_tile, True)
    for tokens in map(int, opt.lengths.split(',')):
        for name in opt.variants.split(','):
            gc.collect()
            torch.cuda.empty_cache()
            fn = methods[name]
            args = make_args(opt.batch, opt.heads, tokens, opt.dim, device='cuda',
                             dtype=torch.float32, requires_grad=True)
            if name == 'fixed_gemm':
                args = (*args[:3], *(x[0, :, 0].detach().clone().requires_grad_()
                                    for x in args[3:]))
            row = dict(tokens=tokens, implementation=name)
            for training in (False, True):
                mode = 'forward_backward' if training else 'forward'
                print(f'START batch={opt.batch} tokens={tokens} {name} {mode}', flush=True)
                start = time.monotonic()
                try:
                    row[mode] = benchmark(fn, args, opt.reps, opt.rounds, training)
                    print(f'DONE {name} {mode}: {row[mode]["median_ms"]:.3f} ms; '
                          f'extra peak {row[mode]["extra_peak_mib"]:.1f} MiB; '
                          f'setup+measure {time.monotonic()-start:.1f}s', flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    row[mode] = dict(error='cuda_out_of_memory', message=str(exc).splitlines()[0])
                    for x in args:
                        x.grad = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f'OOM {name} {mode}', flush=True)
                report['rows'] = [r for r in report['rows']
                                  if (r['tokens'], r['implementation']) != (tokens, name)] + [row]
                save(report, dest)
            del args
    print(f'SAVED {dest}', flush=True)


if __name__ == '__main__':
    main()
