"""Cost of the fused exact tanh*sech window against the trained sine-4 operator.

Same scope and timing as benchmark_free_phase_windows: write + read
(read = RoPE(Q) @ G.T) with forward and full backward, torch.compile(fullgraph),
batch 128, 8 heads, T=81, D=104. Also sweeps the kernel tile shapes and checks
one batch/head of every case against direct FP64 pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import tanhsech_phase_window as ts
from .benchmark_dynamic_phase_stdp import make_args
from .benchmark_free_phase_windows import make_function, timed
from .research_free_phase_windows import DEST, direct_write as sine_direct


def tanhsech_fused(q, k, v, pk, pv):
    return q @ ts.fused_write(k, v, pk, pv).transpose(-1, -2)


def tanhsech_direct(q, k, v, pk, pv):
    return q @ ts.direct_write(k, v, pk, pv).transpose(-1, -2)


def check(fn, args, reference_write):
    torch.manual_seed(981)
    for a in args:
        a.grad = None
    actual = fn(*args)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    selected = tuple(a[:1, :1].detach().cpu().double().requires_grad_() for a in args)
    q, k, v, pk, pv = selected
    expected = q @ reference_write(k, v, pk, pv).transpose(-1, -2)
    reference = torch.autograd.grad((expected * cotangent[:1, :1].cpu().double()).sum(), selected)
    rel = lambda x, y: float((x - y).norm() / y.norm())
    result = dict(output_relative_l2=rel(actual[:1, :1].detach().cpu().double(), expected.detach()),
                  gradient_relative_l2={n: rel(a.grad[:1, :1].cpu().double(), r)
                                        for n, a, r in zip(('Q', 'K', 'V', 'phaseK', 'phaseV'), args, reference)})
    for a in args:
        a.grad = None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--out', type=Path, required=True)
    opt = ap.parse_args()
    torch.set_float32_matmul_precision('highest')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    args = make_args(opt.batch, 8, 81, 104, device='cuda', dtype=torch.float32, requires_grad=True)
    fit = next(r for r in json.loads((DEST / 'shape_study.json').read_text())['rows']
               if r['family'] == 'optimized_frequencies' and r['epsilon'] == .5 and r['modes'] == 4)
    omega = torch.tensor(fit['frequencies'], device='cuda')
    coefficient = torch.tensor(fit['coefficients'], device='cuda')
    sine_reference = lambda k, v, pk, pv: sine_direct(k, v, pk, pv, omega.cpu().double(), coefficient.cpu().double())
    report = dict(protocol=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, batch=opt.batch, heads=8,
                                tokens=81, channels=104, compiled='fullgraph', matmul_precision='highest',
                                timing='CUDA events, forward + full backward of write+read, 5 rounds x 15 repeats',
                                scope='excludes phase generator, projections, FFN, norms and optimizer'),
                  tile_sweep=[], rows=[])

    # Tile sweep for the fused kernel; the best shapes are kept for the comparison.
    best = None
    for fwd, bwd in (((16, 128, 1), (32, 32, 1)), ((16, 128, 1), (64, 16, 2)), ((16, 128, 2), (32, 32, 4))):
        ts.set_blocks(fwd, bwd)
        torch._dynamo.reset()
        row = dict(forward_block=list(fwd), backward_block=list(bwd),
                   **timed(torch.compile(tanhsech_fused, fullgraph=True), args))
        report['tile_sweep'].append(row)
        print('TILE', fwd, bwd, round(row['forward_backward_median_ms'], 3), flush=True)
        if best is None or row['forward_backward_median_ms'] < best['forward_backward_median_ms']:
            best = row
    ts.set_blocks(best['forward_block'], best['backward_block'])
    torch._dynamo.reset()

    cases = {
        'fixed_exp (baseline, shared phases)': (make_function('fixed_exp'), None),
        'sine4 eps0.5 FP32 (trained model)': (make_function('manual_4', omega, coefficient), sine_reference),
        'sine4 eps0.5 BF16 features': (make_function('bf16_4', omega, coefficient), sine_reference),
        'tanhsech direct pairs (torch.compile)': (tanhsech_direct, ts.direct_write),
        'tanhsech fused Triton (exact)': (tanhsech_fused, ts.direct_write),
    }
    for name, (fn, reference) in cases.items():
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        row = dict(name=name, **timed(compiled, args))
        if reference is not None:
            row['correctness_vs_fp64'] = check(compiled, args, reference)
        report['rows'].append(row)
        print('MEASURED', name, round(row['forward_backward_median_ms'], 3), 'ms',
              round(row['extra_peak_mib']), 'MiB', flush=True)
        opt.out.parent.mkdir(parents=True, exist_ok=True)
        opt.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    report['chosen_blocks'] = dict(forward=list(ts.BLOCK_FWD), backward=list(ts.BLOCK_BWD))
    opt.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('SAVED', opt.out)


if __name__ == '__main__':
    main()
