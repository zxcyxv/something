"""GPU costs of exact signed-exp and finite-feature smooth-window operators.

Compilation is done while existing training runs. Only the timed block pauses
the explicitly supplied training PID, with both a finally block and an external
watchdog that resumes it on timeout or parent death. Never terminates training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

from .benchmark_dynamic_phase_stdp import fixed_gemm, split_g, make_args
from .research_free_phase_windows import (DEST, feature_write, two_feature_write, manual_feature_write, bf16_feature_write,
                                        loop_write, smooth_window, direct_write)
from .biexponential_phase_window import split_write as biexp_write, direct_write as biexp_direct


def process_identity(pid):
    # Comm may contain spaces, so split only after the final parenthesis.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


@contextmanager
def paused_training(pid, timeout=120):
    if pid is None:
        yield
        return
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'lt.research_kv_collapse' not in command:
        raise ValueError(f'PID {pid} is not the expected research training process')
    identity = process_identity(pid)
    before = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    if before in ('T', 't'):
        raise RuntimeError('Training was already stopped; refusing to take ownership of its pause')
    watchdog_code = '''import os,select,signal,sys
from pathlib import Path
pid,identity,timeout=int(sys.argv[1]),sys.argv[2],float(sys.argv[3])
ready=select.select([sys.stdin],[],[],timeout)[0]
message=sys.stdin.buffer.read(1) if ready else b''
if message != b'd':
 try:
  actual=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
  if actual == identity: os.kill(pid,signal.SIGCONT)
 except ProcessLookupError: pass
 except FileNotFoundError: pass
'''
    watchdog = subprocess.Popen([sys.executable, '-c', watchdog_code, str(pid), identity, str(timeout)],
                                stdin=subprocess.PIPE, start_new_session=True)
    try:
        os.kill(pid, signal.SIGSTOP)
        time.sleep(.3)
        print(f'PAUSED pid={pid}; automatic resume deadline={timeout}s', flush=True)
        yield
    finally:
        try:
            if process_identity(pid) == identity:
                os.kill(pid, signal.SIGCONT)
                print(f'RESUMED pid={pid}', flush=True)
        finally:
            try:
                watchdog.stdin.write(b'd'); watchdog.stdin.flush(); watchdog.stdin.close()
            except BrokenPipeError:
                pass
            watchdog.wait(timeout=5)


def make_function(name, frequencies=None, coefficients=None, epsilon=.35):
    if name == 'fixed_exp':
        return lambda q, k, v, pk, pv: fixed_gemm(q, k, v, pk[0, :, 0], pv[0, :, 0])
    if name == 'exact_dynamic_exp':
        return split_g
    if name == 'biexponential':
        return lambda q,k,v,pk,pv:q @ biexp_write(k,v,pk,pv).transpose(-1,-2)
    if name == 'direct_smooth':
        def fn(q, k, v, pk, pv):
            window = smooth_window(pv[..., :, None] - pk[..., None, :], epsilon)
            g = (v[..., :, None] * k[..., None, :] * window).mean(-3)
            return q @ g.transpose(-1, -2)
        return fn
    operator = (bf16_feature_write if name.startswith('bf16_') else loop_write if name.startswith('loop') else two_feature_write if name.startswith('two_')
                else manual_feature_write if name.startswith('manual_') else feature_write)
    def fn(q, k, v, pk, pv):
        g = operator(k, v, pk, pv, frequencies, coefficients)
        return q @ g.transpose(-1, -2)
    return fn


def timed(fn, args, repeats=15, rounds=5):
    def call():
        for arg in args:
            arg.grad = None
        output = fn(*args)
        output.square().mean().backward()
        return output
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    for arg in args:
        arg.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    samples = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            call()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    for arg in args:
        arg.grad = None
    return dict(forward_backward_median_ms=statistics.median(samples), rounds_ms=samples,
                extra_peak_mib=(peak - base) / 2**20, process_peak_mib=peak / 2**20)


def accuracy_check(fn, args, omega, coefficient, name=None):
    """Validate one independent batch/head at full T,D against direct FP64 CPU."""
    torch.manual_seed(981)
    cotangent = torch.randn_like(args[0])
    for arg in args:
        arg.grad = None
    actual = fn(*args)
    (actual * cotangent).sum().backward()
    selected = tuple(a[:1, :1].detach().cpu().double().requires_grad_() for a in args)
    q, k, v, pk, pv = selected
    if name == 'biexponential':
        direct = biexp_direct(k,v,pk,pv)
    elif name == 'exact_dynamic_exp':
        delta = pv[..., :, None]-pk[..., None, :]
        direct = (v[..., :, None]*k[..., None, :]*delta.sign()*(-delta.abs()).exp()).mean(-3)
    elif name == 'direct_smooth':
        direct = (v[..., :, None]*k[..., None, :]*smooth_window(pv[..., :, None]-pk[..., None, :],.35)).mean(-3)
    else:
        direct = direct_write(k, v, pk, pv, omega.detach().cpu().double(), coefficient.detach().cpu().double())
    expected = q @ direct.transpose(-1, -2)
    gradients = torch.autograd.grad((expected * cotangent[:1, :1].cpu().double()).sum(), selected)
    error = actual[:1, :1].detach().cpu().double() - expected.detach()
    result = dict(output_relative_l2=float(error.norm() / expected.detach().norm()),
                  output_max_abs_error=float(error.abs().max()), gradients={})
    for name, arg, reference in zip(('Q', 'K', 'V', 'phaseK', 'phaseV'), args, gradients):
        delta = arg.grad[:1, :1].cpu().double() - reference
        result['gradients'][name] = dict(relative_l2=float(delta.norm() / reference.norm()),
                                         max_abs_error=float(delta.abs().max()))
    assert result['output_relative_l2'] < .01, result
    assert max(x['relative_l2'] for x in result['gradients'].values()) < .01, result
    for arg in args:
        arg.grad = None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pause-pid', type=int)
    ap.add_argument('--precision', choices=('highest', 'high'), default='highest')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--tokens', type=int, default=81)
    ap.add_argument('--cases', default='fixed_exp,exact_dynamic_exp,manual_4,manual_8,bf16_4,bf16_8,biexponential')
    ap.add_argument('--out', type=Path)
    opt = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(opt.precision)
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    fits = json.loads((DEST / 'shape_study.json').read_text())['rows']
    args = make_args(opt.batch, 8, opt.tokens, 104, device='cuda', dtype=torch.float32, requires_grad=True)
    compiled, metadata = {}, {}
    output = opt.out or DEST / f'gpu_kernel_{opt.precision}_b{opt.batch}_t{opt.tokens}.json'
    report = dict(protocol=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, batch=opt.batch,
                               heads=8, tokens=opt.tokens, channels=104, precision=opt.precision,
                               allow_tf32=torch.backends.cuda.matmul.allow_tf32, compiled=True,
                               timing='CUDA events, forward + full backward, 5 rounds x 15 repeats',
                               scope='write + read only; excludes phase generator, projections, FFN, norms and optimizer',
                               existing_training_pause_pid=opt.pause_pid), rows=[])
    for name in opt.cases.split(','):
        omega = coefficient = None
        if name.startswith(('packed_', 'loop_', 'two_', 'manual_', 'bf16_')):
            modes = int(name.rsplit('_', 1)[1])
            row = next(r for r in fits if r['family'] == 'optimized_frequencies'
                       and r['epsilon'] == .35 and r['modes'] == modes)
            omega = torch.tensor(row['frequencies'], device='cuda')
            coefficient = torch.tensor(row['coefficients'], device='cuda')
        fn = torch.compile(make_function(name, omega, coefficient), fullgraph=True, dynamic=False)
        print('COMPILE', name, flush=True)
        for a in args:
            a.grad = None
        loss = fn(*args).square().mean()
        loss.backward(); torch.cuda.synchronize()
        del loss
        for a in args:
            a.grad = None
        compiled[name] = fn
        metadata[name] = dict(name=name)
        if omega is not None:
            metadata[name].update(modes=omega.numel(), frequencies=omega.tolist(), coefficients=coefficient.tolist(),
                                  correctness=accuracy_check(fn, args, omega, coefficient))
        elif name in ('biexponential','exact_dynamic_exp','direct_smooth'):
            metadata[name]['correctness'] = accuracy_check(fn,args,None,None,name)
            if name == 'biexponential':
                saved=args[4][...,0].detach().clone()
                with torch.no_grad():args[4][...,0].copy_(args[3][...,0])
                metadata[name]['with_exact_ties_correctness']=accuracy_check(fn,args,None,None,name)
                with torch.no_grad():args[4][...,0].copy_(saved)
        print('READY', name, flush=True)
    # No compilation or numerical verification takes place in the timed pause.
    with paused_training(opt.pause_pid, timeout=180):
        for name, fn in compiled.items():
            if opt.pause_pid:
                state = Path(f'/proc/{opt.pause_pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                if state not in ('T', 't'):
                    raise RuntimeError('Watchdog resumed training; rejecting contended timing')
            row = dict(metadata[name], **timed(fn, args))
            report['rows'].append(row)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('MEASURED', name, row['forward_backward_median_ms'], flush=True)
    report['existing_training_resumed'] = True if opt.pause_pid else None
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    main()
