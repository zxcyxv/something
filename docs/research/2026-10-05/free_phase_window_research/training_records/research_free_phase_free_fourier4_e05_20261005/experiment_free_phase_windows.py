"""Isolated October 5 research controls; no changes to historical variants.

All controls retain the existing projections, RoPE, optimizer and two
post-residual RMSNorms. The optional local head allows K/V channel ordering
to reverse as a function of the current token's activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from . import train as t
from .kv_stability import ExponentialPhaseCurrentReadInner, ORIGINAL_MODEL_ID
from .research_free_phase_windows import DEST, manual_feature_write, bf16_feature_write, sine_window


def model_class(window='fourier', dynamic=True, modes=8, epsilon=.35, generator='dense', feature_precision='float32'):
    class LocalFreePhaseInner(ExponentialPhaseCurrentReadInner):
        def __init__(self, config):
            if config.kv_write_reduction != 'mean':
                raise ValueError('This research protocol requires token-mean writes.')
            super().__init__(config)
            if window == 'fourier':
                rows = json.loads((DEST / 'shape_study.json').read_text())['rows']
                fit = next(r for r in rows if r['family'] == 'optimized_frequencies'
                           and r['epsilon'] == epsilon and r['modes'] == modes)
                self.register_buffer('window_frequencies', torch.tensor(fit['frequencies'], dtype=torch.float32))
                self.register_buffer('window_coefficients', torch.tensor(fit['coefficients'], dtype=torch.float32))
            if dynamic:
                # Preserve baseline initialization, including every original random draw.
                for layer in self.layers:
                    if generator == 'dense':
                        layer.phase_local = torch.nn.Linear(2 * self.dh, 2 * self.dh, bias=False)
                        torch.nn.init.zeros_(layer.phase_local.weight)
                    elif generator == 'diagonal':
                        layer.phase_local_gain = torch.nn.Parameter(torch.zeros(self.H, 2 * self.dh))
                    else:
                        raise ValueError(generator)

        def phases(self, layer, rotated_key=None, value=None):
            if not dynamic:
                return super().phases(layer)
            dtype = torch.float64 if rotated_key.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=rotated_key.device.type, enabled=False):
                activity = torch.cat((rotated_key.to(dtype), value.to(dtype)), -1)
                correction = (layer.phase_local(activity) if generator == 'dense' else
                              activity * layer.phase_local_gain[None, :, None])
                dk, dv = correction.split(self.dh, -1)
                return tuple(self.phase_limit * (offset.to(dtype)[None, :, None] + change).tanh()
                             for offset, change in ((layer.theta_k_raw, dk), (layer.theta_v_raw, dv)))

        def window(self, delta):
            if window == 'exponential':
                return delta.sign() * (-delta.abs()).exp()
            return sine_window(delta, self.window_frequencies, self.window_coefficients)

        def memory_step(self, layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
            with torch.autocast(device_type=q.device.type, enabled=False):
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                q, k, v = (x.to(dtype) for x in (q, k, v))
                tables = self.rope_tables(layer)
                qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
                pk, pv = self.phases(layer, kr, v)
                if not dynamic:
                    delta = pv[:, :, None] - pk[:, None, :]
                    current = (v.transpose(-1, -2) @ kr) * self.window(delta)[None] / k.shape[-2]
                elif window == 'exponential':
                    vm, vp = v * (-pv).exp(), v * pv.exp()
                    kp, km = kr * pk.exp(), kr * (-pk).exp()
                    left, right = pv[..., :, None] > pk[..., None, :], pv[..., :, None] < pk[..., None, :]
                    current = torch.where(left, vm[..., :, None] * kp[..., None, :],
                                          torch.where(right, -vp[..., :, None] * km[..., None, :], 0.)).mean(-3)
                else:
                    operator = bf16_feature_write if feature_precision == 'bfloat16' else manual_feature_write
                    current = operator(kr, v, pk, pv, self.window_frequencies, self.window_coefficients)
                return qr @ current.transpose(-1, -2), current, None, None
    return LocalFreePhaseInner


def configuration():
    cfg = dict(t.CFG)
    cfg.update(json.loads(Path('configs/kv_phase_local_warp_exp_full_research.json').read_text()))
    cfg.update(data_npz=str(Path('data/sudoku_lt_1k.npz').resolve()), run_selftests=False,
               num_processes=1, log_every=16, save_every_steps=0, keep_last=3,
               milestone_every=0, init_from=None, resume_from=None, require_resume=False)
    return cfg


def preflight(cfg, out, steps):
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    torch.manual_seed(cfg['seed'])
    x, y, *_ = t.load_data(cfg)
    batch = next(t.eval_batches(x[:128], y[:128], 128, 0, 1))
    with torch.device(device):
        base = t.ACTLossHead(t.LT(dict(cfg, batch_size=128, seq_len=81, num_puzzle_identifiers=1)),
                             q_weight=cfg['q_weight'])
    base.train()
    opts, lrs = t.create_optimizers(base, cfg, 1)
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    compiled = torch.compile(base, dynamic=False)
    state = t.TrainState()
    report = dict(protocol='Isolated full optimizer steps; batch 128, hidden 832, 8 blocks, BF16 projections, FP32 memory, activation checkpoint',
                  steps=[], parameters=sum(p.numel() for p in base.parameters()), gradient_checks=[])
    old_check = t._check_finite_gradients
    def check(model, loss, device, ws):
        old_check(model, loss, device, ws)
        if state.step in (0, steps - 1):
            report['gradient_checks'].append({n:float(p.grad.norm()) for n,p in model.named_parameters()
                                              if ('phase_local' in n or 'theta_' in n) and p.grad is not None})
    t._check_finite_gradients = check
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(steps):
            started = time.monotonic()
            metrics = t.train_batch(compiled, base, state, batch, cfg, opts, lrs, 390625, 0, 1, device)
            torch.cuda.synchronize()
            row = dict(step=state.step, seconds=time.monotonic()-started, **metrics)
            report['steps'].append(row)
            print('PREFLIGHT', row, flush=True)
    finally:
        t._check_finite_gradients = old_check
    report.update(median_seconds_after_warmup=statistics.median(r['seconds'] for r in report['steps'][3:]),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  phase_local_weight_norms={n:float(p.detach().norm()) for n,p in base.named_parameters() if 'phase_local' in n})
    assert all(all(value > 0 for value in row.values()) for row in report['gradient_checks'])
    (out / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', choices=('fourier','exponential'), default='fourier')
    ap.add_argument('--fixed', action='store_true')
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--epsilon', type=float, default=.35)
    ap.add_argument('--precision', choices=('highest','high'), default='highest')
    ap.add_argument('--generator', choices=('dense','diagonal'), default='dense')
    ap.add_argument('--feature-precision', choices=('float32','bfloat16'), default='float32')
    ap.add_argument('--steps', type=int, default=3008)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    opt = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(opt.precision)
    out = opt.out.resolve()
    if (out / 'config.json').exists() and not opt.preflight:
        raise RuntimeError('Use a new output directory; this entry point never silently resumes.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    name = f"{'fixed' if opt.fixed else 'local_free'}_{opt.window}_r{opt.modes}_e{opt.epsilon}_{opt.generator}_{opt.feature_precision}"
    cfg.update(out_dir=str(out), max_steps=opt.steps, max_hours=None, research_variant=name,
               research_matmul_precision=opt.precision)
    t.KVSTDPInner = model_class(opt.window, not opt.fixed, opt.modes, opt.epsilon, opt.generator, opt.feature_precision)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + name
    protocol = dict(variant=name, window=opt.window, modes=opt.modes, epsilon=opt.epsilon, generator=opt.generator,
                    precision=opt.precision, feature_precision=opt.feature_precision, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    phase='(pi/2)*tanh(theta_role + zero_initialized_Linear(concat(RoPE(K_n),V_n))_role)',
                    phase_input='token-local, shared 2Dh -> 2Dh linear over heads/tokens; no token pooling',
                    phase_dynamic=not opt.fixed, ordering='free to reverse for dynamic controls',
                    phase_initialization='same original theta and other parameter draws; zero correction',
                    write='G_ij=mean_n V_ni K_nj L(phiV_ni-phiK_nj); read=RoPE(Q)@G.T',
                    normalization='RMSNorm immediately after each attention/FFN residual; FP32 mean square, eps1e-5, no affine; count=2',
                    window_note='Fourier is exact for its chosen bounded-domain surrogate; not the original exponential. Frozen frequencies and coefficients.',
                    scope='Short controlled research experiment; max_steps applies only to this new diagnostic run',
                    checkpoints='Only trainer terminal save; keep_last=3; all step logs retained',
                    original_training='local_warp_exp stopped by user at step 30550; not resumed')
    if opt.generator == 'diagonal':
        protocol.update(phase='phiK=(pi/2)*tanh(thetaK+aK*RoPE(K_n)); phiV=(pi/2)*tanh(thetaV+aV*V_n)',
                        phase_input='token-local own-channel activity, learned zero-initialized gain per head/channel/role',
                        limitation='Phase and activity projections are tied by channel gains; free ordering, but less phase expressivity than an independent dense projection')
    for module in (Path(__file__), Path(__file__).with_name('research_free_phase_windows.py')):
        content = module.read_bytes()
        (out / module.name).write_bytes(content)
        protocol[module.name+'_sha256'] = hashlib.sha256(content).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if opt.preflight:
        preflight(cfg, out, opt.steps)
        return
    original_batch = t.train_batch
    began = time.monotonic()
    def batch(model, base, state, *args, **kwargs):
        start = time.monotonic()
        result = original_batch(model, base, state, *args, **kwargs)
        row = dict(step=state.step, seconds=time.monotonic()-start, elapsed=time.monotonic()-began,
                   segment=int(state.carry.steps.max()), **result)
        with (out / 'train.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        return result
    t.train_batch = batch
    print('RESEARCH', json.dumps(protocol), flush=True)
    t.main(cfg)
    path = t.find_latest_checkpoint(str(out))
    (out / 'finished.json').write_text(json.dumps(dict(checkpoint=path, stopped=t._STOP_REQUESTED,
                                                     elapsed=time.monotonic()-began))+'\n')


if __name__ == '__main__':
    main()
