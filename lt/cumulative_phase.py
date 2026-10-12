"""Cumulative V-K phase offset on the pairangle STDP window (2026-10-12).

Every cell (and head) carries a relative phase offset lambda that integrates a bounded, data-dependent
angular velocity across blocks and segments, and the timing term of the general window reads the
content lag shifted by it:

    omega_{n,b}  = omega_max * tanh(W_h [k_hat_n ; v_n] + c_h)        W, c zero-initialized: omega = 0
    lambda_{n,b} = lambda_{n,b-1} + omega_{n,b}                         lambda = 0 for a fresh puzzle
    G_b          = sum_n v_n kr_n^T (1 + alpha_h sin(phiV_n,A - phiK_n,B + lambda_{n,b}))     (R = 1)

Time reading: a block is one cycle of the common carrier (T = 2 pi), the pair angles are latencies on it,
and omega is the per-cycle drift of the cell's V latency against its K latency. A rotation common to a
cell's V and K pairs (a change of the common carrier) cancels in phiV - phiK, so the relative offset is
the only phase integrated over blocks; |omega| < pi keeps a per-cycle shift unambiguous on the circle.
The Hebbian carrier v kr^T and the read stay the current block's. lambda (wrapped to [-pi, pi)) is stored
in the key_trace carry slot (detached at segment boundaries) and reset for fresh puzzles. The fused
kernel (lt/pairangle_fused.py, offset mode) computes the unit phasors itself and shifts the V phase by lambda.
With omega = 0 the model is the 2026-10-10 pairangle model (alpha = 1, R = 1, unrotated K phase).

offset_mode 'head': one offset per cell and head; 'pair': one per cell, head and V pair.

Training (same harness as lt/experiment_free_phase_windows.py; a new output directory each run):
    python -m lt.cumulative_phase --out runs/<name> --steps 50800 --save-every 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from . import train as t
from .experiment_free_phase_windows import configuration, model_class
from .kv_stability import ORIGINAL_MODEL_ID
from .pairangle_fused import pairangle_attention_offset


def offset_model_class(offset_mode='head', omega_max=math.pi, kernel='triton', feature_precision='bfloat16'):
    if offset_mode not in ('head', 'pair'):
        raise ValueError(offset_mode)
    if not 0 < omega_max <= math.pi:
        raise ValueError('omega_max must be in (0, pi] so that a per-block shift is unambiguous.')
    Base = model_class('pairangle', modes=1, feature_precision=feature_precision, qk_l2=True, write_sum=True,
                       tau_phi=2.0, phase_frame='unrotated', dc_hebbian=True, dc_alpha_init=1.0,
                       boundary_ffn='swiglu', kernel=kernel)
    bf16 = feature_precision == 'bfloat16'

    def gemm(a, b):
        if bf16 and a.dtype == torch.float32:
            return (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float()
        return a @ b

    class CumulativePhaseInner(Base):
        def __init__(self, config):
            super().__init__(config)
            # Added after every original draw (zeros: no RNG use), so the rest keeps its initial values.
            width = 1 if offset_mode == 'head' else self.dh // 2
            for layer in self.layers:
                layer.omega_proj = torch.nn.Parameter(torch.zeros(self.H, width, 2 * self.dh))
                layer.omega_bias = torch.nn.Parameter(torch.zeros(self.H, width))

        def unit(self, x):
            """[..., D] -> unit phasor of each pair, [..., P, 2] (cos 1, sin 0 at the origin)."""
            pair = x.reshape(*x.shape[:-1], self.dh // 2, 2)
            angle = torch.atan2(pair[..., 1], pair[..., 0])
            return torch.stack((angle.cos(), angle.sin()), -1)

        def phase_offset(self, layer, k, v, e_k, fresh):
            """omega from the head's own (unit key, value); returns (omega, lambda), both [B, H, T, W]."""
            # W [k_hat; v] without materializing k_hat or the concatenation: the key half is applied to the raw
            # key and divided by its norm afterwards; the products run in the projections' dtype.
            dtype = torch.float64 if k.dtype == torch.float64 else torch.float32
            w = layer.omega_proj
            wk = torch.einsum('bhtc,hwc->bhtw', k, w[..., :self.dh].to(k.dtype)).to(dtype)
            wv = torch.einsum('bhtc,hwc->bhtw', v, w[..., self.dh:].to(v.dtype)).to(dtype)
            norm = torch.linalg.vector_norm(k.to(dtype), dim=-1, keepdim=True) + self.config.eps
            omega = omega_max * torch.tanh(wk / norm + wv + layer.omega_bias.to(dtype)[None, :, None, :])
            if e_k is None:
                return omega, omega
            prev = e_k.to(dtype)
            if fresh is not None:
                prev = torch.where(fresh.view(-1, 1, 1, 1), torch.zeros_like(prev), prev)
            return omega, torch.remainder(prev + omega + math.pi, 2 * math.pi) - math.pi

        def pair_offset(self, lam):
            """[B, H, T, W] -> one offset per V pair, [B, H, T, P]."""
            return lam.expand(*lam.shape[:-1], self.dh // 2)

        def phasors(self, k, v, lam):
            """Timing phasors (torch path): K pairs as they are, V pairs shifted by lambda; both [..., D]."""
            pv = torch.atan2(v.reshape(*v.shape[:-1], self.dh // 2, 2)[..., 1],
                             v.reshape(*v.shape[:-1], self.dh // 2, 2)[..., 0]) + self.pair_offset(lam)
            zv = torch.stack((pv.cos(), pv.sin()), -1)
            return self.unit(k).reshape(k.shape), zv.reshape(v.shape)

        def memory_step(self, layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
            with torch.autocast(device_type=q.device.type, enabled=False):
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                _, lam = self.phase_offset(layer, k, v, e_k, fresh)
                kf, vf = k.to(dtype), v.to(dtype)
                if kernel == 'triton' and q.is_cuda and dtype != torch.float64:
                    angles = (layer.theta[..., 0, None] * self.pos_u
                              + layer.theta[..., 1, None] * self.pos_w).transpose(-1, -2)
                    read, current = pairangle_attention_offset(q, k, v, self.pair_offset(lam), angles,
                                                               layer.stdp_alpha, self.window_coefficients,
                                                               self.config.eps, True, bf16, False)
                    return read, current, lam, None
                zk, zv = self.phasors(kf, vf, lam)
                qf = q.to(dtype)
                qf = qf / (torch.linalg.vector_norm(qf, dim=-1, keepdim=True) + self.config.eps)
                kh = kf / (torch.linalg.vector_norm(kf, dim=-1, keepdim=True) + self.config.eps)
                tables = self.rope_tables(layer)
                qr, kr = (self.apply_rope(x, layer, tables) for x in (qf, kh))
                re = lambda z: z[..., 0::2].repeat_interleave(2, -1)
                im = lambda z: z[..., 1::2].repeat_interleave(2, -1)
                c = self.window_coefficients.to(dtype)[0]
                timing = c * (gemm((vf * im(zv)).transpose(-1, -2), kr * re(zk))
                              - gemm((vf * re(zv)).transpose(-1, -2), kr * im(zk)))
                alpha = layer.stdp_alpha.to(dtype)[None, :, None, None]
                current = gemm(vf.transpose(-1, -2), kr) + alpha * timing          # token sum
                return gemm(qr, current.transpose(-1, -2)), current, lam, None

    return CumulativePhaseInner


def make_model(protocol, cfg, device, state=None, seed=None):
    """Model from a run's protocol (analysis scripts); loads `state` strictly when given."""
    t.KVSTDPInner = offset_model_class(protocol['offset_mode'], protocol['omega_max'], protocol.get('kernel', 'triton'),
                                       protocol.get('feature_precision', 'bfloat16'))
    if seed is not None:
        torch.manual_seed(seed)
    model_cfg = dict(cfg, batch_size=cfg['global_batch_size'], seq_len=81, num_puzzle_identifiers=1)
    with torch.device(device):
        base = t.ACTLossHead(t.LT(model_cfg), q_weight=cfg['q_weight'])
    if state is not None:
        base.load_state_dict(state, strict=True)
    base.eval()
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--offset-mode', choices=('head', 'pair'), default='head')
    ap.add_argument('--omega-max', type=float, default=math.pi, help='bound on the per-block angular velocity')
    ap.add_argument('--kernel', choices=('triton', 'torch'), default='triton')
    ap.add_argument('--feature-precision', choices=('float32', 'bfloat16'), default='bfloat16')
    ap.add_argument('--steps', type=int, default=50800, help='absolute stopping step; 0 runs the full epoch schedule')
    ap.add_argument('--save-every', type=int, default=10000)
    ap.add_argument('--precision', choices=('highest', 'high'), default='highest')
    ap.add_argument('--preflight', action='store_true', help='a few isolated optimizer steps; reports time and memory')
    ap.add_argument('--out', type=Path, required=True)
    opt = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(opt.precision)
    out = opt.out.resolve()
    if (out / 'config.json').exists() and not opt.preflight:
        raise RuntimeError('Use a new output directory; this entry point never silently resumes.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    cfg.update(save_at_boundary=False, activation_checkpoint=False)
    name = (f'cumphase_{opt.offset_mode}_w{opt.omega_max:g}_pairangle_tau2_r1_unrotated_dca1_swiglu'
            f"{'_bf16' if opt.feature_precision == 'bfloat16' else ''}_qkl2_sum")
    cfg.update(out_dir=str(out), max_steps=opt.steps or None, max_hours=None, save_every_steps=opt.save_every,
               research_variant=name, research_matmul_precision=opt.precision)
    t.KVSTDPInner = offset_model_class(opt.offset_mode, opt.omega_max, opt.kernel, opt.feature_precision)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + name
    protocol = dict(
        variant=name, window='pairangle', modes=1, tau_phi=2.0, phase_frame='unrotated', dc_hebbian=True,
        dc_alpha_init=1.0, boundary_ffn='swiglu', qk_l2=True, write_sum=True, v_norm='none', kernel=opt.kernel,
        feature_precision=opt.feature_precision, precision=opt.precision, offset_mode=opt.offset_mode,
        omega_max=opt.omega_max, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        mechanism=('cumulative relative phase offset per cell and ' + ('head' if opt.offset_mode == 'head' else 'head and V pair')
                   + ': omega = omega_max tanh(W_h [k_hat; v] + c_h) (zero init), lambda_b = wrap(lambda_{b-1} + omega_b), '
                   'lambda = 0 for fresh puzzles, carried across blocks and segments (detached at boundaries); '
                   'fused offset kernel shifts the V pair phase by lambda; '
                   'timing term alpha_h sin(phiV - phiK + lambda); Hebbian carrier and read are the current block; '
                   'omega = 0 is the 2026-10-10 pairangle alpha=1 R=1 model'),
        weight_decay_note='omega_proj and omega_bias get the trainer default weight decay (pulls omega toward 0)',
        baseline='2026-10-10 R=1 8-block run pairangle_r1_dca1_swiglu_bf16fused_20261010 (eval log) and the '
                 '2026-10-12 phase-trace runs (ptrace0.2, ptraceL0.2)',
        scope='Short controlled research experiment' if opt.steps else 'Full epoch schedule (max_steps=None)')
    here = Path(__file__)
    for module in (here, here.with_name('experiment_free_phase_windows.py'), here.with_name('pairangle_fused.py'),
                   here.with_name('kv_stability.py'), Path(t.__file__)):
        content = module.read_bytes()
        (out / module.name).write_bytes(content)
        protocol[module.name + '_sha256'] = hashlib.sha256(content).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    if opt.preflight:
        from .experiment_free_phase_windows import preflight
        preflight(cfg, out, 12)
        return
    original_batch = t.train_batch
    began = time.monotonic()

    def batch(model, base, state, *args, **kwargs):
        start = time.monotonic()
        result = original_batch(model, base, state, *args, **kwargs)
        row = dict(step=state.step, seconds=time.monotonic() - start, elapsed=time.monotonic() - began,
                   segment=int(state.carry.steps.max()), **result)
        with (out / 'train.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        return result
    t.train_batch = batch
    print('RESEARCH', json.dumps(protocol), flush=True)
    t.main(cfg)
    path = t.find_latest_checkpoint(str(out))
    (out / 'finished.json').write_text(json.dumps(dict(checkpoint=path, stopped=t._STOP_REQUESTED,
                                                     elapsed=time.monotonic() - began)) + '\n')


if __name__ == '__main__':
    main()
