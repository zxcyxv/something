"""Isolated October 5 research controls; no changes to historical variants.

All controls retain the existing projections, RoPE, optimizer and two
post-residual RMSNorms. The optional local head allows K/V channel ordering
to reverse as a function of the current token's activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from . import train as t
from .kv_stability import ExponentialPhaseCurrentReadInner, ORIGINAL_MODEL_ID
from .research_free_phase_windows import DEST, manual_feature_write, bf16_feature_write, two_feature_write, sine_window
from .biexponential_phase_window import window as biexp_window, split_write as biexp_write
from . import tanhsech_phase_window as tanhsech
from .pairangle_fused import pairangle_attention


class TransposedLinear(torch.nn.Module):
    """Output projection tied to the value projection: out(x) = x @ W_v."""
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        return torch.nn.functional.linear(x, self.source.weight.t())


def periodized_tanhsech(tau_phi, modes):
    """Sine series of the 2pi-periodized tanh*sech window on phase differences.

    L(t) = tanh(t/tau) sech(t/tau) = -tau d/dt sech(t/tau); its transform is
    -i pi tau^2 nu sech(pi tau nu / 2), so the periodization (common frequency,
    all spike pairings over cycles) is sum_r b_r sin(r delta) with
    b_r = tau_phi^2 r sech(pi r tau_phi / 2), tau_phi = omega tau. Returns the
    coefficients and the scale that sets the window peak to 1.
    """
    r = torch.arange(1, modes + 1, dtype=torch.float64)
    b = tau_phi ** 2 * r / torch.cosh(math.pi * r * tau_phi / 2)
    d = torch.linspace(0, math.pi, 20001, dtype=torch.float64)
    peak = (b[None] * torch.sin(d[:, None] * r[None])).sum(-1).max()
    return b, float(1 / peak)


def model_class(window='fourier', dynamic=True, modes=8, epsilon=.35, generator='dense', feature_precision='float32',
                window_scale=1.0, tie_qk=False, tie_vo=False, qk_l2=False, write_sum=False, tau_phi=2.0,
                phase_floor=0.5, v_norm='none', tie_all=False, phase_kappa=1.0, phase_omega=0.0, phase_frame='rotated', dc_hebbian=False, dc_alpha_init=0.0, boundary_ffn='bilinear', kernel='torch'):
    if boundary_ffn not in ('bilinear', 'swiglu'):
        raise ValueError(boundary_ffn)
    if kernel not in ('torch', 'triton'):
        raise ValueError(kernel)
    if kernel == 'triton' and not (window == 'pairangle' and qk_l2 and write_sum and v_norm == 'none'):
        raise ValueError('The fused kernel covers the pairangle window with QK L2, token-sum writes and raw V.')
    if v_norm not in ('none', 'unit', 'unit_gain'):
        raise ValueError(v_norm)
    if phase_frame not in ('rotated', 'unrotated'):
        raise ValueError(phase_frame)
    if phase_frame != 'rotated' and window != 'pairangle':
        raise ValueError('phase_frame applies only to the pairangle window')
    if dc_hebbian and window != 'pairangle':
        raise ValueError('dc_hebbian applies only to the pairangle window')
    if window == 'relaxphase':
        # Relaxed phase state per complex channel, advanced by its own activity across the
        # recurrence (fixed rule, no learned map). The window reads phase *history*, never
        # the current pattern. Carried in the key_trace slot; zero for fresh puzzles.
        dynamic = False
        if not (qk_l2 and write_sum):
            raise ValueError('The relaxed-phase window is defined on unit addresses with token-sum writes.')
    if window in ('complex', 'pairangle', 'softangle'):
        # Phases come from the address itself; no phase parameters.
        # 'complex' (2026-10-08, archived): real PxP G = sum |v||k| W(delta) replaced the
        # Hebbian product and lost the q/k phase match. 'pairangle' keeps the real
        # 104x104 Hebbian write and only sources the per-token phases from the pair angles.
        # 'softangle' replaces the unit phasor by z/sqrt(|z|^2 + (floor*rms)^2): a weak
        # channel has low phase coherence, so its window fades smoothly instead of
        # flipping on a noisy atan2; no atan2 anywhere.
        dynamic = False
        if not (qk_l2 and write_sum):
            raise ValueError('Address-angle phase windows are defined on unit Q/K addresses with token-sum writes.')
    pairangle_bf16 = window == 'pairangle' and feature_precision == 'bfloat16'
    def gemm(a, b):
        if pairangle_bf16 and a.dtype == torch.float32:
            return (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float()
        return a @ b
    class LocalFreePhaseInner(ExponentialPhaseCurrentReadInner):
        def __init__(self, config):
            if config.kv_write_reduction != 'mean':
                raise ValueError('This research protocol requires token-mean writes.')
            super().__init__(config)
            if window in ('complex', 'pairangle', 'softangle', 'relaxphase'):
                # theta_k_raw/theta_v_raw stay allocated (same draws as the other
                # controls) but receive no gradient: the phase is arg of the address.
                b, scale = periodized_tanhsech(tau_phi, modes)
                self.register_buffer('window_coefficients', (scale * b).to(torch.float32))
                self.register_buffer('window_frequencies', torch.arange(1, modes + 1, dtype=torch.float32))
                self.window_scale_factor = scale
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
            if v_norm == 'unit_gain':
                # Unit value per token/head times a learned per-head gain, initialised
                # to sqrt(dh) so the write keeps its initial scale (|v| ~ sqrt(dh) at init).
                for layer in self.layers:
                    layer.v_gain_raw = torch.nn.Parameter(torch.full((self.H,), float(t.inv_softplus(math.sqrt(self.dh)))))
            if dc_hebbian:
                # General STDP window c0 + alpha_h * W(delta) with c0 = 1: the DC Fourier term of
                # the periodized window is its net integral (the rate-Hebbian part), the sine
                # terms are the timing part. alpha_h per head, unbounded; zero init is B-only
                # (the phase path then gets no gradient), init 1 gives the window 1 + W.
                for layer in self.layers:
                    layer.stdp_alpha = torch.nn.Parameter(torch.full((self.H,), float(dc_alpha_init)))
            # Weight tying is applied after every original draw, so the remaining
            # parameters keep their original initial values.
            for layer in self.layers:
                if tie_all:
                    # K = V = Q = one shared projection (Ba-style auto-associative fast weight).
                    layer.k_proj = layer.q_proj
                    layer.v_proj = layer.q_proj
                if tie_qk:
                    layer.q_proj = layer.k_proj
                if tie_vo:
                    layer.out_proj = TransposedLinear(layer.v_proj)

        def boundary(self, layer, hidden):
            if boundary_ffn == 'bilinear':
                return super().boundary(layer, hidden)
            # SwiGLU control: same RMSNorm input and b_gate_up/b_down weights (same draws,
            # zero-initialized b_down); only the gate changes from g/2 to silu(g).
            hidden = self.phi(hidden)
            g, u = layer.b_gate_up(hidden).chunk(2, dim=-1)
            return hidden + layer.b_down(torch.nn.functional.silu(g) * u)

        def phases(self, layer, rotated_key=None, value=None, key=None):
            if window == 'pairangle':
                # Per-token phase of each RoPE pair (complex channel), shared by its two
                # real components. 'rotated': angle of the unit-L2 rotated address, which
                # carries theta_j*pos_n. 'unrotated': angle of the address before RoPE, so
                # the window is a per-pair scalar of content only and commutes with the
                # rotation; the read stays a function of pos_t - pos_n.
                def angle(x):
                    pair = x.reshape(*x.shape[:-1], self.dh // 2, 2)
                    return torch.atan2(pair[..., 1], pair[..., 0]).repeat_interleave(2, -1)
                return angle(key if phase_frame == 'unrotated' else rotated_key), angle(value)
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
            if window == 'biexponential':
                return biexp_window(delta)
            if window == 'tanhsech':
                return window_scale * tanhsech.window(delta)
            if window == 'hebbian':
                return torch.ones_like(delta)
            return sine_window(delta, self.window_frequencies, self.window_coefficients)

        def memory_step(self, layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
            if kernel == 'triton' and q.is_cuda and q.dtype != torch.float64:
                # Fused L2 norm + RoPE + pairangle write + read (lt/pairangle_fused.py); same
                # function as the torch path below, without materializing feature tensors.
                if self.config.rope_type != 'learned_2d':
                    raise ValueError('The fused kernel takes learned 2D RoPE angles.')
                with torch.autocast(device_type=q.device.type, enabled=False):
                    angles = (layer.theta[..., 0, None] * self.pos_u
                              + layer.theta[..., 1, None] * self.pos_w).transpose(-1, -2)
                    alpha = layer.stdp_alpha if dc_hebbian else torch.ones(self.H, device=q.device)
                    read, current = pairangle_attention(q, k, v, angles, alpha, self.window_coefficients,
                                                        self.config.eps, dc_hebbian, pairangle_bf16,
                                                        phase_frame == 'rotated')
                return read, current, None, None
            with torch.autocast(device_type=q.device.type, enabled=False):
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                q, k, v = (x.to(dtype) for x in (q, k, v))
                if qk_l2:
                    # v1.7 address denominator: per token/head, before spatial rotation.
                    q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + self.config.eps)
                    k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.config.eps)
                if v_norm != 'none':
                    # Hopfield-style fixed pattern norm on the value side (v1.7 denominator norm+eps).
                    v = v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + self.config.eps)
                    if v_norm == 'unit_gain':
                        v = v * torch.nn.functional.softplus(layer.v_gain_raw).to(dtype)[None, :, None, None]
                tables = self.rope_tables(layer)
                qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
                if window == 'relaxphase':
                    return self.relaxed_phase_step(layer, qr, kr, k, v, e_k, fresh)
                if window == 'complex':
                    current = self.complex_write(kr, v)
                    return self.complex_read(qr, current), current, None, None
                current = self.window_write(layer, kr, v, k)
                if write_sum:
                    # Every write path is a token mean; restore the token sum (x T=81).
                    current = current * k.shape[-2]
                return gemm(qr, current.transpose(-1, -2)), current, None, None

        def complex_write(self, kr, v):
            """G_ij = sum_n |v_ni| |k_nj| W(arg v_ni - arg k_nj), W = sum_r c_r sin(r delta).

            RoPE pairs are the complex channels (P = dh/2). sin(r delta) =
            Im[(v_hat)^r conj(k_hat)^r], so each harmonic is one complex GEMM:
            Im[A_r^T conj(B_r)] = Im(A_r)^T Re(B_r) - Re(A_r)^T Im(B_r) with
            A_r = v v_hat^(r-1), B_r = k k_hat^(r-1). Token sum, no 1/T.
            """
            pairs = lambda x: x.reshape(*x.shape[:-1], self.dh // 2, 2).unbind(-1)
            ax, ay = pairs(v)
            bx, by = pairs(kr)
            def unit(x, y):
                m = torch.sqrt(x * x + y * y)
                return x / (m + 1e-6), y / (m + 1e-6)
            ux, uy = unit(ax, ay)
            wx, wy = unit(bx, by)
            c = self.window_coefficients.to(ax.dtype)
            current = None
            for r in range(c.shape[0]):
                term = ay.transpose(-1, -2) @ bx - ax.transpose(-1, -2) @ by
                current = c[r] * term if current is None else current + c[r] * term
                if r + 1 < c.shape[0]:
                    ax, ay = ax * ux - ay * uy, ax * uy + ay * ux
                    bx, by = bx * wx - by * wy, bx * wy + by * wx
            return current

        def complex_read(self, qr, current):
            """Apply the real P x P coupling to Re(q) and Im(q) of each complex channel."""
            qc = qr.reshape(*qr.shape[:-1], self.dh // 2, 2)
            gt = current.transpose(-1, -2)
            read = torch.stack((qc[..., 0] @ gt, qc[..., 1] @ gt), -1)
            return read.reshape(*qr.shape[:-1], self.dh)

        def relaxed_phase_step(self, layer, qr, kr, k_unit, v, phase, fresh):
            """Window on the carried phase state, then advance it by this block's activity.

            Uses the plain-autograd feature write: the custom-backward operator
            (manual_feature_write) traced by dynamo evolved the phase differently
            from eager when its phase inputs came from the carried state (key_trace
            max difference 0.87 rad after one segment, torch 2.11). With plain ops
            the residual is BF16 rounding feeding the integrator (~1e-2 rad,
            gradient cosine 0.995), and the step compiles at full speed.

            phi_{k+1} = phi_k + omega + kappa * |z_j|  per complex channel j (pair magnitude of
            the unit address, 0..1); both real components of a pair share the phase. The
            write uses phi_k (history before this block), so block 0 of a fresh puzzle has
            zero window and the write switches on as history accumulates.
            """
            P = self.dh // 2
            magnitude = k_unit.reshape(*k_unit.shape[:-1], P, 2).norm(dim=-1)        # [B,H,T,P]
            if phase is None:
                phase = torch.zeros_like(magnitude)
            else:
                phase = phase.to(magnitude.dtype)
                if fresh is not None:
                    phase = torch.where(fresh.view(-1, 1, 1, 1), torch.zeros_like(phase), phase)
            # Distinct tensors for the two phase inputs: the custom autograd operator
            # receives K- and V-side phases separately (aliased inputs miscompile).
            pk = phase.repeat_interleave(2, -1)
            pv = phase.repeat_interleave(2, -1)
            current = two_feature_write(kr, v, pk, pv, self.window_frequencies, self.window_coefficients) * kr.shape[-2]
            read = qr @ current.transpose(-1, -2)
            advanced = phase + phase_omega + phase_kappa * magnitude
            return read, current, advanced, None

        def soft_phasor(self, x):
            """Unit phasor of each RoPE pair times a detached coherence gate.

            rho = |z| / sqrt(|z|^2 + (floor * rms_pairs|z|)^2) in [0, 1) is treated as
            a constant: it only decides how much a channel's timing is trusted and
            never pulls the magnitudes (an attached gate rewarded concentrating V
            energy into a few channels and blew the value scale up, 2026-10-08).
            Gradients flow through the direction; rho bounds them near |z| = 0.
            """
            pair = x.reshape(*x.shape[:-1], self.dh // 2, 2)
            sq = pair.square().sum(-1).detach()                          # gate input only
            c2 = phase_floor ** 2 * sq.mean(-1, keepdim=True)            # (floor * rms)^2 per token/head
            rho = torch.sqrt(sq / (sq + c2))
            # Direction through atan2 -> cos/sin. The algebraic unit vector
            # z * rsqrt(|z|^2 + eps) has a correct forward but a wrong inductor
            # backward under compile + activation checkpointing (torch 2.11,
            # 2026-10-08: compiled/eager gradient cosine ~0); atan2 compiles correctly.
            angle = torch.atan2(pair[..., 1], pair[..., 0])
            return angle.cos() * rho, angle.sin() * rho

        def soft_angle_write(self, kr, v):
            """G_ab = mean_n V_na K_nb sum_r c_r Im[(v~_i)^r conj(k~_j)^r], i=pair(a), j=pair(b).

            With unit phasors this is the pairangle window; the soft phasor scales
            harmonic r by (rho_i rho_j)^r, rho = |z|/sqrt(|z|^2+(floor*rms)^2).
            """
            ux, uy = self.soft_phasor(v)
            wx, wy = self.soft_phasor(kr)
            ax, ay, bx, by = ux, uy, wx, wy
            c = self.window_coefficients.to(v.dtype)
            rep2 = lambda z: z.repeat_interleave(2, -1)
            current = None
            for r in range(c.shape[0]):
                vs, vc = v * rep2(ay), v * rep2(ax)
                kc, ks = kr * rep2(bx), kr * rep2(by)
                term = vs.transpose(-1, -2) @ kc - vc.transpose(-1, -2) @ ks
                current = c[r] * term if current is None else current + c[r] * term
                if r + 1 < c.shape[0]:
                    ax, ay = ax * ux - ay * uy, ax * uy + ay * ux
                    bx, by = bx * wx - by * wy, bx * wy + by * wx
            return current / kr.shape[-2]

        def window_write(self, layer, kr, v, k=None):
                if window == 'hebbian':
                    # Plain current KV outer product: no window, phases unused.
                    return v.transpose(-1, -2) @ kr / kr.shape[-2]
                pk, pv = self.phases(layer, kr, v, k)
                if window == 'pairangle':
                    # G_ab = mean_n V_na K_nb W(phi_pair(a),n - phi_pair(b),n), W = sum_r c_r sin(r d).
                    # feature_precision='bfloat16': BF16 GEMM operands (Hebbian, timing, read),
                    # FP32 outputs; norms, atan2 and sin/cos stay FP32.
                    write = bf16_feature_write if pairangle_bf16 else manual_feature_write
                    timing = write(kr, v, pk, pv, self.window_frequencies, self.window_coefficients)
                    if dc_hebbian:
                        alpha = layer.stdp_alpha.to(timing.dtype)[None, :, None, None]
                        return gemm(v.transpose(-1, -2), kr) / kr.shape[-2] + alpha * timing
                    return timing
                if window == 'softangle':
                    return self.soft_angle_write(kr, v)
                if not dynamic:
                    delta = pv[:, :, None] - pk[:, None, :]
                    current = (v.transpose(-1, -2) @ kr) * self.window(delta)[None] / kr.shape[-2]
                elif window == 'exponential':
                    vm, vp = v * (-pv).exp(), v * pv.exp()
                    kp, km = kr * pk.exp(), kr * (-pk).exp()
                    left, right = pv[..., :, None] > pk[..., None, :], pv[..., :, None] < pk[..., None, :]
                    current = torch.where(left, vm[..., :, None] * kp[..., None, :],
                                          torch.where(right, -vp[..., :, None] * km[..., None, :], 0.)).mean(-3)
                elif window == 'biexponential':
                    current = biexp_write(kr,v,pk,pv)
                elif window == 'tanhsech':
                    current = window_scale * tanhsech.write(kr, v, pk, pv)
                else:
                    operator = bf16_feature_write if feature_precision == 'bfloat16' else manual_feature_write
                    current = operator(kr, v, pk, pv, self.window_frequencies, self.window_coefficients)
                return current
    return LocalFreePhaseInner


def exclude_phase_gain_from_decay():
    """Put the token-local phase gain/projection in the no-decay group."""
    if getattr(t._is_no_decay, 'phase_local_excluded', False):
        return
    original = t._is_no_decay
    def rule(name, p, *args, **kwargs):
        return 'phase_local' in name or original(name, p, *args, **kwargs)
    rule.phase_local_excluded = True
    t._is_no_decay = rule


def configuration():
    cfg = dict(t.CFG)
    config_path = Path(__file__).resolve().parents[1] / 'configs/free_phase_window_research.json'
    cfg.update(json.loads(config_path.read_text()))
    cfg.update(data_npz=str(Path('data/sudoku_lt_1k.npz').resolve()), run_selftests=False,
               num_processes=1, log_every=16, save_every_steps=0, keep_last=25,
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
    ap.add_argument('--window', choices=('fourier','exponential','biexponential','tanhsech','hebbian','complex','pairangle','softangle','relaxphase'), default='fourier')
    ap.add_argument('--tie-all', action='store_true', help='K = V = Q = one shared projection')
    ap.add_argument('--phase-kappa', type=float, default=1.0, help='relaxphase: phase advance per block per unit pair magnitude (rad)')
    ap.add_argument('--phase-omega', type=float, default=0.0, help='relaxphase: common phase advance per block (rad); cancels in same-block pairs')
    ap.add_argument('--phase-frame', choices=('rotated', 'unrotated'), default='rotated',
                    help='pairangle: K phase from the RoPE-rotated address (includes theta*pos_n) or the address before RoPE')
    ap.add_argument('--dc-hebbian', action='store_true',
                    help='pairangle: window 1 + alpha_h * W(delta), alpha_h learned per head (zero init = B-only)')
    ap.add_argument('--dc-alpha-init', type=float, default=0.0, help='dc-hebbian: initial alpha_h')
    ap.add_argument('--kernel', choices=('torch', 'triton'), default='torch',
                    help='pairangle: torch ops (default) or the fused Triton write+read kernel')
    ap.add_argument('--boundary-ffn', choices=('bilinear', 'swiglu'), default='bilinear',
                    help='FFN after the attention RMSNorm: h + D(g/2 * u) (default) or h + D(silu(g) * u)')
    ap.add_argument('--v-norm', choices=('none', 'unit', 'unit_gain'), default='none',
                    help='value normalisation per token/head: unit, or unit times a learned per-head gain (init sqrt(dh))')
    ap.add_argument('--phase-floor', type=float, default=0.5,
                    help='softangle only: phasor magnitude floor as a fraction of the per-token RMS pair magnitude')
    ap.add_argument('--tau-phi', type=float, default=2.0,
                    help='complex window only: window time constant in phase units (omega*tau); fixes the sine coefficients')
    ap.add_argument('--fixed', action='store_true')
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--epsilon', type=float, default=.35)
    ap.add_argument('--precision', choices=('highest','high'), default='highest')
    ap.add_argument('--generator', choices=('dense','diagonal'), default='dense')
    ap.add_argument('--feature-precision', choices=('float32','bfloat16'), default='float32')
    ap.add_argument('--window-scale', type=float, default=1.0,
                    help='constant multiplying the tanhsech window; 2 gives peak 1 and slope 2 at the origin')
    ap.add_argument('--tie-qk', action='store_true', help='W_q = W_k (one shared projection)')
    ap.add_argument('--tie-vo', action='store_true', help='W_o = W_v^T (output tied to value projection)')
    ap.add_argument('--qk-l2', action='store_true', help='L2-normalize Q/K per token/head before RoPE (v1.7 denominator norm+eps)')
    ap.add_argument('--write-sum', action='store_true', help='token-sum write instead of token mean (no 1/T)')
    ap.add_argument('--phase-gain-no-decay', action='store_true',
                    help='exclude phase_local parameters from weight decay')
    ap.add_argument('--steps', type=int, default=3008, help='absolute stopping step; 0 runs the full epoch schedule')
    ap.add_argument('--save-every', type=int, default=0, help='periodic checkpoint interval in steps; 0 saves only at eval boundaries')
    ap.add_argument('--no-boundary-save', action='store_true', help='save only every --save-every steps and at stop')
    ap.add_argument('--no-activation-checkpoint', action='store_true',
                    help='keep block activations instead of recomputing them (same values, more memory)')
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
    if opt.window_scale != 1 and opt.window != 'tanhsech':
        raise ValueError('--window-scale applies only to the tanhsech window')
    name = f"{'fixed' if opt.fixed else 'local_free'}_{opt.window}_r{opt.modes}_e{opt.epsilon}_{opt.generator}_{opt.feature_precision}"
    if opt.window == 'complex':
        name = f'complex_phase_tau{opt.tau_phi:g}_r{opt.modes}'
    if opt.window == 'pairangle':
        name = f'pairangle_phase_tau{opt.tau_phi:g}_r{opt.modes}'
    if opt.window == 'softangle':
        name = f'softangle_phase_tau{opt.tau_phi:g}_r{opt.modes}_f{opt.phase_floor:g}'
    if opt.window == 'relaxphase':
        name = f'relaxphase_tau{opt.tau_phi:g}_r{opt.modes}_k{opt.phase_kappa:g}_w{opt.phase_omega:g}'
    if opt.window == 'hebbian' and opt.tie_all:
        name = 'hebbian'
    if opt.phase_frame != 'rotated':
        name += '_' + opt.phase_frame
    if opt.dc_hebbian:
        name += '_dc' + (f'a{opt.dc_alpha_init:g}' if opt.dc_alpha_init else '')
    if opt.boundary_ffn != 'bilinear':
        name += '_' + opt.boundary_ffn
    if opt.window == 'pairangle' and opt.feature_precision == 'bfloat16':
        name += '_bf16'
    if opt.tie_all:
        name += '_tieall'
    if opt.window_scale != 1:
        name += f'_x{opt.window_scale:g}'
    if opt.tie_qk:
        name += '_tieqk'
    if opt.tie_vo:
        name += '_tievo'
    if opt.v_norm != 'none':
        name += '_v' + opt.v_norm
    if opt.qk_l2:
        name += '_qkl2'
    if opt.write_sum:
        name += '_sum'
    if opt.phase_gain_no_decay:
        name += '_gainwd0'
        exclude_phase_gain_from_decay()
    cfg.update(phase_gain_no_decay=opt.phase_gain_no_decay)
    if opt.no_boundary_save:
        cfg['save_at_boundary'] = False
    if opt.no_activation_checkpoint:
        cfg['activation_checkpoint'] = False
    cfg.update(out_dir=str(out), max_steps=opt.steps or None, max_hours=None, save_every_steps=opt.save_every, research_variant=name,
               research_matmul_precision=opt.precision)
    t.KVSTDPInner = model_class(opt.window, not opt.fixed, opt.modes, opt.epsilon, opt.generator, opt.feature_precision,
                                opt.window_scale, opt.tie_qk, opt.tie_vo, opt.qk_l2, opt.write_sum, opt.tau_phi,
                                opt.phase_floor, opt.v_norm, opt.tie_all, opt.phase_kappa, opt.phase_omega, opt.phase_frame, opt.dc_hebbian, opt.dc_alpha_init, opt.boundary_ffn, opt.kernel)
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
                    scope=('Short controlled research experiment; max_steps applies only to this new diagnostic run' if opt.steps
                           else 'Full epoch schedule (max_steps=None)'),
                    checkpoints=(f'Periodic saves every {opt.save_every} steps plus' if opt.save_every else 'No periodic step saves;')
                                + ' trainer saves at evaluation boundaries and terminal; keep_last=25; all step logs retained',
                    original_training='local_warp_exp stopped by user at step 30550; not resumed')
    if opt.generator == 'diagonal':
        protocol.update(phase='phiK=(pi/2)*tanh(thetaK+aK*RoPE(K_n)); phiV=(pi/2)*tanh(thetaV+aV*V_n)',
                        phase_input='token-local own-channel activity, learned zero-initialized gain per head/channel/role',
                        limitation='Phase and activity projections are tied by channel gains; free ordering, but less phase expressivity than an independent dense projection')
    if opt.window == 'biexponential':
        protocol.update(window_note='C1 signed difference of exponentials, slow=1, fast=0.1; normalized peak=1. Exact mathematical window and ordinary derivative at zero; not the discontinuous original.',
                        phase_zero_derivative='C*(1/fast-1/slow), true derivative at zero; no surrogate')
    protocol.update(tie_qk=opt.tie_qk, tie_vo=opt.tie_vo, qk_l2=opt.qk_l2, write_sum=opt.write_sum, v_norm=opt.v_norm,
                    v_norm_note={'none': 'V unnormalized', 'unit': 'V = v/(||v||_2+eps) per token/head before the write',
                                 'unit_gain': 'V = softplus(g_h) * v/(||v||_2+eps), g_h per head, init sqrt(dh), no weight decay'}[opt.v_norm])
    if opt.window == 'complex':
        b, scale = periodized_tanhsech(opt.tau_phi, opt.modes)
        protocol.update(
            tau_phi=opt.tau_phi,
            phase='phiK_nj = arg of RoPE pair j of the unit-L2 K address of token n (includes theta_j*pos_n); phiV_ni = arg of pair i of V (unnormalized)',
            phase_input='the address direction itself; no phase parameters (theta_k_raw/theta_v_raw allocated but unused, no gradient)',
            phase_dynamic=True, ordering='free; phase on the circle (2pi-periodic)',
            write='G_ij = sum_n |v_ni| |k_nj| W(phiV_ni - phiK_nj), P=dh/2 complex channels, token sum (no 1/T)',
            window_note='2pi-periodized tanh*sech: W(d) = s * sum_r b_r sin(r d), b_r = tau_phi^2 r sech(pi r tau_phi/2); s sets peak 1. Exact sine series of the periodization (common-frequency all-pairings STDP), computed as R complex GEMMs.',
            window_coefficients=[float(x) for x in (scale * b)], window_scale_factor=scale,
            read='G (real PxP) applied to Re(q) and Im(q) of each complex channel of the unit-L2 RoPE Q; then out_proj',
            interpretation='phase-locked oscillators at a common frequency; omega enters only through tau_phi',
            baseline='2026-10-08 B-only + QK L2 + token sum (same seed/harness)')
    if opt.window == 'softangle':
        b, scale = periodized_tanhsech(opt.tau_phi, opt.modes)
        protocol.update(
            tau_phi=opt.tau_phi, phase_floor=opt.phase_floor,
            phase='soft phasor per RoPE pair: z~ = rho * z/|z|, rho = |z| / sqrt(|z|^2 + (floor * rms_pairs|z|)^2) DETACHED (coherence gate is a constant for the gradient); K = unit-L2 rotated address (includes theta*pos_n), V unnormalized; no atan2',
            coherence_gate='detached; attached version (first softangle run, stopped at ~1.4k steps) inflated V by concentrating energy into few channels',
            phase_input='the address direction itself; no phase parameters (theta_k_raw/theta_v_raw allocated but unused, no gradient)',
            phase_dynamic=True, ordering='free; phase on the circle (2pi-periodic)',
            write='G_ab = sum_n V_na K_nb sum_r c_r Im[(v~_i)^r conj(k~_j)^r]; equals the pairangle window times (rho_i rho_j)^r, rho=|z|/sqrt(|z|^2+(floor*rms)^2); token sum (no 1/T)',
            window_note='2pi-periodized tanh*sech: W(d) = s * sum_r b_r sin(r d), b_r = tau_phi^2 r sech(pi r tau_phi/2); s sets peak 1. Weak channels (|z| << floor*rms) have low phase coherence and their window fades smoothly to 0.',
            window_coefficients=[float(x) for x in (scale * b)], window_scale_factor=scale,
            read='RoPE(Q) @ G.T unchanged',
            interpretation='phase-locked oscillators at a common frequency with magnitude-dependent coherence; omega enters only through tau_phi',
            baseline='2026-10-08 pairangle (same except unit phasors via atan2) and B-only + QK L2 + token sum')
    protocol.update(tie_all=opt.tie_all)
    if opt.window == 'relaxphase':
        b, scale = periodized_tanhsech(opt.tau_phi, opt.modes)
        protocol.update(
            tau_phi=opt.tau_phi, phase_kappa=opt.phase_kappa, phase_omega=opt.phase_omega,
            phase='relaxed state per complex channel carried across blocks (key_trace slot): phi_{k+1} = phi_k + omega + kappa*|z_j|, |z_j| = pair magnitude of the unit address (0..1); phi=0 for fresh puzzles; both pair components share the phase; no learned phase map',
            phase_input='history of own activity only; the window at block k uses phi_k (before this block)',
            phase_dynamic=True, ordering='free; phase on the circle (2pi-periodic); pre = channel with the larger accumulated activity (leads)',
            write='G_ab = sum_n V_na K_nb W(phi_n,pair(a) - phi_n,pair(b)); token sum; G rebuilt every block (no accumulation; history lives in phi)',
            window_note='2pi-periodized tanh*sech: W(d) = s * sum_r b_r sin(r d), b_r = tau_phi^2 r sech(pi r tau_phi/2); s sets peak 1',
            window_coefficients=[float(x) for x in (scale * b)], window_scale_factor=scale,
            read='RoPE(Q) @ G.T unchanged',
            interpretation='STDP as time-lagged Hebbian on one population: timing is a dynamical observable (accumulated drive), not a function of the current pattern',
            baseline='same model with --window hebbian (window off)')
    if opt.window == 'pairangle':
        b, scale = periodized_tanhsech(opt.tau_phi, opt.modes)
        protocol.update(
            tau_phi=opt.tau_phi,
            phase='phi_n,pair(a) = atan2 of RoPE pair of the unit-L2 K address (includes theta*pos_n) for K components; atan2 of the V pair for V components; both components of a pair share the phase',
            phase_input='the address direction itself; no phase parameters (theta_k_raw/theta_v_raw allocated but unused, no gradient)',
            phase_dynamic=True, ordering='free; phase on the circle (2pi-periodic)',
            write='G_ab = sum_n V_na K_nb W(phiV_n,a - phiK_n,b), real 104x104 Hebbian product times the per-token window; token sum (no 1/T)',
            window_note='2pi-periodized tanh*sech: W(d) = s * sum_r b_r sin(r d), b_r = tau_phi^2 r sech(pi r tau_phi/2); s sets peak 1. Computed with the verified feature-write operator (R sin/cos feature GEMMs).',
            window_coefficients=[float(x) for x in (scale * b)], window_scale_factor=scale,
            read='RoPE(Q) @ G.T unchanged (q/k phase match kept inside the Hebbian product)',
            interpretation='phase-locked oscillators at a common frequency; omega enters only through tau_phi',
            baseline='2026-10-08 B-only + QK L2 + token sum (same seed/harness); 2026-10-06 tanhsech diagonal differs only in the phase source and window periodization')
        if opt.phase_frame == 'unrotated':
            protocol.update(
                phase='phi_n,pair(a) = atan2 of the pair of the unit-L2 K address BEFORE RoPE (no theta*pos_n) for K components; atan2 of the V pair for V components; both components of a pair share the phase',
                position_invariance='the window is a per-pair scalar of content only, so it commutes with the RoPE rotation; the read depends on positions only through pos_t - pos_n, as in B-only',
                baseline='2026-10-08 pairangle (identical except phase_frame=rotated) and B-only + QK L2 + token sum (same seed/harness)')
    protocol.update(phase_frame=opt.phase_frame, dc_hebbian=opt.dc_hebbian, dc_alpha_init=opt.dc_alpha_init, boundary_ffn=opt.boundary_ffn, kernel=opt.kernel)
    if opt.dc_hebbian:
        protocol.update(
            write='G_ab = sum_n V_na K_nb (1 + alpha_h W(phiV_n,a - phiK_n,b)) = G_hebbian + alpha_h G_pairangle; token sum (no 1/T)',
            window_general='general STDP window c0 + alpha_h W: c0 = net integral of the time window (rate-Hebbian term, Kempter et al. 1999), '
                           'W = odd timing part; c0 fixed to 1 (scale absorbed by V/out_proj), alpha_h learned per head, unbounded; init alpha_h = ' + f'{opt.dc_alpha_init:g}' + (' (step 0 equals B-only)' if opt.dc_alpha_init == 0 else ''),
            alpha_weight_decay='none (1-D parameter)',
            baseline='2026-10-09 pairangle unrotated (alpha -> window only) and 2026-10-08 B-only + QK L2 + token sum (alpha = 0)')
    if opt.qk_l2 or opt.write_sum:
        protocol.update(write=('G_ij=' + ('sum' if opt.write_sum else 'mean') + '_n V_ni K_nj L(phiV_ni-phiK_nj); read=RoPE(Q)@G.T'
                               + ('; Q,K = x/(||x||_2+eps) per token/head before RoPE, eps=config eps' if opt.qk_l2 else '')))
    protocol.update(phase_gain_weight_decay=0.0 if opt.phase_gain_no_decay else 'trainer default (1.0)')
    if opt.window == 'hebbian':
        protocol.update(window_note='No STDP window: G = mean_n V_n RoPE(K_n)^T. Phase parameters exist but receive no gradient.')
    if opt.window == 'tanhsech':
        protocol.update(window_note='Exact L(d)=tanh(d)sech(d): odd, slope 1 at 0, peak 1/2 at asinh(1), tail 2exp(-|d|). '
                                    'Fused Triton pair kernel (FP32, rcp.approx) on CUDA; direct pairs on CPU/FP64. No surrogate, no feature approximation.',
                        window_scale_factor=opt.window_scale,
                        window_scale=(f'L multiplied by {opt.window_scale:g}: peak {opt.window_scale/2:g}, slope {opt.window_scale:g} at 0; '
                                      'the sine-4 control peaks near 1.03 with slope 3.11.'))
    for module in (Path(__file__), Path(__file__).with_name('research_free_phase_windows.py'),
                   Path(__file__).with_name('biexponential_phase_window.py'),
                   Path(__file__).with_name('tanhsech_phase_window.py'),
                   Path(__file__).with_name('kv_stability.py'), Path(t.__file__)):
        content = module.read_bytes()
        (out / module.name).write_bytes(content)
        protocol[module.name+'_sha256'] = hashlib.sha256(content).hexdigest()
    fit_content=(DEST/'shape_study.json').read_bytes()
    (out/'shape_study.json').write_bytes(fit_content)
    protocol['shape_study_sha256']=hashlib.sha256(fit_content).hexdigest()
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
