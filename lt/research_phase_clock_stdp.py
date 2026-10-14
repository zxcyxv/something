"""Phase-clock STDP: block-axis accumulated phases with phase-tagged eligibility traces.

Isolated research variant (2026-10-13); historical variants are unchanged.

Forward pass per block (one shared layer, same harness as the 2026-10-10 dca1 run):
  h <- h + sqrt(d) inj;  q, k, v = heads(W_q h), heads(W_k h), heads(W_v h)   (K != V)
  q, k <- x / (||x||_2 + eps) per token/head;  qr, kr = RoPE(q), RoPE(k)  (learned 2D);  v raw
  Clock (Mamba-3-style generator, accumulated along the recurrence, per cell and pair):
    dt  = softplus(W_dt h + b_dt)              [B, H, T]     one clock, shared by K and V
    w_K = pi tanh(W_fK h + b_fK)               [B, H, T, P]  angular velocity of the K neurons
    w_V = pi tanh(W_fV h + b_fV)               [B, H, T, P]  angular velocity of the V neurons
    phi_r <- wrap(phi_r + dt w_r)              the post-update phase tags this block's events
    lam_r = exp(-a_r dt),  a_r = softplus(raw_r)   [H, P] per role (K, V)
  Phase-tagged traces (complex, stored as real/imag halves; the K trace before RoPE):
    E_K <- lam_K E_K + (1 - lam_K) k e^{-i phi_K}             includes this block's event
    D_V  = lam_V E_V                                          past V only, decayed to now
    S    = sum_n [ (v e^{+i phi_V}) RoPE(E_K)^T + D_V (kr e^{-i phi_K})^T ]
    E_V <- D_V + (1 - lam_V) v e^{+i phi_V}
  Every (post, pre) event pair since the last reset is counted exactly once, with weight
    (1 - lam_earlier) * exp(-a * sum of dt over the elapsed blocks) * e^{i(phi_post - phi_pre)},
  so Im S is the odd (STDP) window and Re S the even one. Same-block pairs enter only via E_K.
  (1 - lam) is the zero-order-hold input gain of the decay over one block of length dt.
  Write and read (current increment only; history lives in the traces and phases):
    G = sum_n v kr^T + alpha_h Im S;   read = qr G^T
  h <- h + W_o read;  h <- RMS(h);  h <- h + D(silu(g) u), (g, u) = U h;  h <- RMS(h)

Initialization: W_dt = W_f = b_f = 0, so phi = 0 and Im S = 0 exactly; step 0 equals B-only.
alpha_h = alpha_init (default 1), so the frequency head still receives a gradient
(d sin / d phi = 1 at 0), while dt, the decay rates and alpha start with zero gradient.
dt = dt_init and lam = lambda_init at initialization. All added parameters are constant
initialized and created without random draws, after every original draw.

Carry layout (harness slots; detached at segment boundaries; zeroed for fresh puzzles):
  coupling    = cat(phi_K, phi_V)        [B, H, T, 2P]
  key_trace   = cat(Re E_K, Im E_K)      [B, H, T, 2D]   (before RoPE)
  value_trace = cat(Re E_V, Im E_V)      [B, H, T, 2D]

Run `python -m lt.research_phase_clock_stdp --selftest` for the CPU/FP64 checks.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from . import train as t
from .kv_stability import ExponentialPhaseCurrentReadInner, ORIGINAL_MODEL_ID


TWO_PI = 2 * math.pi
STATE_NAMES = ('current_hidden', 'coupling', 'key_trace', 'value_trace')
CLOCK_PARAMETERS = ('clock_freq.weight', 'clock_freq.bias', 'clock_dt.weight', 'clock_dt.bias',
                    'clock_lam_raw', 'stdp_alpha')


def wrap_phase(x):
    """Map to [-pi, pi); the gradient is 1 almost everywhere."""
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def _rep2(x):
    """[..., P] per pair -> [..., D]: both real components of a pair share the value."""
    return x.repeat_interleave(2, -1)


def advance_clock(clock, dt, freq_k, freq_v, rate_k, rate_v, state, omega_bar=None, psi_rate_k=None, psi_rate_v=None,
                  k_unit=None, v=None, bound=None, omega_t=None):
    """Advance the clock state by one block; return this block's event tags and retentions.

    dt [B,H,T,1]; freq_k, freq_v [B,H,T,P] (angular velocities of the K and V neurons); rate_k, rate_v
    (trace decay per unit time) broadcast to [B,H,T,P]. state is already zeroed for fresh lanes.
    clock 'integrate': state = (phase_k, phase_v); phase_r <- wrap(phase_r + dt freq_r) (2026-10-13 run).
    clock 'leaky':     state = (psi_k, psi_v, carrier [B,H,T,1]). The carrier is one clock per cell and head,
                       shared by K and V: carrier <- wrap(carrier + dt omega_bar). The neuron-specific
                       velocity is a deviation that integrates with leak, psi_r <- exp(-psi_rate_r dt) psi_r
                       + dt freq_r, so the relative latency psi_V - psi_K stays bounded and content-driven
                       instead of drifting over the circle. Tags: phase_r = wrap(carrier + psi_r).
    clock 'displacement': no generator. state = (phase_k, phase_v, zk_prev, zv_prev); the tags accumulate each
                       pair's own angular displacement wrap(theta(k) - theta(k-1)), theta = atan2 of the unit
                       key pair (before RoPE) / the raw value pair, given as k_unit, v [B,H,T,D]. dt is 1.
    clock 'bounded':   the displacement clock with block spacing and bounded latency (2026-10-14): state =
                       (phase_k, phase_v, zk_prev, zv_prev, count [B,H,T,1]); the tags are
                       bound * sin(phase_r) + count * omega_t, count = blocks run since the lane's puzzle started (0 in
                       the first block). Same-block pairs see only the bounded latency difference; a pair k blocks apart
                       gets +-k omega_t, so with omega_t > 2 bound and K omega_t + 2 bound < pi every pair up to K blocks
                       apart takes its LTP/LTD sign from the block order.
    Returns (phase_k, phase_v, lam_k, lam_v, new_state), tags and retentions [B,H,T,P].
    """
    lam_k, lam_v = torch.exp(-rate_k * dt), torch.exp(-rate_v * dt)
    if clock == 'integrate':
        phase_k, phase_v = state
        phase_k = wrap_phase(phase_k + dt * freq_k)
        phase_v = wrap_phase(phase_v + dt * freq_v)
        return phase_k, phase_v, lam_k, lam_v, (phase_k, phase_v)
    if clock in ('displacement', 'bounded'):
        # Parameter-free tags observed from the neurons themselves: each pair's own angular displacement
        # between consecutive blocks, accumulated since the puzzle start (a constant per-pair rotation of the
        # basis -- the gauge of the pair angle -- cancels in the displacement). The previous unit phasor is
        # carried; it is zero for a fresh lane, which makes the first displacement 0 (tags start at 0).
        phase_k, phase_v, zk_prev, zv_prev = state[:4]
        dk, zk = _pair_displacement(k_unit, zk_prev)
        dv, zv = _pair_displacement(v, zv_prev)
        phase_k = wrap_phase(phase_k + dk)
        phase_v = wrap_phase(phase_v + dv)
        if clock == 'displacement':
            return phase_k, phase_v, lam_k, lam_v, (phase_k, phase_v, zk, zv)
        count = state[4]
        tag_k = bound * torch.sin(phase_k) + count * omega_t
        tag_v = bound * torch.sin(phase_v) + count * omega_t
        return tag_k, tag_v, lam_k, lam_v, (phase_k, phase_v, zk, zv, count + 1)
    if clock != 'leaky':
        raise ValueError(clock)
    psi_k, psi_v, carrier = state
    carrier = wrap_phase(carrier + dt * omega_bar)
    psi_k = torch.exp(-psi_rate_k * dt) * psi_k + dt * freq_k
    psi_v = torch.exp(-psi_rate_v * dt) * psi_v + dt * freq_v
    return wrap_phase(carrier + psi_k), wrap_phase(carrier + psi_v), lam_k, lam_v, (psi_k, psi_v, carrier)


def _pair_displacement(x, z_prev):
    """Angular displacement of each pair of x [B,H,T,D] from the carried unit phasor z_prev [B,H,T,2P]
    (cos half, sin half; zero means no previous event -> displacement 0), and x's own unit phasor."""
    P = x.shape[-1] // 2
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c0, s0 = z_prev.split(P, -1)
    has = (c0 * c0 + s0 * s0) > 0.5
    cross, dot = c0 * x1 - s0 * x0, c0 * x0 + s0 * x1
    d = torch.atan2(torch.where(has, cross, torch.zeros_like(cross)), torch.where(has, dot, torch.ones_like(dot)))
    r2 = x0 * x0 + x1 * x1
    pos = r2 > 0
    inv = torch.where(pos, torch.rsqrt(torch.where(pos, r2, torch.ones_like(r2))), torch.zeros_like(r2))
    z = torch.cat((torch.where(pos, x0 * inv, torch.ones_like(inv)), x1 * inv), -1)
    return d, z


def phase_clock_step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
    """One block of the phase-clock write; all tensors share one float dtype.

    k_unit [B,H,T,D]: unit key before RoPE; kr = rope(k_unit); v [B,H,T,D]: raw value.
    rope: callable applying this layer's RoPE to [B,H,T,D].
    phase_k, phase_v [B,H,T,P]: the tags of this block's events (already advanced by advance_clock);
    lam_k, lam_v [B,H,T,P]: retentions over this block. traces = (ek_re, ek_im, ev_re, ev_im),
    already zeroed for fresh lanes. Returns (hebb, im_s, new_traces); hebb and im_s are token sums
    [B,H,D,D] with V rows and K columns.
    """
    ek_re, ek_im, ev_re, ev_im = traces
    ck, sk = _rep2(phase_k.cos()), _rep2(phase_k.sin())
    cv, sv = _rep2(phase_v.cos()), _rep2(phase_v.sin())
    lam_k, lam_v = _rep2(lam_k), _rep2(lam_v)
    # K trace, tag e^{-i phi_K}; it already holds this block's event, so same-block pairs
    # are counted here and only here.
    ek_re = lam_k * ek_re + (1 - lam_k) * k_unit * ck
    ek_im = lam_k * ek_im - (1 - lam_k) * k_unit * sk
    # V trace decayed to now, before this block's event.
    dv_re, dv_im = lam_v * ev_re, lam_v * ev_im
    ekr_re, ekr_im = rope(ek_re), rope(ek_im)
    vs, vc = (v * sv).transpose(-1, -2), (v * cv).transpose(-1, -2)
    # Im[(v e^{i phi_V})^T E_K] + Im[D_V^T (kr e^{-i phi_K})]
    im_s = (vs @ ekr_re + vc @ ekr_im
            + dv_im.transpose(-1, -2) @ (kr * ck) - dv_re.transpose(-1, -2) @ (kr * sk))
    hebb = v.transpose(-1, -2) @ kr
    ev_re = dv_re + (1 - lam_v) * v * cv
    ev_im = dv_im + (1 - lam_v) * v * sv
    return hebb, im_s, (ek_re, ek_im, ev_re, ev_im)


class ZeroLinear(torch.nn.Module):
    """Affine map with constant initialization and no random draw."""
    def __init__(self, n_in, n_out, bias_value=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n_out, n_in))
        self.bias = torch.nn.Parameter(torch.full((n_out,), float(bias_value)))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


def model_class(lambda_init=0.1, alpha_init=1.0, dt_init=1.0, boundary_ffn='swiglu', kernel='torch',
                feature_precision='float32', clock_precision='float32', clock='integrate', rho_init=0.5,
                bound=0.14, omega_t=0.34, freeze_lambda=False):
    """kernel 'torch': the block above in plain torch (any device, FP64 checks). 'triton': the fused kernel
    (lt/phase_clock_fused.py) on CUDA, with the phase advance and retentions in torch; feature_precision
    'bfloat16' rounds the kernel's dot operands to BF16 (FP32 accumulation, FP32 traces and phases).
    clock: see advance_clock. 'leaky' adds clock_omega_raw [H] (carrier velocity pi tanh, init 0) and
    clock_psi_lam_raw [2, H, P] (latency leak; rho = rho_init per block at dt = dt_init). 'bounded' uses the fixed
    constants bound and omega_t (no parameters). freeze_lambda keeps clock_lam_raw at its initial value (no gradient)."""
    if boundary_ffn not in ('bilinear', 'swiglu'):
        raise ValueError(boundary_ffn)
    if kernel not in ('torch', 'triton'):
        raise ValueError(kernel)
    if feature_precision not in ('float32', 'bfloat16'):
        raise ValueError(feature_precision)
    if clock_precision not in ('float32', 'bfloat16'):
        raise ValueError(clock_precision)
    if clock not in ('integrate', 'leaky', 'displacement', 'bounded'):
        raise ValueError(clock)
    if not 0 < lambda_init < 1:
        raise ValueError('lambda_init must lie in (0, 1).')
    if not 0 < rho_init < 1:
        raise ValueError('rho_init must lie in (0, 1).')
    if not dt_init > 0:
        raise ValueError('dt_init must be positive.')
    rate_init = -math.log(lambda_init) / dt_init
    psi_rate_init = -math.log(rho_init) / dt_init
    bf16 = feature_precision == 'bfloat16'

    class PhaseClockSTDPInner(ExponentialPhaseCurrentReadInner):
        def __init__(self, config):
            if config.kv_write_reduction != 'mean':
                raise ValueError('This research protocol keeps the token-mean config; the variant writes token sums itself.')
            if kernel == 'triton' and config.rope_type != 'learned_2d':
                raise ValueError('The fused kernel takes the learned 2D RoPE angles.')
            super().__init__(config)
            self.P = self.dh // 2
            # Added after every original draw, constant initialized: the shared parameters keep
            # the initial values of the pairangle runs at the same seed. theta_k_raw/theta_v_raw
            # and trace_lam_raw stay allocated and unused, as in those runs.
            for layer in self.layers:
                if clock not in ('displacement', 'bounded'):
                    layer.clock_dt = ZeroLinear(self.d, self.H, t.inv_softplus(dt_init))
                    layer.clock_freq = ZeroLinear(self.d, 2 * self.H * self.P)
                # 'lam_raw' puts the decay rates in the trainer's no-decay group.
                layer.clock_lam_raw = torch.nn.Parameter(
                    torch.full((2, self.H, self.P), float(t.inv_softplus(rate_init))), requires_grad=not freeze_lambda)
                layer.stdp_alpha = torch.nn.Parameter(torch.full((self.H,), float(alpha_init)))
                if clock == 'leaky':
                    layer.clock_omega_raw = torch.nn.Parameter(torch.zeros(self.H))
                    layer.clock_psi_lam_raw = torch.nn.Parameter(
                        torch.full((2, self.H, self.P), float(t.inv_softplus(psi_rate_init))))
            self.clock = clock
            self.clock_width = {'integrate': 2, 'leaky': 2, 'displacement': 6, 'bounded': 6}[clock] * self.P \
                + (1 if clock in ('leaky', 'bounded') else 0)

        def boundary(self, layer, hidden):
            if boundary_ffn == 'bilinear':
                return super().boundary(layer, hidden)
            # SwiGLU: same RMSNorm input and b_gate_up/b_down weights; gate silu(g) instead of g/2.
            hidden = self.phi(hidden)
            g, u = layer.b_gate_up(hidden).chunk(2, dim=-1)
            return hidden + layer.b_down(torch.nn.functional.silu(g) * u)

        def block(self, layer, h, inj, memory, e_k, e_v, fresh):
            # KVSTDPInner.block, except that the block input also feeds the clock.
            h = h + self.embed_scale * inj
            B, T, _ = h.shape

            def heads(x):
                return x.reshape(B, T, self.H, self.dh).transpose(1, 2)
            q = heads(layer.q_proj(h))
            if self.config.kv_projection_fp32:
                with torch.autocast(device_type=h.device.type, enabled=False):
                    projection_h = h if h.dtype == torch.float64 else h.float()
                    k, v = heads(layer.k_proj(projection_h)), heads(layer.v_proj(projection_h))
            else:
                k, v = heads(layer.k_proj(h)), heads(layer.v_proj(h))
            read, memory, e_k, e_v = self.memory_step(layer, q, k, v, memory, e_k, e_v, fresh, hidden=h)
            read = read.transpose(1, 2).reshape(B, T, self.d).to(h.dtype)
            h = h + layer.out_proj(read)
            h = self.phi(self.boundary(layer, h))
            return h, memory, e_k, e_v

        def clock_inputs(self, layer, hidden, dtype):
            """dt [B,H,T,1]; angular velocities and decay rates for the K and V neurons.

            clock_precision 'float32': the two affine maps run in FP32 (dtype) outside autocast.
            'bfloat16': they run on the block input as it is (BF16 under autocast, like the Q/K/V
            projections); softplus, tanh and everything after are FP32 either way."""
            b, n, _ = hidden.shape
            if clock_precision == 'bfloat16' and hidden.dtype != torch.float64:
                # memory_step runs with autocast disabled, so the cast is explicit (BF16 GEMM, FP32 result).
                h = hidden.to(torch.bfloat16)
                raw_dt, raw_freq = layer.clock_dt(h).to(dtype), layer.clock_freq(h).to(dtype)
            else:
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    h = hidden.to(dtype)
                    raw_dt, raw_freq = layer.clock_dt(h), layer.clock_freq(h)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                dt = torch.nn.functional.softplus(raw_dt).transpose(1, 2)[..., None]
                freq = math.pi * torch.tanh(raw_freq)
                freq = freq.reshape(b, n, 2, self.H, self.P).permute(2, 0, 3, 1, 4)        # [2,B,H,T,P]
                rate = torch.nn.functional.softplus(layer.clock_lam_raw.to(dtype))[:, None, :, None, :]
            return dt, freq[0], freq[1], rate[0], rate[1]

        def unpack_state(self, memory, e_k, e_v, fresh, like):
            """Clock state tuple and the four trace halves, zeroed for fresh lanes (see advance_clock)."""
            B, H, T, D = like.shape
            clk = like.new_zeros(B, H, T, self.clock_width) if memory is None else memory.to(like.dtype)
            ek = like.new_zeros(B, H, T, 2 * D) if e_k is None else e_k.to(like.dtype)
            ev = like.new_zeros(B, H, T, 2 * D) if e_v is None else e_v.to(like.dtype)
            if fresh is not None:
                mask = fresh.view(-1, 1, 1, 1)
                clk, ek, ev = (torch.where(mask, torch.zeros_like(x), x) for x in (clk, ek, ev))
            P = D // 2
            clk = tuple(clk.split(self.clock_split(P), -1))
            ek_re, ek_im = ek.split(D, -1)
            ev_re, ev_im = ev.split(D, -1)
            return clk, (ek_re, ek_im, ev_re, ev_im)

        def clock_split(self, P):
            return {'integrate': (P, P), 'leaky': (P, P, 1), 'displacement': (P, P, 2 * P, 2 * P),
                    'bounded': (P, P, 2 * P, 2 * P, 1)}[self.clock]

        def advance(self, layer, hidden, clk, dtype, k_unit=None, v=None):
            """This block's tags and retentions; returns them with the packed clock state.
            'displacement' and 'bounded' read the tags from k_unit (unit key before RoPE) and v; the others from the block input."""
            if self.clock in ('displacement', 'bounded'):
                B, H, T, D = k_unit.shape
                with torch.autocast(device_type=k_unit.device.type, enabled=False):
                    rate = torch.nn.functional.softplus(layer.clock_lam_raw.to(dtype))[:, None, :, None, :]
                    one = k_unit.new_ones(1, 1, 1, 1, dtype=dtype)
                    phase_k, phase_v, lam_k, lam_v, clk = advance_clock(self.clock, one, None, None, rate[0], rate[1], clk,
                                                                        k_unit=k_unit, v=v, bound=bound, omega_t=omega_t)
                    lam_k, lam_v = (x.expand(B, H, T, D // 2).contiguous() for x in (lam_k, lam_v))
                return phase_k, phase_v, lam_k, lam_v, torch.cat(clk, -1)
            dt, freq_k, freq_v, rate_k, rate_v = self.clock_inputs(layer, hidden, dtype)
            extra = {}
            if self.clock == 'leaky':
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    extra = dict(omega_bar=(math.pi * torch.tanh(layer.clock_omega_raw.to(dtype)))[None, :, None, None],
                                 psi_rate_k=torch.nn.functional.softplus(layer.clock_psi_lam_raw[0].to(dtype))[None, :, None, :],
                                 psi_rate_v=torch.nn.functional.softplus(layer.clock_psi_lam_raw[1].to(dtype))[None, :, None, :])
            phase_k, phase_v, lam_k, lam_v, clk = advance_clock(self.clock, dt, freq_k, freq_v, rate_k, rate_v, clk, **extra)
            return phase_k, phase_v, lam_k, lam_v, torch.cat(clk, -1)

        def memory_step(self, layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None, hidden=None):
            if hidden is None:
                raise RuntimeError('The phase clock reads the block input; call it through this class\'s block().')
            with torch.autocast(device_type=q.device.type, enabled=False):
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                if kernel == 'triton' and q.is_cuda and dtype != torch.float64:
                    return self.fused_memory_step(layer, q, k, v, memory, e_k, e_v, fresh, hidden, dtype)
                q, k, v = (x.to(dtype) for x in (q, k, v))
                q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + self.config.eps)
                k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.config.eps)
                tables = self.rope_tables(layer)

                def rope(x):
                    return self.apply_rope(x, layer, tables)
                qr, kr = rope(q), rope(k)
                clk, traces = self.unpack_state(memory, e_k, e_v, fresh, k)
                phase_k, phase_v, lam_k, lam_v, clk = self.advance(layer, hidden, clk, dtype, k_unit=k, v=v)
                hebb, im_s, (ek_re, ek_im, ev_re, ev_im) = phase_clock_step(k, kr, v, rope, phase_k, phase_v,
                                                                           lam_k, lam_v, traces)
                alpha = layer.stdp_alpha.to(dtype)[None, :, None, None]
                read = qr @ (hebb + alpha * im_s).transpose(-1, -2)
            return read, clk, torch.cat((ek_re, ek_im), -1), torch.cat((ev_re, ev_im), -1)

        def fused_memory_step(self, layer, q, k, v, memory, e_k, e_v, fresh, hidden, dtype):
            """Same block through lt/phase_clock_fused.py: the clock, retentions and fresh-lane reset stay
            in torch ([B, H, T, P] work); the kernel takes the tags and normalizes q/k itself."""
            from .phase_clock_fused import phase_clock_attention
            B, H, T, D = k.shape
            clk = k.new_zeros(B, H, T, self.clock_width, dtype=dtype) if memory is None else memory.to(dtype)
            ek = k.new_zeros(B, H, T, 2 * D, dtype=dtype) if e_k is None else e_k.to(dtype)
            ev = k.new_zeros(B, H, T, 2 * D, dtype=dtype) if e_v is None else e_v.to(dtype)
            if fresh is not None:
                mask = fresh.view(-1, 1, 1, 1)
                clk, ek, ev = (torch.where(mask, torch.zeros_like(x), x) for x in (clk, ek, ev))
            P = D // 2
            clk = tuple(clk.split(self.clock_split(P), -1))
            if self.clock in ('displacement', 'bounded'):
                kf = k.to(dtype)
                kf = kf / (torch.linalg.vector_norm(kf, dim=-1, keepdim=True) + self.config.eps)
                phase_k, phase_v, lam_k, lam_v, clk = self.advance(layer, hidden, clk, dtype, k_unit=kf, v=v.to(dtype))
            else:
                phase_k, phase_v, lam_k, lam_v, clk = self.advance(layer, hidden, clk, dtype)
            angles = (layer.theta[..., 0, None] * self.pos_u + layer.theta[..., 1, None] * self.pos_w).transpose(-1, -2)
            read, _, ek, ev = phase_clock_attention(q, k, v, ek, ev, phase_k, phase_v, lam_k, lam_v, angles,
                                                    layer.stdp_alpha, self.config.eps, bf16)
            return read, clk, ek, ev

    return PhaseClockSTDPInner


def make_model(protocol, cfg, device, state=None, seed=None, kernel=None):
    """Model from a run's protocol (analysis scripts); loads `state` strictly when given.
    kernel overrides the protocol's (e.g. 'torch' to hook phase_clock_step)."""
    t.KVSTDPInner = model_class(protocol['lambda_init'], protocol['alpha_init'], protocol['dt_init'],
                                protocol['boundary_ffn'], kernel or protocol.get('kernel', 'torch'),
                                protocol.get('feature_precision', 'float32'), protocol.get('clock_precision', 'float32'),
                                protocol.get('clock', 'integrate'), protocol.get('rho_init', 0.5),
                                protocol.get('bound', 0.14), protocol.get('omega_t', 0.34), protocol.get('freeze_lambda', False))
    if seed is not None:
        torch.manual_seed(seed)
    model_cfg = dict(cfg, batch_size=cfg['global_batch_size'], seq_len=81, num_puzzle_identifiers=1)
    with torch.device(device):
        base = t.ACTLossHead(t.LT(model_cfg), q_weight=cfg['q_weight'])
    if state is not None:
        base.load_state_dict(state, strict=True)
    base.eval()
    return base


def exclude_clock_from_decay():
    """Put the clock generator weights in the no-decay group (decay rates and alpha already are)."""
    if getattr(t._is_no_decay, 'clock_excluded', False):
        return
    original = t._is_no_decay

    def rule(name, p, *args, **kwargs):
        return '.clock_' in name or original(name, p, *args, **kwargs)
    rule.clock_excluded = True
    t._is_no_decay = rule


# ----------------------------------------------------------------------------- self-tests
def _check_step_against_pairs(generator):
    """The trace recursion equals the explicit complex sum over all (post, pre) event pairs."""
    g = generator
    B, H, T, P, blocks = 2, 2, 5, 3, 6
    D = 2 * P
    angles = torch.rand(1, H, T, P, generator=g) * TWO_PI
    cos, sin = angles.cos(), angles.sin()

    def rope(x):
        x0, x1 = x.reshape(*x.shape[:-1], P, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape(x.shape)
    rate_k = torch.rand(1, H, 1, P, generator=g) * 2 + 0.1
    rate_v = torch.rand(1, H, 1, P, generator=g) * 2 + 0.1
    seq = []
    for i in range(blocks):
        k = torch.randn(B, H, T, D, generator=g)
        k = k / k.norm(dim=-1, keepdim=True)
        v = torch.randn(B, H, T, D, generator=g)
        dt = torch.rand(B, H, T, 1, generator=g) * 1.8 + 0.2
        fk = (torch.rand(B, H, T, P, generator=g) * 2 - 1) * math.pi
        fv = (torch.rand(B, H, T, P, generator=g) * 2 - 1) * math.pi
        fresh = torch.tensor([True, True]) if i == 0 else torch.tensor([True, False]) if i == 3 else None
        seq.append((k, v, dt, fk, fv, fresh))

    clk = tuple(torch.zeros(B, H, T, P) for _ in range(2))
    traces = tuple(torch.zeros(B, H, T, D) for _ in range(4))
    got = []
    for k, v, dt, fk, fv, fresh in seq:
        if fresh is not None:
            mask = fresh.view(-1, 1, 1, 1)
            clk = tuple(torch.where(mask, torch.zeros_like(x), x) for x in clk)
            traces = tuple(torch.where(mask, torch.zeros_like(x), x) for x in traces)
        phase_k, phase_v, lam_k, lam_v, clk = advance_clock('integrate', dt, fk, fv, rate_k, rate_v, clk)
        hebb, im_s, traces = phase_clock_step(k, rope(k), v, rope, phase_k, phase_v, lam_k, lam_v, traces)
        got.append((hebb, im_s))

    start = torch.zeros(B, dtype=torch.long)
    pk, pv = torch.zeros(B, H, T, P), torch.zeros(B, H, T, P)
    phik, phiv, lamk, lamv, starts = [], [], [], [], []
    for i, (k, v, dt, fk, fv, fresh) in enumerate(seq):
        if fresh is not None:
            start = torch.where(fresh, torch.full_like(start, i), start)
            mask = fresh.view(-1, 1, 1, 1)
            pk, pv = torch.where(mask, torch.zeros_like(pk), pk), torch.where(mask, torch.zeros_like(pv), pv)
        pk, pv = pk + dt * fk, pv + dt * fv                      # unwrapped
        phik.append(pk)
        phiv.append(pv)
        lamk.append(_rep2(torch.exp(-rate_k * dt)))
        lamv.append(_rep2(torch.exp(-rate_v * dt)))
        starts.append(start.clone())
    for i in range(blocks):
        k_i, v_i = seq[i][0], seq[i][1]
        post_now = v_i * torch.exp(1j * _rep2(phiv[i]))
        pre_now = rope(k_i) * torch.exp(-1j * _rep2(phik[i]))
        s = torch.zeros(B, H, D, D, dtype=torch.complex128)
        for j in range(i + 1):
            active = (j >= starts[i]).view(-1, 1, 1, 1).to(torch.float64)
            wk, wv = 1 - lamk[j], 1 - lamv[j]
            for r in range(j + 1, i + 1):
                wk, wv = wk * lamk[r], wv * lamv[r]
            pre_then = rope(seq[j][0]) * wk * torch.exp(-1j * _rep2(phik[j]))
            s = s + active * torch.einsum('bhna,bhnj->bhaj', post_now, pre_then)          # pre first or same block
            if j < i:
                post_then = seq[j][1] * wv * torch.exp(1j * _rep2(phiv[j]))
                s = s + active * torch.einsum('bhna,bhnj->bhaj', post_then, pre_now)      # post first
        hebb, im_s = got[i]
        torch.testing.assert_close(im_s, s.imag, rtol=1e-9, atol=1e-11, msg=f'Im S, block {i}')
        torch.testing.assert_close(hebb, torch.einsum('bhna,bhnj->bhaj', v_i, rope(k_i)),
                                   rtol=1e-12, atol=1e-12, msg=f'Hebbian, block {i}')


def _activate_clock(inner, generator):
    g = generator
    with torch.no_grad():
        for layer in inner.layers:
            if hasattr(layer, 'clock_freq'):
                layer.clock_freq.weight.normal_(0, 0.5, generator=g)
                layer.clock_freq.bias.normal_(0, 0.3, generator=g)
                layer.clock_dt.weight.normal_(0, 0.1, generator=g)
            layer.clock_lam_raw.add_(0.3 * torch.randn(layer.clock_lam_raw.shape, generator=g))
            layer.stdp_alpha.fill_(0.7)
            layer.b_down.weight.normal_(0, 0.03, generator=g)
            if hasattr(layer, 'clock_omega_raw'):
                layer.clock_omega_raw.normal_(0, 0.5, generator=g)
                layer.clock_psi_lam_raw.add_(0.3 * torch.randn(layer.clock_psi_lam_raw.shape, generator=g))


def _check_displacement_clock(generator):
    """The displacement clock equals the explicit sum of wrapped pair-angle differences since the lane's reset,
    is invariant to a constant per-pair rotation of the basis (the gauge), and gives finite gradients."""
    g = generator
    B, H, T, P, blocks = 2, 2, 4, 3, 7
    D = 2 * P
    rate = torch.ones(1, H, 1, P)
    ks = [torch.randn(B, H, T, D, generator=g, requires_grad=True) for _ in range(blocks)]
    vs = [torch.randn(B, H, T, D, generator=g, requires_grad=True) for _ in range(blocks)]
    gauge = torch.rand(1, H, 1, P, generator=g) * TWO_PI                      # constant rotation per pair
    def rotate(x, a):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x0 * a.cos() - x1 * a.sin(), x0 * a.sin() + x1 * a.cos()), -1).reshape(x.shape)
    def angle(x):
        return torch.atan2(x[..., 1::2], x[..., 0::2])
    for rotated in (False, True):
        clk = (torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), torch.zeros(B, H, T, D), torch.zeros(B, H, T, D))
        ref_k, ref_v, prev_k, prev_v = torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), None, None
        tags = []
        for i in range(blocks):
            k, v = (rotate(ks[i], gauge), vs[i]) if rotated else (ks[i], vs[i])
            if i == 3:
                mask = torch.tensor([True, False]).view(-1, 1, 1, 1)
                clk = tuple(torch.where(mask, torch.zeros_like(x), x) for x in clk)
                ref_k, ref_v = torch.where(mask, torch.zeros_like(ref_k), ref_k), torch.where(mask, torch.zeros_like(ref_v), ref_v)
                prev_k = torch.where(mask, torch.full_like(prev_k, float('nan')), prev_k)
                prev_v = torch.where(mask, torch.full_like(prev_v, float('nan')), prev_v)
            phase_k, phase_v, _, _, clk = advance_clock('displacement', torch.ones(1, 1, 1, 1), None, None, rate, rate, clk,
                                                        k_unit=k, v=v)
            if prev_k is not None:
                ref_k = ref_k + torch.nan_to_num(wrap_phase(angle(k) - prev_k), nan=0.0)
                ref_v = ref_v + torch.nan_to_num(wrap_phase(angle(v) - prev_v), nan=0.0)
            prev_k, prev_v = angle(k), angle(v)
            torch.testing.assert_close(torch.exp(1j * phase_k), torch.exp(1j * ref_k), rtol=1e-9, atol=1e-11, msg=f'K tag, block {i}')
            torch.testing.assert_close(torch.exp(1j * phase_v), torch.exp(1j * ref_v), rtol=1e-9, atol=1e-11, msg=f'V tag, block {i}')
            tags.append(torch.exp(1j * phase_k) + torch.exp(1j * phase_v))
        if rotated:
            for a, b in zip(tags, tags_plain):
                torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11, msg='gauge invariance')
        else:
            tags_plain = tags
    loss = sum((t.real + t.imag).sum() for t in tags)
    loss.backward()
    assert all(bool(torch.isfinite(x.grad).all()) for x in ks + vs)


def _check_bounded_clock(generator):
    """The bounded clock's tags equal bound * sin(summed wrapped pair-angle differences since the lane's reset)
    + n * omega_t, n = blocks since the reset (0 in the reset block), with a mid-run reset of one lane."""
    g = generator
    B, H, T, P, blocks = 2, 2, 4, 3, 7
    D = 2 * P
    bound, omega_t = 0.14, 0.34
    rate = torch.ones(1, H, 1, P)
    angle = lambda x: torch.atan2(x[..., 1::2], x[..., 0::2])
    clk = (torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), torch.zeros(B, H, T, D), torch.zeros(B, H, T, D),
           torch.zeros(B, H, T, 1))
    ref_k, ref_v, prev_k, prev_v = torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), None, None
    n = torch.zeros(B, 1, 1, 1)
    for i in range(blocks):
        k, v = torch.randn(B, H, T, D, generator=g), torch.randn(B, H, T, D, generator=g)
        if i == 3:
            mask = torch.tensor([True, False]).view(-1, 1, 1, 1)
            clk = tuple(torch.where(mask, torch.zeros_like(x), x) for x in clk)
            ref_k, ref_v = torch.where(mask, torch.zeros_like(ref_k), ref_k), torch.where(mask, torch.zeros_like(ref_v), ref_v)
            prev_k = torch.where(mask, torch.full_like(prev_k, float('nan')), prev_k)
            prev_v = torch.where(mask, torch.full_like(prev_v, float('nan')), prev_v)
            n = torch.where(mask, torch.zeros_like(n), n)
        tag_k, tag_v, _, _, clk = advance_clock('bounded', torch.ones(1, 1, 1, 1), None, None, rate, rate, clk,
                                                 k_unit=k, v=v, bound=bound, omega_t=omega_t)
        if prev_k is not None:
            ref_k = ref_k + torch.nan_to_num(wrap_phase(angle(k) - prev_k), nan=0.0)
            ref_v = ref_v + torch.nan_to_num(wrap_phase(angle(v) - prev_v), nan=0.0)
        prev_k, prev_v = angle(k), angle(v)
        torch.testing.assert_close(tag_k, bound * torch.sin(ref_k) + n * omega_t, rtol=1e-9, atol=1e-11, msg=f'K tag, block {i}')
        torch.testing.assert_close(tag_v, bound * torch.sin(ref_v) + n * omega_t, rtol=1e-9, atol=1e-11, msg=f'V tag, block {i}')
        n = n + 1


def _check_leaky_clock(generator):
    """The leaky clock recursion equals the explicit sums: carrier = wrap(sum dt omega_bar) and
    psi(b) = sum_j [prod_{r=j+1..b} rho(r)] dt(j) freq(j), with a mid-run reset of one lane."""
    g = generator
    B, H, T, P, blocks = 2, 2, 4, 3, 7
    omega_bar = (torch.rand(1, H, 1, 1, generator=g) * 2 - 1) * math.pi
    psi_rate_k = torch.rand(1, H, 1, P, generator=g) * 2 + 0.1
    psi_rate_v = torch.rand(1, H, 1, P, generator=g) * 2 + 0.1
    rate = torch.ones(1, H, 1, P)
    clk = (torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), torch.zeros(B, H, T, 1))
    hist, ref_psi_k, ref_psi_v, ref_car = [], torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), torch.zeros(B, H, T, 1)
    for i in range(blocks):
        dt = torch.rand(B, H, T, 1, generator=g) * 1.8 + 0.2
        fk = (torch.rand(B, H, T, P, generator=g) * 2 - 1) * math.pi
        fv = (torch.rand(B, H, T, P, generator=g) * 2 - 1) * math.pi
        if i == 3:
            mask = torch.tensor([True, False]).view(-1, 1, 1, 1)
            clk = tuple(torch.where(mask, torch.zeros_like(x), x) for x in clk)
            ref_psi_k, ref_psi_v, ref_car = (torch.where(mask, torch.zeros_like(x), x) for x in (ref_psi_k, ref_psi_v, ref_car))
        phase_k, phase_v, _, _, clk = advance_clock('leaky', dt, fk, fv, rate, rate, clk, omega_bar, psi_rate_k, psi_rate_v)
        ref_psi_k = torch.exp(-psi_rate_k * dt) * ref_psi_k + dt * fk
        ref_psi_v = torch.exp(-psi_rate_v * dt) * ref_psi_v + dt * fv
        ref_car = ref_car + dt * omega_bar                        # unwrapped
        for got, ref in ((phase_k, ref_car + ref_psi_k), (phase_v, ref_car + ref_psi_v)):
            torch.testing.assert_close(torch.exp(1j * got), torch.exp(1j * ref), rtol=1e-9, atol=1e-11, msg=f'tag, block {i}')
        torch.testing.assert_close(clk[0], ref_psi_k, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(clk[1], ref_psi_v, rtol=1e-9, atol=1e-11)


def selftest():
    """CPU/FP64 checks with synthetic inputs; no dataset and no claim about accuracy."""
    old_dtype, old_inner = torch.get_default_dtype(), t.KVSTDPInner
    torch.set_default_dtype(torch.float64)
    checks = []
    close = dict(rtol=1e-9, atol=1e-11)
    try:
        _check_step_against_pairs(torch.Generator().manual_seed(11))
        checks.append('trace recursion equals the explicit sum over all (post, pre) event pairs, '
                      'including a mid-run reset of one lane')
        _check_leaky_clock(torch.Generator().manual_seed(17))
        checks.append('leaky clock: carrier and latency recursions equal their explicit sums, with a mid-run reset')
        _check_displacement_clock(torch.Generator().manual_seed(19))
        _check_bounded_clock(torch.Generator().manual_seed(23))
        checks.append('bounded clock: tags equal bound * sin(summed wrapped pair-angle differences) + blocks since reset * omega_t, '
                      'with a mid-run reset')
        checks.append('displacement clock: tags equal the summed wrapped pair-angle differences since reset, are invariant '
                      'to a constant per-pair basis rotation (gauge), finite gradients')

        cfg = dict(t.DEFAULT_CFG, **t.PRESETS['kv_stdp'])
        cfg.update(hidden_size=16, num_heads=2, puzzle_emb_ndim=16, global_batch_size=2, batch_size=2,
                   seq_len=81, num_puzzle_identifiers=1, blocks_per_seg=3, loops=8, amp=False,
                   amp_dtype='float32', forward_dtype='float64', activation_checkpoint=False,
                   compile=False, kv_write_reduction='mean')

        def batch_of(seed):
            gen = torch.Generator().manual_seed(seed)
            return dict(inputs=torch.randint(1, 11, (2, 81), generator=gen),
                        labels=torch.randint(2, 11, (2, 81), generator=gen),
                        puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        batch, alt = batch_of(1), batch_of(2)
        for clock in ('integrate', 'leaky', 'displacement', 'bounded'):
            _model_checks(clock, cfg, batch, alt, checks, close)
    finally:
        t.KVSTDPInner = old_inner
        torch.set_default_dtype(old_dtype)
    for item in checks:
        print('[selftest] PASS ' + item, flush=True)
    print(f'[selftest] {len(checks)}/{len(checks)} passed (CPU, FP64, synthetic inputs).', flush=True)
    return checks


def _model_checks(clock, cfg, batch, alt, checks, close):
    t.KVSTDPInner = model_class(clock=clock)
    tag = f' [{clock}]'

    def build(**overrides):
        torch.manual_seed(5)
        return t.ACTLossHead(t.LT(dict(cfg, **overrides)), q_weight=cfg['q_weight'])

    base = build()
    lt, inner = base.model, base.model.inner
    base.eval()
    if clock in ('displacement', 'bounded'):
        lt.config.blocks_per_seg = 1          # tags are 0 in a lane's first block only (no previous event)
    with torch.no_grad():
        ca, oa = lt(lt.initial_carry(batch), batch)
        for layer in inner.layers:
            layer.stdp_alpha.zero_()
        cb, ob = lt(lt.initial_carry(batch), batch)
        for layer in inner.layers:
            layer.stdp_alpha.fill_(1.0)
    lt.config.blocks_per_seg = 3
    assert torch.equal(oa['logits'], ob['logits']) and torch.equal(ca.current_hidden, cb.current_hidden)
    phases = ca.coupling[..., :2 * inner.P]
    assert int(torch.count_nonzero(phases)) == 0 and ca.coupling.shape[-1] == inner.clock_width
    checks.append(('first block: tags are exactly 0 and the forward equals alpha = 0 (B-only)' if clock in ('displacement', 'bounded')
                   else 'initialization: phases are exactly 0 and the forward equals alpha = 0 (B-only)') + tag)

    _activate_clock(inner, torch.Generator().manual_seed(12))
    with torch.no_grad():
        lt.config.blocks_per_seg = 3
        c1, _ = lt(lt.initial_carry(batch), batch)
        c2, o2 = lt(c1, batch)
        lt.config.blocks_per_seg = 6
        cl, ol = lt(lt.initial_carry(batch), batch)
        lt.config.nograd_blocks, lt.config.blocks_per_seg = 3, 3
        cn, on = lt(lt.initial_carry(batch), batch)
        lt.config.nograd_blocks = 0
    assert float(c2.coupling.abs().max()) > 0 and float(c2.key_trace[..., 8:].abs().max()) > 0
    for name in STATE_NAMES:
        torch.testing.assert_close(getattr(c2, name), getattr(cl, name), msg=name, **close)
        torch.testing.assert_close(getattr(cn, name), getattr(cl, name), msg=name, **close)
    torch.testing.assert_close(o2['logits'], ol['logits'], **close)
    torch.testing.assert_close(on['logits'], ol['logits'], **close)
    checks.append('two 3-block segments and a 3 no-grad + 3 block segment equal one 6-block unroll '
                  '(hidden, clock state, both traces, logits)' + tag)

    with torch.no_grad():
        c1, _ = lt(lt.initial_carry(batch), batch)
        c2, _ = lt(c1, alt)
        cm, _ = lt(replace(c1, halted=torch.tensor([True, False])), alt)
        cf, _ = lt(lt.initial_carry(alt), alt)
    for name in STATE_NAMES:
        torch.testing.assert_close(getattr(cm, name)[0], getattr(cf, name)[0], msg=name, **close)
        torch.testing.assert_close(getattr(cm, name)[1], getattr(c2, name)[1], msg=name, **close)
    checks.append('per-lane reset: a fresh lane equals a fresh run, a continuing lane is unaffected' + tag)

    fresh_model = build()
    fresh_model.train()
    _, loss, *_ = fresh_model(carry=fresh_model.initial_carry(batch), batch=batch, return_keys=set())
    loss.backward()
    layer = fresh_model.model.inner.layers[0]
    if clock in ('displacement', 'bounded'):
        # no generator: the only clock parameters are the decay rates and alpha, both live from the start
        for name in ('clock_lam_raw', 'stdp_alpha'):
            assert float(layer.get_parameter(name).grad.abs().max()) > 0, name
        checks.append('no clock generator; decay rates and alpha receive gradients from the first step' + tag)
        zero = []
    else:
        assert float(layer.clock_freq.weight.grad.abs().max()) > 0
        zero = ['clock_dt.weight', 'clock_dt.bias', 'clock_lam_raw', 'stdp_alpha']
    if clock == 'leaky':
        zero.append('clock_psi_lam_raw')
        # the carrier cancels in same-block pairs; cross-block pairs give it a gradient from block 2 on
        assert float(layer.clock_omega_raw.grad.abs().max()) > 0
    for name in zero:
        grad = layer.get_parameter(name).grad
        assert grad is None or float(grad.abs().max()) == 0.0, name
    if clock not in ('displacement', 'bounded'):
        checks.append('at initialization only the frequency head' + (' and the carrier velocity' if clock == 'leaky' else '')
                      + ' have a gradient (dt, decay, latency leak and alpha are exactly 0)' + tag)

    params = (['clock_lam_raw', 'stdp_alpha'] if clock in ('displacement', 'bounded')
              else list(CLOCK_PARAMETERS) + (['clock_omega_raw', 'clock_psi_lam_raw'] if clock == 'leaky' else []))
    models = []
    for checkpointing in (False, True):
        model = build(activation_checkpoint=checkpointing)
        _activate_clock(model.model.inner, torch.Generator().manual_seed(13))
        model.train()
        model.zero_grad(set_to_none=True)
        _, loss, *_ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
        loss.backward()
        models.append(model)
    layer = models[0].model.inner.layers[0]
    for name in params:
        grad = layer.get_parameter(name).grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0, name
    for (name, p), (other_name, q) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert name == other_name and (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()), name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-11, msg=name)
    checks.append('active clock: finite nonzero gradients for every clock parameter; '
                  'activation checkpoint on/off gradients equal' + tag)


# ----------------------------------------------------------------------------- runner
def configuration():
    cfg = dict(t.CFG)
    config_path = Path(__file__).resolve().parents[1] / 'configs/free_phase_window_research.json'
    cfg.update(json.loads(config_path.read_text()))
    cfg.update(data_npz=str(Path('data/sudoku_lt_1k.npz').resolve()), run_selftests=False,
               num_processes=1, log_every=16, save_every_steps=0, keep_last=25,
               milestone_every=0, init_from=None, resume_from=None, require_resume=False,
               activation_checkpoint=False)        # block activations are kept; no recomputation (2026-10-13 decision)
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
    report = dict(protocol='Isolated full optimizer steps; batch 128, BF16 projections, FP32 clock/traces/memory, no activation checkpoint',
                  steps=[], parameters=sum(p.numel() for p in base.parameters()), gradient_checks=[])
    old_check = t._check_finite_gradients

    def check(model, loss, device, ws):
        old_check(model, loss, device, ws)
        if state.step in (0, steps - 1):
            report['gradient_checks'].append(dict(step=state.step, norms={
                n: float(p.grad.norm()) for n, p in model.named_parameters()
                if ('.clock_' in n or 'stdp_alpha' in n) and p.grad is not None}))
    t._check_finite_gradients = check
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(steps):
            started = time.monotonic()
            metrics = t.train_batch(compiled, base, state, batch, cfg, opts, lrs, 390625, 0, 1, device)
            torch.cuda.synchronize()
            row = dict(step=state.step, seconds=time.monotonic() - started, **metrics)
            report['steps'].append(row)
            print('PREFLIGHT', row, flush=True)
    finally:
        t._check_finite_gradients = old_check
    report.update(median_seconds_after_warmup=statistics.median(r['seconds'] for r in report['steps'][3:])
                  if len(report['steps']) > 3 else None,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  clock_parameter_norms={n: float(p.detach().norm()) for n, p in base.named_parameters()
                                         if '.clock_' in n or 'stdp_alpha' in n})
    first = report['gradient_checks'][0]['norms']
    assert all(math.isfinite(v) for row in report['gradient_checks'] for v in row['norms'].values())
    # At step 0 the frequency head is the only clock parameter with a gradient (see the module doc).
    assert all(v > 0 for n, v in first.items() if '.clock_freq.' in n), first
    (out / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda-init', type=float, default=0.1,
                    help='initial trace retention per block at dt = dt-init (KV-STDP trace_decay_init)')
    ap.add_argument('--alpha-init', type=float, default=1.0, help='initial alpha_h (STDP gain per head)')
    ap.add_argument('--dt-init', type=float, default=1.0, help='initial clock step dt (bias of the dt head)')
    ap.add_argument('--boundary-ffn', choices=('bilinear', 'swiglu'), default='swiglu',
                    help='FFN after the attention RMSNorm: h + D(silu(g) u) (default) or h + D(g/2 * u)')
    ap.add_argument('--clock-no-decay', action='store_true',
                    help='exclude the clock generator weights (dt and frequency heads) from weight decay')
    ap.add_argument('--kernel', choices=('triton', 'torch'), default='triton',
                    help='fused Triton block (lt/phase_clock_fused.py) or the plain torch block')
    ap.add_argument('--feature-precision', choices=('bfloat16', 'float32'), default='bfloat16',
                    help='dot operands of the fused kernel (FP32 accumulation; traces, phases and norms stay FP32)')
    ap.add_argument('--clock', choices=('displacement', 'leaky', 'integrate', 'bounded'), default='displacement',
                    help="'bounded': displacement tags bounded to bound*sin(.) plus blocks-since-start * omega_t (2026-10-14); "
                         "'displacement': no generator; tags = each pair's own accumulated angular displacement (third run); "
                         "'leaky': common carrier per head + leaky neuron latencies (second run); "
                         "'integrate': every neuron integrates its own velocity (first run)")
    ap.add_argument('--rho-init', type=float, default=0.5, help='initial latency retention per block at dt = dt-init (leaky clock)')
    ap.add_argument('--bound', type=float, default=0.14, help='bounded clock: latency amplitude c (tags c sin(phase))')
    ap.add_argument('--omega-t', type=float, default=0.34, help='bounded clock: phase advance per block (block spacing)')
    ap.add_argument('--freeze-lambda', action='store_true', help='keep the trace retentions at --lambda-init (no gradient)')
    ap.add_argument('--clock-precision', choices=('float32', 'bfloat16'), default='float32',
                    help='the dt and frequency affine maps: FP32 outside autocast (default) or BF16 like the Q/K/V projections')
    ap.add_argument('--precision', choices=('highest', 'high'), default='highest')
    ap.add_argument('--steps', type=int, default=3008, help='absolute stopping step; 0 runs the full epoch schedule')
    ap.add_argument('--save-every', type=int, default=0, help='periodic checkpoint interval in steps; 0 saves only at eval boundaries')
    ap.add_argument('--nograd-fixed', type=int, default=0,
                    help='blocks run without gradient at the start of every segment; 0 keeps the config value')
    ap.add_argument('--no-boundary-save', action='store_true', help='save only every --save-every steps and at stop')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--selftest', action='store_true', help='run the CPU/FP64 checks and exit')
    ap.add_argument('--out', type=Path)
    opt = ap.parse_args()
    if opt.selftest:
        selftest()
        return
    if opt.out is None:
        ap.error('--out is required unless --selftest is given')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(opt.precision)
    out = opt.out.resolve()
    if (out / 'config.json').exists() and not opt.preflight:
        raise RuntimeError('Use a new output directory; this entry point never silently resumes.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    name = f'phaseclock_l{opt.lambda_init:g}_a{opt.alpha_init:g}_dt{opt.dt_init:g}'
    if opt.clock == 'leaky':
        name += f'_leaky_rho{opt.rho_init:g}'
    elif opt.clock == 'displacement':
        name += '_disp'
    elif opt.clock == 'bounded':
        name += f'_bounded_c{opt.bound:g}_wT{opt.omega_t:g}'
    if opt.freeze_lambda:
        name += '_lamfix'
    if opt.boundary_ffn != 'bilinear':
        name += '_' + opt.boundary_ffn
    if opt.nograd_fixed:
        name += f'_ng{opt.nograd_fixed}'
        cfg['nograd_fixed'] = opt.nograd_fixed
    if opt.clock_no_decay:
        name += '_clockwd0'
        exclude_clock_from_decay()
    if opt.kernel == 'triton':
        name += '_bf16fused' if opt.feature_precision == 'bfloat16' else '_fp32fused'
    if opt.clock_precision == 'bfloat16':
        name += '_clockbf16'
    if opt.precision == 'high':
        name += '_tf32'
    name += '_qkl2_sum'
    if opt.no_boundary_save:
        cfg['save_at_boundary'] = False
    cfg.update(out_dir=str(out), max_steps=opt.steps or None, max_hours=None, save_every_steps=opt.save_every,
               research_variant=name, research_matmul_precision=opt.precision,
               clock_no_decay=opt.clock_no_decay)
    t.KVSTDPInner = model_class(opt.lambda_init, opt.alpha_init, opt.dt_init, opt.boundary_ffn,
                                opt.kernel, opt.feature_precision, opt.clock_precision, opt.clock, opt.rho_init,
                                opt.bound, opt.omega_t, opt.freeze_lambda)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + name
    ffn = 'h + D(silu(g) u)' if opt.boundary_ffn == 'swiglu' else 'h + D(g/2 * u)'
    protocol = dict(
        variant=name, torch=torch.__version__, gpu=torch.cuda.get_device_name(), precision=opt.precision,
        lambda_init=opt.lambda_init, alpha_init=opt.alpha_init, dt_init=opt.dt_init,
        boundary_ffn=opt.boundary_ffn, clock_no_decay=opt.clock_no_decay,
        kernel=opt.kernel, feature_precision=opt.feature_precision if opt.kernel == 'triton' else 'float32',
        clock_precision=opt.clock_precision, clock=opt.clock, rho_init=opt.rho_init,
        bound=opt.bound, omega_t=opt.omega_t, freeze_lambda=opt.freeze_lambda,
        clock_state=('leaky: carrier Phi <- wrap(Phi + dt pi tanh(clock_omega_raw[h])) per cell and head, shared by K and V; '
                     'latency psi_r <- exp(-softplus(clock_psi_lam_raw[r]) dt) psi_r + dt w_r (bounded, content-driven); '
                     'tags phi_r = wrap(Phi + psi_r); coupling = cat(psi_K, psi_V, Phi) [B,H,T,2P+1]'
                     if opt.clock == 'leaky' else
                     'displacement: no clock generator (clock_dt, clock_freq absent); tags phi_r <- wrap(phi_r + wrap(theta_r(k) - theta_r(k-1))), '
                     'theta_K = atan2 of the unit key pair before RoPE, theta_V = atan2 of the raw value pair; 0 in a lane\'s first block; '
                     'dt = 1, lam_r = exp(-softplus(clock_lam_raw[r])); coupling = cat(phi_K, phi_V, zK_prev, zV_prev) [B,H,T,6P]'
                     if opt.clock == 'displacement' else
                     f'bounded: displacement phases phi_r (as above, carried with zK_prev, zV_prev) and a per-lane block counter n '
                     f'(0 in a puzzle\'s first block); tags = {opt.bound:g} sin(phi_r) + n * {opt.omega_t:g} for K and V; '
                     f'coupling = cat(phi_K, phi_V, zK_prev, zV_prev, n) [B,H,T,6P+1]; lambda '
                     + ('fixed' if opt.freeze_lambda else 'learned') + f' (init {opt.lambda_init:g})'
                     if opt.clock == 'bounded' else
                     'integrate: phi_r <- wrap(phi_r + dt w_r) per neuron; coupling = cat(phi_K, phi_V) [B,H,T,2P]'),
        kernel_note=('lt/phase_clock_fused.py: L2 norm, RoPE, phase tags, trace update, five feature GEMMs and the read '
                     'in one Triton forward kernel; phase advance, retentions and the fresh-lane reset in torch'
                     if opt.kernel == 'triton' else 'plain torch block'),
        block=f'h + sqrt(d) inj -> Q,K,V (AMP projections) -> phase-clock memory (FP32) -> out_proj -> RMSNorm -> {ffn} -> RMSNorm',
        address='Q, K = x/(||x||_2+eps) per token/head before learned 2D RoPE; V raw; K != V; token-sum writes',
        clock_generator=('dt = softplus(W_dt h + b_dt) [B,H,T], shared by K and V; w_role = pi tanh(W_f h + b_f) [B,H,T,P]; '
               'phi_role <- wrap(phi_role + dt w_role) per cell, head and pair along the recurrence (never along cells); '
               'the post-update phase tags the block\'s events; lam_role = exp(-softplus(clock_lam_raw[role]) dt)'),
        traces=('E_K <- lam_K E_K + (1-lam_K) k e^{-i phi_K} (unit K before RoPE, includes this block); '
                'D_V = lam_V E_V; E_V <- D_V + (1-lam_V) v e^{+i phi_V}'),
        write=('G = sum_n v kr^T + alpha_h Im S, S = sum_n [(v e^{i phi_V}) RoPE(E_K)^T + D_V (kr e^{-i phi_K})^T]; '
               'each (post, pre) pair counted once with weight (1-lam_earlier) exp(-a sum dt) e^{i(phi_post-phi_pre)}; '
               'Im S odd (STDP), Re S even (unused); current increment only, no matrix accumulation'),
        read='qr @ G.T',
        initialization=('W_dt = W_f = b_f = 0 -> phi = 0, Im S = 0: step 0 equals B-only; b_dt = inv_softplus(dt_init); '
                        'clock_lam_raw = inv_softplus(-ln(lambda_init)/dt_init); alpha_h = alpha_init; no random draws'),
        carry=('coupling = cat(phi_K, phi_V); key_trace = cat(Re E_K, Im E_K); value_trace = cat(Re E_V, Im E_V); '
               'zeroed for fresh puzzles, detached at segment boundaries'),
        weight_decay=('clock_lam_raw and stdp_alpha: none; clock dt/frequency weights: '
                      + ('none' if opt.clock_no_decay else 'trainer default') + '; biases: none'),
        unused_parameters='theta_k_raw, theta_v_raw, trace_lam_raw: allocated for identical draws, no gradient',
        baseline='2026-10-10 pairangle_r1_dca1_swiglu_bf16fused_20261010 (same harness, phases from address angles, no history)')
    here = Path(__file__)
    for module in (here, here.with_name('phase_clock_fused.py'), here.with_name('pairangle_fused.py'),
                   here.with_name('kv_stability.py'), Path(t.__file__)):
        content = module.read_bytes()
        (out / module.name).write_bytes(content)
        protocol[module.name + '_sha256'] = hashlib.sha256(content).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    if opt.preflight:
        preflight(cfg, out, opt.steps)
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
