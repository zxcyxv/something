"""Bounded phase-clock STDP: a recurrent KV model whose write is a pair-STDP integral over the recurrence.

One shared block, run 8 times per segment and 16 segments per puzzle (docs/architecture.md):

    u    = h + sqrt(d) inj                                   inj = E[x] + E_puzzle (re-injected every block)
    q, k, v = heads(W_q u), heads(W_k u), heads(W_v u)
    q^, k^ = x / (||x||_2 + eps);  q~, k~ = RoPE(q^), RoPE(k^)   (learned 2D RoPE; v raw)
    clock:  phi_r <- wrap(phi_r + wrap(theta_r(now) - theta_r(prev)))   theta = pair angle of k^ (before RoPE) / v
            tau_r = c sin(phi_r) + n omegaT                              n = blocks since the puzzle started
    traces: E_K <- lam E_K + (1 - lam) k^ e^{-i tau_K};   D_V = lam E_V;   E_V <- D_V + (1 - lam) v e^{+i tau_V}
    Im S  = Im sum_t [ (v e^{i tau_V}) RoPE(E_K)^T + D_V (k~ e^{-i tau_K})^T ]          (token sums, V rows, K cols)
    W     = rho W + Im S                                     (model B; rho = 0 is model A: W = Im S)
            or (1 - eta) W + eta Im S                        (outer_eta > 0: EMA form, fixed point W = Im S)
    read  = q~ (sum_t v k~^T + alpha_h W)^T                  (hebb=False drops sum_t v k~^T)
    h'    = RMSNorm(u + W_O read);   h = RMSNorm(h' + W_D(silu(g) * a)),  [g; a] = W_GU h'

Every (post = V, pre = K) event pair of the same cell since the puzzle started is counted exactly once with weight
(1 - lam) lam^k e^{i(tau_post - tau_pre)}, k = block difference, so alpha_h Im S is an odd (STDP) window on top of the
Hebbian carrier. With omegaT > 2c and K omegaT + 2c < pi every pair up to K blocks apart takes its LTP/LTD sign from
the block order. c, omegaT, lam and rho are fixed constants; alpha_h is trained (init 1).

The clock, traces, W and the read are FP32 (FP64 in tests); projections and the FFN follow BF16 autocast where the
GPU has native BF16. kernel 'triton' runs the memory step through lt/kernel.py (rho = 0 only).

event_stdp (control model, tag_bound = 0): the pairs are taken between block-to-block changes (events) instead of
states, and the Hebbian term stays on the states:

    e_x  = x - x_prev for x in (q^, k^, v), soft-thresholded e = d r2 / (r2 + theta^2) when event_theta > 0; 0 at n = 0
    P    = Im S over (e_V, e_K) with tau = n omegaT (no latency)  [+ gamma_h Re S with event_even]
    W    = rho W + P | (1 - eta) W + eta P | P;    read = alpha_h RoPE(e_Q) W^T + q~ (sum_t v k~^T)^T

bands = R > 0 (no clock, tag_bound = 0): the pair window is a sum of R damped complex exponentials with learned poles
z_r = exp(-e^{nu_r} + i theta_r) and complex coefficients a_r (value at or after the key), b_r (value before):

    E_K,r <- z_r E_K,r + gamma_r k~;   D_V,r = z_r E_V,r;   E_V,r <- D_V,r + gamma_r v      gamma_r = sqrt(1 - |z_r|^2)
    S     = sum_r Re(a_r v^T E_K,r + b_r D_V,r^T k~)        window l(D) = sum_r gamma_r Re(a_r z_r^D), D >= 0;
    W     = (1 - eta) W + eta S,   read = q~ (sum_t v k~^T + W)^T          sum_r gamma_r Re(b_r z_r^|D|), D < 0
    hebb=False drops the tonic term and reads this block's write directly: read = q~ (S + W)^T (one window: the
    same-block Hebb is l(0) = sum_r gamma_r Re a_r)
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

TWO_PI = 2 * math.pi


@dataclass
class ModelConfig:
    batch_size: int = 128             # per process (the puzzle-embedding buffer)
    seq_len: int = 81
    grid: int = 9
    num_grids: int = 1                # grids per sample, stacked along rows (in-context demonstrations + query)
    vocab_size: int = 11              # 0 pad, 1 blank, 2..10 digits 1..9
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 832
    hidden_size: int = 832
    num_heads: int = 8
    blocks_per_seg: int = 8
    loops: int = 16                   # segments per puzzle
    mlp_expansion: float = 4.0
    mlp_inter: int = 0                # FFN inner width (0: mlp_expansion * d * 2/3 rounded up to a multiple of 256)
    eps: float = 1e-4                 # Q/K L2 denominator ||x|| + eps
    tag_bound: float = 0.14           # c
    block_omega: float = 0.34         # omegaT
    trace_lambda: float = 0.5         # lam_K = lam_V
    outer_rho: float = 0.0            # rho of W (0: model A)
    outer_eta: float = 0.0            # EMA rate of W: W = (1 - eta) W + eta Im S (0: off; exclusive with outer_rho)
    outer_eta_learn: bool = False     # learn eta per head (sigmoid(eta_raw), initialized at outer_eta; no weight decay)
    alpha_init: float = 1.0           # alpha_h at initialization
    hebb: bool = True                 # include the Hebbian carrier sum_t v k~^T in the write (False: G = alpha Im S)
    event_stdp: bool = False          # STDP between block-to-block changes, read by the query's change (needs tag_bound 0)
    event_theta: float = 0.0          # initial soft threshold of the events (0: linear events, no parameter)
    event_even: bool = False          # add the even (co-change) window Re S with a learned gain gamma_h (init 0)
    kv_ema: bool = False              # plain linear KV memory: W = (1 - eta) W + eta sum_t v k~^T, read q~ W^T (no clock/STDP)
    kv_ema_current: bool = False      # kv_ema read = q~ (sum_t v k~^T + alpha_h W)^T: this block's Hebb at gain 1 plus the EMA
    exp_stdp: bool = False            # exponential pair STDP on scalar channels, L(0) = 0: W = (1 - eta) W + eta dW, read q~ W^T;
                                      # per-head learned lam+, lam-, A+ (init trace_lambda, trace_lambda, 1), A- by equal areas
    exp_hebb: bool = False            # exp_stdp read = q~ (sum_t v k~^T + alpha_h W)^T: B-EMA's form with the exponential window
    bands: int = 0                    # R > 0: clockless window of R complex poles, read q~ (sum_t v k~^T + W)^T (bands_step);
                                      # with hebb False read q~ (S + W)^T
    band_periods: tuple = (2.0, 6.0, 20.0, 64.0)   # initial periods p_r: theta_r = 2 pi / p_r, |z_r| = e^{-1/p_r}
    band_coef_std: float = 0.1        # std of each real component of the initial a_r, b_r
    outer_eta_cap: bool = False       # project the learned eta onto eta <= outer_eta after every optimizer step
    linear_attn: bool = False         # standard recurrent linear attention: read = q~ (sum_t v k~^T)^T (no clock, W, STDP)
    task_film: bool = False           # per-identifier FiLM of the FFN gate keys: g <- g * (1 + gamma[id]), gamma init 0
    task_film_up: bool = False        # per-identifier scale of the FFN up side: a <- a * (1 + w[id]) (linear in each slot's
                                      # output: the update is a task-weighted sum of the slots' fields), w init 0
    task_lowrank: int = 0             # r > 0: per-identifier rank-r change of the gate keys, W_G + U[id] V[id]^T (U init 0)
    puzzle_emb_mode: str = 'add'      # add: puzzle embedding added to every cell's input; prefix: URM/HRM style, padded to
                                      # ceil(puzzle_emb_ndim / d) extra positions in front of the cells (logits drop them);
                                      # film: not injected, a shared projection makes FFN gate scales g <- g * (1 + H e[id])
    kernel: str = 'auto'              # auto | torch | triton
    kernel_bf16: bool = True          # BF16 dot operands in the fused kernel (FP32 accumulation)
    act_ckpt: bool = False            # recompute each block in backward (activation checkpointing; same math, less memory)
    act_ckpt_blocks: int = -1         # with act_ckpt: checkpoint only the first N blocks of a segment (-1: all)
    amp: bool = True                  # BF16 autocast for projections and FFN where native
    forward_dtype: str = 'float32'    # float64 for tests
    act: bool = False                 # URM ACT: q_head on position 0 gives q_halt; in training a lane halts when q_halt > 0
                                      # (or after `loops` segments), with exploration (a random minimum of 2..loops segments
                                      # with probability act_explore); evaluation always runs `loops` segments
    act_explore: float = 0.1
    fast_ffn: int = 0                 # r > 0: fast-weight FFN memory (weight-space delta rule). Per lane dW [r, I]; key
                                      # phi~ = unit(silu(g) * a) (the FFN's own hidden pattern), value v_f = W_fv h' [r];
                                      # dW <- (1 - lam) dW + (eta / T) sum_t (v_f - dW phi~) phi~^T (one gradient step on
                                      # 1/2 sum_t |v_f - dW phi~|^2: eta < 1 and |phi~| = 1 keep the step a contraction);
                                      # FFN output += W_fo dW phi~ (W_fo starts at 0, so the model starts unchanged)
    fast_ffn_eta: float = 0.5         # initial eta (learned, sigmoid)
    fast_ffn_decay: float = 0.02      # initial lam (learned, sigmoid)

    def __post_init__(self):
        if self.seq_len != self.num_grids * self.grid ** 2:
            raise ValueError('seq_len must equal num_grids * grid**2')
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError('the head dimension must be even')
        if not 0 <= self.puzzle_emb_ndim <= self.hidden_size:
            raise ValueError('puzzle_emb_ndim must lie in [0, hidden_size]')
        if not (0 < self.trace_lambda < 1 and 0 <= self.outer_rho < 1 and 0 <= self.outer_eta < 1
                and self.tag_bound >= 0 and self.eps > 0):
            raise ValueError('need 0 < trace_lambda < 1, 0 <= outer_rho < 1, 0 <= outer_eta < 1, tag_bound >= 0, eps > 0')
        if self.outer_rho and self.outer_eta:
            raise ValueError('outer_rho (leaky integral) and outer_eta (EMA) are exclusive')
        if self.outer_eta_learn and not self.outer_eta:
            raise ValueError('outer_eta_learn needs outer_eta > 0 as the initial value')
        if self.kernel not in ('auto', 'torch', 'triton'):
            raise ValueError('kernel must be auto, torch or triton')
        if self.kernel == 'triton' and (self.outer_rho or self.outer_eta or not self.hebb or self.event_stdp
                                        or self.linear_attn):
            raise ValueError('the fused kernel implements outer_rho = outer_eta = 0 with the Hebbian term only')
        if self.event_stdp and self.tag_bound != 0:
            raise ValueError('event_stdp has no in-block latency: set tag_bound = 0')
        if self.event_theta < 0 or ((self.event_theta or self.event_even) and not self.event_stdp):
            raise ValueError('event_theta >= 0; event_theta and event_even need event_stdp')
        if self.kv_ema and (not self.outer_eta or self.event_stdp or not self.hebb):
            raise ValueError('kv_ema needs outer_eta > 0 (the EMA rate), event_stdp off and hebb on')
        if self.kv_ema_current and not self.kv_ema:
            raise ValueError('kv_ema_current needs kv_ema')
        if self.exp_stdp and (self.outer_rho or self.event_stdp or self.kv_ema or self.tag_bound):
            raise ValueError('exp_stdp needs outer_rho = 0, event_stdp and kv_ema off and tag_bound = 0')
        if self.exp_hebb and not self.exp_stdp:
            raise ValueError('exp_hebb needs exp_stdp')
        self.band_periods = tuple(float(p) for p in self.band_periods)
        if self.bands and (self.outer_rho or not self.outer_eta or self.tag_bound or self.event_stdp
                           or self.kv_ema or self.exp_stdp):
            raise ValueError('bands needs outer_eta > 0, outer_rho = 0, tag_bound = 0 and no other memory mode')
        if self.bands < 0 or (self.bands and (len(self.band_periods) != self.bands or min(self.band_periods) <= 0)):
            raise ValueError('bands >= 0 with one positive initial period per band')
        if self.linear_attn and (self.outer_rho or self.outer_eta or self.tag_bound or not self.hebb or self.event_stdp
                                 or self.kv_ema or self.exp_stdp or self.bands):
            raise ValueError('linear_attn needs outer_rho = outer_eta = 0, tag_bound = 0, hebb on and no other memory mode')
        if self.puzzle_emb_mode not in ('add', 'prefix', 'film'):
            raise ValueError('puzzle_emb_mode must be add, prefix or film')
        if self.puzzle_emb_mode == 'film' and not self.puzzle_emb_ndim:
            raise ValueError('puzzle_emb_mode film needs puzzle_emb_ndim > 0')
        if self.mlp_inter < 0:
            raise ValueError('mlp_inter >= 0')
        if self.outer_eta_cap and not self.outer_eta_learn:
            raise ValueError('outer_eta_cap needs outer_eta_learn')
        if not 0 <= self.act_explore <= 1:
            raise ValueError('act_explore must lie in [0, 1]')
        if self.fast_ffn < 0 or not (0 < self.fast_ffn_eta < 1 and 0 < self.fast_ffn_decay < 1):
            raise ValueError('fast_ffn >= 0 with 0 < fast_ffn_eta, fast_ffn_decay < 1')

    @property
    def outer_memory(self) -> bool:
        return bool(self.outer_rho or self.outer_eta)

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Carry:
    """Per-lane state between segments (detached at segment boundaries; zeroed for a fresh puzzle)."""
    hidden: torch.Tensor                            # [B, T, d]
    steps: torch.Tensor                             # [B] segments run on the current puzzle
    halted: torch.Tensor                            # [B] the puzzle is finished; the lane takes a new one
    data: dict                                      # the lanes' current puzzles
    clock: Optional[torch.Tensor] = None            # cat(phi_K, phi_V, zK_prev, zV_prev, n) [B, H, T, 6P+1]
    key_trace: Optional[torch.Tensor] = None        # cat(Re E_K, Im E_K) [B, H, T, 2D] (before RoPE)
    value_trace: Optional[torch.Tensor] = None      # cat(Re E_V, Im E_V) [B, H, T, 2D]
    stdp_w: Optional[torch.Tensor] = None           # W [B, H, D, D] (outer_rho or outer_eta > 0 only)
    prev: Optional[torch.Tensor] = None             # cat(q^, k^, v) of the previous block [B, H, T, 3D] (event_stdp only)
    fresh: Optional[torch.Tensor] = None            # [B] lanes that start a puzzle in this segment
    fast_w: Optional[torch.Tensor] = None           # dW of the fast-weight FFN [B, r, I] (fast_ffn only)


@dataclass
class StepRecord:
    """What one memory step saw; passed to Model.probe (analysis, torch path)."""
    k_unit: torch.Tensor          # k^ before RoPE [B, H, T, D]
    kr: torch.Tensor              # k~
    qr: torch.Tensor              # q~
    v: torch.Tensor
    tag_k: torch.Tensor           # tau_K, tau_V [B, H, T, P]
    tag_v: torch.Tensor
    count: torch.Tensor           # n of this block [B, H, T, 1]
    traces: tuple                 # (Re E_K, Im E_K, Re E_V, Im E_V) before this block
    new_traces: tuple
    hebb: torch.Tensor            # sum_t v k~^T [B, H, D, D]
    im_s: torch.Tensor            # Im S
    rope: Callable


def trunc_normal_(tensor, std=1.0, lower=-2.0, upper=2.0):
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        a, b = math.erf(lower / math.sqrt(2)), math.erf(upper / math.sqrt(2))
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pu, pl = c * math.exp(-0.5 * upper ** 2), c * math.exp(-0.5 * lower ** 2)
        cs = std / math.sqrt(1 - (upper * pu - lower * pl) / z - ((pu - pl) / z) ** 2)
        tensor.uniform_(a, b).erfinv_().mul_(math.sqrt(2) * cs)
        return tensor.clip_(lower * cs, upper * cs)


def rms_norm(x, eps=1e-5):
    """URM RMSNorm: FP32 mean of squares over all hidden channels, no scale or bias, input dtype out."""
    dtype = x.dtype
    xf = x if dtype == torch.float64 else x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(dtype)


def wrap_phase(x):
    """Map to [-pi, pi); the gradient is 1 almost everywhere."""
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def rep2(x):
    """[..., P] per pair -> [..., D]: both components of a pair share the value."""
    return x.repeat_interleave(2, -1)


def pair_displacement(x, z_prev):
    """Angular displacement of each pair of x [B,H,T,D] from the carried unit phasor z_prev [B,H,T,2P]
    (cos half, sin half; zero = no previous event -> displacement 0), and x's own unit phasor."""
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


def clock_step(clock, k_unit, v, bound, omega_t):
    """Event tags of this block and the advanced clock state.

    clock = cat(phi_K, phi_V, zK_prev, zV_prev, n) [B,H,T,6P+1], already zeroed for fresh lanes.
    phi_r accumulates each pair's own wrapped angular displacement since the puzzle start (a constant rotation of the
    pair basis cancels); tau_r = bound sin(phi_r) + n omega_t, n = 0 in the puzzle's first block.
    Returns (tau_K, tau_V, n, new clock). n omega_t is not wrapped: run A was trained this way, and wrapping changes
    BF16 rounding enough to move the trajectory (the phasor is the same).
    """
    P = k_unit.shape[-1] // 2
    phase_k, phase_v, zk, zv, count = clock.split((P, P, 2 * P, 2 * P, 1), -1)
    dk, zk = pair_displacement(k_unit, zk)
    dv, zv = pair_displacement(v, zv)
    phase_k = wrap_phase(phase_k + dk)
    phase_v = wrap_phase(phase_v + dv)
    spacing = count * omega_t
    tag_k = bound * torch.sin(phase_k) + spacing
    tag_v = bound * torch.sin(phase_v) + spacing
    return tag_k, tag_v, count, torch.cat((phase_k, phase_v, zk, zv, count + 1), -1)


def stdp_step(k_unit, kr, v, rope, tag_k, tag_v, lam, traces, real=False):
    """One block of the phase-tagged traces; returns (hebb, Im S, new traces), token sums [B,H,D,D].

    k_unit: unit key before RoPE; kr = rope(k_unit); v raw; tag_k, tag_v [B,H,T,P]; lam: retention (scalar or
    broadcastable to [B,H,T,P]); traces = (Re E_K, Im E_K, Re E_V, Im E_V), zeroed for fresh lanes.
    real=True also returns Re S (the even window, same pairs) as a fourth value.
    """
    ek_re, ek_im, ev_re, ev_im = traces
    ck, sk = rep2(tag_k.cos()), rep2(tag_k.sin())
    cv, sv = rep2(tag_v.cos()), rep2(tag_v.sin())
    if torch.is_tensor(lam):
        lam = rep2(lam)
    # K trace with tag e^{-i tau_K}; it already holds this block's event, so same-block pairs are counted here only.
    ek_re = lam * ek_re + (1 - lam) * k_unit * ck
    ek_im = lam * ek_im - (1 - lam) * k_unit * sk
    # V trace decayed to now, before this block's event.
    dv_re, dv_im = lam * ev_re, lam * ev_im
    ekr_re, ekr_im = rope(ek_re), rope(ek_im)
    vs, vc = (v * sv).transpose(-1, -2), (v * cv).transpose(-1, -2)
    # Im[(v e^{i tau_V})^T E_K] + Im[D_V^T (kr e^{-i tau_K})]
    im_s = (vs @ ekr_re + vc @ ekr_im
            + dv_im.transpose(-1, -2) @ (kr * ck) - dv_re.transpose(-1, -2) @ (kr * sk))
    hebb = v.transpose(-1, -2) @ kr
    if real:
        re_s = (vc @ ekr_re - vs @ ekr_im
                + dv_re.transpose(-1, -2) @ (kr * ck) + dv_im.transpose(-1, -2) @ (kr * sk))
    ev_re = dv_re + (1 - lam) * v * cv
    ev_im = dv_im + (1 - lam) * v * sv
    new = (ek_re, ek_im, ev_re, ev_im)
    return (hebb, im_s, new, re_s) if real else (hebb, im_s, new)


class PuzzleEmbedding(nn.Module):
    """URM sparse puzzle embedding (one row here: every puzzle has identifier 0), trained by SignSGD."""
    def __init__(self, num, dim, batch_size, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.register_buffer('weights', torch.zeros(num, dim))
        self.register_buffer('local_weights', torch.zeros(batch_size, dim, requires_grad=True), persistent=False)
        self.register_buffer('local_ids', torch.zeros(batch_size, dtype=torch.long), persistent=False)

    def forward(self, ids):
        if not self.training:
            return self.weights[ids.long()].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[ids.long()])
            self.local_ids.copy_(ids)
        return self.local_weights.to(self.cast_to)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.hidden_size, cfg.num_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        inter = cfg.mlp_inter or int(cfg.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.ffn_down.weight.zero_()
        if cfg.fast_ffn:                     # W_fv drawn last in LT.__init__ (W_fo stays 0)
            self.ffw_v = nn.Parameter(torch.zeros(cfg.fast_ffn, d))
            self.ffw_out = nn.Parameter(torch.zeros(d, cfg.fast_ffn))
            logit = lambda x: math.log(x / (1 - x))
            self.ffw_eta_raw = nn.Parameter(torch.tensor(logit(cfg.fast_ffn_eta)))
            self.ffw_decay_raw = nn.Parameter(torch.tensor(logit(cfg.fast_ffn_decay)))
        self.rope_theta = nn.Parameter(torch.zeros(H, d // H // 2, 2))     # drawn last (see LT.__init__)
        if not (cfg.kv_ema or cfg.exp_stdp or cfg.bands or cfg.linear_attn) or cfg.kv_ema_current or cfg.exp_hebb:
            self.alpha = nn.Parameter(torch.full((H,), float(cfg.alpha_init)))
        if cfg.outer_eta_learn:                                              # no random draw: the init stays as without it
            self.eta_raw = nn.Parameter(torch.full((H,), math.log(cfg.outer_eta / (1 - cfg.outer_eta))))
        if cfg.event_stdp and cfg.event_theta > 0:                           # theta_i^2 = softplus(raw_i); rows q^/k^, v
            t2 = cfg.event_theta ** 2
            self.event_theta_raw = nn.Parameter(torch.full((2, H), t2 + math.log(-math.expm1(-t2))))
        if cfg.event_even:
            self.gamma_even = nn.Parameter(torch.zeros(H))
        if cfg.exp_stdp:                                                     # rows: logit lam+, logit lam-, log A+
            lam = math.log(cfg.trace_lambda / (1 - cfg.trace_lambda))
            self.window_raw = nn.Parameter(torch.tensor([[lam] * H, [lam] * H, [0.0] * H]))
        if cfg.task_lowrank:                                                 # U zero (start = no change); V drawn in LT
            self.task_u = nn.Parameter(torch.zeros(cfg.num_puzzle_identifiers, cfg.task_lowrank, inter))
            self.task_v = nn.Parameter(torch.zeros(cfg.num_puzzle_identifiers, cfg.task_lowrank, d))
        if cfg.puzzle_emb_mode == 'film' and cfg.puzzle_emb_ndim:           # H: embedding -> gate scales (drawn in LT)
            self.task_film_proj = nn.Linear(cfg.puzzle_emb_ndim, inter, bias=False)
        if cfg.task_film:                                                    # zeros: the start equals the model without it
            self.task_film = nn.Parameter(torch.zeros(cfg.num_puzzle_identifiers, inter))
        if cfg.task_film_up:
            self.task_film_up = nn.Parameter(torch.zeros(cfg.num_puzzle_identifiers, inter))
        if cfg.bands:                                                        # rows nu, theta [2,H,R]; |z| = e^{-1/p}
            p = torch.tensor(cfg.band_periods, dtype=torch.float64)
            self.band_pole = nn.Parameter(torch.stack((-p.log(), 2 * math.pi / p))[:, None].repeat(1, H, 1).float())
            self.band_coef = nn.Parameter(torch.zeros(4, H, cfg.bands))     # Re a, Im a, Re b, Im b; drawn in LT

    def window(self):
        """exp_stdp window per head [H]: (lam+, lam-, A+, A-) with A- lam- = A+ lam+ (equal LTP and LTD areas)."""
        lp, lm = torch.sigmoid(self.window_raw[0]), torch.sigmoid(self.window_raw[1])
        ap = torch.exp(self.window_raw[2])
        return lp, lm, ap, ap * lp / lm

    def bands(self):
        """Per head and band [H,R]: Re z, Im z, gamma = sqrt(1 - |z|^2), Re a, Im a, Re b, Im b."""
        nu, theta = self.band_pole
        mag = torch.exp(-torch.exp(nu))
        return (mag * torch.cos(theta), mag * torch.sin(theta), torch.sqrt(-torch.expm1(-2 * torch.exp(nu))),
                *self.band_coef)

    def band_summary(self):
        """Per head [H]: l(0) = sum_r gamma Re a (same-block pairs) and the window area
        M0 = sum_r gamma Re[(a + b z) / (1 - z)] (the gain on a constant state); per band [R]: head-mean period
        2 pi / theta and e-fold time e^{-nu} in blocks."""
        zr, zi, gamma, ar, ai, br, bi = (x.detach().double() for x in self.bands())
        z, a, b = torch.complex(zr, zi), torch.complex(ar, ai), torch.complex(br, bi)
        l0 = (gamma * a.real).sum(-1)
        area = (gamma * ((a + b * z) / (1 - z)).real).sum(-1)
        nu, theta = self.band_pole.detach().double()
        return l0, area, (2 * math.pi / theta.abs()).mean(0), torch.exp(-nu).mean(0)

    @torch.no_grad()
    def project_eta_(self, cfg):
        """outer_eta_cap: eta <= outer_eta (the W window never shorter than 1 / outer_eta blocks)."""
        if cfg.outer_eta_cap:
            self.eta_raw.clamp_(max=math.log(cfg.outer_eta / (1 - cfg.outer_eta)))

    def eta(self, cfg):
        """Per-head EMA rate of W [H] (learned) or the constant outer_eta."""
        return torch.sigmoid(self.eta_raw) if cfg.outer_eta_learn else cfg.outer_eta


class LT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.forward_dtype = getattr(torch, cfg.forward_dtype)
        d, H = cfg.hidden_size, cfg.num_heads
        self.d, self.H, self.dh = d, H, d // H
        self.embed_scale = math.sqrt(d)
        # Creation order fixes the random draws: embed, classifier, block projections, init_hidden, RoPE.
        self.embed = nn.Embedding(cfg.vocab_size, d)
        trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale)
        self.classifier = nn.Linear(d, cfg.vocab_size)
        self.block = Block(cfg)
        self.puzzle_emb = (PuzzleEmbedding(cfg.num_puzzle_identifiers, cfg.puzzle_emb_ndim, cfg.batch_size,
                                           self.forward_dtype) if cfg.puzzle_emb_ndim > 0 else None)
        self.init_hidden = nn.Buffer(trunc_normal_(torch.empty(d, dtype=self.forward_dtype), std=1.0))
        with torch.no_grad():
            self.block.rope_theta.uniform_(-math.pi / 2, math.pi / 2)
            if cfg.bands:                    # drawn last: every other initial weight matches the runs without bands
                self.block.band_coef.normal_(0.0, cfg.band_coef_std)
            if cfg.task_lowrank:             # drawn last as well
                self.block.task_v.normal_(0.0, 1.0 / math.sqrt(d))
            if cfg.puzzle_emb_mode == 'film' and cfg.puzzle_emb_ndim:      # the table starts at 0, so H must not
                self.block.task_film_proj.weight.normal_(0.0, 1.0 / math.sqrt(cfg.puzzle_emb_ndim))
        self.prefix_len = (-(cfg.puzzle_emb_ndim // -d) if cfg.puzzle_emb_mode == 'prefix' and cfg.puzzle_emb_ndim else 0)
        self.n_pos = cfg.seq_len + self.prefix_len         # positions in the recurrent state (prefix first)
        T, g = cfg.seq_len, cfg.grid
        rope_dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        self.register_buffer('pos_row', (torch.arange(T) // g).to(rope_dtype), persistent=False)
        self.register_buffer('pos_col', (torch.arange(T) % g).to(rope_dtype), persistent=False)
        self.probe: Optional[Callable[[StepRecord], Optional[torch.Tensor]]] = None
        self.block_fn: Optional[Callable] = None          # compiled block_step (lt.train compile_scope 'block')
        self.q_halt_logits: Optional[torch.Tensor] = None   # [B] of the last segment (act)
        if cfg.fast_ffn:                     # drawn last as well
            with torch.no_grad():
                self.block.ffw_v.normal_(0.0, 1.0 / math.sqrt(d))
        if cfg.act:                          # created last: every other initial weight matches the runs without ACT
            self.q_head = nn.Linear(d, 1, dtype=self.forward_dtype)
            with torch.no_grad():            # URM: weight 0, bias -5 (no early halting at the start)
                self.q_head.weight.zero_()
                self.q_head.bias.fill_(-5.0)

    # ------------------------------------------------------------------ parts
    def rope_angles(self):
        """[H, T, P] learned 2D RoPE angles theta_row * row + theta_col * col; prefix positions get angle 0."""
        th = self.block.rope_theta
        a = (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)
        if self.prefix_len:
            a = torch.cat([a.new_zeros(a.shape[0], self.prefix_len, a.shape[-1]), a], 1)
        return a

    def rope(self, x, tables):
        cos, sin = tables
        x0, x1 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def injection(self, batch):
        inj = self.embed(batch['inputs'].long())
        if self.puzzle_emb is not None:
            pe = self.puzzle_emb(batch['puzzle_identifiers'])
            if self.config.puzzle_emb_mode == 'film':            # read by block_step, not injected
                self.task_gamma = self.block.task_film_proj(pe.to(self.block.task_film_proj.weight.dtype))
                return inj
            if self.prefix_len:                                  # URM/HRM: concatenated in front, not added
                pe = F.pad(pe, (0, self.prefix_len * self.d - pe.shape[-1])).view(-1, self.prefix_len, self.d)
                return torch.cat([pe.to(inj.dtype), inj], 1)
            pe = F.pad(pe, (0, self.d - pe.shape[-1]))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def use_kernel(self, device, dtype):
        if (self.probe is not None or self.config.outer_memory or not self.config.hebb or self.config.event_stdp
                or self.config.linear_attn or device.type != 'cuda' or dtype == torch.float64):
            return False
        if self.config.kernel == 'auto':
            return torch.cuda.get_device_capability(device) >= (8, 0)
        return self.config.kernel == 'triton'

    # ------------------------------------------------------------------ memory
    def memory_step(self, q, k, v, state):
        """One block of the STDP memory in FP32 (FP64 on the test path). state = (clock, E_K, E_V, W), already
        zeroed for fresh lanes. Returns read [B,H,T,D] and the new state."""
        cfg = self.config
        if cfg.event_stdp:
            return self.event_memory_step(q, k, v, state)
        if cfg.kv_ema:
            return self.kv_ema_step(q, k, v, state)
        if cfg.exp_stdp:
            return self.exp_stdp_step(q, k, v, state)
        if cfg.bands:
            return self.bands_step(q, k, v, state)
        if cfg.linear_attn:
            return self.linear_attn_step(q, k, v, state)
        clock, ek, ev, w, _ = state
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            k_unit = k.to(dtype)
            k_unit = k_unit / (torch.linalg.vector_norm(k_unit, dim=-1, keepdim=True) + cfg.eps)
            tag_k, tag_v, count, clock = clock_step(clock.to(dtype), k_unit, v.to(dtype), cfg.tag_bound, cfg.block_omega)
            angles = self.rope_angles().to(dtype)
            if self.use_kernel(q.device, dtype):
                # The kernel takes the projections as they are (BF16 under autocast) and normalizes q, k itself.
                from .kernel import phase_clock_attention
                lam = torch.full_like(tag_k, cfg.trace_lambda)
                read, _, ek, ev = phase_clock_attention(q, k, v, ek, ev, tag_k, tag_v, lam, lam, angles,
                                                        self.block.alpha, cfg.eps, cfg.kernel_bf16)
                return read, (clock, ek, ev, None, None)
            q, v = q.to(dtype), v.to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            rope = lambda x: self.rope(x, tables)
            q_unit = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + cfg.eps)
            qr, kr = rope(q_unit), rope(k_unit)
            D = k.shape[-1]
            traces = (*ek.to(dtype).split(D, -1), *ev.to(dtype).split(D, -1))
            hebb, im_s, new = stdp_step(k_unit, kr, v, rope, tag_k, tag_v, cfg.trace_lambda, traces)
            if self.probe is not None:
                swap = self.probe(StepRecord(k_unit, kr, qr, v, tag_k, tag_v, count, traces, new, hebb, im_s, rope))
                im_s = im_s if swap is None else swap
            w, im_s = self.accumulate(w, im_s, dtype)
            alpha = self.block.alpha.to(dtype)[None, :, None, None]
            write = hebb + alpha * im_s if cfg.hebb else alpha * im_s
            read = qr @ write.transpose(-1, -2)
        return read, (clock, torch.cat(new[:2], -1), torch.cat(new[2:], -1), w, None)

    def accumulate(self, w, p, dtype):
        """W of model B (leaky or EMA form) for this block's write p; returns (W, what the read uses)."""
        cfg = self.config
        if cfg.outer_rho:
            w = cfg.outer_rho * w.to(dtype) + p
            return w, w
        if cfg.outer_eta:
            eta = self.block.eta(cfg)
            if torch.is_tensor(eta):
                eta = eta.to(dtype)[None, :, None, None]
            w = (1 - eta) * w.to(dtype) + eta * p
            return w, w
        return w, p

    def kv_ema_step(self, q, k, v, state):
        """kv_ema: W = (1 - eta) W + eta sum_t v k~^T, read = q~ W^T (kv_ema_current: q~ (sum_t v k~^T + alpha_h W)^T).
        The clock and traces are carried unchanged."""
        cfg = self.config
        clock, ek, ev, w, prev = state
        if self.probe is not None:
            raise NotImplementedError('the analysis probe reads the phase-clock model only')
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            angles = self.rope_angles().to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            qr, kr = self.rope(unit(q.to(dtype)), tables), self.rope(unit(k.to(dtype)), tables)
            hebb = v.to(dtype).transpose(-1, -2) @ kr
            w, m = self.accumulate(w, hebb, dtype)
            if cfg.kv_ema_current:
                m = hebb + self.block.alpha.to(dtype)[None, :, None, None] * m
            read = qr @ m.transpose(-1, -2)
        return read, (clock, ek, ev, w, prev)

    def exp_stdp_step(self, q, k, v, state):
        """exp_stdp: pair STDP on the scalar channels of k~ (pre) and v (post) of the same cell, with the per-head window
        L(s) = A+ (1 - lam+) lam+^s (s > 0, pre first), -A- (1 - lam-) lam-^|s| (s < 0), L(0) = 0, A- lam- = A+ lam+:

            xbar <- lam+ xbar + (1 - lam+) k~,   ybar <- lam- ybar + (1 - lam-) v         (real traces)
            dW = A+ sum_t v (lam+ xbar_prev)^T - A- sum_t (lam- ybar_prev) k~^T          (LTP at the post, LTD at the pre)
            W = (1 - eta) W + eta dW,   read = q~ W^T            (outer_eta = 0: read = q~ dW^T, this block's write only)
            exp_hebb: read = q~ (sum_t v k~^T + alpha_h W)^T     (the Hebbian term at gain 1, as in B-EMA)

        xbar, ybar are kept in the first D columns of the E_K, E_V slots; the clock is carried unchanged."""
        cfg = self.config
        clock, ek, ev, w, prev = state
        if self.probe is not None:
            raise NotImplementedError('the analysis probe reads the phase-clock model only')
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            angles = self.rope_angles().to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            qr, kr = self.rope(unit(q.to(dtype)), tables), self.rope(unit(k.to(dtype)), tables)
            v = v.to(dtype)
            D = k.shape[-1]
            lp, lm, ap, am = (x.to(dtype)[None, :, None, None] for x in self.block.window())
            xbar, ybar = ek[..., :D].to(dtype), ev[..., :D].to(dtype)
            dw = ap * (v.transpose(-1, -2) @ (lp * xbar)) - am * ((lm * ybar).transpose(-1, -2) @ kr)
            w, m = self.accumulate(w, dw, dtype)
            if cfg.exp_hebb:
                m = v.transpose(-1, -2) @ kr + self.block.alpha.to(dtype)[None, :, None, None] * m
            read = qr @ m.transpose(-1, -2)
            ek = torch.cat([lp * xbar + (1 - lp) * kr, ek[..., D:].to(dtype)], -1)
            ev = torch.cat([lm * ybar + (1 - lm) * v, ev[..., D:].to(dtype)], -1)
        return read, (clock, ek, ev, w, prev)

    def linear_attn_step(self, q, k, v, state):
        """linear_attn: read = q~ (sum_t v k~^T)^T, the cells of this block only; every carried state is unchanged."""
        cfg = self.config
        if self.probe is not None:
            raise NotImplementedError('the analysis probe reads the phase-clock model only')
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            angles = self.rope_angles().to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            qr, kr = self.rope(unit(q.to(dtype)), tables), self.rope(unit(k.to(dtype)), tables)
            read = qr @ (v.to(dtype).transpose(-1, -2) @ kr).transpose(-1, -2)
        return read, state

    def bands_step(self, q, k, v, state):
        """bands: the complex-pole pair window (module docstring). E_K, E_V hold (Re, Im) x R bands of [T,D] traces
        of k~ and v; the bands are merged before the token sums, so S costs two [D,T] @ [T,D] products for any R:

            S = v^T sum_r Re(a_r E_K,r) + (sum_r Re(b_r D_V,r))^T k~

        The clock is carried unchanged."""
        cfg = self.config
        clock, ek, ev, w, prev = state
        if self.probe is not None:
            raise NotImplementedError('the analysis probe reads the phase-clock model only')
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            angles = self.rope_angles().to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            qr, kr = self.rope(unit(q.to(dtype)), tables), self.rope(unit(k.to(dtype)), tables)
            v = v.to(dtype)
            B, H, T, D = v.shape
            zr, zi, gamma, ar, ai, br, bi = (x.to(dtype)[None, :, None, :, None] for x in self.block.bands())
            ek_re, ek_im = ek.to(dtype).reshape(B, H, T, 2, cfg.bands, D).unbind(3)
            ev_re, ev_im = ev.to(dtype).reshape(B, H, T, 2, cfg.bands, D).unbind(3)
            ek_re, ek_im = zr * ek_re - zi * ek_im + gamma * kr.unsqueeze(3), zr * ek_im + zi * ek_re   # this block in
            dv_re, dv_im = zr * ev_re - zi * ev_im, zr * ev_im + zi * ev_re                              # this block out
            k_mix = (ar * ek_re - ai * ek_im).sum(3)
            v_mix = (br * dv_re - bi * dv_im).sum(3)
            s = v.transpose(-1, -2) @ k_mix + v_mix.transpose(-1, -2) @ kr
            w, m = self.accumulate(w, s, dtype)
            read = qr @ ((v.transpose(-1, -2) @ kr if cfg.hebb else s) + m).transpose(-1, -2)
            ev_re = dv_re + gamma * v.unsqueeze(3)
            ek = torch.stack((ek_re, ek_im), 3).reshape(B, H, T, -1)
            ev = torch.stack((ev_re, dv_im), 3).reshape(B, H, T, -1)
        return read, (clock, ek, ev, w, prev)

    def event_memory_step(self, q, k, v, state):
        """event_stdp: one block of STDP between changes (events) of q^, k^, v, plus the Hebbian read of the states.
        state = (clock, E_K, E_V, W, prev); only n (the clock's last channel) is used from the clock."""
        cfg = self.config
        clock, ek, ev, w, prev = state
        if self.probe is not None:
            raise NotImplementedError('the analysis probe reads the phase-clock model only')
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            q_unit, k_unit, v = unit(q.to(dtype)), unit(k.to(dtype)), v.to(dtype)
            clock = clock.to(dtype)
            count = clock[..., -1:]
            started = (count > 0).to(dtype)
            theta = (F.softplus(self.block.event_theta_raw.to(dtype)) if cfg.event_theta > 0 else None)

            def event(d, i):                     # soft threshold per vector: d r2 / (r2 + theta_i^2), direction kept
                if theta is not None:
                    r2 = d.square().sum(-1, keepdim=True)
                    d = d * (r2 / (r2 + theta[i][None, :, None, None]))
                return d * started

            D = k.shape[-1]
            pq, pk, pv = prev.to(dtype).split(D, -1)
            e_q, e_k, e_v = event(q_unit - pq, 0), event(k_unit - pk, 0), event(v - pv, 1)
            angles = self.rope_angles().to(dtype)
            tables = angles.cos()[None], angles.sin()[None]
            rope = lambda x: self.rope(x, tables)
            tag = (count * cfg.block_omega).expand(*count.shape[:-1], D // 2)
            traces = (*ek.to(dtype).split(D, -1), *ev.to(dtype).split(D, -1))
            out = stdp_step(e_k, rope(e_k), e_v, rope, tag, tag, cfg.trace_lambda, traces, real=cfg.event_even)
            p, new = out[1], out[2]
            if cfg.event_even:
                p = p + self.block.gamma_even.to(dtype)[None, :, None, None] * out[3]
            w, m = self.accumulate(w, p, dtype)
            alpha = self.block.alpha.to(dtype)[None, :, None, None]
            read = rope(e_q) @ (alpha * m).transpose(-1, -2)
            if cfg.hebb:
                kr = rope(k_unit)
                read = read + rope(q_unit) @ (v.transpose(-1, -2) @ kr).transpose(-1, -2)
            clock = torch.cat((clock[..., :-1], count + 1), -1)
        return read, (clock, torch.cat(new[:2], -1), torch.cat(new[2:], -1), w,
                      torch.cat((q_unit, k_unit, v), -1))

    def block_step(self, h, inj, state):
        """One recurrent block: returns the new hidden state and memory state."""
        L = self.block
        u = h + self.embed_scale * inj
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.dh).transpose(1, 2)
        q, k, v = heads(L.q_proj(u)), heads(L.k_proj(u)), heads(L.v_proj(u))
        extended = len(state) > 5                      # (memory state..., fast_w); a bare memory state is accepted too
        state, fast_w = state[:5], (state[5] if extended else None)
        read, state = self.memory_step(q, k, v, state)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(u.dtype)
        h = rms_norm(u + L.out_proj(read))
        g, a = L.ffn_gate_up(h).chunk(2, dim=-1)
        if self.config.task_film:
            g = g * (1 + L.task_film[self.task_ids].to(g.dtype)[:, None, :])
        if self.config.task_film_up:
            a = a * (1 + L.task_film_up[self.task_ids].to(a.dtype)[:, None, :])
        if self.config.puzzle_emb_mode == 'film' and self.puzzle_emb is not None:
            g = g * (1 + self.task_gamma.to(g.dtype)[:, None, :])
        if self.config.task_lowrank:
            u, v = L.task_u[self.task_ids].to(g.dtype), L.task_v[self.task_ids].to(g.dtype)    # [B,r,I], [B,r,d]
            g = g + torch.einsum('btd,brd,bri->bti', h.to(g.dtype), v, u)
        m = F.silu(g) * a
        out = L.ffn_down(m)
        if self.config.fast_ffn:
            out, fast_w = self.fast_ffn_step(h, m, out, fast_w)
        return rms_norm(h + out), ((*state, fast_w) if extended else state)

    def fast_ffn_step(self, h, m, out, fast_w):
        """Fast-weight FFN (ModelConfig.fast_ffn): one delta-rule step of dW on this block's (phi~, v_f) pairs, then
        the read W_fo dW phi~ added to the FFN output. FP32 (FP64 on the test path)."""
        L, cfg = self.block, self.config
        with torch.autocast(device_type=h.device.type, enabled=False):
            dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
            phi = m.to(dtype)
            phi = phi / (torch.linalg.vector_norm(phi, dim=-1, keepdim=True) + cfg.eps)        # [B, T, I]
            vf = h.to(dtype) @ L.ffw_v.to(dtype).T                                           # [B, T, r]
            eta = torch.sigmoid(L.ffw_eta_raw.to(dtype)) / phi.shape[1]
            lam = torch.sigmoid(L.ffw_decay_raw.to(dtype))
            w = fast_w.to(dtype)                                                             # [B, r, I]
            err = vf - phi @ w.transpose(-1, -2)                                             # [B, T, r]
            w = (1 - lam) * w + eta * err.transpose(-1, -2) @ phi
            read = (phi @ w.transpose(-1, -2)) @ L.ffw_out.to(dtype).T                       # [B, T, d]
        return out + read.to(out.dtype), w

    def initial_state(self, B, device, dtype=torch.float32):
        P, D = self.dh // 2, self.dh
        z = lambda *s: torch.zeros(B, self.H, *s, device=device, dtype=dtype)
        trace = 2 * D * max(self.config.bands, 1)
        return (z(self.n_pos, 6 * P + 1), z(self.n_pos, trace), z(self.n_pos, trace),
                z(D, D) if self.config.outer_memory else None,
                z(self.n_pos, 3 * D) if self.config.event_stdp else None)

    # ------------------------------------------------------------------ segments
    def initial_carry(self, batch):
        B = batch['inputs'].shape[0]
        device = self.init_hidden.device
        return Carry(hidden=torch.empty(B, self.n_pos, self.d, dtype=self.forward_dtype, device=device),
                     steps=torch.zeros(B, dtype=torch.int32, device=device),
                     halted=torch.ones(B, dtype=torch.bool, device=device),
                     data={k: torch.empty_like(v, device=device) for k, v in batch.items()})

    def _amp(self, device):
        if not (self.config.amp and device.type == 'cuda' and self.forward_dtype == torch.float32):
            return False
        try:
            return torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            return torch.cuda.get_device_capability(device)[0] >= 8

    def forward(self, carry: Carry, batch: dict):
        """One segment (blocks_per_seg blocks). Halted lanes take the puzzle in `batch` and start fresh
        (URM lane replacement); returns the new carry (detached) and the logits [B, T, vocab]."""
        fresh = carry.halted
        data = {k: torch.where(fresh.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                for k, v in carry.data.items()}
        h = torch.where(fresh.view(-1, 1, 1), self.init_hidden, carry.hidden)
        B, device = h.shape[0], h.device
        dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        if carry.clock is None:
            state = self.initial_state(B, device, dtype)
        else:
            mask = fresh.view(-1, 1, 1, 1)
            state = tuple(None if x is None else torch.where(mask, torch.zeros_like(x), x)
                          for x in (carry.clock, carry.key_trace, carry.value_trace, carry.stdp_w, carry.prev))
        fast = None
        if self.config.fast_ffn:
            fast = carry.fast_w
            fast = (torch.zeros(B, self.config.fast_ffn, self.block.ffn_down.in_features, device=device, dtype=dtype)
                    if fast is None else torch.where(fresh.view(-1, 1, 1), torch.zeros_like(fast), fast))
        state = (*state, fast)
        amp = self._amp(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            inj = self.injection(data)
            self.task_ids = data['puzzle_identifiers'].long()           # read by block_step (task_film)
            ckpt = self.config.act_ckpt and torch.is_grad_enabled()
            step = self.block_fn or self.block_step
            n_ckpt = self.config.blocks_per_seg if self.config.act_ckpt_blocks < 0 else self.config.act_ckpt_blocks
            for i in range(self.config.blocks_per_seg):
                if ckpt and i < n_ckpt:
                    h, state = torch.utils.checkpoint.checkpoint(step, h, inj, state, use_reentrant=False)
                else:
                    h, state = step(h, inj, state)
            logits = self.classifier(h[:, self.prefix_len:])
        if amp:
            h, logits = h.float(), logits.float()
        steps = torch.where(fresh, 0, carry.steps) + 1
        halted = steps >= self.config.loops
        if self.config.act:
            q = self.q_head(h[:, 0].to(self.q_head.weight.dtype)).float().squeeze(-1)   # prefix token (URM: position 0)
            self.q_halt_logits = q
            if self.training and self.config.loops > 1:
                with torch.no_grad():
                    halted = halted | (q > 0)
                    explore = torch.rand_like(q) < self.config.act_explore
                    min_steps = explore * torch.randint_like(steps, low=2, high=self.config.loops + 1)
                    halted = halted & (steps >= min_steps)
        clock, ek, ev, w, prev, fast = (None if x is None else x.detach() for x in state)
        new = Carry(hidden=h.detach(), steps=steps, halted=halted, data=data,
                    clock=clock, key_trace=ek, value_trace=ev, stdp_w=w, prev=prev, fast_w=fast)
        return new, logits


def convert_legacy_state_dict(sd: dict, cfg: ModelConfig) -> dict:
    """Weights of a pre-refactor checkpoint (lt/research_phase_clock_stdp.py run A, or the Kaggle model-B cell)
    -> this module's names. The unused legacy parameters are dropped; the frozen retention is checked."""
    if not any(k.startswith('model.inner.') for k in sd):
        return sd
    rename = {'embed.weight': 'embed.weight', 'w_cls.weight': 'classifier.weight', 'w_cls.bias': 'classifier.bias',
              'puzzle_emb.weights': 'puzzle_emb.weights', 'init_hidden': 'init_hidden',
              'layers.0.q_proj.weight': 'block.q_proj.weight', 'layers.0.k_proj.weight': 'block.k_proj.weight',
              'layers.0.v_proj.weight': 'block.v_proj.weight', 'layers.0.out_proj.weight': 'block.out_proj.weight',
              'layers.0.b_gate_up.weight': 'block.ffn_gate_up.weight', 'layers.0.b_down.weight': 'block.ffn_down.weight',
              'layers.0.theta': 'block.rope_theta', 'layers.0.stdp_alpha': 'block.alpha'}
    dropped = ('layers.0.trace_lam_raw', 'layers.0.theta_k_raw', 'layers.0.theta_v_raw', 'layers.0.clock_lam_raw')
    out = {}
    for key, value in sd.items():
        name = key[len('model.inner.'):]
        if name in rename:
            out[rename[name]] = value
        elif name not in dropped:
            raise KeyError(f'unexpected legacy key {key}')
    lam_raw = sd.get('model.inner.layers.0.clock_lam_raw')
    if lam_raw is not None:
        lam = torch.exp(-F.softplus(lam_raw.double()))
        if not torch.allclose(lam, torch.full_like(lam, cfg.trace_lambda), atol=1e-6):
            raise ValueError(f'legacy retention {float(lam.mean()):.4f} != trace_lambda {cfg.trace_lambda}')
    return out
