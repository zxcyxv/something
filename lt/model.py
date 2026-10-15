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
    read  = q~ (sum_t v k~^T + alpha_h W)^T                  (hebb=False drops sum_t v k~^T)
    h'    = RMSNorm(u + W_O read);   h = RMSNorm(h' + W_D(silu(g) * a)),  [g; a] = W_GU h'

Every (post = V, pre = K) event pair of the same cell since the puzzle started is counted exactly once with weight
(1 - lam) lam^k e^{i(tau_post - tau_pre)}, k = block difference, so alpha_h Im S is an odd (STDP) window on top of the
Hebbian carrier. With omegaT > 2c and K omegaT + 2c < pi every pair up to K blocks apart takes its LTP/LTD sign from
the block order. c, omegaT, lam and rho are fixed constants; alpha_h is trained (init 1).

The clock, traces, W and the read are FP32 (FP64 in tests); projections and the FFN follow BF16 autocast where the
GPU has native BF16. kernel 'triton' runs the memory step through lt/kernel.py (rho = 0 only).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

TWO_PI = 2 * math.pi


@dataclass
class ModelConfig:
    batch_size: int = 128             # per process (the puzzle-embedding buffer)
    seq_len: int = 81
    grid: int = 9
    vocab_size: int = 11              # 0 pad, 1 blank, 2..10 digits 1..9
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 832
    hidden_size: int = 832
    num_heads: int = 8
    blocks_per_seg: int = 8
    loops: int = 16                   # segments per puzzle
    mlp_expansion: float = 4.0
    eps: float = 1e-4                 # Q/K L2 denominator ||x|| + eps
    tag_bound: float = 0.14           # c
    block_omega: float = 0.34         # omegaT
    trace_lambda: float = 0.5         # lam_K = lam_V
    outer_rho: float = 0.0            # rho of W (0: model A)
    alpha_init: float = 1.0           # alpha_h at initialization
    hebb: bool = True                 # include the Hebbian carrier sum_t v k~^T in the write (False: G = alpha Im S)
    kernel: str = 'auto'              # auto | torch | triton
    kernel_bf16: bool = True          # BF16 dot operands in the fused kernel (FP32 accumulation)
    amp: bool = True                  # BF16 autocast for projections and FFN where native
    forward_dtype: str = 'float32'    # float64 for tests

    def __post_init__(self):
        if self.seq_len != self.grid ** 2:
            raise ValueError('seq_len must equal grid**2')
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError('the head dimension must be even')
        if not 0 <= self.puzzle_emb_ndim <= self.hidden_size:
            raise ValueError('puzzle_emb_ndim must lie in [0, hidden_size]')
        if not (0 < self.trace_lambda < 1 and 0 <= self.outer_rho < 1 and self.tag_bound >= 0 and self.eps > 0):
            raise ValueError('need 0 < trace_lambda < 1, 0 <= outer_rho < 1, tag_bound >= 0, eps > 0')
        if self.kernel not in ('auto', 'torch', 'triton'):
            raise ValueError('kernel must be auto, torch or triton')
        if self.kernel == 'triton' and (self.outer_rho or not self.hebb):
            raise ValueError('the fused kernel implements outer_rho = 0 with the Hebbian term only')

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
    stdp_w: Optional[torch.Tensor] = None           # W [B, H, D, D] (model B only)
    fresh: Optional[torch.Tensor] = None            # [B] lanes that start a puzzle in this segment


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


def stdp_step(k_unit, kr, v, rope, tag_k, tag_v, lam, traces):
    """One block of the phase-tagged traces; returns (hebb, Im S, new traces), token sums [B,H,D,D].

    k_unit: unit key before RoPE; kr = rope(k_unit); v raw; tag_k, tag_v [B,H,T,P]; lam: retention (scalar or
    broadcastable to [B,H,T,P]); traces = (Re E_K, Im E_K, Re E_V, Im E_V), zeroed for fresh lanes.
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
    ev_re = dv_re + (1 - lam) * v * cv
    ev_im = dv_im + (1 - lam) * v * sv
    return hebb, im_s, (ek_re, ek_im, ev_re, ev_im)


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
        inter = int(cfg.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.ffn_down.weight.zero_()
        self.rope_theta = nn.Parameter(torch.zeros(H, d // H // 2, 2))     # drawn last (see LT.__init__)
        self.alpha = nn.Parameter(torch.full((H,), float(cfg.alpha_init)))


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
        T, g = cfg.seq_len, cfg.grid
        rope_dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        self.register_buffer('pos_row', (torch.arange(T) // g).to(rope_dtype), persistent=False)
        self.register_buffer('pos_col', (torch.arange(T) % g).to(rope_dtype), persistent=False)
        self.probe: Optional[Callable[[StepRecord], Optional[torch.Tensor]]] = None

    # ------------------------------------------------------------------ parts
    def rope_angles(self):
        """[H, T, P] learned 2D RoPE angles theta_row * row + theta_col * col."""
        th = self.block.rope_theta
        return (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)

    def rope(self, x, tables):
        cos, sin = tables
        x0, x1 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def injection(self, batch):
        inj = self.embed(batch['inputs'].long())
        if self.puzzle_emb is not None:
            pe = self.puzzle_emb(batch['puzzle_identifiers'])
            pe = F.pad(pe, (0, self.d - pe.shape[-1]))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def use_kernel(self, device, dtype):
        if (self.probe is not None or self.config.outer_rho or not self.config.hebb or device.type != 'cuda'
                or dtype == torch.float64):
            return False
        if self.config.kernel == 'auto':
            return torch.cuda.get_device_capability(device) >= (8, 0)
        return self.config.kernel == 'triton'

    # ------------------------------------------------------------------ memory
    def memory_step(self, q, k, v, state):
        """One block of the STDP memory in FP32 (FP64 on the test path). state = (clock, E_K, E_V, W), already
        zeroed for fresh lanes. Returns read [B,H,T,D] and the new state."""
        cfg = self.config
        clock, ek, ev, w = state
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
                return read, (clock, ek, ev, None)
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
            if cfg.outer_rho:
                w = cfg.outer_rho * w.to(dtype) + im_s
                im_s = w
            alpha = self.block.alpha.to(dtype)[None, :, None, None]
            write = hebb + alpha * im_s if cfg.hebb else alpha * im_s
            read = qr @ write.transpose(-1, -2)
        return read, (clock, torch.cat(new[:2], -1), torch.cat(new[2:], -1), w)

    def block_step(self, h, inj, state):
        """One recurrent block: returns the new hidden state and memory state."""
        L = self.block
        u = h + self.embed_scale * inj
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.dh).transpose(1, 2)
        q, k, v = heads(L.q_proj(u)), heads(L.k_proj(u)), heads(L.v_proj(u))
        read, state = self.memory_step(q, k, v, state)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(u.dtype)
        h = rms_norm(u + L.out_proj(read))
        g, a = L.ffn_gate_up(h).chunk(2, dim=-1)
        return rms_norm(h + L.ffn_down(F.silu(g) * a)), state

    def initial_state(self, B, device, dtype=torch.float32):
        P, D = self.dh // 2, self.dh
        z = lambda *s: torch.zeros(B, self.H, *s, device=device, dtype=dtype)
        return (z(self.config.seq_len, 6 * P + 1), z(self.config.seq_len, 2 * D), z(self.config.seq_len, 2 * D),
                z(D, D) if self.config.outer_rho else None)

    # ------------------------------------------------------------------ segments
    def initial_carry(self, batch):
        B = batch['inputs'].shape[0]
        device = self.init_hidden.device
        return Carry(hidden=torch.empty(B, self.config.seq_len, self.d, dtype=self.forward_dtype, device=device),
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
                          for x in (carry.clock, carry.key_trace, carry.value_trace, carry.stdp_w))
        amp = self._amp(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            inj = self.injection(data)
            for _ in range(self.config.blocks_per_seg):
                h, state = self.block_step(h, inj, state)
            logits = self.classifier(h)
        if amp:
            h, logits = h.float(), logits.float()
        steps = torch.where(fresh, 0, carry.steps) + 1
        clock, ek, ev, w = (None if x is None else x.detach() for x in state)
        new = Carry(hidden=h.detach(), steps=steps, halted=steps >= self.config.loops, data=data,
                    clock=clock, key_trace=ek, value_trace=ev, stdp_w=w)
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
