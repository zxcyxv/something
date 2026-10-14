# -*- coding: utf-8 -*-
# LT phase-clock STDP "B" -- Kaggle single cell (same harness as lt/train.py, self-contained).
# Paste this whole file into ONE notebook cell and run it. Settings live in CFG below.
# Data: attach sudoku_lt_1k.npz as a Kaggle Input (or set CFG['data_npz']); see README_phaseclock_bounded_leaky.md.
#
# Model B = 2026-10-13 run 3 (displacement clock, plain torch path, SwiGLU, Q/K L2, token sums, alpha_h init 1)
# with three fixed changes (constants below, no extra trainable parameters):
#   1. bounded latency + block spacing:  tag_r = c sin(phi_r) + n omegaT   (r = K, V)
#        phi_r = run-3 displacement tag (accumulated wrapped pair-angle displacement since the puzzle start),
#        n = blocks this lane has run since its puzzle started (0 in the first block); c = 0.14, omegaT = 0.34.
#   2. trace retention lambda_K = lambda_V = 0.5, frozen (clock_lam_raw has requires_grad=False).
#   3. leaky outer integral of the STDP increment:  W_b = rho W_{b-1} + Im S_b  (rho = 0.5);
#        read = RoPE(q) (Hebb_b + alpha_h W_b)^T  instead of  RoPE(q) (Hebb_b + alpha_h Im S_b)^T.
# Carry: coupling = cat(phi_K, phi_V, zK_prev, zV_prev, n) [B,H,T,6P+1]; key_trace/value_trace = phase-tagged
#        complex traces (re, im) [B,H,T,2D]; stdp_w = W [B,H,D,D]. All FP32, zeroed for fresh puzzles, detached at
#        segment boundaries, saved with every training checkpoint (exact resume).
# Harness (unchanged from lt/train.py): batch 128, 16 segments x 8 blocks, carry detached per segment, URM-style
# lane replacement (fresh lanes), stablemax loss, AdamATan2, lr 1e-4 constant after 2000 warmup, weight decay 1.0,
# EMA 0.999, eval every 250 epochs (held-out 2048, EMA, 16 segments), milestone every 10000 steps (512 puzzles,
# 128 segments). No activation checkpointing. Q/K/V/out/FFN projections follow AMP ('auto': BF16 where native BF16
# exists, otherwise FP32 -- T4/P100 run FP32; FP16 is never used). Clock, traces, W, read stay FP32.
# Outputs go to CFG['out_dir']; rerunning the cell resumes from the latest step_*.pt there.
# max_steps is an ABSOLUTE stopping step.

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

CFG = {'data_npz': '/kaggle/input/datasets/jrjinwoo/sudoku-lt-1k/sudoku_lt_1k.npz',  # missing -> local fallback / search
 'out_dir': '/kaggle/working/lt_phaseclock_bounded_leaky',
 'resume_from': None,            # checkpoint file or folder (e.g. a previous session's output attached as Input)
 'require_resume': False,
 'max_steps': 50800,             # absolute stopping step
 'max_hours': 11.5,              # save and stop cleanly before the Kaggle session limit
 'save_every_steps': 2000,
 'keep_last': 2,
 'milestone_every': 10000,
 'milestone_extrap_segs': 128,
 'milestone_extrap_n': 512,
 'eval_interval': 250,           # epochs between held-out evaluations (250 epochs = 1953 steps)
 'test_size': 2048,
 'log_every': 250,
 'run_selftests': True,          # CPU FP64 self-test of model B before training (a few seconds)
 'selftest_only': False,         # True: run only the self-test and stop
 # ---- model B constants (fixed, not trained) ----
 'b_tag_bound': 0.14,            # c
 'b_block_omega': 0.34,          # omegaT
 'b_trace_lambda': 0.5,          # lambda_K = lambda_V
 'b_outer_rho': 0.5,             # rho
 'b_alpha_init': 1.0,            # alpha_h initial value (trained)
 # ---- architecture / harness (run 3 = configs/free_phase_window_research.json on top of lt/train.py CFG) ----
 'num_aug': 1000,
 'hidden_size': 832,
 'num_heads': 8,
 'loops': 16,
 'blocks_per_seg': 8,
 'num_layers': 1,
 'grid': 9,
 'vocab_size': 11,
 'mlp_expansion': 4.0,
 'eps': 0.0001,
 'puzzle_emb_ndim': 832,
 'memory_type': 'kv_stdp',
 'trace_decay_init': 0.1,        # init of the unused trace_lam_raw (kept so initial draws match run 3)
 'trace_decay_mode': 'pair',
 'kv_write_reduction': 'mean',   # protocol value; model B writes token SUMS itself (as run 3)
 'kv_qk_rmsnorm': False,
 'kv_qk_l2norm': False,          # Q/K L2 normalization happens inside model B (as run 3)
 'kv_projection_fp32': False,
 'rope_type': 'learned_2d',
 'rope_base': 10000.0,
 'forward_dtype': 'float32',
 'amp': True,
 'amp_dtype': 'auto',            # auto: BF16 if native, else FP32 (T4/P100). 'bfloat16' errors where unsupported.
 'compile': 'auto',              # auto: torch.compile on CUDA capability >= 7.0 (T4 yes, P100 no -> eager)
 'inductor_no_persist': True,
 'global_batch_size': 128,
 'epochs': 50000,
 'lr': 0.0001,
 'lr_min_ratio': 1.0,
 'lr_warmup_steps': 2000,
 'lr_rewarm_start': None,
 'lr_rewarm_steps': 10000,
 'lr_rewarm_from_ratio': 0.1,
 'weight_decay': 1.0,
 'qkv_no_decay': False,
 'beta1': 0.9,
 'beta2': 0.95,
 'puzzle_emb_lr': 0.0001,
 'puzzle_emb_weight_decay': 1.0,
 'grad_accum_steps': 1,
 'q_weight': 0.5,
 'seed': 0,
 'ema': True,
 'ema_rate': 0.999,
 'stop_check_every': 25,
 'dataloader_workers': 1,
 'num_processes': 'auto',        # one process per visible GPU (torchrun when > 1); batch is split across ranks
 'late_sup_prob': 0.0,
 'late_sup_min': 16,
 'late_sup_max': 112,
 'nograd_fixed': 0,              # full BPTT over the 8 blocks of a segment (run 3)
 'nograd_every': 0,
 'nograd_start': 0,
 'nograd_max': 16}

# Self-contained trainer source (run in a separate process; no repository imports)
_TRAINER_SOURCE = r'''# -*- coding: utf-8 -*-
"""LT Sudoku trainer for phase-clock STDP model B (self-contained; no repository imports).

Harness: lt/train.py (data, augmentation pool, stablemax loss, AdamATan2, EMA, checkpoint/resume with per-rank carry
and RNG, evaluation, milestone extrapolation, torchrun launcher). Model: lt/research_phase_clock_stdp.py run 3
(displacement clock, torch path, SwiGLU boundary, two URM residual RMSNorms, Q/K L2, token-sum writes) with the
bounded-latency + block-spacing tags, frozen lambda = 0.5 and the leaky outer integral W of the STDP increment.
One optimizer step is one segment; hidden, clock, traces and W detach only at segment boundaries.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

MODEL_ID_KV_STDP = "lt-kv-stdp-fast-weight-v1"
RESEARCH_VARIANT = "phaseclock_disp_bounded_leakyW_lfrozen_swiglu_torch_qkl2_sum"
MODEL_ID_B = MODEL_ID_KV_STDP + ":research-" + RESEARCH_VARIANT
IGNORE_LABEL_ID = -100
_STOP_REQUESTED = False


def model_id_of(cfg):
    return MODEL_ID_B


def model_tag_of(cfg):
    return "phaseclock_bounded_leaky"

# DEFAULT_CFG is the notebook CFG (the launcher passes the full CFG as JSON on top of it).
DEFAULT_CFG = {'data_npz': None,
 'out_dir': '/kaggle/working/lt_phaseclock_bounded_leaky',
 'resume_from': None,
 'require_resume': False,
 'max_steps': 50800,
 'max_hours': 11.5,
 'save_every_steps': 2000,
 'keep_last': 2,
 'milestone_every': 10000,
 'milestone_extrap_segs': 128,
 'milestone_extrap_n': 512,
 'eval_interval': 250,
 'test_size': 2048,
 'log_every': 250,
 'run_selftests': True,
 'selftest_only': False,
 'b_tag_bound': 0.14,
 'b_block_omega': 0.34,
 'b_trace_lambda': 0.5,
 'b_outer_rho': 0.5,
 'b_alpha_init': 1.0,
 'num_aug': 1000,
 'hidden_size': 832,
 'num_heads': 8,
 'loops': 16,
 'blocks_per_seg': 8,
 'num_layers': 1,
 'grid': 9,
 'vocab_size': 11,
 'mlp_expansion': 4.0,
 'eps': 0.0001,
 'puzzle_emb_ndim': 832,
 'memory_type': 'kv_stdp',
 'trace_decay_init': 0.1,
 'trace_decay_mode': 'pair',
 'kv_write_reduction': 'mean',
 'kv_qk_rmsnorm': False,
 'kv_qk_l2norm': False,
 'kv_projection_fp32': False,
 'rope_type': 'learned_2d',
 'rope_base': 10000.0,
 'forward_dtype': 'float32',
 'amp': True,
 'amp_dtype': 'auto',
 'compile': 'auto',
 'inductor_no_persist': True,
 'global_batch_size': 128,
 'epochs': 50000,
 'lr': 0.0001,
 'lr_min_ratio': 1.0,
 'lr_warmup_steps': 2000,
 'lr_rewarm_start': None,
 'lr_rewarm_steps': 10000,
 'lr_rewarm_from_ratio': 0.1,
 'weight_decay': 1.0,
 'qkv_no_decay': False,
 'beta1': 0.9,
 'beta2': 0.95,
 'puzzle_emb_lr': 0.0001,
 'puzzle_emb_weight_decay': 1.0,
 'grad_accum_steps': 1,
 'q_weight': 0.5,
 'seed': 0,
 'ema': True,
 'ema_rate': 0.999,
 'stop_check_every': 25,
 'dataloader_workers': 1,
 'num_processes': 'auto',
 'late_sup_prob': 0.0,
 'late_sup_min': 16,
 'late_sup_max': 112,
 'nograd_fixed': 0,
 'nograd_every': 0,
 'nograd_start': 0,
 'nograd_max': 16}


# 1. Initialization / sparse puzzle embedding (lt/train.py) ---------------------------------
def trunc_normal_init_(tensor, std=1.0, lower=-2.0, upper=2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            a, b = math.erf(lower / math.sqrt(2)), math.erf(upper / math.sqrt(2))
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pu, pl = c * math.exp(-0.5 * upper**2), c * math.exp(-0.5 * lower**2)
            cs = std / math.sqrt(1 - (upper*pu - lower*pl)/z - ((pu-pl)/z)**2)
            tensor.uniform_(a, b).erfinv_().mul_(math.sqrt(2)*cs)
            tensor.clip_(lower*cs, upper*cs)
    return tensor


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.register_buffer("weights", trunc_normal_init_(
            torch.empty(num_embeddings, embedding_dim), std=init_std))
        self.register_buffer("local_weights", torch.zeros(
            batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.register_buffer("local_ids", torch.zeros(batch_size, dtype=torch.long), persistent=False)

    def forward(self, inputs):
        if not self.training:
            return self.weights[inputs.long()].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs.long()])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, embedding, world_size, lr=0.0, weight_decay=0.0):
        self.embedding = embedding
        super().__init__([embedding.weights, embedding.local_weights, embedding.local_ids],
                         dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        e = self.embedding
        grad, ids = e.local_weights.grad, e.local_ids
        # Called only after the common finite-gradient check.
        if grad is None:
            grad = torch.zeros_like(e.local_weights)
        if group["world_size"] > 1:
            all_g = torch.empty((len(grad)*group["world_size"], grad.shape[1]), device=grad.device)
            all_i = torch.empty(len(ids)*group["world_size"], dtype=ids.dtype, device=ids.device)
            dist.all_gather_into_tensor(all_g, grad.contiguous())
            dist.all_gather_into_tensor(all_i, ids.contiguous())
            grad, ids = all_g, all_i
        unique, inverse = ids.unique(return_inverse=True)
        summed = torch.zeros((len(unique), grad.shape[1]), dtype=grad.dtype, device=grad.device)
        summed.scatter_add_(0, inverse[:, None].expand_as(grad), grad)
        p = e.weights[unique]
        p.mul_(1 - group["lr"]*group["weight_decay"]).add_(summed.sign(), alpha=-group["lr"])
        e.weights[unique] = p



# 2. Model B --------------------------------------------------------------------
@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    coupling: Optional[torch.Tensor] = None       # clock state cat(phi_K, phi_V, zK_prev, zV_prev, n) [B,H,T,6P+1]
    fresh: Optional[torch.Tensor] = None          # [B] bool: this lane starts a new puzzle in this segment
    trace: Optional[torch.Tensor] = None          # unused (address-model slot), always None
    key_trace: Optional[torch.Tensor] = None      # cat(Re E_K, Im E_K) [B,H,T,2D] (before RoPE)
    value_trace: Optional[torch.Tensor] = None    # cat(Re E_V, Im E_V) [B,H,T,2D]
    stdp_w: Optional[torch.Tensor] = None         # leaky outer integral W of Im S [B,H,D(V),D(K)]


@dataclass
class LTConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int = 832
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8
    num_layers: int = 1
    mlp_expansion: float = 4.0
    memory_type: str = "kv_stdp"
    trace_decay_init: float = 0.1   # init of the unused trace_lam_raw (identical initial draws to run 3)
    trace_decay_mode: str = "pair"
    kv_write_reduction: str = "mean"
    kv_qk_rmsnorm: bool = False
    kv_qk_l2norm: bool = False
    kv_projection_fp32: bool = False
    rope_type: str = "learned_2d"
    rope_base: float = 10000.0
    eps: float = 1e-4               # Q/K L2 denominator norm + eps
    amp: bool = True
    amp_dtype: str = "auto"
    forward_dtype: str = "float32"
    b_tag_bound: float = 0.14       # c
    b_block_omega: float = 0.34     # omegaT
    b_trace_lambda: float = 0.5     # lambda_K = lambda_V (frozen)
    b_outer_rho: float = 0.5        # rho of W
    b_alpha_init: float = 1.0       # alpha_h init (trained)
    nograd_blocks: int = 0          # set by the harness per step (0 in this protocol)

    def __post_init__(self):
        if self.memory_type != "kv_stdp":
            raise ValueError("Model B uses the kv_stdp protocol (memory_type='kv_stdp').")
        if self.kv_write_reduction != "mean":
            raise ValueError("This protocol keeps the token-mean config; model B writes token sums itself.")
        if self.kv_qk_rmsnorm or self.kv_qk_l2norm:
            raise ValueError("Model B normalizes Q/K (L2) itself; keep kv_qk_rmsnorm = kv_qk_l2norm = False.")
        if self.rope_type != "learned_2d":
            raise ValueError("Model B uses the learned 2D RoPE.")
        if self.trace_decay_mode not in ("head", "pair") or not 0 < self.trace_decay_init < 1:
            raise ValueError("trace_decay_mode head|pair and trace_decay_init in (0, 1) are required.")
        if self.num_heads < 1 or self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 2:
            raise ValueError("hidden_size / num_heads must be even.")
        if not 0 <= self.puzzle_emb_ndim <= self.hidden_size:
            raise ValueError("puzzle_emb_ndim must lie between 0 and hidden_size")
        if not (math.isfinite(self.eps) and self.eps > 0):
            raise ValueError("Q/K normalization requires finite eps > 0.")
        if self.amp_dtype not in ("auto", "bfloat16", "float32"):
            raise ValueError("amp_dtype must be auto, bfloat16, or float32")
        if not (0 < self.b_trace_lambda < 1 and 0 <= self.b_outer_rho < 1 and self.b_tag_bound >= 0
                and math.isfinite(self.b_block_omega) and math.isfinite(self.b_alpha_init)):
            raise ValueError("B constants: 0 < lambda < 1, 0 <= rho < 1, c >= 0, finite omegaT and alpha_init.")

    @classmethod
    def from_dict(cls, d: dict) -> "LTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


TWO_PI = 2 * math.pi


def wrap_phase(x):
    """Map to [-pi, pi); the gradient is 1 almost everywhere."""
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def _rep2(x):
    """[..., P] per pair -> [..., D]: both real components of a pair share the value."""
    return x.repeat_interleave(2, -1)


def _pair_displacement(x, z_prev):
    """Angular displacement of each pair of x [B,H,T,D] from the carried unit phasor z_prev [B,H,T,2P]
    (cos half, sin half; zero means no previous event -> displacement 0), and x's own unit phasor. (run 3)"""
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


def bounded_clock_step(clk, k_unit, v, bound, omega_t):
    """Model B tags for this block and the advanced clock state.

    clk = cat(phi_K, phi_V, zK_prev, zV_prev, n) [B,H,T,6P+1], already zeroed for fresh lanes.
    phi_r <- wrap(phi_r + wrap(theta_r(now) - theta_r(prev)))   (run-3 displacement tag; 0 in a lane's first block)
    tag_r  = bound * sin(phi_r) + n * omega_t,  n = blocks since the lane's puzzle started (0 in the first block).
    n * omega_t is wrapped to [-pi, pi) (identical phasor, better FP32 precision for long extrapolation).
    """
    P = k_unit.shape[-1] // 2
    phase_k, phase_v, zk, zv, count = clk.split((P, P, 2 * P, 2 * P, 1), -1)
    dk, zk = _pair_displacement(k_unit, zk)
    dv, zv = _pair_displacement(v, zv)
    phase_k = wrap_phase(phase_k + dk)
    phase_v = wrap_phase(phase_v + dv)
    spacing = wrap_phase(count * omega_t)
    tag_k = bound * torch.sin(phase_k) + spacing
    tag_v = bound * torch.sin(phase_v) + spacing
    return tag_k, tag_v, torch.cat((phase_k, phase_v, zk, zv, count + 1), -1)


def phase_clock_step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
    """One block of the run-3 phase-clock write (verbatim); all tensors share one float dtype.

    k_unit [B,H,T,D]: unit key before RoPE; kr = rope(k_unit); v [B,H,T,D]: raw value.
    phase_k, phase_v [B,H,T,P]: the tags of this block's events; lam_k, lam_v: retentions (broadcast to [B,H,T,P]).
    traces = (ek_re, ek_im, ev_re, ev_im), already zeroed for fresh lanes. Returns (hebb, im_s, new_traces);
    hebb and im_s are token sums [B,H,D,D] with V rows and K columns.
    """
    ek_re, ek_im, ev_re, ev_im = traces
    ck, sk = _rep2(phase_k.cos()), _rep2(phase_k.sin())
    cv, sv = _rep2(phase_v.cos()), _rep2(phase_v.sin())
    lam_k, lam_v = _rep2(lam_k), _rep2(lam_v)
    # K trace, tag e^{-i phi_K}; it already holds this block's event, so same-block pairs are counted here only.
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


class KVSTDPLayer(nn.Module):
    """lt/train.py KVSTDPLayer: separate Q/K/V/out projections, (unused) trace lambda, FFN, learned 2D RoPE theta."""
    def __init__(self, config):
        super().__init__()
        d, H = config.hidden_size, config.num_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.head_dim = d // H
        logit = math.log(config.trace_decay_init / (1 - config.trace_decay_init))
        self.trace_decay_mode = config.trace_decay_mode
        lam_shape = (H, self.head_dim // 2) if self.trace_decay_mode == "pair" else (H,)
        self.trace_lam_raw = nn.Parameter(torch.full(lam_shape, logit))     # unused by B (no gradient)
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()
        # Initialized at the end of the Inner so the other draws are unchanged.
        self.theta = nn.Parameter(torch.zeros(H, d // H // 2, 2))


class PhaseClockBInner(nn.Module):
    """Model B. Block (one shared layer):
      h <- h + sqrt(d) inj;  q, k, v = heads(W_q h), heads(W_k h), heads(W_v h);  q, k <- x / (||x||_2 + eps)
      qr, kr = RoPE(q), RoPE(k) (learned 2D);  tags (bounded_clock_step);  lam = exp(-softplus(clock_lam_raw)) = 0.5
      E_K <- lam E_K + (1-lam) k e^{-i tag_K};  D_V = lam E_V;  E_V <- D_V + (1-lam) v e^{+i tag_V}
      Im S = Im sum_n [(v e^{i tag_V})^T RoPE(E_K) + D_V^T (kr e^{-i tag_K})]     (token sums)
      W <- rho W + Im S;   read = qr (sum_n v kr^T + alpha_h W)^T
      h <- RMS(h + W_o read);  h <- RMS(h + D(silu(g) u)), (g, u) = U h
    Parameter creation order and random draws equal run 3 (lt/train.py KVSTDPInner -> PhaseCurrentReadInner's unused
    theta_k_raw/theta_v_raw -> constant clock_lam_raw [2,H,P] and stdp_alpha [H]), so a fixed seed gives the same
    initial weights as run 3.
    """
    norm_eps = 1e-5                   # URM RMSNorm epsilon
    phase_limit = math.pi / 2         # unused theta_k_raw / theta_v_raw (kept for identical draws)
    phase_init_limit = math.pi / 4

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        if T != g * g or config.num_layers < 1 or config.blocks_per_seg < 1:
            raise ValueError("KV memory requires seq_len=grid**2 and positive layer/block counts")
        self.d, self.H, self.dh = d, H, d // H
        self.P = self.dh // 2
        self.gamma, self.embed_scale = 1.0 / d, math.sqrt(d)
        self.embed = nn.Embedding(config.vocab_size, d)
        with torch.no_grad():
            trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
        self.w_cls = nn.Linear(d, config.vocab_size)
        self.layers = nn.ModuleList([KVSTDPLayer(config) for _ in range(config.num_layers)])
        if config.num_layers > 1:
            sd0 = self.layers[0].state_dict()
            for layer in self.layers[1:]:
                layer.load_state_dict({k: v.clone() for k, v in sd0.items()})
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if self.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, self.puzzle_emb_ndim,
                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        self.init_hidden = nn.Buffer(trunc_normal_init_(
            torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)
        positions = torch.stack((torch.arange(T) // g, torch.arange(T) % g), dim=-1)
        rope_dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        self.register_buffer("pos_u", positions[:, 0].to(rope_dtype), persistent=False)
        self.register_buffer("pos_w", positions[:, 1].to(rope_dtype), persistent=False)
        with torch.no_grad():
            self.layers[0].theta.uniform_(-math.pi / 2, math.pi / 2)
            for layer in self.layers[1:]:
                layer.theta.copy_(self.layers[0].theta)
        # PhaseCurrentReadInner's channel phases: allocated for identical draws, unused (no gradient).
        for layer in self.layers:
            for name in ("theta_k_raw", "theta_v_raw"):
                angle = torch.empty(self.H, self.dh).uniform_(-self.phase_init_limit, self.phase_init_limit)
                layer.register_parameter(name, nn.Parameter(torch.atanh(angle / self.phase_limit)))
        # Model B: frozen retention (lambda = exp(-softplus(raw)) = b_trace_lambda) and the trained STDP gain.
        rate = -math.log(config.b_trace_lambda)
        for layer in self.layers:
            layer.clock_lam_raw = nn.Parameter(torch.full((2, self.H, self.P), float(inv_softplus(rate))),
                                               requires_grad=False)
            layer.stdp_alpha = nn.Parameter(torch.full((self.H,), float(config.b_alpha_init)))
        self.clock_width = 6 * self.P + 1

    # ---------------------------------------------------------------- parts
    def phi(self, hidden):
        """URM RMSNorm: FP32 mean of squares over all hidden channels, no scale/bias, input dtype out."""
        dtype = hidden.dtype
        hidden = hidden.float() if dtype != torch.float64 else hidden
        return (hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.norm_eps)).to(dtype)

    def boundary(self, layer, hidden):
        """SwiGLU FFN on the RMS-normalized attention residual: u + D(silu(g) * v), (g, v) = U u."""
        hidden = self.phi(hidden)
        g, u = layer.b_gate_up(hidden).chunk(2, dim=-1)
        return hidden + layer.b_down(F.silu(g) * u)

    def rope_tables(self, L):
        angles = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w
        angles = angles.transpose(-1, -2)[None]         # [1,H,T,Dk/2]
        return angles.cos(), angles.sin()

    def apply_rope(self, x, L=None, tables=None):
        cos, sin = self.rope_tables(self.layers[0] if L is None else L) if tables is None else tables
        pairs = x.reshape(*x.shape[:-1], self.dh // 2, 2)
        x0, x1 = pairs.unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def injection(self, batch):
        inj = self.embed(batch["inputs"].to(torch.long))
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.d - self.puzzle_emb_ndim
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def empty_carry(self, batch_size, device=None):
        device = self.init_hidden.device if device is None else device
        return LTCarry(current_hidden=torch.empty(batch_size, self.config.seq_len, self.d,
                                                  dtype=self.forward_dtype, device=device))

    def reset_carry(self, reset_flag, carry):
        return replace(carry, current_hidden=torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def forward(self, carry, batch):
        # lt/train.py precision rule: BF16 autocast only with native BF16 (amp_dtype auto/bfloat16), else FP32.
        device = carry.current_hidden.device
        enabled = self.config.amp and device.type == "cuda" and self.config.amp_dtype != "float32"
        if enabled and self.config.amp_dtype == "auto":
            try:
                enabled = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                enabled = torch.cuda.get_device_capability(device)[0] >= 8
        if enabled:
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                nc, logits = self._forward(carry, batch)
            return replace(nc, current_hidden=nc.current_hidden.float()), logits.float()
        return self._forward(carry, batch)

    # ---------------------------------------------------------------- memory
    def memory_step(self, layer, q, k, v, clk=None, e_k=None, e_v=None, w=None, fresh=None):
        """One block of model B's memory in FP32 (FP64 on the self-test path); returns (read, clk, E_K, E_V, W)."""
        cfg = self.config
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + cfg.eps)
            k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + cfg.eps)
            tables = self.rope_tables(layer)

            def rope(x):
                return self.apply_rope(x, layer, tables)
            qr, kr = rope(q), rope(k)
            B, H, T, D = k.shape
            clk = k.new_zeros(B, H, T, self.clock_width) if clk is None else clk.to(dtype)
            ek = k.new_zeros(B, H, T, 2 * D) if e_k is None else e_k.to(dtype)
            ev = k.new_zeros(B, H, T, 2 * D) if e_v is None else e_v.to(dtype)
            w = k.new_zeros(B, H, D, D) if w is None else w.to(dtype)
            if fresh is not None:
                mask = fresh.view(-1, 1, 1, 1)
                clk, ek, ev, w = (torch.where(mask, torch.zeros_like(x), x) for x in (clk, ek, ev, w))
            tag_k, tag_v, clk = bounded_clock_step(clk, k, v, cfg.b_tag_bound, cfg.b_block_omega)
            lam = torch.exp(-F.softplus(layer.clock_lam_raw.to(dtype)))[:, None, :, None, :]   # [2,1,H,1,P]
            ek_re, ek_im = ek.split(D, -1)
            ev_re, ev_im = ev.split(D, -1)
            hebb, im_s, (ek_re, ek_im, ev_re, ev_im) = phase_clock_step(
                k, kr, v, rope, tag_k, tag_v, lam[0], lam[1], (ek_re, ek_im, ev_re, ev_im))
            w = cfg.b_outer_rho * w + im_s
            alpha = layer.stdp_alpha.to(dtype)[None, :, None, None]
            read = qr @ (hebb + alpha * w).transpose(-1, -2)
        return read, clk, torch.cat((ek_re, ek_im), -1), torch.cat((ev_re, ev_im), -1), w

    def block(self, layer, h, inj, clk, e_k, e_v, w, fresh):
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
        read, clk, e_k, e_v, w = self.memory_step(layer, q, k, v, clk, e_k, e_v, w, fresh)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(h.dtype)
        h = h + layer.out_proj(read)
        h = self.phi(self.boundary(layer, h))
        return h, clk, e_k, e_v, w

    def _forward(self, carry, batch):
        h, inj = carry.current_hidden, self.injection(batch)
        clk, e_k, e_v, w = carry.coupling, carry.key_trace, carry.value_trace, carry.stdp_w
        fresh = carry.fresh
        if self.config.nograd_blocks:
            with torch.no_grad():
                for _ in range(self.config.nograd_blocks):
                    for L in self.layers:
                        h, clk, e_k, e_v, w = self.block(L, h, inj, clk, e_k, e_v, w, fresh)
                        fresh = None
        for _ in range(self.config.blocks_per_seg):
            for L in self.layers:
                h, clk, e_k, e_v, w = self.block(L, h, inj, clk, e_k, e_v, w, fresh)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=clk.detach(), trace=None,
                       key_trace=e_k.detach(), value_trace=e_v.detach(), stdp_w=w.detach(),
                       fresh=None), self.w_cls(h)


class LT(nn.Module):
    """URM harness interface (lt/train.py). No ACT: halted = steps >= loops; q logits are constant."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
        self.inner = PhaseClockBInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        device = self.inner.init_hidden.device
        return LTCarry(current_hidden=self.inner.empty_carry(B, device=device).current_hidden,
                       steps=torch.zeros((B,), dtype=torch.int32, device=device),
                       halted=torch.ones((B,), dtype=torch.bool, device=device),
                       current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()})

    def forward(self, carry, batch, compute_target_q: bool = False):
        inner = self.inner.reset_carry(carry.halted, carry)
        inner = replace(inner, fresh=carry.halted.clone())
        steps = torch.where(carry.halted, 0, carry.steps)
        data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        inner, logits = self.inner(inner, data)
        q = torch.full((logits.shape[0],), -5.0, device=logits.device, dtype=torch.float32)
        outputs = {"logits": logits, "q_halt_logits": q, "q_continue_logits": q}
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.config.loops
        return replace(inner, steps=steps, halted=halted, current_data=data), outputs


# 3. Loss and optimizer -------------------------------------------------------
def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1/(1-x+epsilon), x+1)


def log_stablemax(x, dim=-1):
    sx = s(x)
    return torch.log(sx/sx.sum(dim=dim,keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    lp = log_stablemax(logits.double(),dim=-1)
    valid = labels != ignore_index
    lab = torch.where(valid,labels,0).long()
    chosen = torch.gather(lp,-1,lab.unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid,chosen,0)


class ACTLossHead(nn.Module):
    def __init__(self,model,loss_type="stablemax_cross_entropy",q_weight=0.5):
        super().__init__()
        self.model,self.loss_fn,self.q_weight = model,globals()[loss_type],q_weight

    def initial_carry(self,batch):
        return self.model.initial_carry(batch)

    def forward(self,return_keys:set,**kwargs):
        carry,out = self.model(**kwargs)
        y = carry.current_data["labels"]
        with torch.no_grad():
            pred = out["logits"].argmax(-1)
            mask = y != IGNORE_LABEL_ID
            counts = mask.sum(-1)
            divisor = counts.clamp_min(1).unsqueeze(-1)
            ok = mask & (pred==y)
            exact = ok.sum(-1)==counts
            valid = carry.halted & (counts>0)
            metrics = dict(count=valid.sum(),
                accuracy=torch.where(valid,(ok.float()/divisor).sum(-1),0).sum(),
                exact_accuracy=(valid & exact).sum(),
                q_halt_accuracy=(valid & ((out["q_halt_logits"]>=0)==exact)).sum(),
                steps=torch.where(valid,carry.steps,0).sum())
        lm = (self.loss_fn(out["logits"],y,IGNORE_LABEL_ID)/divisor).sum()
        qloss = F.binary_cross_entropy_with_logits(out["q_halt_logits"],exact.float(),reduction="sum")
        metrics.update(lm_loss=lm.detach(),q_halt_loss=qloss.detach())
        out["preds"] = pred
        returned = {k:out[k].detach() for k in return_keys if k in out}
        return carry,lm+self.q_weight*qloss,metrics,returned,carry.halted.all()


class AdamATan2(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),weight_decay=1e-2):
        super().__init__(params,dict(lr=lr,betas=betas,weight_decay=weight_decay))

    @torch.no_grad()
    def step(self,closure=None):
        for g in self.param_groups:
            b1,b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                st["step"] += 1
                t,m,v = st["step"],st["m"],st["v"]
                m.mul_(b1).add_(p.grad,alpha=1-b1)
                v.mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
                p.mul_(1-g["lr"]*g["weight_decay"])
                p.add_(torch.atan2(m/(1-b1**t),(v/(1-b2**t)).sqrt()),alpha=-g["lr"])


NO_DECAY_KEYS = ("psi","theta","alpha_raw","gamma_raw","inj_gate","st_gain",
                 "gain_raw","eta_raw","lam_raw","beta","mu")


def _is_no_decay(name,p,qkv_no_decay=False):
    return (p.ndim<=1 or name.endswith(".b") or any(k in name for k in NO_DECAY_KEYS)
            or (qkv_no_decay and name.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight"))))


def create_optimizers(base,cfg,world_size):
    opts,lrs = [],[]
    if base.model.puzzle_emb is not None:
        opts.append(CastedSparseEmbeddingSignSGD_Distributed(base.model.puzzle_emb,world_size,
                    weight_decay=cfg["puzzle_emb_weight_decay"]))
        lrs.append(cfg["puzzle_emb_lr"])
    named = [(n,p) for n,p in base.named_parameters() if p.requires_grad]
    qkv_no_decay = cfg.get("qkv_no_decay",False)
    opts.append(AdamATan2([
        {"params":[p for n,p in named if _is_no_decay(n,p,qkv_no_decay)],"weight_decay":0.0},
        {"params":[p for n,p in named if not _is_no_decay(n,p,qkv_no_decay)],"weight_decay":cfg["weight_decay"]}],
        lr=0,betas=(cfg["beta1"],cfg["beta2"])))
    lrs.append(cfg["lr"])
    return opts,lrs


def lr_at(step,base_lr,cfg,planned_steps):
    """기존 스케줄 (warmup 후 전체 계획 스텝에 걸친 cosine, lr_min_ratio=1 이면 상수)에 재가열을 곱한다:
    lr_rewarm_start 부터 lr_rewarm_steps 동안 lr_rewarm_from_ratio 배 → 1 배로 선형, 이후 1 배.
    init_from 으로 새 optimizer 를 만들어 이어 학습할 때 쓴다. lr_rewarm_start=None 이면 기존과 같다."""
    lr = cosine_schedule_with_warmup_lr_lambda(step,base_lr=base_lr,num_warmup_steps=cfg["lr_warmup_steps"],
                                               num_training_steps=planned_steps,min_ratio=cfg["lr_min_ratio"])
    start = cfg.get("lr_rewarm_start")
    if start is None or step<start:
        return lr
    r0,t = cfg["lr_rewarm_from_ratio"],min(1.0,(step-start)/max(1,cfg["lr_rewarm_steps"]))
    return lr*(r0+(1-r0)*t)


def cosine_schedule_with_warmup_lr_lambda(current_step,*,base_lr,num_warmup_steps,
                                         num_training_steps,min_ratio=0.0,num_cycles=0.5):
    if current_step < num_warmup_steps:
        return base_lr*current_step/max(1,num_warmup_steps)
    progress = (current_step-num_warmup_steps)/max(1,num_training_steps-num_warmup_steps)
    return base_lr*(min_ratio+max(0.,(1-min_ratio)*.5*(1+math.cos(2*math.pi*num_cycles*progress))))


# 4. EMA ---------------------------------------------------------------------
class EMAHelper:
    def __init__(self,mu=0.999):
        self.mu,self.shadow = mu,{}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self,state):
        self.shadow = state

    def register(self,module):
        self.shadow = {n:p.detach().clone() for n,p in module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self,module):
        for n,p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.mu).add_(p,alpha=1-self.mu)


class _EMASwap:
    def __init__(self,module,ema):
        self.module,self.ema,self.backup = module,ema,{}

    def __enter__(self):
        if self.ema is not None:
            with torch.no_grad():
                for n,p in self.module.named_parameters():
                    if n in self.ema.shadow:
                        self.backup[n] = p.detach().clone()
                        p.copy_(self.ema.shadow[n])
        return self.module

    def __exit__(self,*exc):
        with torch.no_grad():
            for n,p in self.module.named_parameters():
                if n in self.backup:
                    p.copy_(self.backup[n])
        self.backup.clear()
        return False

# 5. Fixed augmentation pool / data ------------------------------------------
_ROW_OF, _COL_OF = np.arange(81)//9, np.arange(81)%9


def _draw_aug_params(rng):
    digits = np.pad(rng.permutation(np.arange(1,10)),(1,0))
    trans = rng.random() < 0.5
    bands = rng.permutation(3)
    rows = np.concatenate([b*3+rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    cols = np.concatenate([s*3+rng.permutation(3) for s in stacks])
    return digits,trans,rows,cols


def _apply_aug(x,digits,trans,rows,cols):
    mapping = rows[_ROW_OF]*9+cols[_COL_OF]
    if trans:
        x = x.T
    return digits[x.flatten()[mapping].reshape(9,9)]


class SudokuTrainDataset(IterableDataset):
    def __init__(self,inputs,labels,*,seed,num_aug,global_batch_size,rank,world_size,
                 epochs_per_iter,start_iter,total_iters,skip_batches=0):
        super().__init__()
        self.inputs,self.labels = inputs,labels
        self.seed,self.num_aug = seed,num_aug
        self.gbs,self.rank,self.world_size = global_batch_size,rank,world_size
        self.local_bs = global_batch_size//world_size
        self.epochs_per_iter,self.start_iter = epochs_per_iter,start_iter
        self.total_iters,self.skip_batches = total_iters,skip_batches
        self.n_groups,self.gsize = len(inputs),1+num_aug
        self._aug_key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def _augmented(self,gid,aug_idx):
        if aug_idx == 0:
            return self.inputs[gid],self.labels[gid]
        key = np.array([self._aug_key0,np.uint64(gid*self.gsize+aug_idx)],dtype=np.uint64)
        rng = np.random.Generator(np.random.Philox(key=key))
        prm = _draw_aug_params(rng)
        return _apply_aug(self.inputs[gid],*prm),_apply_aug(self.labels[gid],*prm)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        if wi is not None and wi.num_workers != 1:
            raise RuntimeError("dataloader_workers must be 0 or 1.")
        skip = self.skip_batches
        for it in range(self.start_iter,self.total_iters):
            rng = np.random.Generator(np.random.Philox(seed=self.seed+it+1))
            groups = np.concatenate([rng.permutation(self.n_groups) for _ in range(self.epochs_per_iter)])
            start = 0
            while start < len(groups):
                gsel,asel = [],[]
                while start < len(groups) and len(gsel) < self.gbs:
                    g = int(groups[start])
                    pid = int(rng.integers(g*self.gsize,(g+1)*self.gsize))
                    start += 1
                    rng.choice(1,1,replace=False)
                    gsel.append(g)
                    asel.append(pid-g*self.gsize)
                if len(gsel) < self.gbs:
                    break
                if skip:
                    skip -= 1
                    continue
                off = self.rank*self.local_bs
                inp = np.empty((self.local_bs,81),dtype=np.int32)
                lab = np.empty_like(inp)
                for j in range(self.local_bs):
                    x,y = self._augmented(gsel[off+j],asel[off+j])
                    inp[j],lab[j] = x.reshape(81)+1,y.reshape(81)+1
                yield it,dict(inputs=torch.from_numpy(inp),labels=torch.from_numpy(lab),
                              puzzle_identifiers=torch.zeros(self.local_bs,dtype=torch.int32))


def eval_batches(inputs,labels,gbs,rank,world_size):
    """One yield per GLOBAL batch on every rank, including None for an empty shard."""
    lbs = gbs//world_size
    for start in range(0,len(inputs),gbs):
        end = min(start+gbs,len(inputs))
        ls,le = start+rank*lbs,min(start+(rank+1)*lbs,end)
        if le <= ls:
            yield None
        else:
            yield dict(inputs=torch.from_numpy(inputs[ls:le].reshape(-1,81).astype(np.int32)+1),
                       labels=torch.from_numpy(labels[ls:le].reshape(-1,81).astype(np.int32)+1),
                       puzzle_identifiers=torch.zeros(le-ls,dtype=torch.int32))


def _find_npz(path):
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"CFG['data_npz'] does not exist: {p}")
    candidates = set()
    for root in (Path('/kaggle/input'),Path.cwd()/"data",Path.cwd()):
        if root.is_dir():
            pattern = '**/sudoku_lt_1k.npz' if str(root)=='/kaggle/input' else 'sudoku_lt_1k.npz'
            candidates.update(str(p.resolve()) for p in root.glob(pattern))
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise RuntimeError("Multiple NPZ files; set CFG['data_npz'] explicitly:\n"+'\n'.join(sorted(candidates)))
    raise FileNotFoundError("Attach sudoku_lt_1k.npz as Kaggle Input, or set CFG['data_npz']. "
                            "Required keys: train_inputs, train_labels, test_inputs, test_labels; raw digits 0..9.")


def _check_boards(x,y,name):
    if x.ndim == 2 and x.shape[1] == 81:
        x = x.reshape(-1,9,9)
    if y.ndim == 2 and y.shape[1] == 81:
        y = y.reshape(-1,9,9)
    if x.shape != y.shape or x.ndim != 3 or x.shape[1:] != (9,9) or len(x)==0:
        raise ValueError(f"{name}: expected matching nonempty [N,9,9] arrays; got {x.shape}, {y.shape}.")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"{name}: nonfinite board values.")
    if not ((x == x.astype(np.int64)).all() and (y == y.astype(np.int64)).all()):
        raise ValueError(f"{name}: boards must contain integer digits.")
    if x.min()<0 or x.max()>9 or y.min()<1 or y.max()>9:
        raise ValueError(f"{name}: raw inputs must be 0..9 and solutions 1..9. Do NOT pre-add 1.")
    x,y = x.astype(np.uint8),y.astype(np.uint8)
    if not ((x==0)|(x==y)).all():
        raise ValueError(f"{name}: clues do not match solutions.")
    digits = np.arange(1,10)
    boxes = y.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
    if not ((np.sort(y,axis=2)==digits).all() and
            (np.sort(y,axis=1)==digits[None,:,None]).all() and
            (np.sort(boxes,axis=2)==digits).all()):
        raise ValueError(f"{name}: at least one solution is not a valid Sudoku.")
    return np.ascontiguousarray(x),np.ascontiguousarray(y)


def load_data(cfg):
    path = _find_npz(cfg["data_npz"])
    with np.load(path,allow_pickle=False) as z:
        need = ("train_inputs","train_labels","test_inputs","test_labels")
        if not set(need).issubset(z.files):
            raise ValueError(f"NPZ keys must include {need}; found {z.files}")
        tr_x,tr_y = _check_boards(z[need[0]],z[need[1]],"train")
        ntest = int(cfg["test_size"])
        te_x,te_y = _check_boards(z[need[2]][:ntest],z[need[3]][:ntest],"test")
    digest = hashlib.sha256()
    for a in (tr_x,tr_y,te_x,te_y):
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return tr_x,tr_y,te_x,te_y,path,digest.hexdigest()


# 6. Checkpoints: weights + optimizers + EMA + per-rank carry and RNG ----------
@dataclass
class TrainState:
    step: int = 0
    iter_id: int = 0           # iterator to resume NEXT
    batch_in_iter: int = 0     # number of already consumed global batches in that iterator
    carry: Any = None
    in_step: bool = False


def _cpu_tree(obj):
    if isinstance(obj,torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj,dict):
        return {k:_cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return type(obj)(_cpu_tree(v) for v in obj)
    return obj


def _carry_dict(carry):
    return None if carry is None else {f.name:_cpu_tree(getattr(carry,f.name)) for f in fields(carry)}


def _rng_state(device):
    return dict(torch=torch.random.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device).cpu() if device.type=="cuda" else None)


def _restore_rng(state,device):
    torch.random.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type=="cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(),device)


_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


def find_latest_checkpoint(path):
    if not path:
        return None
    if os.path.isfile(path):
        return str(path)
    choices = []
    for p in glob.glob(os.path.join(str(path),"**","step_*.pt"),recursive=True):
        m = _CKPT_RE.search(os.path.basename(p))
        if m:
            choices.append((int(m.group(1)),os.path.getmtime(p),p))
    return max(choices)[2] if choices else None


def _collect_rank_states(ts,device,rank,ws):
    local = dict(carry=_carry_dict(ts.carry),rng=_rng_state(device))
    if ws==1:
        return [local]
    gathered = [None]*ws if rank==0 else None
    dist.gather_object(local,gathered,dst=0)
    return gathered


def save_training_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device,keep_last=None):
    """Every rank must call. Atomic file replace; cursor means 'next unread batch'."""
    if ts.in_step:
        raise RuntimeError("Refusing to save a partially completed optimizer step.")
    rank_states = _collect_rank_states(ts,device,rank,ws)
    status = [None]
    path = os.path.join(out_dir,f"step_{ts.step}.pt")
    if rank==0:
        try:
            os.makedirs(out_dir,exist_ok=True)
            raw = _cpu_tree(base.state_dict())
            with _EMASwap(base,ema):
                ema_sd = _cpu_tree(base.state_dict())
            ck = dict(model_id=model_id_of(cfg),step=ts.step,iter_id=ts.iter_id,batch_in_iter=ts.batch_in_iter,
                      world_size=ws,model_state_dict=ema_sd,raw_model_state_dict=raw,
                      ema_shadow=_cpu_tree(ema.shadow) if ema else None,
                      optimizer_states=_cpu_tree([o.state_dict() for o in optimizers]),
                      rank_states=rank_states,cfg=dict(cfg))
            tmp = path+".tmp"
            torch.save(ck,tmp)
            os.replace(tmp,path)
            keep = cfg["keep_last"] if keep_last is None else keep_last
            if keep>0:
                old = []
                for p in glob.glob(os.path.join(out_dir,"step_*.pt")):
                    m = _CKPT_RE.search(os.path.basename(p))
                    if m:
                        old.append((int(m.group(1)),p))
                for _,p in sorted(old)[:-keep]:
                    os.remove(p)
        except Exception as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    if ws>1:
        dist.broadcast_object_list(status,src=0)
    if status[0]:
        raise RuntimeError(f"Checkpoint save failed: {status[0]}")
    return path


def _restore_carry(rank_states,rank,ws,gbs,device):
    saved = [r["carry"] for r in rank_states]
    if all(c is None for c in saved):
        return None
    if any(c is None for c in saved):
        raise ValueError("Checkpoint contains inconsistent rank carries.")
    lbs = gbs//ws
    start,end = rank*lbs,(rank+1)*lbs
    def merge(parts):
        if all(p is None for p in parts):
            return None
        if isinstance(parts[0],dict):
            return {k:merge([p[k] for p in parts]) for k in parts[0]}
        all_t = torch.cat(parts,dim=0)
        if all_t.shape[0] != gbs:
            raise ValueError("Saved carry batch size does not match global_batch_size.")
        return all_t[start:end].to(device).clone()
    return LTCarry(**{k:merge([c[k] for c in saved]) for k in saved[0]})


# Resume protocol: every model field (including the B constants) and the data/optimizer recipe must match.
_RESUME_KEYS = tuple(f.name for f in fields(LTConfig) if f.name not in
                     ("batch_size", "amp", "amp_dtype", "nograd_blocks")) + (
    "global_batch_size","epochs","eval_interval","num_aug","seed","grad_accum_steps",
    "lr","lr_min_ratio","lr_warmup_steps","weight_decay","qkv_no_decay","beta1","beta2","puzzle_emb_lr",
    "puzzle_emb_weight_decay","q_weight","ema","ema_rate","data_fingerprint",
    "lr_rewarm_start","lr_rewarm_steps","lr_rewarm_from_ratio","late_sup_prob","late_sup_min","late_sup_max",
    "nograd_fixed","nograd_every","nograd_start","nograd_max")


def _effective_recipe(cfg):
    """Values compared on resume; sub-options of disabled features are ignored."""
    r = {k: cfg.get(k) for k in _RESUME_KEYS}
    if r["lr_rewarm_start"] is None:
        r["lr_rewarm_steps"] = r["lr_rewarm_from_ratio"] = None
    if not r["late_sup_prob"]:
        r["late_sup_min"] = r["late_sup_max"] = None
    if not r["nograd_every"]:
        r["nograd_start"] = r["nograd_max"] = None
    if r["rope_type"] == "learned_2d":
        r["rope_base"] = None
    return r


def load_training_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device):
    # Only load checkpoints you trust: weights_only=False is needed for RNG/optimizer objects.
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("model_id") != model_id_of(cfg):
        raise ValueError(f"Checkpoint model_id={ck.get('model_id')!r} but this run is {model_id_of(cfg)!r}. "
                         "Resume needs the same architecture.")
    was,now = _effective_recipe(ck["cfg"]),_effective_recipe(cfg)
    changed = {k:(was[k],now[k]) for k in _RESUME_KEYS if was[k]!=now[k]}
    if changed:
        raise ValueError(f"Resume config/data mismatch (start a new out_dir for a new experiment): {changed}")
    base.load_state_dict(ck["raw_model_state_dict"],strict=True,assign=False)
    if len(optimizers)!=len(ck["optimizer_states"]):
        raise ValueError("Optimizer count mismatch.")
    for opt,state in zip(optimizers,ck["optimizer_states"]):
        opt.load_state_dict(state)
        if isinstance(opt,CastedSparseEmbeddingSignSGD_Distributed):
            opt.param_groups[0]["world_size"] = ws
        opt.zero_grad(set_to_none=True)
    if ema is not None:
        shadow = ck.get("ema_shadow")
        if shadow is None or set(shadow)!=set(ema.shadow):
            raise ValueError("EMA shadow is missing or incompatible.")
        ema.shadow = {n:v.to(device) for n,v in shadow.items()}
    ts = TrainState(step=int(ck["step"]),iter_id=int(ck["iter_id"]),
                    batch_in_iter=int(ck["batch_in_iter"]),
                    carry=_restore_carry(ck["rank_states"],rank,ws,cfg["global_batch_size"],device))
    if ws==ck["world_size"]:
        _restore_rng(ck["rank_states"][rank]["rng"],device)
    elif rank==0:
        print(f"[LT] resharded saved carry (h, clock, traces, W): {ck['world_size']} -> {ws} ranks; "
              "bitwise equality across GPU layouts is not promised.",flush=True)
    return ts


# 7. Evaluation / segment extrapolation --------------------------------------
_METRIC_KEYS = ("accuracy","count","exact_accuracy","lm_loss","q_halt_accuracy","q_halt_loss","steps")


def stop_requested(device,deadline):
    flag = torch.tensor(int(_STOP_REQUESTED or time.monotonic()>=deadline),dtype=torch.int32,device=device)
    if dist.is_initialized():
        dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


@torch.no_grad()
def evaluate(base,eval_in,eval_lb,cfg,rank,ws,device,step,ema,deadline=float("inf")):
    # Use eager base for eval; it avoids a second compile graph and uneven-batch recompiles.
    totals = torch.zeros(len(_METRIC_KEYS),dtype=torch.float64,device=device)
    interrupted = False
    training = base.training
    with _EMASwap(base,ema):
        base.eval()
        try:
            for batch in eval_batches(eval_in,eval_lb,cfg["global_batch_size"],rank,ws):
                if stop_requested(device,deadline):
                    interrupted = True
                    break
                if batch is None:
                    continue
                batch = {k:v.to(device) for k,v in batch.items()}
                carry = base.initial_carry(batch)
                for _ in range(cfg["loops"]):
                    carry,_,metrics,_,_ = base(carry=carry,batch=batch,return_keys=set())
                totals += torch.stack([metrics[k].double() for k in _METRIC_KEYS])
        finally:
            base.train(training)
    if ws>1:
        dist.reduce(totals,dst=0)
    if rank!=0:
        return None
    d = dict(zip(_METRIC_KEYS,totals.cpu().tolist()))
    n = int(d["count"])
    print(f"[EVAL] step {step} blocks/seg {getattr(base.model.config,'nograd_blocks',0)}+{cfg['blocks_per_seg']} "
          f"acc {d['accuracy']/max(n,1):.4f} "
          f"exact {int(d['exact_accuracy'])}/{n}"+(" [partial: time/stop]" if interrupted else ""),flush=True)
    return {k:v/max(n,1) if k!="count" else v for k,v in d.items()}


@torch.no_grad()
def extrapolate(base,eval_in,eval_lb,cfg,rank,ws,device,step,ema,segs,out_txt,deadline=float("inf")):
    lt = base.model
    loops0,training = lt.config.loops,base.training
    limit = cfg.get("milestone_extrap_n")
    if limit:
        eval_in,eval_lb = eval_in[:int(limit)],eval_lb[:int(limit)]  # GLOBAL subset, not per rank
    sums = torch.zeros(3,segs,dtype=torch.float64,device=device)
    counts = torch.zeros(segs,dtype=torch.float64,device=device)
    interrupted = False
    t0 = time.monotonic()
    weights = "ema" if ema is not None else "raw"
    if rank==0:
        print(f"[EXTRAP] start step={step} weights={weights} segs={segs} global_n={len(eval_in)}",flush=True)
    with _EMASwap(base,ema):
        base.eval()
        try:
            lt.config.loops = segs+1
            for batch in eval_batches(eval_in,eval_lb,cfg["global_batch_size"],rank,ws):
                b = None if batch is None else {k:v.to(device) for k,v in batch.items()}
                carry = None if b is None else lt.initial_carry(b)
                prev = None
                for si in range(segs):
                    # Every rank participates, even if it has no examples in the final shard.
                    if stop_requested(device,deadline):
                        interrupted = True
                        break
                    if b is None:
                        continue
                    carry,out = lt(carry,b)
                    pred,y = out["logits"].argmax(-1),b["labels"]
                    mask = y!=IGNORE_LABEL_ID
                    nvalid = mask.sum(-1)
                    ok = mask & (pred==y)
                    sums[0,si] += (ok.sum(-1).double()/nvalid.clamp_min(1)).sum()
                    sums[1,si] += (ok.sum(-1)==nvalid).sum()
                    if prev is not None:
                        sums[2,si] += (pred!=prev).double().mean(-1).sum()
                    counts[si] += len(y)
                    prev = pred
                if interrupted:
                    break
        finally:
            lt.config.loops = loops0
            base.train(training)
    if ws>1:
        dist.reduce(sums,dst=0)
        dist.reduce(counts,dst=0)
    if rank!=0:
        return None
    nn = counts.cpu().numpy()
    aa = (sums[0]/counts.clamp_min(1)).cpu().numpy()
    ee = sums[1].cpu().numpy()
    cc = (sums[2]/counts.clamp_min(1)).cpu().numpy()
    elapsed = time.monotonic()-t0
    rows = [dict(segment=si+1,acc=float(aa[si]),exact=int(ee[si]),n=int(nn[si]),
                 exact_percent=100*float(ee[si])/int(nn[si]),churn=float(cc[si]))
            for si in range(segs) if nn[si]>0]
    by_segment = {row["segment"]:row for row in rows}
    train_row,final_row = by_segment.get(loops0),by_segment.get(segs)
    # On interruption, different segments can have different sample counts.
    # Compare only segments evaluated on the same largest prefix of puzzles.
    comparison_n = max((row["n"] for row in rows),default=0)
    comparable = [row for row in rows if row["n"]==comparison_n]
    best_exact = max(comparable,key=lambda row:row["exact"]) if comparable else None
    best_acc = max(comparable,key=lambda row:row["acc"]) if comparable else None
    lines = [f"# step={step} weights={weights} segs={segs} global_n={len(eval_in)} "
             f"elapsed={elapsed:.1f}s partial={interrupted}",
             f"# {model_id_of(cfg)}; per segment {getattr(lt.config,'nograd_blocks',0)}+{cfg['blocks_per_seg']} blocks "
             f"(no-grad+grad); train segments={loops0}",
             "# seg acc exact n exact_percent churn"]
    for row in rows:
        lines.append(f"{row['segment']:4d} {row['acc']:.6f} {row['exact']:6d} {row['n']:6d} "
                     f"{row['exact_percent']:.4f} {row['churn']:.6f}"+
                     ("  <-train" if row["segment"]==loops0 else ""))

    def summary(label,row,segment=None):
        if row is None:
            return f"# {label}: seg{segment} not evaluated" if segment is not None else f"# {label}: not evaluated"
        return (f"# {label}: seg{row['segment']} acc={row['acc']:.6f} "
                f"exact={row['exact']}/{row['n']} ({row['exact_percent']:.4f}%) churn={row['churn']:.6f}")

    lines += ["",summary("train",train_row,loops0),
              summary(f"best_acc [same n={comparison_n}]",best_acc),
              summary(f"best_exact [same n={comparison_n}]",best_exact),
              summary("final",final_row,segs)]
    if final_row is None and rows:
        lines.append(summary("last measured",rows[-1]))
    if train_row is not None and best_exact is not None and train_row["n"]==comparison_n:
        lines.append(f"# best-train: exact={best_exact['exact']-train_row['exact']:+d} "
                     f"percentage_points={best_exact['exact_percent']-train_row['exact_percent']:+.4f}")
    result = dict(acc=aa.tolist(),exact=ee.tolist(),churn=cc.tolist(),count=nn.tolist(),
                  step=int(step),weights=weights,segs=int(segs),target_n=len(eval_in),n=comparison_n,
                  partial=interrupted,elapsed_seconds=elapsed,model_id=model_id_of(cfg),
                  blocks_per_segment=int(cfg["blocks_per_seg"]),nograd_blocks=int(getattr(lt.config,'nograd_blocks',0)),
                  train_segment=int(loops0),
                  train=train_row,best_acc=best_acc,best_exact=best_exact,final=final_row,
                  last_measured=rows[-1] if rows else None,best_comparison_n=comparison_n,segments=rows)
    # The complete table is printed as well as persisted, so notebook logs suffice.
    print("\n".join(lines),flush=True)
    out_txt = os.fspath(out_txt)
    directory = os.path.dirname(os.path.abspath(out_txt))
    os.makedirs(directory,exist_ok=True)
    with open(out_txt+".tmp","w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    os.replace(out_txt+".tmp",out_txt)
    out_json = os.path.splitext(out_txt)[0]+".json"
    with open(out_json+".tmp","w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    os.replace(out_json+".tmp",out_json)
    out_jsonl = os.path.join(directory,"extrap_results.jsonl")
    with open(out_jsonl,"a",encoding="utf-8") as f:
        f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
    print(f"[EXTRAP] step {step} weights={weights} -> {out_txt}, {out_json}, {out_jsonl}"+
          (" [partial: time/stop]" if interrupted else ""),flush=True)
    return result

# 8. Training ----------------------------------------------------------------
def _allreduce_parameter_grads(base,device,ws):
    if ws==1:
        return
    params = [p for p in base.parameters() if p.requires_grad]
    used = torch.tensor([int(p.grad is not None) for p in params],dtype=torch.int32,device=device)
    dist.all_reduce(used,op=dist.ReduceOp.MAX)
    grads = []
    for p,is_used in zip(params,used.cpu().tolist()):
        if is_used:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
    if grads:
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat,op=dist.ReduceOp.SUM)       # SUM, never average
        off = 0
        for g in grads:
            g.copy_(flat[off:off+g.numel()].view_as(g))
            off += g.numel()


def _check_finite_gradients(base,loss,device,ws):
    flags = [torch.isfinite(loss.detach()).all()]
    flags.extend(torch.isfinite(p.grad).all() for p in base.parameters() if p.grad is not None)
    pe = base.model.puzzle_emb
    if pe is not None and pe.local_weights.grad is not None:
        flags.append(torch.isfinite(pe.local_weights.grad).all())
    bad = (~torch.stack(flags).all()).to(torch.int32)
    if ws>1:
        dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError("Nonfinite loss/gradient BEFORE optimizer step. "
                                 "Stopped without overwriting the last committed checkpoint.")


def train_batch(model,base,ts,batch,cfg,optimizers,lrs,total_steps,rank,world_size,device):
    planned_steps,ws = total_steps,world_size
    ts.in_step = True
    # 평가·외삽은 마지막 학습 스텝의 깊이를 그대로 쓴다 (config 는 다음 train_batch 에서 다시 정해진다).
    base.model.config.nograd_blocks = nograd_at(ts.step,cfg)
    batch = {k:v.to(device,non_blocking=True) for k,v in batch.items()}
    if ts.carry is None:
        ts.carry = base.initial_carry(batch)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    nc,loss,metrics,_,_ = model(carry=ts.carry,batch=batch,return_keys=set())
    (loss/cfg["global_batch_size"]).backward()
    _allreduce_parameter_grads(base,device,ws)
    _check_finite_gradients(base,loss,device,ws)
    lr = 0.0
    for opt,base_lr in zip(optimizers,lrs):
        lr = lr_at(ts.step,base_lr,cfg,planned_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
    ts.carry,ts.step = nc,ts.step+1
    extra = late_supervision_extra(cfg,ts.step,nc)
    if extra:
        ts.carry = run_unsupervised_segments(base,nc,extra)
    ts.in_step = False
    vals = torch.stack([metrics[k].float() for k in _METRIC_KEYS])
    if ws>1:
        dist.reduce(vals,dst=0)
    if rank!=0:
        return None
    d = dict(zip(_METRIC_KEYS,vals.cpu().tolist()))
    n = max(d["count"],1)
    result = {k:v/(cfg["global_batch_size"] if k.endswith("loss") else n) for k,v in d.items()}
    result.update(lr=lr,_count_raw=d["count"])
    return result


def nograd_at(step,cfg):
    """세그먼트 앞의 no-grad 블록 수. nograd_start 스텝에 1개로 시작해 nograd_every 스텝마다 1씩 늘고
    nograd_max 에서 멈춘다. 예: start=100000, every=10000 이면 100k 에 1, 110k 에 2, 130k 에 4."""
    if int(cfg.get("nograd_fixed",0) or 0)>0:
        return int(cfg["nograd_fixed"])                 # 스케줄 없이 매 세그먼트 같은 개수
    every,start = int(cfg.get("nograd_every",0) or 0),int(cfg.get("nograd_start",0))
    if every<=0 or step<start:
        return 0
    return min(int(cfg["nograd_max"]),1+(step-start)//every)


def late_supervision_extra(cfg,step,carry):
    """후반 감독: 학습 지평(loops)의 마지막 세그먼트를 막 마친 배치를, 확률 late_sup_prob 로
    R∈[late_sup_min, late_sup_max] 세그먼트 더 (gradient 없이) 돌린 뒤 한 세그먼트를 더 감독한다.
    8블록 BPTT 는 그대로이고, 감독받는 상태의 분포만 긴 지평으로 넓힌다. 결정은 (seed, step) 로만
    정해지므로 모든 rank 에서 같고 재개해도 같다. 반환: 추가로 돌릴 세그먼트 수 (0 이면 없음)."""
    prob = float(cfg.get("late_sup_prob",0.0))
    if prob<=0 or not bool(carry.halted.all()) or int(carry.steps.max())!=cfg["loops"]:
        return 0
    rng = np.random.Generator(np.random.Philox(key=np.array([cfg["seed"],step],dtype=np.uint64)))
    if rng.random()>=prob:
        return 0
    return int(rng.integers(cfg["late_sup_min"],cfg["late_sup_max"]+1))


@torch.no_grad()
def run_unsupervised_segments(base,carry,extra):
    """같은 퍼즐을 extra 세그먼트 더 진행한다. 끝나면 halted=False 로 두어 다음 학습 스텝이
    같은 퍼즐의 seg(loops+extra+1) 을 감독하고, 그 뒤에 새 퍼즐로 넘어간다 (steps ≥ loops)."""
    lt,training = base.model,base.training
    base.eval()
    try:
        for _ in range(extra):
            carry = replace(carry,halted=torch.zeros_like(carry.halted))
            carry,_ = lt(carry,carry.current_data)
    finally:
        base.train(training)
    return replace(carry,halted=torch.zeros_like(carry.halted))


def resolve_out_dir(cfg):
    if cfg["out_dir"]:
        return os.path.abspath(os.path.expanduser(str(cfg["out_dir"])))
    root = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    return os.path.join(root, "lt_" + model_tag_of(cfg))


def init_distributed():
    ws = int(os.environ.get("WORLD_SIZE","1"))
    rank = int(os.environ.get("RANK","0"))
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK","0"))
        torch.cuda.set_device(local)
        device = torch.device("cuda",local)
    else:
        device = torch.device("cpu")
    if ws>1:
        # Long first compilation on one GPU must not time out another rank's allreduce.
        from datetime import timedelta
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo",timeout=timedelta(minutes=60))
    return rank,ws,device


def _resolve_precision(cfg,device):
    requested = cfg["amp_dtype"]
    if requested not in ("auto","bfloat16","float32"):
        raise ValueError("amp_dtype: auto | bfloat16 | float32. FP16 is deliberately not used.")
    supported = False
    if device.type=="cuda":
        try:
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            supported = torch.cuda.get_device_capability(device)[0]>=8
    if requested=="bfloat16" and cfg["amp"] and not supported:
        raise ValueError("Native BF16 is unavailable; use amp_dtype='auto' or 'float32'.")
    cfg["amp_dtype"] = "bfloat16" if cfg["amp"] and supported and requested!="float32" else "float32"


def _validate_kv_memory_update(update, rho):
    if update not in ("additive", "ema", "leaky"):
        raise ValueError("kv_memory_update must be additive, ema or leaky.")
    if (isinstance(rho, bool) or not isinstance(rho, (int, float))
            or not math.isfinite(rho) or not 0 <= rho < 1):
        raise ValueError("kv_memory_rho must be a finite number in [0, 1).")


def validate_run_cfg(cfg):
    _validate_kv_memory_update(cfg.get("kv_memory_update","additive"),cfg.get("kv_memory_rho",0.95))
    if not isinstance(cfg.get("kv_trace_activity_detach",False),bool):
        raise ValueError("kv_trace_activity_detach must be a boolean.")
    if not isinstance(cfg.get("allow_trace_activity_fork",False),bool):
        raise ValueError("allow_trace_activity_fork must be a boolean.")
    if cfg.get("allow_trace_activity_fork",False) and not (
            cfg.get("memory_type") == "kv_stdp" and cfg.get("kv_trace_activity_detach",False)
            and cfg.get("require_resume",False) and cfg.get("resume_from")):
        raise ValueError("allow_trace_activity_fork requires KV-STDP activity detach and an explicit required resume.")
    if cfg.get("kv_write_reduction","mean") not in ("mean","sum"):
        raise ValueError("kv_write_reduction must be mean or sum.")
    if not isinstance(cfg.get("qkv_no_decay",False),bool):
        raise ValueError("qkv_no_decay must be a boolean.")
    if not isinstance(cfg.get("kv_qk_rmsnorm",False),bool):
        raise ValueError("kv_qk_rmsnorm must be a boolean.")
    if not isinstance(cfg.get("kv_qk_l2norm",False),bool):
        raise ValueError("kv_qk_l2norm must be a boolean.")
    if cfg.get("kv_qk_rmsnorm",False) and cfg.get("kv_qk_l2norm",False):
        raise ValueError("kv_qk_rmsnorm and kv_qk_l2norm are mutually exclusive.")
    if not isinstance(cfg.get("kv_projection_fp32",False),bool):
        raise ValueError("kv_projection_fp32 must be a boolean.")
    if cfg["grad_accum_steps"]!=1:
        raise ValueError("grad_accum_steps must be 1: one segment equals one optimizer step. "
                         "The supplied train.py did not actually implement gradient accumulation.")
    if cfg["dataloader_workers"] not in (0,1):
        raise ValueError("dataloader_workers must be 0 or 1.")
    if cfg["epochs"]<=0 or cfg["eval_interval"]<=0 or cfg["epochs"]%cfg["eval_interval"]:
        raise ValueError("epochs must be a positive multiple of eval_interval (in EPOCHS, not steps).")
    if min(cfg["global_batch_size"],cfg["test_size"],cfg["log_every"],cfg["stop_check_every"])<=0:
        raise ValueError("Batch, test_size, log_every, and stop_check_every must be positive.")
    if cfg["num_aug"]<0 or (cfg["max_hours"] is not None and cfg["max_hours"]<=0):
        raise ValueError("num_aug>=0 and max_hours>0 or None are required.")
    if cfg["max_steps"] is not None and cfg["max_steps"]<0:
        raise ValueError("max_steps must be nonnegative or None; it is an ABSOLUTE stopping step.")
    if cfg["milestone_every"] and cfg["milestone_extrap_segs"]<1:
        raise ValueError("milestone_extrap_segs must be positive.")
    if not 0<=cfg["lr_min_ratio"]<=1:
        raise ValueError("0<=lr_min_ratio<=1 is required.")
    if cfg.get("lr_rewarm_start") is not None and (cfg["lr_rewarm_start"]<0 or cfg["lr_rewarm_steps"]<=0
                                                   or not 0<cfg["lr_rewarm_from_ratio"]<=1):
        raise ValueError("lr_rewarm_start>=0 needs lr_rewarm_steps>0 and 0<lr_rewarm_from_ratio<=1.")
    if min(cfg.get("nograd_fixed",0),cfg.get("nograd_every",0),cfg.get("nograd_start",0),cfg.get("nograd_max",0))<0:
        raise ValueError("nograd_every, nograd_start, nograd_max must be nonnegative.")
    if cfg.get("late_sup_prob",0.0)>0 and not (cfg["late_sup_prob"]<=1 and 0<=cfg["late_sup_min"]<=cfg["late_sup_max"]):
        raise ValueError("late_sup_prob in (0,1] needs 0<=late_sup_min<=late_sup_max.")


def _signal_stop(signum,frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _resolve_compile(cfg,device):
    """compile: True | False | 'auto' (CUDA capability >= 7.0; Triton/Inductor do not support P100)."""
    value = cfg["compile"]
    if value == "auto":
        value = device.type=="cuda" and torch.cuda.get_device_capability(device)>=(7,0)
    if not isinstance(value,bool):
        raise ValueError("compile must be True, False or 'auto'.")
    cfg["compile"] = value


def main(cfg):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    cfg = dict(cfg)
    validate_run_cfg(cfg)
    start = time.monotonic()
    deadline = float("inf") if cfg["max_hours"] is None else start+cfg["max_hours"]*3600
    # Self-tests run before CUDA initialization or distributed collectives.
    if cfg["run_selftests"] and int(os.environ.get("RANK","0"))==0:
        selftest()
    rank,ws,device = init_distributed()
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,_signal_stop)
    torch.manual_seed(cfg["seed"]+rank)
    np.random.seed(cfg["seed"]+rank)
    random.seed(cfg["seed"]+rank)
    if cfg["global_batch_size"]%ws:
        raise ValueError("global_batch_size must be divisible by the number of processes.")
    _resolve_precision(cfg,device)
    _resolve_compile(cfg,device)
    tr_x,tr_y,te_x,te_y,path,fingerprint = load_data(cfg)
    cfg.update(data_npz=path,data_fingerprint=fingerprint)
    gbs,lbs = cfg["global_batch_size"],cfg["global_batch_size"]//ws
    planned_steps = int(cfg["epochs"]*len(tr_x)/gbs)
    steps_per_iter = cfg["eval_interval"]*len(tr_x)//gbs
    total_iters = cfg["epochs"]//cfg["eval_interval"]
    actual_steps = steps_per_iter*total_iters
    if steps_per_iter==0:
        raise ValueError("eval_interval * number_of_train_puzzles must be >= global_batch_size.")
    stop_at = actual_steps if cfg["max_steps"] is None else min(cfg["max_steps"],actual_steps)
    mcfg = dict(cfg,batch_size=lbs,seq_len=cfg["grid"]**2,num_puzzle_identifiers=1)
    with torch.device(device):
        base = ACTLossHead(LT(mcfg),q_weight=cfg["q_weight"])
    base.train()
    if ws>1:
        with torch.no_grad():
            for p in list(base.parameters())+list(base.buffers()):
                dist.broadcast(p,src=0)
    optimizers,lrs = create_optimizers(base,cfg,ws)
    ema = EMAHelper(cfg["ema_rate"]) if cfg["ema"] else None
    if ema is not None:
        ema.register(base)
    out_dir = resolve_out_dir(cfg)
    cfg["out_dir"] = out_dir
    os.makedirs(out_dir,exist_ok=True)
    search_at = cfg["resume_from"] if cfg["resume_from"] is not None else out_dir
    path = find_latest_checkpoint(search_at)
    if cfg["resume_from"] is not None and not path:
        raise FileNotFoundError(f"Explicit resume_from contains no checkpoint: {search_at}")
    if cfg["require_resume"] and not path:
        raise FileNotFoundError("require_resume=True but no checkpoint exists.")
    ts = load_training_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device) if path else TrainState()
    if ts.step != ts.iter_id*steps_per_iter+ts.batch_in_iter:
        raise ValueError("Checkpoint cursor/step inconsistency.")
    if ts.batch_in_iter>=steps_per_iter or ts.iter_id>total_iters:
        raise ValueError("Checkpoint cursor is outside the dataset protocol.")
    if rank==0:
        with open(os.path.join(out_dir,"config.json"),"w",encoding="utf-8") as f:
            json.dump(dict(cfg,model_id=model_id_of(cfg)),f,ensure_ascii=False,indent=2)
    if cfg["compile"] and cfg["inductor_no_persist"]:
        try:
            import torch._inductor.config as ic
            ic.triton.persistent_reductions = False
        except (AttributeError,ImportError) as exc:
            if rank==0:
                print(f"[LT] optional Inductor setting unavailable: {exc}",flush=True)
    model = torch.compile(base,dynamic=False) if cfg["compile"] else base
    if rank==0:
        layer = base.model.inner.layers[0]
        print(f"[LT] {model_id_of(cfg)} torch={torch.__version__} device={device} "
              f"gpu={torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'} ranks={ws} local_bs={lbs}",flush=True)
        print(f"[LT] params={sum(p.numel() for p in base.parameters()):,} "
              f"trainable={sum(p.numel() for p in base.parameters() if p.requires_grad):,} amp={cfg['amp_dtype']} "
              f"activation_checkpoint=False compile={cfg['compile']}",flush=True)
        print(f"[LT] data={cfg['data_npz']} train={len(tr_x)} test={len(te_x)}",flush=True)
        print(f"[LT] planned steps={planned_steps}; actual steps={actual_steps}; stop_at={stop_at}; "
              f"1 step={cfg['blocks_per_seg']} blocks; loops={cfg['loops']} segments",flush=True)
        lam = torch.exp(-F.softplus(layer.clock_lam_raw.detach().double()))
        print(f"[LT] B: tag = c sin(phi_disp) + n omegaT, c={cfg['b_tag_bound']:g}, omegaT={cfg['b_block_omega']:g}; "
              f"lambda_K=lambda_V={float(lam.mean()):.6f} frozen (requires_grad={layer.clock_lam_raw.requires_grad}); "
              f"W <- rho W + Im S, rho={cfg['b_outer_rho']:g}; read = RoPE(q)(Hebb + alpha_h W)^T, "
              f"alpha_h={[round(x,4) for x in layer.stdp_alpha.detach().cpu().tolist()]}",flush=True)
        print("[LT] B: Q/K L2 (norm+eps) before learned 2D RoPE; V raw; token-sum writes; "
              "h <- RMS(h + W_o read) -> RMS(u + D(silu(g) v)); clock/traces/W/read FP32; "
              "projections/FFN follow AMP",flush=True)
        print(f"[LT] lr={cfg['lr']} warmup={cfg['lr_warmup_steps']} min_ratio={cfg['lr_min_ratio']} "
              f"weight_decay={cfg['weight_decay']} ema={cfg['ema_rate']} batch={gbs} nograd_fixed={cfg.get('nograd_fixed',0)}",flush=True)
        how = "RESUME "+str(path) if path else "NEW RUN"
        print(f"[LT] {how} step={ts.step} "
              f"next_iter={ts.iter_id} consumed_batches={ts.batch_in_iter} out={out_dir}",flush=True)
        if device.type=="cpu":
            print("[LT] CPU execution: suitable for self-tests, not a full d=832 training run.",flush=True)
    ds = SudokuTrainDataset(tr_x,tr_y,seed=cfg["seed"],num_aug=cfg["num_aug"],global_batch_size=gbs,
         rank=rank,world_size=ws,epochs_per_iter=cfg["eval_interval"],start_iter=ts.iter_id,
         total_iters=total_iters,skip_batches=ts.batch_in_iter)
    # Dedicated loader RNG means loader creation does not consume the model RNG on resume.
    generator = torch.Generator().manual_seed(cfg["seed"]+rank)
    kwargs = dict(batch_size=None,num_workers=cfg["dataloader_workers"],
                  pin_memory=device.type=="cuda",generator=generator)
    if cfg["dataloader_workers"]:
        kwargs.update(prefetch_factor=4,persistent_workers=False,multiprocessing_context="spawn")
    loader = DataLoader(ds,**kwargs)
    last_full = None
    last_eval_step = -1
    stopped = stop_requested(device,deadline)
    began = time.monotonic()
    jsonl = os.path.join(out_dir,"train.jsonl")
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pbar = None
    if rank==0 and sys.stdout.isatty():
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=stop_at,initial=min(ts.step,stop_at),dynamic_ncols=True)
        except ImportError:
            pass
    try:
        if not stopped and ts.step<stop_at:
            for it,batch in loader:
                if int(it)!=ts.iter_id:
                    raise RuntimeError(f"Data cursor mismatch: yielded {it}, expected {ts.iter_id}.")
                started = time.monotonic()
                metrics = train_batch(model,base,ts,batch,cfg,optimizers,lrs,planned_steps,rank,ws,device)
                if ema is not None:
                    ema.update(base)
                ts.batch_in_iter += 1
                boundary = ts.batch_in_iter==steps_per_iter
                if boundary:
                    ts.iter_id += 1
                    ts.batch_in_iter = 0
                if rank==0:
                    if pbar is not None:
                        pbar.update(1)
                    # Per-step record as in the research runner (seconds = one optimizer step on this rank).
                    with open(jsonl,"a",encoding="utf-8") as f:
                        f.write(json.dumps(dict(step=ts.step,seconds=time.monotonic()-started,
                                                elapsed=time.monotonic()-began,
                                                segment=int(ts.carry.steps.max()),
                                                **{k:v for k,v in metrics.items() if not k.startswith("_")}),
                                           allow_nan=False)+"\n")
                    if metrics["_count_raw"]>0:
                        last_full = (ts.step,metrics)
                    if ts.step%cfg["log_every"]==0:
                        msg = f"[LT] step {ts.step} lm_loss {metrics['lm_loss']:.5f} lr {metrics['lr']:.3g}"
                        if last_full:
                            fs,fm = last_full
                            msg += f" acc {fm['accuracy']:.4f} exact {fm['exact_accuracy']:.4f} [halt step {fs}]"
                        if device.type=="cuda":
                            msg += f" peak {torch.cuda.max_memory_allocated(device)/2**30:.2f}GiB"
                        print(msg,flush=True)
                if ts.step%cfg["stop_check_every"]==0 or boundary or ts.step>=stop_at:
                    stopped = stop_requested(device,deadline)
                if stopped:
                    break                         # Save FIRST; skip expensive final evaluation.
                due_save = cfg["save_every_steps"] and ts.step%cfg["save_every_steps"]==0
                due_milestone = cfg["milestone_every"] and ts.step%cfg["milestone_every"]==0
                if (boundary and cfg.get("save_at_boundary",True)) or due_save:
                    save_training_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device)
                if boundary:
                    evaluate(base,te_x,te_y,cfg,rank,ws,device,ts.step,ema,deadline)
                    last_eval_step = ts.step
                if due_milestone:
                    mdir = os.path.join(out_dir,"milestones")
                    save_training_checkpoint(mdir,ts,base,optimizers,ema,cfg,rank,ws,device,keep_last=0)
                    extrapolate(base,te_x,te_y,cfg,rank,ws,device,ts.step,ema,
                                cfg["milestone_extrap_segs"],os.path.join(mdir,f"extrap_step_{ts.step}.txt"),deadline)
                if ts.step>=stop_at:
                    break
                if (boundary or due_milestone) and stop_requested(device,deadline):
                    stopped = True
                    break
        # Save before evaluation so interruption during the final eval loses no training progress.
        p = save_training_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device)
        stopped = stopped or stop_requested(device,deadline)
        if not stopped and last_eval_step!=ts.step:
            evaluate(base,te_x,te_y,cfg,rank,ws,device,ts.step,ema,deadline)
        if rank==0:
            print(f"[LT] saved step={ts.step}: {p}"+(" [time limit / stop requested]" if stopped else ""),flush=True)
            if device.type=="cuda":
                print(f"[LT] peak allocated {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB, "
                      f"reserved {torch.cuda.max_memory_reserved(device)/2**30:.2f} GiB (rank 0)",flush=True)
    except BaseException:
        # Never write a mixed pre/post-optimizer checkpoint or start collectives after a rank has failed.
        if rank==0:
            print("[LT] run failed; last atomically saved checkpoint remains intact. "
                  "No partially completed step was saved.",flush=True)
        raise
    finally:
        if pbar is not None:
            pbar.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    return ts.step


# 9. CPU FP64 self-tests of model B (synthetic inputs; no dataset, no claim about accuracy) -------------------------
def _selftest_cfg():
    return dict(DEFAULT_CFG, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, global_batch_size=2, batch_size=2,
                seq_len=81, num_puzzle_identifiers=1, blocks_per_seg=3, loops=4, amp=False, amp_dtype="float32",
                forward_dtype="float64", compile=False, epochs=4, eval_interval=2, lr_warmup_steps=0, lr=1e-3,
                puzzle_emb_lr=1e-3, ema_rate=0.9, num_aug=10, run_selftests=False, dataloader_workers=0)


def _selftest_batch(seed, n=2):
    gen = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 11, (n, 81), generator=gen),
                labels=torch.randint(2, 11, (n, 81), generator=gen),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32))


def _pair_angle(x):
    return torch.atan2(x[..., 1::2], x[..., 0::2])


def _selftest_tags(g, close):
    """(a) tags = c sin(sum of wrapped pair-angle displacements since the lane's reset) + n omegaT, n = blocks since
    the reset (0 in the reset block), K and V, with a mid-run reset of one lane; the clock state carries phi and n."""
    B, H, T, P, blocks = 2, 2, 4, 3, 9
    D = 2 * P
    c, om = 0.14, 0.34
    clk = torch.zeros(B, H, T, 6 * P + 1)
    ref_k, ref_v, prev_k, prev_v = torch.zeros(B, H, T, P), torch.zeros(B, H, T, P), None, None
    n = torch.zeros(B, 1, 1, 1)
    for i in range(blocks):
        k = torch.randn(B, H, T, D, generator=g)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-4)
        v = torch.randn(B, H, T, D, generator=g)
        if i == 4:
            mask = torch.tensor([True, False]).view(-1, 1, 1, 1)
            clk = torch.where(mask, torch.zeros_like(clk), clk)
            ref_k, ref_v = torch.where(mask, torch.zeros_like(ref_k), ref_k), torch.where(mask, torch.zeros_like(ref_v), ref_v)
            prev_k = torch.where(mask, torch.full_like(prev_k, float("nan")), prev_k)
            prev_v = torch.where(mask, torch.full_like(prev_v, float("nan")), prev_v)
            n = torch.where(mask, torch.zeros_like(n), n)
        tag_k, tag_v, clk = bounded_clock_step(clk, k, v, c, om)
        if prev_k is not None:
            ref_k = ref_k + torch.nan_to_num(wrap_phase(_pair_angle(k) - prev_k), nan=0.0)
            ref_v = ref_v + torch.nan_to_num(wrap_phase(_pair_angle(v) - prev_v), nan=0.0)
        prev_k, prev_v = _pair_angle(k), _pair_angle(v)
        for got, ref, role in ((tag_k, ref_k, "K"), (tag_v, ref_v, "V")):
            expected = c * torch.sin(ref) + n * om                       # unwrapped explicit formula
            torch.testing.assert_close(torch.exp(1j * got), torch.exp(1j * expected), msg=f"{role} tag, block {i}", **close)
            assert bool(((got - wrap_phase(n * om)).abs() <= c + 1e-12).all()), "latency bound"
        torch.testing.assert_close(clk[..., -1:], (n + 1).expand(B, H, T, 1), rtol=0, atol=0)
        n = n + 1


def _selftest_memory(g, close):
    """(b) Memory path against explicit sums: Im S_b = Im sum over all (post, pre) event pairs since the lane's reset
    with weights (1 - lam) lam^lag e^{i(tag_post - tag_pre)} (general lam_K != lam_V); W_b = sum_j rho^(b-j) Im S_j
    since the reset; read = RoPE(q_hat) (Hebb_b + alpha W_b)^T. Mid-run reset of one lane; rho = 0.5 and 0
    (rho = 0 is the run-3 read)."""
    cfg = _selftest_cfg()
    torch.manual_seed(3)
    inner = LT(cfg).inner
    layer = inner.layers[0]
    H, D, P, T, B, blocks = inner.H, inner.dh, inner.P, 81, 2, 6
    with torch.no_grad():
        layer.clock_lam_raw.add_(0.5 * torch.randn(layer.clock_lam_raw.shape, generator=g))
        layer.stdp_alpha.copy_(torch.tensor([0.7, -1.3]))
    lam = torch.exp(-F.softplus(layer.clock_lam_raw.detach()))
    lam_k, lam_v = _rep2(lam[0])[None, :, None, :], _rep2(lam[1])[None, :, None, :]
    alpha = layer.stdp_alpha.detach()[None, :, None, None]
    tables = inner.rope_tables(layer)

    def rope(x):
        return inner.apply_rope(x, layer, tables)
    seq = []
    for i in range(blocks):
        q, k, v = (torch.randn(B, H, T, D, generator=g) for _ in range(3))
        fresh = torch.tensor([True, True]) if i == 0 else torch.tensor([True, False]) if i == 3 else None
        seq.append((q, k, v, fresh))
    c, om = cfg["b_tag_bound"], cfg["b_block_omega"]
    for rho in (0.5, 0.0):
        inner.config.b_outer_rho = rho
        clk = ek = ev = w = None
        got = []
        with torch.no_grad():
            for q, k, v, fresh in seq:
                read, clk, ek, ev, w = inner.memory_step(layer, q, k, v, clk, ek, ev, w, fresh)
                got.append((read, w))
        # explicit reference
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + cfg["eps"])
        start = torch.zeros(B, dtype=torch.long)
        tk, tv, starts = [], [], []
        acc_k = acc_v = prev_k = prev_v = None
        for i, (q, k, v, fresh) in enumerate(seq):
            kh = unit(k)
            if fresh is not None:
                start = torch.where(fresh, torch.full_like(start, i), start)
            m = (start == i).view(-1, 1, 1, 1)
            if acc_k is None:
                acc_k, acc_v = torch.zeros(B, H, T, P), torch.zeros(B, H, T, P)
            else:
                acc_k = torch.where(m, torch.zeros_like(acc_k), acc_k + wrap_phase(_pair_angle(kh) - prev_k))
                acc_v = torch.where(m, torch.zeros_like(acc_v), acc_v + wrap_phase(_pair_angle(v) - prev_v))
            prev_k, prev_v = _pair_angle(kh), _pair_angle(v)
            n = (i - start).view(-1, 1, 1, 1).to(torch.float64)
            tk.append(c * torch.sin(acc_k) + n * om)
            tv.append(c * torch.sin(acc_v) + n * om)
            starts.append(start.clone())
        ims = []
        for i, (q, k, v, _) in enumerate(seq):
            post_now = v * torch.exp(1j * _rep2(tv[i]))
            pre_now = rope(unit(k)) * torch.exp(-1j * _rep2(tk[i]))
            s = torch.zeros(B, H, D, D, dtype=torch.complex128)
            for j in range(i + 1):
                active = (j >= starts[i]).view(-1, 1, 1, 1).to(torch.float64)
                pre_then = rope(unit(seq[j][1])) * (1 - lam_k) * lam_k ** (i - j) * torch.exp(-1j * _rep2(tk[j]))
                s = s + active * torch.einsum("bhna,bhnj->bhaj", post_now, pre_then)       # pre first or same block
                if j < i:
                    post_then = seq[j][2] * (1 - lam_v) * lam_v ** (i - j) * torch.exp(1j * _rep2(tv[j]))
                    s = s + active * torch.einsum("bhna,bhnj->bhaj", post_then, pre_now)   # post first
            ims.append(s.imag)
        for i, (q, k, v, _) in enumerate(seq):
            w_ref = torch.zeros(B, H, D, D)
            for j in range(i + 1):
                active = (j >= starts[i]).view(-1, 1, 1, 1).to(torch.float64)
                w_ref = w_ref + active * rho ** (i - j) * ims[j]
            hebb = torch.einsum("bhna,bhnj->bhaj", v, rope(unit(k)))
            read_ref = rope(unit(q)) @ (hebb + alpha * w_ref).transpose(-1, -2)
            torch.testing.assert_close(got[i][1], w_ref, msg=f"W, block {i}, rho {rho}", **close)
            torch.testing.assert_close(got[i][0], read_ref, msg=f"read, block {i}, rho {rho}", **close)
        assert float(got[-1][1][1].abs().max()) > 1e-3 and float(got[2][1].abs().max()) > 1e-3


def _selftest_model(close):
    """(c) segment split equivalence, (d) per-lane reset, (e) gradients."""
    cfg = _selftest_cfg()
    batch, alt = _selftest_batch(1), _selftest_batch(2)

    def build():
        torch.manual_seed(5)
        model = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
        with torch.no_grad():
            layer = model.model.inner.layers[0]
            layer.b_down.weight.normal_(0, 0.03)
            layer.stdp_alpha.copy_(torch.tensor([0.7, 1.2]))
        return model
    base = build()
    lt = base.model
    base.eval()
    names = ("current_hidden", "coupling", "key_trace", "value_trace", "stdp_w")
    with torch.no_grad():
        lt.config.blocks_per_seg = 3
        c1, _ = lt(lt.initial_carry(batch), batch)
        c2, o2 = lt(c1, batch)
        lt.config.blocks_per_seg = 6
        cl, ol = lt(lt.initial_carry(batch), batch)
        lt.config.nograd_blocks, lt.config.blocks_per_seg = 3, 3
        cn, on = lt(lt.initial_carry(batch), batch)
        lt.config.nograd_blocks = 0
    assert float(c2.stdp_w.abs().max()) > 0 and float(c2.coupling[..., :2 * lt.inner.P].abs().max()) > 0
    assert torch.equal(c2.coupling[..., -1], torch.full_like(c2.coupling[..., -1], 6.0))
    for name in names:
        torch.testing.assert_close(getattr(c2, name), getattr(cl, name), msg=name, **close)
        torch.testing.assert_close(getattr(cn, name), getattr(cl, name), msg=name, **close)
    torch.testing.assert_close(o2["logits"], ol["logits"], **close)
    torch.testing.assert_close(on["logits"], ol["logits"], **close)
    checks = ["(c) two 3-block segments and a 3 no-grad + 3 block segment equal one 6-block unroll "
              "(hidden, clock state incl. block counter, both traces, W, logits)"]

    with torch.no_grad():
        lt.config.blocks_per_seg = 3
        c1, _ = lt(lt.initial_carry(batch), batch)
        c2, _ = lt(c1, alt)
        cm, _ = lt(replace(c1, halted=torch.tensor([True, False])), alt)
        cf, _ = lt(lt.initial_carry(alt), alt)
    for name in names:
        torch.testing.assert_close(getattr(cm, name)[0], getattr(cf, name)[0], msg=name, **close)
        torch.testing.assert_close(getattr(cm, name)[1], getattr(c2, name)[1], msg=name, **close)
    assert cm.coupling[0, ..., -1].eq(3).all() and cm.coupling[1, ..., -1].eq(6).all()
    checks.append("(d) per-lane reset: a fresh lane equals a fresh run (counter n, clock, traces, W zeroed), "
                  "a continuing lane is unaffected")

    model = build()
    model.train()
    layer = model.model.inner.layers[0]
    _, loss, *_ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
    loss.backward()
    unused = {"model.inner.layers.0.theta_k_raw", "model.inner.layers.0.theta_v_raw",
              "model.inner.layers.0.trace_lam_raw", "model.inner.layers.0.clock_lam_raw"}
    for name, p in model.named_parameters():
        if name in unused:
            assert p.grad is None, name
        else:
            assert p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()), name
            assert float(p.grad.abs().max()) > 0, name
    assert not layer.clock_lam_raw.requires_grad and layer.clock_lam_raw.grad is None
    assert float(model.model.puzzle_emb.local_weights.grad.abs().max()) > 0
    opts, _ = create_optimizers(model, cfg, 1)
    assert all(p is not layer.clock_lam_raw for o in opts for g_ in o.param_groups for p in g_["params"])
    ema = EMAHelper(0.9)
    ema.register(model)
    assert "model.inner.layers.0.clock_lam_raw" not in ema.shadow and "model.inner.layers.0.stdp_alpha" in ema.shadow
    torch.testing.assert_close(torch.exp(-F.softplus(layer.clock_lam_raw)),
                               torch.full_like(layer.clock_lam_raw, cfg["b_trace_lambda"]), rtol=1e-12, atol=1e-12)
    # Directional derivatives against central differences (5-block segment, so alpha reaches W_b through
    # rho W_{b-1}): along alpha alone and along a random direction over every trainable parameter. A W detached
    # between blocks inside the segment changes both.
    model.zero_grad(set_to_none=True)
    model.model.config.blocks_per_seg = 5
    _, loss, *_ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
    loss.backward()
    gen = torch.Generator().manual_seed(23)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    directions = []
    for only_alpha in (True, False):
        d = {n: (torch.randn(p.shape, generator=gen) if (not only_alpha or n.endswith("stdp_alpha")) else torch.zeros_like(p))
             for n, p in trainable}
        norm = math.sqrt(sum(float(x.square().sum()) for x in d.values()))
        directions.append({n: x / norm for n, x in d.items()})
    results = []
    for d in directions:
        analytic = sum(float((p.grad * d[n]).sum()) for n, p in trainable if p.grad is not None)
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                for n, p in trainable:
                    p.add_(sign * 1e-6 * d[n])
                _, lp, *_ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
                for n, p in trainable:
                    p.sub_(sign * 1e-6 * d[n])
                values.append(float(lp))
        fd = (values[0] - values[1]) / 2e-6
        assert abs(fd - analytic) <= 1e-6 * max(1.0, abs(analytic)), (fd, analytic)
        results.append((analytic, fd))
    model.model.config.blocks_per_seg = 3
    checks.append("(e) finite nonzero gradients for alpha and every trainable parameter (unused run-3 slots and "
                  "lambda: no gradient); lambda frozen at 0.5, outside optimizer/EMA; directional derivatives through the W "
                  "accumulation (5-block segment) equal central differences: alpha %.6e vs %.6e, all params %.6e vs %.6e"
                  % (results[0] + results[1]))
    return checks


def _selftest_resume(cfg, checks, device):
    """(f) checkpoint save -> load restores carry (incl. W and the counter) and the next step exactly."""
    import tempfile
    batches = [_selftest_batch(10 + j) for j in range(3)]
    torch.manual_seed(117)
    base = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
    opts, lrs = create_optimizers(base, cfg, 1)
    ema = EMAHelper(cfg["ema_rate"]); ema.register(base)
    ts = TrainState()

    def update(model, state, optimizers, rates, shadow, data):
        out = train_batch(model, model, state, data, cfg, optimizers, rates, 16, 0, 1, device)
        shadow.update(model); state.batch_in_iter += 1
        return out
    for j in range(2):
        update(base, ts, opts, lrs, ema, batches[j])
    names = ("current_hidden", "coupling", "key_trace", "value_trace", "stdp_w", "steps", "halted")
    for name in names[:5]:
        assert not getattr(ts.carry, name).requires_grad and torch.isfinite(getattr(ts.carry, name)).all()
    assert float(ts.carry.stdp_w.abs().max()) > 0 and ts.carry.coupling[..., -1].eq(6).all()
    with tempfile.TemporaryDirectory() as directory:
        path = save_training_checkpoint(directory, ts, base, opts, ema, cfg, 0, 1, device)
        saved = {n: getattr(ts.carry, n).clone() for n in names}
        expected = update(base, ts, opts, lrs, ema, batches[2])
        resumed = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
        ro, rl = create_optimizers(resumed, cfg, 1)
        rema = EMAHelper(cfg["ema_rate"]); rema.register(resumed)
        rs = load_training_checkpoint(path, resumed, ro, rema, cfg, 0, 1, device)
        assert rs.step == 2 and rs.batch_in_iter == 2
        for name in names:
            assert torch.equal(getattr(rs.carry, name), saved[name]), name
        actual = update(resumed, rs, ro, rl, rema, batches[2])
        assert actual["lm_loss"] == expected["lm_loss"] and actual["accuracy"] == expected["accuracy"], (actual, expected)
        for name, value in base.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[name]), name
        for name, value in ema.shadow.items():
            assert torch.equal(value, rema.shadow[name]), name
        for name in names:
            assert torch.equal(getattr(ts.carry, name), getattr(rs.carry, name)), name
        bad = dict(cfg, b_outer_rho=0.25)
        other = ACTLossHead(LT(bad), q_weight=cfg["q_weight"])
        oo, _ = create_optimizers(other, bad, 1)
        oe = EMAHelper(cfg["ema_rate"]); oe.register(other)
        try:
            load_training_checkpoint(path, other, oo, oe, bad, 0, 1, device)
            raise AssertionError("resume with a different rho was accepted")
        except ValueError:
            pass
    checks.append("(f) checkpoint save -> load: identical carry (h, clock incl. counter, traces, W), identical next-step "
                  "loss/accuracy, weights, EMA and carry after it; a changed B constant refuses to resume")


def selftest():
    """CPU FP64 checks (a)-(f) of model B with synthetic inputs."""
    device = torch.device("cpu")
    old_dtype, old_threads, rng0 = torch.get_default_dtype(), torch.get_num_threads(), _rng_state(device)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(min(2, old_threads))
    close = dict(rtol=1e-9, atol=1e-11)
    checks = []
    began = time.monotonic()
    try:
        _selftest_tags(torch.Generator().manual_seed(19), close)
        checks.append("(a) tag = c sin(sum of wrapped pair-angle displacements since reset) + n omegaT for K and V, "
                      "n = blocks since the puzzle start, with a mid-run reset of one lane; |tag - n omegaT| <= c")
        _selftest_memory(torch.Generator().manual_seed(11), close)
        checks.append("(b) Im S equals the explicit sum over all (post, pre) event pairs; W equals the explicit geometric "
                      "sum of Im S increments since reset (rho 0.5 and 0); read = RoPE(q)(Hebb + alpha W)^T; mid-run reset")
        checks += _selftest_model(close)
        _selftest_resume(_selftest_cfg(), checks, device)
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_num_threads(old_threads)
        _restore_rng(rng0, device)
    for item in checks:
        print("[selftest] PASS " + item, flush=True)
    print(f"[selftest] {len(checks)}/{len(checks)} passed (CPU, FP64, synthetic inputs, {time.monotonic()-began:.1f}s).",
          flush=True)
    return checks


def _cli():
    ap = argparse.ArgumentParser(description="Self-contained LT phase-clock STDP model B trainer")
    ap.add_argument("--config",help="JSON overrides for DEFAULT_CFG")
    ap.add_argument("--data")
    ap.add_argument("--out_dir")
    ap.add_argument("--resume_from")
    ap.add_argument("--max_steps",type=int)
    ap.add_argument("--no_compile",action="store_true")
    ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--local-rank","--local_rank",type=int,default=0)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    cfg = dict(DEFAULT_CFG)
    if a.config:
        with open(a.config,encoding="utf-8") as f:
            overrides = json.load(f)
        unknown = set(overrides)-set(cfg)
        if unknown:
            raise ValueError(f"Unknown CFG keys: {sorted(unknown)}")
        cfg.update(overrides)
    for attr,key in (("data","data_npz"),("out_dir","out_dir"),("resume_from","resume_from"),("max_steps","max_steps")):
        value = getattr(a,attr)
        if value is not None:
            cfg[key] = value
    if a.no_compile:
        cfg["compile"] = False
    if cfg.get("selftest_only"):
        selftest()
        return
    main(cfg)


if __name__=="__main__":
    _cli()

'''


def _resolve_data_path(cfg):
    """Kaggle Input path when it exists; otherwise the repository/local copy; otherwise None (the trainer then
    searches /kaggle/input/**/sudoku_lt_1k.npz, ./data and the working directory)."""
    candidates = [cfg.get("data_npz")]
    here = globals().get("__file__")
    if here:
        candidates.append(Path(here).resolve().parents[1] / "data" / "sudoku_lt_1k.npz")
    candidates += [Path.cwd() / "data" / "sudoku_lt_1k.npz", Path.cwd() / "sudoku_lt_1k.npz"]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return str(Path(c).expanduser().resolve())
    return None


def launch_lt_one_cell():
    """Runs the trainer synchronously in a new process (torchrun for several GPUs); the notebook's CUDA state and
    argv are not inherited. Interrupting the cell asks the trainer to save at the next completed step."""
    work_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    tag = "phaseclock_bounded_leaky"
    runtime = Path(tempfile.mkdtemp(prefix=f"_lt_{tag}_", dir=str(work_root)))
    trainer_path = runtime / f"train_{tag}.py"
    config_path = runtime / "config.json"
    cfg = dict(CFG)
    cfg["data_npz"] = _resolve_data_path(cfg)
    log_dir = Path(cfg["out_dir"]).expanduser().resolve() if cfg["out_dir"] else work_root / f"lt_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg["out_dir"] = str(log_dir)
    log_path = log_dir / "train.log"
    trainer_path.write_text(_TRAINER_SOURCE, encoding="utf-8")
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.get("selftest_only"):
        nproc, cmd = 1, [sys.executable, str(trainer_path), "--selftest"]
        ngpu = None
    else:
        # GPU query in a separate process as well; the notebook itself never imports torch.
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
            check=True, capture_output=True, text=True,
        )
        ngpu = int(probe.stdout.strip().splitlines()[-1])
        requested = CFG["num_processes"]
        nproc = max(1, ngpu) if requested == "auto" else int(requested)
        if nproc < 1 or (ngpu > 0 and nproc > ngpu):
            raise ValueError(f"num_processes={nproc}, available GPUs={ngpu}")
        if CFG["global_batch_size"] % nproc:
            raise ValueError("global_batch_size must be divisible by num_processes.")
        cmd = [sys.executable]
        if nproc > 1:
            cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={nproc}"]
        cmd += [str(trainer_path), "--config", str(config_path)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    print(f"[Kaggle] GPU={ngpu}, processes={nproc}; trainer: {trainer_path}; data: {cfg['data_npz']}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, start_new_session=(os.name == "posix"))
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    log_file.write(f"\n[Kaggle] launch {tag}: " + str(trainer_path) + "\n")
    print(f"[Kaggle] training log: {log_path}", flush=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        rc = proc.wait()
    except KeyboardInterrupt:
        print("\n[Kaggle] stop requested; the trainer saves at the next completed step.", flush=True)
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGINT)
        else:
            proc.terminate()
        try:
            output, _ = proc.communicate(timeout=120)
            if output:
                print(output, end="", flush=True)
                log_file.write(output)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            print("[Kaggle] killed; the last saved checkpoint is intact.", flush=True)
        raise
    finally:
        proc.stdout.close()
        log_file.close()
    if rc:
        raise RuntimeError(f"Trainer exited with code {rc}; see the log above.")


if __name__ == "__main__":
    # In a notebook cell: launch. As a script with arguments (e.g. --selftest, --config run.json): run the trainer here.
    if "__file__" in globals() and "ipykernel" not in sys.modules and len(sys.argv) > 1:
        exec(compile(_TRAINER_SOURCE, __file__ + "::trainer", "exec"), globals())
    else:
        launch_lt_one_cell()
else:
    # Imported (e.g. as __mp_main__ by a spawned DataLoader worker): define the trainer's classes and functions.
    exec(compile(_TRAINER_SOURCE, globals().get("__file__", "phaseclock_bounded_leaky_cell.py") + "::trainer", "exec"), globals())
