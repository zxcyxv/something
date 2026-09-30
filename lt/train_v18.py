# -*- coding: utf-8 -*-
# LT v1.8 — independent complex Q/K, old-old trace subtraction, hidden_size=512.
# Kaggle: paste this entire file into one cell and connect sudoku_lt_1k.npz.
# Local: python lt/train_v18.py --data data/sudoku_lt_1k.npz --out_dir runs/v18_qk_delta512
# Q and K each have 32 complex channels per head (8 heads, 64 real coordinates).
# Read/write share psi; there is no Attn_Beta or beta parameter.
# Separate Q/K traces share rho/omega so the raw old-old term is rho^2*C(previous).
# Existing v1.8 selective gates, current agree, EMA, loss and data harness are retained.
# This 512-dimensional architecture starts fresh; old 832-dimensional checkpoints differ.
# One update is 8 blocks; carry is detached only at segment boundaries.

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

CFG = {'data_npz': '/kaggle/input/datasets/jrjinwoo/sudoku-lt-1k/sudoku_lt_1k.npz',
 'num_aug': 1000,
 'test_size': 2048,
 'hidden_size': 512,
 'num_heads': 8,
 'loops': 16,
 'blocks_per_seg': 8,
 'num_layers': 1,
 'grid': 9,
 'vocab_size': 11,
 'mlp_expansion': 4.0,
 'alpha_init': 0.1,
 'dist_decay': True,
 'eps': 0.0001,
 'psi_zero': False,
 'puzzle_emb_ndim': 512,
 'legacy_gauge': False,
 'block_order': 'post',
 'use_trace': True,
 'forward_dtype': 'float32',
 'amp': True,
 'amp_dtype': 'bfloat16',
 'activation_checkpoint': True,
 'stdp': True,
 'stdp_eta_init': 0.05,
 'stdp_gain_init': 1.0,
 'stdp_lam_init': 0.25,
 'stdp_gain_fixed': -1.0,
 'stdp_lam_fixed': -1.0,
 'global_batch_size': 128,
 'epochs': 50000,
 'lr': 0.0001,
 'lr_min_ratio': 1.0,
 'lr_warmup_steps': 2000,
 'lr_rewarm_start': None,
 'lr_rewarm_steps': 10000,
 'lr_rewarm_from_ratio': 0.1,
 'weight_decay': 1.0,
 'beta1': 0.9,
 'beta2': 0.95,
 'puzzle_emb_lr': 0.0001,
 'puzzle_emb_weight_decay': 1.0,
 'grad_accum_steps': 1,
 'q_weight': 0.5,
 'seed': 0,
 'ema': True,
 'ema_rate': 0.999,
 'eval_interval': 250,
 'compile': True,
 'inductor_no_persist': True,
 'out_dir': '/kaggle/working/lt_v18_qk_delta512',
 'resume_from': None,
 'require_resume': False,
 'init_from': None,
 'keep_last': 2,
 'save_every_steps': 2000,
 'milestone_every': 10000,
 'milestone_extrap_segs': 128,
 'milestone_extrap_n': 512,
 'max_hours': 11.5,
 'max_steps': None,
 'log_every': 250,
 'stop_check_every': 25,
 'dataloader_workers': 1,
 'run_selftests': True,
 'num_processes': 'auto',
 'address_projection': 'split_linear',
 'trace_rho_init': 0.5,
 'inj_gate_init': 0.25,
 'gamma_init': 0.1,
 'plastic_select': True,
 'select_g_max': 4.0,
 'late_sup_prob': 0.0,
 'late_sup_min': 16,
 'late_sup_max': 112,
 'nograd_fixed': 0,
 'nograd_every': 0,
 'nograd_start': 0,
 'nograd_max': 16}

_TRAINER_SOURCE = r'''# -*- coding: utf-8 -*-
"""LT v1.8: independent complex Q/K, trace-pair subtraction, d=512.

Read and write use the same psi/spatial kernel. The write removes carried
old-Q x old-K contributions with CURRENT Q/K trace norms on both terms.
Q/K have separate traces but share channelwise mu=rho*exp(i*omega), so
the removed raw correlation is rho^2 times the previous correlation.
The existing selective keep/write/read gates and training harness are retained.
Paste the complete outer file into one Kaggle cell; no repository imports.
"""
import argparse
import copy
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
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint

MODEL_ID = "lt-v18-split-qk-trace-delta-v1"
MODEL_ID_V18 = "lt-v18-split-qk-trace-delta-selective-v1"
IGNORE_LABEL_ID = -100
_STOP_REQUESTED = False


def model_id_of(cfg):
    return MODEL_ID_V18 if cfg.get("plastic_select") else MODEL_ID

# DEFAULT_CFG is also copied to the top of the one-cell notebook.
DEFAULT_CFG = {'data_npz': None,
 'num_aug': 1000,
 'test_size': 2048,
 'hidden_size': 512,
 'num_heads': 8,
 'loops': 16,
 'blocks_per_seg': 8,
 'num_layers': 1,
 'grid': 9,
 'vocab_size': 11,
 'mlp_expansion': 4.0,
 'alpha_init': 0.1,
 'dist_decay': True,
 'eps': 0.0001,
 'psi_zero': False,
 'puzzle_emb_ndim': 512,
 'legacy_gauge': False,
 'block_order': 'post',
 'use_trace': True,
 'forward_dtype': 'float32',
 'amp': True,
 'amp_dtype': 'bfloat16',
 'activation_checkpoint': True,
 'stdp': True,
 'stdp_eta_init': 0.05,
 'stdp_gain_init': 1.0,
 'stdp_lam_init': 0.25,
 'stdp_gain_fixed': -1.0,
 'stdp_lam_fixed': -1.0,
 'global_batch_size': 128,
 'epochs': 50000,
 'lr': 0.0001,
 'lr_min_ratio': 1.0,
 'lr_warmup_steps': 2000,
 'lr_rewarm_start': None,
 'lr_rewarm_steps': 10000,
 'lr_rewarm_from_ratio': 0.1,
 'weight_decay': 1.0,
 'beta1': 0.9,
 'beta2': 0.95,
 'puzzle_emb_lr': 0.0001,
 'puzzle_emb_weight_decay': 1.0,
 'grad_accum_steps': 1,
 'q_weight': 0.5,
 'seed': 0,
 'ema': True,
 'ema_rate': 0.999,
 'eval_interval': 250,
 'compile': True,
 'inductor_no_persist': True,
 'out_dir': None,
 'resume_from': None,
 'require_resume': False,
 'init_from': None,
 'keep_last': 2,
 'save_every_steps': 2000,
 'milestone_every': 10000,
 'milestone_extrap_segs': 128,
 'milestone_extrap_n': 512,
 'max_hours': 11.5,
 'max_steps': None,
 'log_every': 250,
 'stop_check_every': 25,
 'dataloader_workers': 1,
 'run_selftests': True,
 'num_processes': 'auto',
 'address_projection': 'split_linear',
 'trace_rho_init': 0.5,
 'inj_gate_init': 0.25,
 'gamma_init': 0.1,
 'plastic_select': True,
 'select_g_max': 4.0,
 'late_sup_prob': 0.0,
 'late_sup_min': 16,
 'late_sup_max': 112,
 'nograd_fixed': 0,
 'nograd_every': 0,
 'nograd_start': 0,
 'nograd_max': 16}

PRESETS = {
    "v1.8": dict(hidden_size=512, puzzle_emb_ndim=512, num_heads=8,
                 legacy_gauge=False, block_order="post", use_trace=True,
                 address_projection="split_linear", plastic_select=True),
}

# 1. Initialization / sparse puzzle embedding ---------------------------------
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


# 2. v1.8 model: independent complex Q/K, rotating traces, delta writer
@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    coupling: Optional[torch.Tensor] = None       # STDP 결합 기억 w [B,H,T,T]
    fresh: Optional[torch.Tensor] = None          # [B] bool — 이 퍼즐의 w·z 가 아직 초기화 전
    trace: Optional[torch.Tensor] = None          # Q/K 흔적 [B,T,H,p,2,2]: penultimate Q/K, last real/imag


@dataclass
class LTConfig:
    """Independent Q/K address projections with the shared rotating trace rule.

    Both traces use the same channelwise rho/omega. Only the writer removes
    old-old activity pairs; normalization, agree, EMA and selective gates remain.
    """
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int = 512
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8     # 세그먼트당 블록 수 (× num_layers)
    num_layers: int = 1         # 가중치 벌 수. 블록 k → layers[k % num_layers]
    mlp_expansion: float = 4.0
    alpha_init: float = 0.1
    legacy_gauge: bool = False  # True: 주입 계수·γ 를 학습 스칼라로 (9/1 원본). False: √d 고정, γ=1/d, 임베딩 init 1/√d
    inj_gate_init: float = 0.25
    gamma_init: float = 0.1
    dist_decay: bool = True     # 거리 감쇠 e^{−α‖Δ‖₁} (헤드별 α)
    eps: float = 1e-4
    amp: bool = True
    amp_dtype: str = "auto"  # outer harness resolves auto before compilation
    activation_checkpoint: bool = True  # recompute each block; no within-segment detach
    forward_dtype: str = "float32"
    address_projection: str = "split_linear"  # independent wq/wk [H,dh,d]
    psi_zero: bool = False      # ψ=0; independent Q/K still allow asymmetry
    stdp: bool = True           # corrected trace write x current agree; EMA memory
    stdp_eta_init: float = 0.05
    stdp_gain_init: float = 1.0
    stdp_lam_init: float = 0.25
    stdp_gain_fixed: float = -1.0   # ≥0 이면 g 를 이 값으로 고정
    stdp_lam_fixed: float = -1.0    # ≥0 이면 λ 를 이 값으로 고정
    block_order: str = "post"   # 주입 → 스텝 → 경계 → Φ
    use_trace: bool = True      # separate Q/K traces; shared μ_j = ρ_j e^{iω_j}
    trace_rho_init: float = 0.5
    plastic_select: bool = True     # v1.8: state-dependent keep/write/read gates
    select_g_max: float = 4.0       # gain bound only; corrected G need not be <=1
    nograd_blocks: int = 0          # 세그먼트 앞에서 gradient 없이 돌리는 블록 수 (하네스가 스텝마다 정한다)

    def __post_init__(self):
        if self.amp_dtype not in ("auto", "bfloat16", "float32"):
            raise ValueError("amp_dtype must be auto, bfloat16, or float32")
        if self.address_projection != "split_linear":
            raise ValueError("This v1.8 uses independent split_linear Q/K projections")
        if not self.use_trace or not self.stdp:
            raise ValueError("This v1.8 requires use_trace=True and stdp=True")
        if not 0 < self.trace_rho_init < 1:
            raise ValueError("trace_rho_init must lie in (0, 1)")
        if self.eps <= 0 or self.puzzle_emb_ndim > self.hidden_size:
            raise ValueError("eps must be positive; puzzle_emb_ndim must not exceed hidden_size")
        if self.plastic_select:
            if not self.stdp:
                raise ValueError("plastic_select requires stdp=True")
            if self.stdp_gain_fixed >= 0 or self.stdp_lam_fixed >= 0:
                raise ValueError("plastic_select generates g and λ; stdp_gain_fixed/stdp_lam_fixed must stay -1")
            if not 0 < self.stdp_gain_init < self.select_g_max:
                raise ValueError("stdp_gain_init must lie in (0, select_g_max)")

    @classmethod
    def from_dict(cls, d: dict) -> "LTConfig":
        known = {f.name for f in fields(cls)}
        dd = {k: v for k, v in d.items() if k in known}
        if "use_trace" not in d and "trace_rho_init" in d:     # v1.6/v1.7 체크포인트 cfg 호환 (use_trace 키가 없던 시절)
            dd["use_trace"] = True
        return cls(**dd)


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LTLayer(nn.Module):
    """One layer: independent real/imag stacked complex Q and K projections."""

    def __init__(self, config: "LTConfig", H, d, dh, p) -> None:
        super().__init__()
        self.wq = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        self.wk = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        if config.psi_zero:
            self.register_buffer("psi", torch.zeros(H, p), persistent=False)
        else:
            self.psi = nn.Parameter(torch.rand(H, p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(config.alpha_init)))
        w_sh = torch.zeros(H, dh, d)
        for m in range(H):
            w_sh[m, :, m * dh:(m + 1) * dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, dh, d) / math.sqrt(d))
        if config.use_trace:
            _lg = math.log(config.trace_rho_init / (1 - config.trace_rho_init))
            self.mu_rho_raw = nn.Parameter(torch.full((H, p), _lg))
            self.mu_omega = nn.Parameter((torch.rand(H, p) * 2 - 1) * (math.pi / 2))
        if config.stdp:
            lg = lambda x: math.log(x / (1 - x))
            if config.plastic_select:
                # v1.8: 게이트 로짓 = sel_w·q + bias.  sel_w=0 이면 헤드별 상수 게이트이고,
                # 그 값은 v1.71 의 η·g·λ 와 같다 (A = exp(−Δ) = 1−η).
                self.sel_w = nn.Parameter(torch.zeros(3, H, d))            # [Δ, g, λ] × 헤드 × 입력
                self.dt_bias = nn.Parameter(torch.full((H,), inv_softplus(-math.log1p(-config.stdp_eta_init))))
                self.gsel_bias = nn.Parameter(torch.full((H,), lg(config.stdp_gain_init / config.select_g_max)))
                self.lsel_bias = nn.Parameter(torch.full((H,), lg(config.stdp_lam_init)))
            else:
                self.eta_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_eta_init)))
                self.lam_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_lam_init)))
                self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(config.stdp_gain_init)))
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()

    @property
    def alpha(self): return F.softplus(self.alpha_raw)


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        assert T == g * g and d % H == 0 and (d // H) % 2 == 0
        self.d, self.H = d, H
        self.dh = d // H
        self.p = self.dh // 2
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.embed = nn.Embedding(config.vocab_size, d)
        if config.legacy_gauge:
            self.embed_scale = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
            self.gamma_raw = nn.Parameter(torch.tensor(inv_softplus(config.gamma_init)))
        else:
            self.gamma = 1.0 / d
            self.embed_scale = math.sqrt(d)
            with torch.no_grad():
                trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
        self.w_cls = nn.Linear(d, config.vocab_size)
        assert config.num_layers >= 1
        assert config.block_order in ("pre", "post"), f"block_order: pre | post (받은 값 {config.block_order})"
        self.stdp = config.stdp
        self.use_trace = config.use_trace
        self.layers = nn.ModuleList([LTLayer(config, H, d, self.dh, self.p) for _ in range(config.num_layers)])
        if config.num_layers > 1:
            sd0 = self.layers[0].state_dict()
            for Lx in self.layers[1:]:
                Lx.load_state_dict({k: v.clone() for k, v in sd0.items()})
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)

    # ---------------------------------------------------------------- 부품 (L = 레이어)
    def W_QK(self, L):
        """Independent complex projections: ((Aq,Bq), (Ak,Bk))."""
        return ((L.wq[:, :self.p], L.wq[:, self.p:]),
                (L.wk[:, :self.p], L.wk[:, self.p:]))

    def kernel(self, L):
        """One psi/spatial phase kernel, shared by instantaneous read and trace write."""
        decay_h = (torch.exp(-L.alpha[:, 0, None, None] * self.l1)
                   if self.config.dist_decay else
                   torch.ones_like(self.l1).expand(L.alpha.shape[0], -1, -1))
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w
        A = (ppos + L.psi[..., None] / 2).permute(2, 0, 1)
        B = (ppos - L.psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def attn_xy(self, qxy, kxy, kc):
        """Re(Q K*) after phase rotation and spatial decay; [B,H,T,T]."""
        qx, qy = qxy
        kx, ky = kxy
        decay_h, cosA, sinA, cosB, sinB = kc
        qr = qx * cosA - qy * sinA
        qi = qx * sinA + qy * cosA
        kr = kx * cosB - ky * sinB
        ki = kx * sinB + ky * cosB
        a = (torch.einsum('bthj,bnhj->bhtn', qr, kr) +
             torch.einsum('bthj,bnhj->bhtn', qi, ki))
        return a * decay_h.unsqueeze(0)

    def phi(self, h):
        g = F.softplus(self.gamma_raw) if hasattr(self, "gamma_raw") else self.gamma
        return h / torch.sqrt(1.0 + g * h.pow(2).sum(-1, keepdim=True))


    def injection(self, batch):
        inj = self.embed(batch["inputs"].to(torch.long))
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.d - self.puzzle_emb_ndim
            if pad > 0: pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    # ---------------------------------------------------------------- 세그먼트

    def empty_carry(self, batch_size, device=None):
        device = self.init_hidden.device if device is None else device
        return LTCarry(current_hidden=torch.empty(batch_size, self.config.seq_len, self.d,
                                                  dtype=self.forward_dtype, device=device))


    def reset_carry(self, reset_flag, carry):
        return replace(carry, current_hidden=torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))


    def forward(self, carry, batch):
        # Main resolves auto once before compilation; direct LT callers can also
        # use auto. CPU and GPUs without native BF16 use the original FP32 path.
        device = carry.current_hidden.device
        enabled = self.config.amp and device.type == "cuda" and self.config.amp_dtype != "float32"
        if enabled and self.config.amp_dtype == "auto":
            try:
                enabled = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:
                enabled = torch.cuda.get_device_capability(device)[0] >= 8
        if enabled:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nc, logits = self._forward(carry, batch)
            return replace(nc, current_hidden=nc.current_hidden.float()), logits.float()
        return self._forward(carry, batch)


    def addr_raw(self, h, AB):
        """Real/imag components of one complex linear projection, [B,T,H,p]."""
        A, Bm = AB
        return torch.einsum('btd,hjd->bthj', h, A), torch.einsum('btd,hjd->bthj', h, Bm)

    def _norm(self, xy):
        return (xy[0].square() + xy[1].square()).sum(-1, keepdim=True).sqrt() + self.config.eps

    def _unit(self, xy):
        norm = self._norm(xy)
        return xy[0] / norm, xy[1] / norm

    def trace_step(self, L, uq, uk, ztr, fresh):
        """Separate Q/K traces, shared mu: Z_new=mu*Z_old+sqrt(1-rho^2)*U.

        Packed trace [B,T,H,p,Q/K,real/imag]. Reset both traces before carrying.
        Return current traces and carried traces for exact old-old subtraction.
        """
        rho = torch.sigmoid(L.mu_rho_raw)
        cw, sw = torch.cos(L.mu_omega), torch.sin(L.mu_omega)
        gin = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-6))

        def advance(u, role):
            if ztr is None:
                ox, oy = torch.zeros_like(u[0]), torch.zeros_like(u[1])
            else:
                ox, oy = ztr[..., role, 0], ztr[..., role, 1]
                if fresh is not None:
                    mask = fresh.view(-1, 1, 1, 1)
                    ox = torch.where(mask, torch.zeros_like(ox), ox)
                    oy = torch.where(mask, torch.zeros_like(oy), oy)
            carry = (rho * (cw * ox - sw * oy), rho * (sw * ox + cw * oy))
            current = (carry[0] + gin * u[0], carry[1] + gin * u[1])
            return current, carry

        zq, cq = advance(uq, 0)
        zk, ck = advance(uk, 1)
        packed = torch.stack((torch.stack(zq, dim=-1), torch.stack(zk, dim=-1)), dim=-2)
        return zq, zk, cq, ck, packed

    def write_window(self, zq, zk, cq, ck, kc):
        """Remove rho_j^2*C_j(previous) using CURRENT Q/K denominators.

        Carry is mu*Z_old. Sharing mu makes cq*conj(ck)=rho^2*oldQ*conj(oldK).
        Normalizing each carried trace separately would subtract a different term.
        No clipping or extra stabilizer is applied.
        """
        nq, nk = self._norm(zq), self._norm(zk)
        new_q, new_k = (zq[0] / nq, zq[1] / nq), (zk[0] / nk, zk[1] / nk)
        old_q, old_k = (cq[0] / nq, cq[1] / nq), (ck[0] / nk, ck[1] / nk)
        return self.attn_xy(new_q, new_k, kc) - self.attn_xy(old_q, old_k, kc)

    def step(self, L, h, AB, kc, w=None, fresh=None, ztr=None, apply_phi=True):
        """Instantaneous Q/K read; corrected Q/K trace write x current value agree."""
        uq, uk = self.addr_raw(h, AB[0]), self.addr_raw(h, AB[1])
        a = self.attn_xy(self._unit(uq), self._unit(uk), kc)
        v = torch.einsum('btd,hcd->bthc', h, L.w_sh)
        zq, zk, cq, ck, ztr_new = self.trace_step(L, uq, uk, ztr, fresh)
        win = self.write_window(zq, zk, cq, ck, kc)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.config.eps)
        agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
        G = win * agree
        if self.config.plastic_select:
            keep, write, gain, lam = self.select_gates(L, h)
        else:
            gain = F.softplus(L.gain_raw) if self.config.stdp_gain_fixed < 0 else float(self.config.stdp_gain_fixed)
            eta = torch.sigmoid(L.eta_raw)
            keep, write = 1 - eta, eta
            lam = torch.sigmoid(L.lam_raw) if self.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(self.config.stdp_lam_fixed))
        tgt = gain * G
        if w is None:
            w = tgt
        else:
            w = keep * w + write * tgt
            w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
        a = (1 - lam) * a + lam * w
        o = torch.einsum('bhtn,bnhc->bthc', a, v)
        f = torch.einsum('bthc,hcd->btd', o, L.w_sh)
        hout = self.phi(h + f) if apply_phi else (h + f)
        return hout, w, ztr_new

    def select_gates(self, L, h):
        """v1.8 선택적 가소성. h 는 스텝 입력(주입 후 q) [B,T,d].

        Δ_t = softplus(s^Δ_t),  A_tn = exp(−(Δ_t+Δ_n)/2)        기억 유지 (쌍 대칭, ZOH 이산화)
        g_tn = g_max·σ((s^g_t+s^g_n)/2)                            쓰기 이득 (쌍 대칭, 상한)
        λ_t = σ(s^λ_t)                                              읽는 칸 t 의 기억 보간
        w ← A⊙w + (1−A)⊙g⊙G,  a_eff = (1−λ_t)a + λ_t w.  1−A 는 expm1 로 계산해 긴 기억에서도 정밀하다.
        반환: keep=A, write=1−A, gain [B,H,T,T]; lam [B,H,T,1]. 전부 FP32 (bf16 에서는 1−A 가 뭉개진다).
        """
        with torch.autocast(device_type=h.device.type, enabled=False):
            s = torch.einsum('btd,khd->kbht', h.float(), L.sel_w.float())          # [3,B,H,T]
            dt = F.softplus(s[0] + L.dt_bias.float()[:, None])
            dpair = 0.5 * (dt[..., :, None] + dt[..., None, :])
            gl = s[1] + L.gsel_bias.float()[:, None]
            gain = self.config.select_g_max * torch.sigmoid(0.5 * (gl[..., :, None] + gl[..., None, :]))
            lam = torch.sigmoid(s[2] + L.lsel_bias.float()[:, None])[..., None]
            return torch.exp(-dpair), -torch.expm1(-dpair), gain, lam

    def boundary(self, L, h):
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5 * g * u)

    def block(self, L, h, inj, AB, kc, w, fresh, ztr):
        # Layer and fresh are explicit arguments so recomputation cannot pick up
        # another loop iteration's layer or reset mask.
        pre = self.config.block_order == "pre"
        if pre:
            h = self.boundary(L, h)
        h = h + self.embed_scale * inj
        h, w, ztr = self.step(L, h, AB, kc, w, fresh, ztr, apply_phi=pre)
        if not pre:
            h = self.boundary(L, h)
            h = self.phi(h)
        return h, w, ztr

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        ABs = [self.W_QK(L) for L in self.layers]
        kcs = [self.kernel(L) for L in self.layers]
        w = carry.coupling if self.stdp else None; fresh = carry.fresh if self.stdp else None
        ztr = carry.trace if self.use_trace else None
        if self.config.nograd_blocks:
            # TRM 식: 세그먼트 앞부분은 gradient 없이 상태만 진행하고, 뒤의 blocks_per_seg 블록만 역전파한다.
            with torch.no_grad():
                for _ in range(self.config.nograd_blocks):
                    for li, L in enumerate(self.layers):
                        h, w, ztr = self.block(L, h, inj, ABs[li], kcs[li], w, fresh, ztr)
                        fresh = None
        for _ in range(self.config.blocks_per_seg):
            for li, L in enumerate(self.layers):
                AB, kc = ABs[li], kcs[li]
                if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                    h, w, ztr = checkpoint(self.block, L, h, inj, AB, kc, w, fresh, ztr,
                                           use_reentrant=False, preserve_rng_state=False)
                else:
                    h, w, ztr = self.block(L, h, inj, AB, kc, w, fresh, ztr)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=(w.detach() if w is not None else None),
                       trace=(ztr.detach() if ztr is not None else None), fresh=None), self.w_cls(h)


class LT(nn.Module):
    """URM 하네스 인터페이스. ACT 없음 (halted = steps ≥ loops). q 로짓은 상수."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
        self.inner = LT_Inner(self.config)

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
            steps = steps + 1; halted = steps >= self.config.loops
        return LTCarry(current_hidden=inner.current_hidden, steps=steps, halted=halted, current_data=data,
                       coupling=inner.coupling, trace=inner.trace), outputs



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
                 "gain_raw","eta_raw","lam_raw","mu")


def _is_no_decay(name,p):
    return p.ndim<=1 or name.endswith(".b") or any(k in name for k in NO_DECAY_KEYS)


def create_optimizers(base,cfg,world_size):
    opts,lrs = [],[]
    if base.model.puzzle_emb is not None:
        opts.append(CastedSparseEmbeddingSignSGD_Distributed(base.model.puzzle_emb,world_size,
                    weight_decay=cfg["puzzle_emb_weight_decay"]))
        lrs.append(cfg["puzzle_emb_lr"])
    named = [(n,p) for n,p in base.named_parameters() if p.requires_grad]
    opts.append(AdamATan2([
        {"params":[p for n,p in named if _is_no_decay(n,p)],"weight_decay":0.0},
        {"params":[p for n,p in named if not _is_no_decay(n,p)],"weight_decay":cfg["weight_decay"]}],
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


_RESUME_KEYS = tuple(f.name for f in fields(LTConfig) if f.name not in
                     ("batch_size","amp","amp_dtype","activation_checkpoint",
                      "nograd_blocks")) + (
    "global_batch_size","epochs","eval_interval","num_aug","seed","grad_accum_steps",
    "lr","lr_min_ratio","lr_warmup_steps","weight_decay","beta1","beta2","puzzle_emb_lr",
    "puzzle_emb_weight_decay","q_weight","ema","ema_rate","data_fingerprint",
    "lr_rewarm_start","lr_rewarm_steps","lr_rewarm_from_ratio","late_sup_prob","late_sup_min","late_sup_max",
    "nograd_fixed","nograd_every","nograd_start","nograd_max")
# Defaults for optional training schedule keys.
_LEGACY_DEFAULTS = dict(plastic_select=False,select_g_max=4.0,lr_rewarm_start=None,lr_rewarm_steps=0,
                        lr_rewarm_from_ratio=1.0,late_sup_prob=0.0,late_sup_min=16,late_sup_max=112,
                        nograd_fixed=0,nograd_every=0,nograd_start=0,nograd_max=16)


def _effective_recipe(cfg):
    """재개 비교용: 꺼진 옵션의 하위 값(예: 재가열 없음일 때 lr_rewarm_steps)은 학습에 영향이 없으므로 지운다."""
    r = {k:cfg.get(k,_LEGACY_DEFAULTS.get(k)) for k in _RESUME_KEYS}
    if r["lr_rewarm_start"] is None:
        r["lr_rewarm_steps"] = r["lr_rewarm_from_ratio"] = None
    if not r["late_sup_prob"]:
        r["late_sup_min"] = r["late_sup_max"] = None
    if not r["plastic_select"]:
        r["select_g_max"] = None
    if not r["nograd_every"]:
        r["nograd_start"] = r["nograd_max"] = None
    return r


def load_training_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device):
    # Only load checkpoints you trust: weights_only=False is needed for RNG/optimizer objects.
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("model_id") != model_id_of(cfg):
        raise ValueError(f"Checkpoint model_id={ck.get('model_id')!r} but this run is {model_id_of(cfg)!r}. "
                         "Resume requires this split-Q/K delta architecture with matching dimensions.")
    old = ck["cfg"]
    was,now = _effective_recipe(old),_effective_recipe(cfg)
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
        print(f"[LT] resharded saved h/w/z/data: {ck['world_size']} -> {ws} ranks; "
              "bitwise equality across GPU layouts is not promised.",flush=True)
    return ts


# 6b. Warm start: weights (raw + EMA) from another run, fresh optimizer ----------
# Warm start is supported only for this same split-Q/K delta architecture.
_INIT_KEYS = ("vocab_size","puzzle_emb_ndim","hidden_size","num_heads","loops","grid","blocks_per_seg",
              "num_layers","mlp_expansion","legacy_gauge","dist_decay","eps","forward_dtype","address_projection",
              "psi_zero","stdp","stdp_gain_fixed","stdp_lam_fixed","block_order","use_trace",
              "global_batch_size","epochs","eval_interval","num_aug","seed","data_fingerprint")


def init_from_checkpoint(path,base,ema,cfg,device):
    """raw·EMA 가중치만 가져오고 optimizer 는 새로 만든다. step 과 데이터 커서는 원본을 이어받는다
    (원본 런의 같은 구간과 같은 배치를 본다). carry 는 새로 시작한다."""
    ck = torch.load(path,map_location="cpu",weights_only=False)
    src = ck.get("model_id")
    if src != model_id_of(cfg):
        raise ValueError(f"init_from needs the same split-Q/K delta architecture; got {src!r}")
    old = ck["cfg"]
    keys = (*_INIT_KEYS, "plastic_select", "select_g_max")
    changed = {k:(old.get(k),cfg.get(k)) for k in keys if old.get(k)!=cfg.get(k)}
    if changed:
        raise ValueError(f"init_from architecture/data protocol mismatch: {changed}")
    base.load_state_dict(ck["raw_model_state_dict"],strict=True,assign=False)
    if ema is not None:
        shadow = ck.get("ema_shadow")
        if shadow is None:
            raise ValueError("init_from checkpoint has no EMA shadow; set ema=False or use another checkpoint.")
        if set(shadow)!=set(ema.shadow):
            raise ValueError(f"EMA keys differ: {sorted(set(shadow)^set(ema.shadow))[:6]}")
        ema.shadow = {n:v.to(device=device,dtype=ema.shadow[n].dtype).clone() for n,v in shadow.items()}
    cfg.update(init_from_model_id=src,init_from_step=int(ck["step"]))
    return TrainState(step=int(ck["step"]),iter_id=int(ck["iter_id"]),batch_in_iter=int(ck["batch_in_iter"]))


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
    return os.path.join(root,"lt_v18_qk_delta512")


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


def validate_run_cfg(cfg):
    if cfg["grad_accum_steps"]!=1:
        raise ValueError("grad_accum_steps must be 1: one segment equals one optimizer step. "
                         "The supplied train.py did not actually implement gradient accumulation.")
    if cfg["dataloader_workers"] not in (0,1):
        raise ValueError("dataloader_workers must be 0 or 1.")
    if cfg["epochs"]<=0 or cfg["eval_interval"]<=0 or cfg["epochs"]%cfg["eval_interval"]:
        raise ValueError("epochs must be a positive multiple of eval_interval (in EPOCHS, not steps).")
    if min(cfg["global_batch_size"],cfg["test_size"],cfg["log_every"],cfg["stop_check_every"])<=0:
        raise ValueError("Batch, test_size, log_every, and stop_check_every must be positive.")
    if cfg["num_aug"]<0 or cfg["max_hours"]<=0:
        raise ValueError("num_aug>=0 and max_hours>0 are required.")
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


def main(cfg):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    cfg = dict(cfg)
    validate_run_cfg(cfg)
    start = time.monotonic()
    deadline = start+cfg["max_hours"]*3600
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
    if path:
        ts = load_training_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device)
    elif cfg.get("init_from"):
        ts = init_from_checkpoint(cfg["init_from"],base,ema,cfg,device)
    else:
        ts = TrainState()
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
    if cfg["compile"] and cfg.get("nograd_every",0) and not cfg.get("nograd_fixed",0):
        # no-grad 블록 수가 바뀔 때마다 한 번씩 다시 컴파일한다. 기본 한도(8)를 넘으면 eager 로 떨어지므로 올린다.
        import torch._dynamo as dynamo      # "import torch._dynamo" 는 main 안에서 torch 를 지역 이름으로 만든다
        for name in ("cache_size_limit","recompile_limit"):
            if hasattr(dynamo.config,name):
                setattr(dynamo.config,name,max(getattr(dynamo.config,name),int(cfg["nograd_max"])+8))
    model = torch.compile(base,dynamic=False) if cfg["compile"] else base
    if rank==0:
        print(f"[LT] {model_id_of(cfg)} torch={torch.__version__} device={device} ranks={ws} local_bs={lbs}",flush=True)
        print(f"[LT] params={sum(p.numel() for p in base.parameters()):,} amp={cfg['amp_dtype']} "
              f"activation_checkpoint={cfg['activation_checkpoint']} compile={cfg['compile']}",flush=True)
        print("[LT] independent complex Q/K; shared psi read/write kernel; "
              "old-old subtraction with current Q/K norms", flush=True)
        print(f"[LT] data={cfg['data_npz']} train={len(tr_x)} test={len(te_x)}",flush=True)
        print(f"[LT] planned steps={planned_steps}; actual steps={actual_steps}; "
              f"1 step={cfg['blocks_per_seg']} blocks; loops={cfg['loops']} segments",flush=True)
        print(f"[LT] v1.8: projection={cfg['address_projection']} "
              f"order={cfg['block_order']} address_trace={cfg['use_trace']} "
              f"rho_init={cfg['trace_rho_init']} plastic_select={cfg.get('plastic_select')} "
              f"g_max={cfg.get('select_g_max')}",flush=True)
        print(f"[LT] lr={cfg['lr']} min_ratio={cfg['lr_min_ratio']} rewarm_start={cfg.get('lr_rewarm_start')} "
              f"rewarm_steps={cfg.get('lr_rewarm_steps')} rewarm_from_ratio={cfg.get('lr_rewarm_from_ratio')} "
              f"late_sup_prob={cfg.get('late_sup_prob',0.0)} nograd_fixed={cfg.get('nograd_fixed',0)} "
              f"nograd_every={cfg.get('nograd_every',0)} "
              f"nograd_start={cfg.get('nograd_start',0)} nograd_max={cfg.get('nograd_max',16)} "
              f"late_sup_extra=[{cfg.get('late_sup_min')},{cfg.get('late_sup_max')}]",flush=True)
        how = ("RESUME "+str(path) if path else
               f"INIT_FROM {cfg['init_from']} ({cfg.get('init_from_model_id')})" if cfg.get("init_from") else "NEW RUN")
        print(f"[LT] {how} step={ts.step} "
              f"next_iter={ts.iter_id} consumed_batches={ts.batch_in_iter} out={out_dir}",flush=True)
        if device.type=="cpu":
            print("[LT] CPU execution: suitable for self-tests and small smoke runs.",flush=True)
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
                    if metrics["_count_raw"]>0:
                        last_full = (ts.step,metrics)
                    if ts.step%cfg["log_every"]==0:
                        msg = f"[LT] step {ts.step} lm_loss {metrics['lm_loss']:.5f} lr {metrics['lr']:.3g}"
                        if last_full:
                            fs,fm = last_full
                            msg += f" acc {fm['accuracy']:.4f} exact {fm['exact_accuracy']:.4f} [halt step {fs}]"
                        print(msg,flush=True)
                if ts.step%cfg["stop_check_every"]==0 or boundary or ts.step>=stop_at:
                    stopped = stop_requested(device,deadline)
                if stopped:
                    break                         # Save FIRST; skip expensive final evaluation.
                due_save = cfg["save_every_steps"] and ts.step%cfg["save_every_steps"]==0
                due_milestone = cfg["milestone_every"] and ts.step%cfg["milestone_every"]==0
                if boundary or due_save:
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

# 9. CPU self-tests (no real dataset and no claims about training accuracy) -----
def selftest():
    """CPU checks for complex delta arithmetic and the complete training harness."""
    import tempfile
    old_threads = torch.get_num_threads()
    device = torch.device("cpu")
    rng0 = _rng_state(device)
    torch.set_num_threads(2)
    checks = []
    try:
        cfg = dict(DEFAULT_CFG, hidden_size=16, num_heads=2, puzzle_emb_ndim=16,
                   global_batch_size=2, batch_size=2, seq_len=81, num_puzzle_identifiers=1,
                   blocks_per_seg=2, loops=3, amp=False, amp_dtype="float32",
                   activation_checkpoint=False, compile=False, epochs=4, eval_interval=2,
                   lr_warmup_steps=0, lr=1e-3, puzzle_emb_lr=1e-3, ema_rate=0.9,
                   num_aug=10, run_selftests=False, dataloader_workers=0, data_fingerprint="selftest")
        torch.manual_seed(172)
        model = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
        inner, L = model.model.inner, model.model.inner.layers[0]
        assert L.wq.data_ptr() != L.wk.data_ptr() and not hasattr(L, "beta")
        assert not hasattr(L, "wc") and not hasattr(L, "wc_raw")
        checks.append("independent Q/K projections; beta parameter absent")
        h = torch.randn(2, 81, 16)
        q, k = [inner.addr_raw(h, AB) for AB in inner.W_QK(L)]
        previous = torch.randn(2, 81, 2, 4, 2, 2)
        zq, zk, cq, ck, packed = inner.trace_step(L, q, k, previous, None)
        assert packed.shape == previous.shape
        kc = inner.kernel(L)
        zqc, zkc = torch.complex(*zq), torch.complex(*zk)
        cqc, ckc = torch.complex(*cq), torch.complex(*ck)
        aq, ak = zqc - cqc, zkc - ckc
        raw = torch.einsum('bthj,bnhj->bhtnj', zqc, zkc.conj())
        rho = torch.sigmoid(L.mu_rho_raw)
        pq, pk = torch.complex(previous[..., 0, 0], previous[..., 0, 1]), torch.complex(previous[..., 1, 0], previous[..., 1, 1])
        removed = torch.einsum('bthj,bnhj->bhtnj', pq, pk.conj()) * rho.square()[None,:,None,None,:]
        expanded = (torch.einsum('bthj,bnhj->bhtnj', cqc, ak.conj()) +
                    torch.einsum('bthj,bnhj->bhtnj', aq, ckc.conj()) +
                    torch.einsum('bthj,bnhj->bhtnj', aq, ak.conj()))
        torch.testing.assert_close(raw - removed, expanded, rtol=2e-5, atol=3e-6)
        phase_q = torch.complex(kc[1], kc[2])
        phase_k = torch.complex(kc[3], kc[4])
        expected = torch.einsum('bhtnj,thj,nhj->bhtn', (raw - removed).to(torch.complex64), phase_q, phase_k.conj()).real
        nq, nk = inner._norm(zq), inner._norm(zk)
        denominator = nq.squeeze(-1).permute(0,2,1)[..., :, None] * nk.squeeze(-1).permute(0,2,1)[..., None, :]
        expected = expected * kc[0].unsqueeze(0) / denominator
        torch.testing.assert_close(inner.write_window(zq, zk, cq, ck, kc), expected, rtol=2e-5, atol=2e-6)
        checks.append("rho^2 old-old removal and current-normalized complex kernel agree")
        rzq, rzk, rcq, rck, _ = inner.trace_step(L, q, k, previous, torch.ones(2, dtype=torch.bool))
        assert rcq[0].count_nonzero() == 0 and rck[0].count_nonzero() == 0
        torch.testing.assert_close(inner.write_window(rzq, rzk, rcq, rck, kc),
                                   inner.attn_xy(inner._unit(rzq), inner._unit(rzk), kc), rtol=0, atol=0)
        checks.append("fresh puzzle resets both traces; first write unchanged by subtraction")
        batch = dict(inputs=torch.randint(1,11,(2,81)), labels=torch.randint(2,11,(2,81)),
                     puzzle_identifiers=torch.zeros(2,dtype=torch.int32))
        states = []
        for checkpointing in (False, True):
            model.zero_grad(set_to_none=True)
            model.model.config.activation_checkpoint = checkpointing
            _, loss, _, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
            loss.backward()
            grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in grads.values())
            for name in ("wq", "wk", "mu_rho_raw", "mu_omega", "sel_w"):
                assert getattr(L,name).grad is not None and getattr(L,name).grad.norm() > 0, name
            states.append(grads)
        assert states[0].keys() == states[1].keys()
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=2e-5, atol=3e-6, msg=name)
        checks.append("finite nonzero Q/K/trace/gate gradients; activation checkpoint parity")
        model.model.config.activation_checkpoint = False
        opts, lrs = create_optimizers(model,cfg,1)
        ema = EMAHelper(cfg["ema_rate"]); ema.register(model)
        ts = TrainState()
        def update(net, state, optimizers, rates, shadow):
            train_batch(net,net,state,batch,cfg,optimizers,rates,16,0,1,device)
            shadow.update(net); state.batch_in_iter += 1
        for _ in range(2): update(model,ts,opts,lrs,ema)
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_checkpoint(directory,ts,model,opts,ema,cfg,0,1,device)
            update(model,ts,opts,lrs,ema)
            resumed = ACTLossHead(LT(cfg),q_weight=cfg["q_weight"])
            ro, rl = create_optimizers(resumed,cfg,1)
            rema = EMAHelper(cfg["ema_rate"]); rema.register(resumed)
            rs = load_training_checkpoint(path,resumed,ro,rema,cfg,0,1,device)
            update(resumed,rs,ro,rl,rema)
            for name,value in model.state_dict().items():
                assert torch.equal(value,resumed.state_dict()[name]), name
            for name,value in ema.shadow.items():
                assert torch.equal(value,rema.shadow[name]), name
            for name in ("current_hidden","coupling","trace","steps","halted"):
                assert torch.equal(getattr(ts.carry,name),getattr(rs.carry,name)), name
        checks.append("checkpoint resume reproduces next weights/EMA and separate Q/K traces")
        for item in checks:
            print("[selftest] PASS "+item,flush=True)
        print(f"[selftest] {len(checks)}/{len(checks)} groups passed (CPU).",flush=True)
        return checks
    finally:
        _restore_rng(rng0,device)
        torch.set_num_threads(old_threads)


def _cli():
    ap = argparse.ArgumentParser(description="Self-contained LT v1.8 split-Q/K trace-delta trainer")
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
    main(cfg)


if __name__=="__main__":
    _cli()

'''


def launch_lt_one_cell():
    """새 프로세스에서 동기 실행. 노트북의 CUDA 상태와 argparse 인자를 상속하지 않습니다."""
    work_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    tag = "v18_qk_delta512"
    runtime = Path(tempfile.mkdtemp(prefix=f"_lt_{tag}_", dir=str(work_root)))
    trainer_path = runtime / f"train_{tag}.py"
    config_path = runtime / "config.json"
    cfg = dict(CFG)
    log_dir = Path(cfg["out_dir"]).expanduser().resolve() if cfg["out_dir"] else work_root / f"lt_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg["out_dir"] = str(log_dir)
    log_path = log_dir / "train.log"
    trainer_path.write_text(_TRAINER_SOURCE, encoding="utf-8")
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # GPU 조회도 별도 프로세스에서 합니다. 부모 노트북에서는 torch를 import할 필요가 없습니다.
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
        check=True, capture_output=True, text=True,
    )
    ngpu = int(probe.stdout.strip().splitlines()[-1])
    requested = CFG["num_processes"]
    nproc = max(1, ngpu) if requested == "auto" else int(requested)
    if nproc < 1 or (ngpu > 0 and nproc > ngpu):
        raise ValueError(f"num_processes={nproc}, 사용 가능한 GPU={ngpu}")
    if CFG["global_batch_size"] % nproc:
        raise ValueError("global_batch_size는 num_processes로 나누어떨어져야 합니다.")

    cmd = [sys.executable]
    if nproc > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={nproc}"]
    cmd += [str(trainer_path), "--config", str(config_path)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    print(f"[Kaggle] GPU={ngpu}, processes={nproc}; 실행 파일: {trainer_path}", flush=True)
    print("[Kaggle] 모델 생성 후 학습을 시작합니다." +
          (" 첫 학습 호출에 컴파일이 포함됩니다." if CFG["compile"] else " compile=False: eager 실행입니다."), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, start_new_session=(os.name == "posix"))
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    log_file.write(f"\n[Kaggle] launch {tag}: " + str(trainer_path) + "\n")
    print(f"[Kaggle] 훈련 로그: {log_path}", flush=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        rc = proc.wait()
    except KeyboardInterrupt:
        print("\n[Kaggle] 중단 요청을 전달했습니다. 완료된 스텝 경계에서 저장하도록 요청합니다.", flush=True)
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
            print("[Kaggle] 강제 종료되었습니다. 마지막으로 저장된 체크포인트는 유지됩니다.", flush=True)
        raise
    finally:
        proc.stdout.close()
        log_file.close()
    if rc:
        raise RuntimeError(f"LT 학습 프로세스가 종료 코드 {rc}로 실패했습니다. 위 오류 로그를 확인하세요.")

if __name__ == "__main__":
    # The original one-cell launcher ignores Jupyter's kernel arguments.
    # Explicit shell arguments also allow `python lt/train_v18.py --selftest`.
    if "__file__" in globals() and "ipykernel" not in sys.modules and len(sys.argv) > 1:
        exec(compile(_TRAINER_SOURCE, __file__ + "::trainer", "exec"), globals())
    else:
        launch_lt_one_cell()
else:
    # Keep LT/model helpers importable for the existing research scripts.
    exec(compile(_TRAINER_SOURCE, globals().get("__file__", "train_v18.py") + "::trainer", "exec"), globals())
