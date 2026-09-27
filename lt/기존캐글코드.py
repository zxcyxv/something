# -*- coding: utf-8 -*-
# LT v1.1(pre) + 값 흔적 STDP — Kaggle 단일 셀 완성본
# 아래 CFG만 수정하고 셀 전체를 실행합니다. 외부 저장소/추가 셀은 필요 없습니다.
# 데이터: sudoku_lt_1k.npz를 Kaggle Input으로 연결하거나 data_npz에 경로 지정.
# 신규 모델입니다. 원래 v1.1 체크포인트의 자동 변환/재개는 하지 않습니다.
# W0=1 고정, A=1에서 학습 시작, rho=0.5에서 학습 시작이 기본 설정입니다.
# A=0 절제: stdp_A_fixed=0.0 / 직전 블록만 사용: value_trace_rho_fixed=0.0
# 1 step = 세그먼트 1개 = blocks_per_seg 블록. grad_accum_steps=1을 유지합니다.
# GPU가 BF16을 직접 지원하지 않으면 FP32를 사용합니다. FP16은 쓰지 않습니다.
# activation_checkpoint=True는 블록을 재계산해 메모리를 줄이며 h/w/x를 끊지 않습니다.
# 셀을 다시 실행하면 out_dir의 마지막 체크포인트에서 h/w/x/퍼즐/옵티마이저까지 재개합니다.
# 다른 실험을 시작할 때는 out_dir을 바꾸세요. max_steps는 추가 스텝이 아닌 절대 종료 스텝입니다.

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

CFG = dict(
    data_npz=None,                 # None: find sudoku_lt_1k.npz under /kaggle/input
    num_aug=1000, test_size=2048,
    hidden_size=832, num_heads=8, loops=16, blocks_per_seg=8, num_layers=1,
    grid=9, vocab_size=11, mlp_expansion=4.0, alpha_init=0.1,
    dist_decay=True, eps=1e-4, psi_zero=False, puzzle_emb_ndim=832,
    legacy_gauge=False, block_order="pre", use_trace=False,
    forward_dtype="float32", amp=True,
    amp_dtype="auto",             # native BF16 if supported; otherwise FP32 (no FP16)
    activation_checkpoint=True,    # recompute blocks; does NOT detach h/w/x
    stdp=True, stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_init=0.25,
    stdp_gain_fixed=-1.0, stdp_lam_fixed=-1.0,
    stdp_w0_init=1.0, stdp_w0_fixed=1.0,   # None -> learn unrestricted W0
    stdp_A_init=1.0, stdp_A_fixed=None,     # None -> softplus(A_raw); 0.0 -> exact A=0
    value_trace_rho_init=0.5, value_trace_rho_fixed=None,  # 0.0 -> exact one-block trace
    global_batch_size=128, epochs=50000,
    lr=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000, weight_decay=1.0,
    beta1=0.9, beta2=0.95, puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0,
    grad_accum_steps=1, q_weight=0.5,
    seed=0, ema=True, ema_rate=0.999, eval_interval=250,
    compile=True, inductor_no_persist=True,
    out_dir=None,                 # /kaggle/working/lt_value_window
    resume_from=None,             # explicit file/directory has priority over out_dir
    require_resume=False, keep_last=2, save_every_steps=2000,
    milestone_every=10000, milestone_extrap_segs=128, milestone_extrap_n=512,
    max_hours=11.5, max_steps=None, log_every=250, stop_check_every=25,
    dataloader_workers=1, run_selftests=True,
    num_processes="auto",         # notebook wrapper only; 1 to use one GPU
)


# 아래 문자열에 모델부터 self-test까지 학습 프로그램 전체가 포함되어 있습니다.
_TRAINER_SOURCE = r'''# -*- coding: utf-8 -*-
"""LT v1.1(pre, sqrt(d)) + delayed-read value-trace STDP.

Self-contained adaptation of the train.py supplied in this conversation.
No repository imports. This is a NEW model, not a bitwise reproduction of v1.1.

One optimizer step = one model(carry, batch) = blocks_per_seg blocks.
Read w_old -> write w_new -> update x. Detach only at the segment boundary.
x_new = rho*x_old + (1-rho)*vhat, so the lag-s weight is
A*(1-rho)*rho**(s-1), not A*rho**(s-1). The recurrence is kept verbatim.

The notebook wrapper below runs this source in a fresh Python/torchrun process;
it does not fork a CUDA-initialized notebook or require accelerate.
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
from dataclasses import dataclass, fields, replace
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

MODEL_ID = "lt-v11-value-trace-stdp-delayed-v1"
IGNORE_LABEL_ID = -100
_STOP_REQUESTED = False

# DEFAULT_CFG is also copied to the top of the one-cell notebook.
DEFAULT_CFG = dict(
    data_npz=None,                 # None: find sudoku_lt_1k.npz under /kaggle/input
    num_aug=1000, test_size=2048,
    hidden_size=832, num_heads=8, loops=16, blocks_per_seg=8, num_layers=1,
    grid=9, vocab_size=11, mlp_expansion=4.0, alpha_init=0.1,
    dist_decay=True, eps=1e-4, psi_zero=False, puzzle_emb_ndim=832,
    legacy_gauge=False, block_order="pre", use_trace=False,
    forward_dtype="float32", amp=True,
    amp_dtype="auto",             # native BF16 if supported; otherwise FP32 (no FP16)
    activation_checkpoint=True,    # recompute blocks; does NOT detach h/w/x
    stdp=True, stdp_eta_init=0.05, stdp_gain_init=1.0, stdp_lam_init=0.25,
    stdp_gain_fixed=-1.0, stdp_lam_fixed=-1.0,
    stdp_w0_init=1.0, stdp_w0_fixed=1.0,   # None -> learn unrestricted W0
    stdp_A_init=1.0, stdp_A_fixed=None,     # None -> softplus(A_raw); 0.0 -> exact A=0
    value_trace_rho_init=0.5, value_trace_rho_fixed=None,  # 0.0 -> exact one-block trace
    global_batch_size=128, epochs=50000,
    lr=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000, weight_decay=1.0,
    beta1=0.9, beta2=0.95, puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0,
    grad_accum_steps=1, q_weight=0.5,
    seed=0, ema=True, ema_rate=0.999, eval_interval=250,
    compile=True, inductor_no_persist=True,
    out_dir=None,                 # /kaggle/working/lt_value_window
    resume_from=None,             # explicit file/directory has priority over out_dir
    require_resume=False, keep_last=2, save_every_steps=2000,
    milestone_every=10000, milestone_extrap_segs=128, milestone_extrap_n=512,
    max_hours=11.5, max_steps=None, log_every=250, stop_check_every=25,
    dataloader_workers=1, run_selftests=True,
    num_processes="auto",         # notebook wrapper only; 1 to use one GPU
)

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


# 2. Model -------------------------------------------------------------------
@dataclass
class LTCarry:
    current_hidden: torch.Tensor                  # [B,T,d]
    steps: torch.Tensor                           # [B]
    halted: torch.Tensor                          # [B]
    current_data: Dict[str, torch.Tensor]
    coupling: Optional[torch.Tensor] = None        # w [B,H,T,T]
    value_trace: Optional[torch.Tensor] = None     # x [B,T,H,dh]; NOT address trace z


@dataclass
class LTConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 832
    hidden_size: int = 832
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8
    num_layers: int = 1
    mlp_expansion: float = 4.0
    alpha_init: float = 0.1
    dist_decay: bool = True
    eps: float = 1e-4
    psi_zero: bool = False
    legacy_gauge: bool = False
    block_order: str = "pre"
    use_trace: bool = False
    forward_dtype: str = "float32"
    amp: bool = True
    amp_dtype: str = "float32"  # resolved by main, before compilation
    activation_checkpoint: bool = True
    stdp: bool = True
    stdp_eta_init: float = 0.05
    stdp_gain_init: float = 1.0
    stdp_lam_init: float = 0.25
    stdp_gain_fixed: float = -1.0
    stdp_lam_fixed: float = -1.0
    stdp_w0_init: float = 1.0
    stdp_w0_fixed: Optional[float] = 1.0
    stdp_A_init: float = 1.0
    stdp_A_fixed: Optional[float] = None
    value_trace_rho_init: float = 0.5
    value_trace_rho_fixed: Optional[float] = None

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def __post_init__(self):
        d, H = self.hidden_size, self.num_heads
        if self.seq_len != self.grid**2 or d % H or (d//H) % 2:
            raise ValueError("Require T=grid^2, d%H=0 and an even dh=d/H.")
        if self.legacy_gauge or self.block_order != "pre" or self.use_trace:
            raise ValueError("This implementation is v1.1/pre/sqrt(d), with value trace only.")
        if self.num_layers != 1:
            raise ValueError("num_layers=1: one shared layer, exactly blocks_per_seg blocks per step.")
        if self.forward_dtype != "float32":
            raise ValueError("Keep carry/parameters in float32. Mixed precision is controlled by amp_dtype.")
        if not 0 <= self.puzzle_emb_ndim <= d or min(self.loops, self.blocks_per_seg) < 1:
            raise ValueError("Invalid puzzle embedding width, loops, or blocks_per_seg.")
        if self.eps <= 0 or self.alpha_init <= 0:
            raise ValueError("eps and alpha_init must be positive.")
        if self.stdp:
            for name in ("stdp_eta_init", "stdp_lam_init"):
                if not 0 < getattr(self, name) < 1:
                    raise ValueError(f"{name} must be strictly between 0 and 1.")
            if self.stdp_gain_init <= 0 or self.stdp_lam_fixed > 1:
                raise ValueError("stdp_gain_init>0; fixed lambda must be in [0,1] or negative (learned).")
            if self.stdp_A_fixed is None and self.stdp_A_init <= 0:
                raise ValueError("Learned A requires A_init>0; use stdp_A_fixed=0.0 for exact A=0.")
            if self.stdp_A_fixed is not None and self.stdp_A_fixed < 0:
                raise ValueError("stdp_A_fixed must be nonnegative or None.")
            r = self.value_trace_rho_fixed
            if r is None and not 0 < self.value_trace_rho_init < 1:
                raise ValueError("Learned rho requires 0<rho_init<1; use rho_fixed=0.0 for exact rho=0.")
            if r is not None and not 0 <= r < 1:
                raise ValueError("value_trace_rho_fixed must be in [0,1) or None.")


def inv_softplus(y):
    return y + math.log(-math.expm1(-y))


def logit(x):
    return math.log(x / (1 - x))


class LTLayer(nn.Module):
    def __init__(self, c, H, d, dh, p):
        super().__init__()
        self.wc_raw = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        if c.psi_zero:
            self.register_buffer("psi", torch.zeros(H, p), persistent=False)
        else:
            self.psi = nn.Parameter(torch.rand(H, p)*2*math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, p, 2)*2-1)*(math.pi/2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(c.alpha_init)))
        ws = torch.zeros(H, dh, d)
        for m in range(H):
            ws[m, :, m*dh:(m+1)*dh] = torch.eye(dh)
        self.w_sh = nn.Parameter(ws + 0.01*torch.randn(H, dh, d)/math.sqrt(d))
        if c.stdp:
            shape = (H, 1, 1)
            self.eta_raw = nn.Parameter(torch.full(shape, logit(c.stdp_eta_init)))
            # Keep the original names for these learned parameters.
            self.lam_raw = nn.Parameter(torch.full(shape, logit(c.stdp_lam_init)),
                                        requires_grad=c.stdp_lam_fixed < 0)
            self.gain_raw = nn.Parameter(torch.full(shape, inv_softplus(c.stdp_gain_init)),
                                         requires_grad=c.stdp_gain_fixed < 0)
            if c.stdp_w0_fixed is None:
                self.w0 = nn.Parameter(torch.full(shape, float(c.stdp_w0_init)))
            if c.stdp_A_fixed is None:
                self.A_raw = nn.Parameter(torch.full(shape, inv_softplus(c.stdp_A_init)))
            if c.value_trace_rho_fixed is None:
                self.rho_raw = nn.Parameter(torch.full(shape, logit(c.value_trace_rho_init)))
            # No beta parameter and no address trace/mu parameters.
        inter = int(c.mlp_expansion*d*2/3 + 255)//256*256
        self.b_gate_up = nn.Linear(d, 2*inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)


class LT_Inner(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.d, self.H, self.dh = c.hidden_size, c.num_heads, c.hidden_size//c.num_heads
        self.p = self.dh//2
        T, g, d = c.seq_len, c.grid, self.d
        r = torch.arange(T, dtype=torch.float32)//g
        col = torch.arange(T, dtype=torch.float32)%g
        self.register_buffer("pos_u", r, persistent=False)
        self.register_buffer("pos_w", col, persistent=False)
        self.register_buffer("l1", (r[:,None]-r[None]).abs()+(col[:,None]-col[None]).abs(), persistent=False)
        self.embed = nn.Embedding(c.vocab_size, d)
        self.embed_scale, self.gamma = math.sqrt(d), 1.0/d
        trunc_normal_init_(self.embed.weight, std=1.0/math.sqrt(d))
        self.w_cls = nn.Linear(d, c.vocab_size)
        self.layers = nn.ModuleList([LTLayer(c, self.H, d, self.dh, self.p)])
        if c.puzzle_emb_ndim:
            self.puzzle_emb = CastedSparseEmbedding(c.num_puzzle_identifiers, c.puzzle_emb_ndim,
                                                   c.batch_size, 0, torch.float32)
        self.register_buffer("init_hidden", trunc_normal_init_(torch.empty(d), std=1.0))

    def W_C(self, L):
        # QR is kept in FP32 even under autocast.
        with torch.autocast(L.wc_raw.device.type, enabled=False):
            Q, _ = torch.linalg.qr(L.wc_raw.transpose(-1,-2), mode="reduced")
        AB = Q.transpose(-1,-2)
        return AB[:, :self.p], AB[:, self.p:]

    def kernel(self, L, psi=None):
        psi = L.psi if psi is None else psi
        decay = (torch.exp(-L.alpha[:,0,None,None]*self.l1) if self.config.dist_decay
                 else torch.ones_like(self.l1).expand(self.H,-1,-1))
        pp = L.theta[...,0,None]*self.pos_u + L.theta[...,1,None]*self.pos_w
        aa = (pp + psi[...,None]/2).permute(2,0,1)
        bb = (pp - psi[...,None]/2).permute(2,0,1)
        return decay, aa.cos(), aa.sin(), bb.cos(), bb.sin()

    def attn_xy(self, xy, kc):
        # Accumulate address inner products in FP32, not in BF16.
        with torch.autocast(xy[0].device.type, enabled=False):
            x, y = xy
            decay, ca, sa, cb, sb = kc
            qx, qy = x*ca-y*sa, x*sa+y*ca
            kx, ky = x*cb-y*sb, x*sb+y*cb
            return (torch.einsum("bthj,bnhj->bhtn", qx, kx) +
                    torch.einsum("bthj,bnhj->bhtn", qy, ky))*decay[None]

    def phi(self, h):
        h = h.float()
        return h / torch.sqrt(1.0 + self.gamma*h.square().sum(-1, keepdim=True))

    def addr(self, h, AB):
        a, b = AB
        ux = torch.einsum("btd,hjd->bthj", h, a).float()
        uy = torch.einsum("btd,hjd->bthj", h, b).float()
        # vector_norm has a defined zero subgradient; denominator is norm+eps, not sqrt(norm^2+eps).
        norm = torch.linalg.vector_norm(torch.cat((ux,uy), dim=-1), dim=-1, keepdim=True)
        return ux/(norm+self.config.eps), uy/(norm+self.config.eps)

    def injection(self, data):
        inj = self.embed(data["inputs"].long())
        if self.config.puzzle_emb_ndim:
            pe = self.puzzle_emb(data["puzzle_identifiers"])
            if self.config.puzzle_emb_ndim < self.d:
                pe = F.pad(pe, (0, self.d-self.config.puzzle_emb_ndim))
            inj = inj + pe[:,None]
        return inj

    def plasticity_scalars(self, L):
        c = self.config
        eta = L.eta_raw.sigmoid()
        lam = L.lam_raw.sigmoid() if c.stdp_lam_fixed < 0 else c.stdp_lam_fixed
        gain = F.softplus(L.gain_raw) if c.stdp_gain_fixed < 0 else c.stdp_gain_fixed
        w0 = L.w0 if c.stdp_w0_fixed is None else c.stdp_w0_fixed
        A = F.softplus(L.A_raw) if c.stdp_A_fixed is None else c.stdp_A_fixed
        rho = L.rho_raw.sigmoid() if c.value_trace_rho_fixed is None else c.value_trace_rho_fixed
        return eta, lam, gain, w0, A, rho

    def boundary(self, L, h):
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(0.5*g*u)

    def step(self, L, h, AB, kc, kc0, w, x):
        """The user's pseudocode: read old w, then write, then update x."""
        uh = self.addr(h, AB)
        v = torch.einsum("btd,hcd->bthc", h, L.w_sh)
        a = self.attn_xy(uh, kc)
        if self.config.stdp:
            eta, lam, gain, w0, A, rho = self.plasticity_scalars(L)
            a_eff = (1-lam)*a + lam*w             # IMPORTANT: OLD w; zero on a fresh puzzle
        else:
            a_eff = a
        o = torch.einsum("bhtn,bnhc->bthc", a_eff, v)
        f = torch.einsum("bthc,hcd->btd", o, L.w_sh)
        hout = self.phi(h + f)                     # read finished before the write
        if self.config.stdp:
            with torch.autocast(h.device.type, enabled=False):
                vf = v.float()
                vv = vf / (torch.linalg.vector_norm(vf, dim=-1, keepdim=True)+self.config.eps)
                K = self.attn_xy(uh, kc0)          # psi=0; symmetric; beta does not exist
                agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
                M = torch.einsum("bthc,bnhc->bhtn", vv, x)
                anti = M - M.transpose(-1,-2)      # transpose token axes ONLY
                tgt = gain*K*(w0*agree + A*anti)
                w = (1-eta)*w + eta*tgt            # fresh first write is eta*tgt, NOT tgt
                if isinstance(rho, torch.Tensor):
                    rho = rho.view(1,1,self.H,1)    # rho [H,1,1] -> value axes [1,1,H,1]
                x = rho*x + (1-rho)*vv             # old trace used above, update only now
        return hout, w, x

    def block(self, h, inj, AB, kc, kc0, w, x):
        L = self.layers[0]
        h = self.boundary(L, h)
        h = h + self.embed_scale*inj
        return self.step(L, h, AB, kc, kc0, w, x)

    def forward(self, carry, data):
        h, w, x = carry.current_hidden, carry.coupling, carry.value_trace
        c, L = self.config, self.layers[0]
        enabled = c.amp and c.amp_dtype == "bfloat16" and h.device.type == "cuda"
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=enabled):
            inj = self.injection(data)
            AB, kc = self.W_C(L), self.kernel(L)
            kc0 = self.kernel(L, torch.zeros_like(L.psi)) if c.stdp else None
            for _ in range(c.blocks_per_seg):
                if c.activation_checkpoint and self.training and torch.is_grad_enabled():
                    h,w,x = checkpoint(self.block, h,inj,AB,kc,kc0,w,x,
                                       use_reentrant=False, preserve_rng_state=False)
                else:
                    h,w,x = self.block(h,inj,AB,kc,kc0,w,x)
            logits = self.w_cls(h).float()
        # No detach inside the block loop. Gradients flow through writes/traces within a segment.
        new = replace(carry, current_hidden=h.detach(),
                      coupling=None if w is None else w.detach(),
                      value_trace=None if x is None else x.detach())
        return new, logits


class LT(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
        self.inner = LT_Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch):
        c, device = self.config, self.inner.init_hidden.device
        B = batch["inputs"].shape[0]
        h = torch.zeros(B,c.seq_len,c.hidden_size,device=device)
        return LTCarry(h, torch.zeros(B,dtype=torch.int32,device=device),
                       torch.ones(B,dtype=torch.bool,device=device),
                       {k: torch.zeros_like(v,device=device) for k,v in batch.items()},
                       torch.zeros(B,c.num_heads,c.seq_len,c.seq_len,device=device) if c.stdp else None,
                       torch.zeros(B,c.seq_len,c.num_heads,c.hidden_size//c.num_heads,device=device) if c.stdp else None)

    def forward(self, carry, batch, compute_target_q=False):
        fresh = carry.halted
        h = torch.where(fresh[:,None,None], self.inner.init_hidden, carry.current_hidden)
        if self.config.stdp:
            mask = fresh[:,None,None,None]
            w = torch.where(mask, torch.zeros_like(carry.coupling), carry.coupling)
            x = torch.where(mask, torch.zeros_like(carry.value_trace), carry.value_trace)
        else:
            w = x = None
        data = {k: torch.where(fresh.view((-1,)+(1,)*(v.ndim-1)), batch[k], v)
                for k,v in carry.current_data.items()}
        inner = replace(carry,current_hidden=h,coupling=w,value_trace=x,current_data=data)
        inner, logits = self.inner(inner,data)
        steps = torch.where(fresh,0,carry.steps)+1
        halted = steps >= self.config.loops
        q = torch.full((logits.shape[0],),-5.0,dtype=torch.float32,device=logits.device)
        return replace(inner,steps=steps,halted=halted), {
            "logits":logits,"q_halt_logits":q,"q_continue_logits":q}


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
                 "gain_raw","eta_raw","lam_raw","w0","A_raw","rho_raw")


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


def save_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device,keep_last=None):
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
            ck = dict(model_id=MODEL_ID,step=ts.step,iter_id=ts.iter_id,batch_in_iter=ts.batch_in_iter,
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
                     ("batch_size","amp","amp_dtype","activation_checkpoint")) + (
    "global_batch_size","epochs","eval_interval","num_aug","seed","grad_accum_steps",
    "lr","lr_min_ratio","lr_warmup_steps","weight_decay","beta1","beta2","puzzle_emb_lr",
    "puzzle_emb_weight_decay","q_weight","ema","ema_rate","data_fingerprint")


def load_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device):
    # Only load checkpoints you trust: weights_only=False is needed for RNG/optimizer objects.
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("model_id") != MODEL_ID:
        raise ValueError("This is not a value-trace delayed-STDP checkpoint. "
                         "Old v1.1 beta/address-trace checkpoints cannot be resumed as this model.")
    old = ck["cfg"]
    changed = {k:(old.get(k),cfg.get(k)) for k in _RESUME_KEYS if old.get(k)!=cfg.get(k)}
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
        print(f"[LT] resharded saved h/w/x/data: {ck['world_size']} -> {ws} ranks; "
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
    print(f"[EVAL] step {step} acc {d['accuracy']/max(n,1):.4f} "
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
    lines = [f"# step={step} weights={'ema' if ema else 'raw'} segs={segs} "
             f"elapsed={time.monotonic()-t0:.1f}s partial={interrupted}",
             f"# {MODEL_ID}; per segment {cfg['blocks_per_seg']} blocks; train segments={loops0}",
             "# seg acc exact n exact_percent churn"]
    for si in range(segs):
        if nn[si]:
            lines.append(f"{si+1:4d} {aa[si]:.6f} {int(ee[si]):6d} {int(nn[si]):6d} "
                         f"{100*ee[si]/nn[si]:.4f} {cc[si]:.6f}"+
                         ("  <-train" if si+1==loops0 else ""))
    if not interrupted and nn[-1]>0:
        be = int(np.argmax(ee))
        lines.append(f"# best_exact_seg={be+1} exact={int(ee[be])}/{int(nn[be])}")
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)),exist_ok=True)
    with open(out_txt+".tmp","w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    os.replace(out_txt+".tmp",out_txt)
    print(f"[EXTRAP] step {step} -> {out_txt}"+(" [partial]" if interrupted else ""),flush=True)
    return dict(acc=aa.tolist(),exact=ee.tolist(),churn=cc.tolist(),count=nn.tolist())

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


def train_batch(model,base,ts,batch,cfg,optimizers,lrs,planned_steps,rank,ws,device):
    ts.in_step = True
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
        lr = cosine_schedule_with_warmup_lr_lambda(ts.step,base_lr=base_lr,
             num_warmup_steps=cfg["lr_warmup_steps"],num_training_steps=planned_steps,
             min_ratio=cfg["lr_min_ratio"])
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
    ts.carry,ts.step = nc,ts.step+1
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


def resolve_out_dir(cfg):
    if cfg["out_dir"]:
        return os.path.abspath(os.path.expanduser(str(cfg["out_dir"])))
    root = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    return os.path.join(root,"lt_value_window")


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
    ts = (load_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device) if path else TrainState())
    if ts.step != ts.iter_id*steps_per_iter+ts.batch_in_iter:
        raise ValueError("Checkpoint cursor/step inconsistency.")
    if ts.batch_in_iter>=steps_per_iter or ts.iter_id>total_iters:
        raise ValueError("Checkpoint cursor is outside the dataset protocol.")
    if rank==0:
        with open(os.path.join(out_dir,"config.json"),"w",encoding="utf-8") as f:
            json.dump(dict(cfg,model_id=MODEL_ID),f,ensure_ascii=False,indent=2)
    if cfg["compile"] and cfg["inductor_no_persist"]:
        try:
            import torch._inductor.config as ic
            ic.triton.persistent_reductions = False
        except (AttributeError,ImportError) as exc:
            if rank==0:
                print(f"[LT] optional Inductor setting unavailable: {exc}",flush=True)
    model = torch.compile(base,dynamic=False) if cfg["compile"] else base
    if rank==0:
        print(f"[LT] {MODEL_ID} torch={torch.__version__} device={device} ranks={ws} local_bs={lbs}",flush=True)
        print(f"[LT] params={sum(p.numel() for p in base.parameters()):,} amp={cfg['amp_dtype']} "
              f"activation_checkpoint={cfg['activation_checkpoint']} compile={cfg['compile']}",flush=True)
        print(f"[LT] data={cfg['data_npz']} train={len(tr_x)} test={len(te_x)}",flush=True)
        print(f"[LT] planned steps={planned_steps}; actual steps={actual_steps}; "
              f"1 step={cfg['blocks_per_seg']} blocks; loops={cfg['loops']} segments",flush=True)
        print(f"[LT] W0={cfg['stdp_w0_fixed']} A_fixed={cfg['stdp_A_fixed']} "
              f"rho_fixed={cfg['value_trace_rho_fixed']} (None means learned)",flush=True)
        print(f"[LT] {'RESUME '+str(path) if path else 'NEW RUN'} step={ts.step} "
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
                    save_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device)
                if boundary:
                    evaluate(base,te_x,te_y,cfg,rank,ws,device,ts.step,ema,deadline)
                    last_eval_step = ts.step
                if due_milestone:
                    mdir = os.path.join(out_dir,"milestones")
                    save_checkpoint(mdir,ts,base,optimizers,ema,cfg,rank,ws,device,keep_last=0)
                    extrapolate(base,te_x,te_y,cfg,rank,ws,device,ts.step,ema,
                                cfg["milestone_extrap_segs"],os.path.join(mdir,f"extrap_step_{ts.step}.txt"),deadline)
                if ts.step>=stop_at:
                    break
                if (boundary or due_milestone) and stop_requested(device,deadline):
                    stopped = True
                    break
        # Save before evaluation so interruption during the final eval loses no training progress.
        p = save_checkpoint(out_dir,ts,base,optimizers,ema,cfg,rank,ws,device)
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
    import tempfile
    old_threads = torch.get_num_threads()
    rng0 = _rng_state(torch.device("cpu"))
    torch.set_num_threads(min(2,old_threads))
    checks = []
    try:
        torch.manual_seed(17)
        cfg = dict(DEFAULT_CFG,hidden_size=16,num_heads=2,puzzle_emb_ndim=16,
                   global_batch_size=2,batch_size=2,seq_len=81,blocks_per_seg=3,loops=3,
                   amp=False,amp_dtype="float32",activation_checkpoint=False,compile=False,
                   epochs=4,eval_interval=2,lr_warmup_steps=0,lr=1e-3,puzzle_emb_lr=1e-3,
                   ema_rate=0.9,run_selftests=False,dataloader_workers=0)
        base = ACTLossHead(LT(cfg),q_weight=cfg["q_weight"])
        inner,L = base.model.inner,base.model.inner.layers[0]
        B,T,H,dh = 2,81,2,8
        h = torch.randn(B,T,16)
        w = torch.randn(B,H,T,T)*0.03
        x = F.normalize(torch.randn(B,T,H,dh),dim=-1)*0.5
        AB = inner.W_C(L)
        kc,kc0 = inner.kernel(L),inner.kernel(L,torch.zeros_like(L.psi))

        # Independent complex-number implementation of the specified attention.
        def reference_attention(uh,psi):
            ux,uy = uh
            z = torch.complex(ux,uy)
            pp = L.theta[...,0,None]*inner.pos_u + L.theta[...,1,None]*inner.pos_w
            qa = (pp+psi[...,None]/2).permute(2,0,1)
            ka = (pp-psi[...,None]/2).permute(2,0,1)
            q = z*torch.exp(1j*qa)
            k = z*torch.exp(1j*ka)
            return torch.einsum("bthj,bnhj->bhtn",q,k.conj()).real*torch.exp(-L.alpha[:,0,None,None]*inner.l1)[None]

        with torch.no_grad():
            uh = inner.addr(h,AB)
            ar = reference_attention(uh,L.psi)
            Kr = reference_attention(uh,torch.zeros_like(L.psi))
            torch.testing.assert_close(ar,inner.attn_xy(uh,kc),rtol=2e-5,atol=2e-6)
            v = torch.einsum("btd,hcd->bthc",h,L.w_sh)
            vv = v/(v.norm(dim=-1,keepdim=True)+cfg["eps"])
            eta,lam,gain,w0,A,rho = inner.plasticity_scalars(L)
            agree = torch.einsum("bthc,bnhc->bhtn",vv,vv)
            M = torch.einsum("bthc,bnhc->bhtn",vv,x)
            anti = M-M.transpose(-1,-2)
            read = torch.einsum("bhtn,bnhc->bthc",(1-lam)*ar+lam*w,v)
            href = inner.phi(h+torch.einsum("bthc,hcd->btd",read,L.w_sh))
            wref = (1-eta)*w+eta*gain*Kr*(w0*agree+A*anti)
            xref = rho.view(1,1,H,1)*x+(1-rho.view(1,1,H,1))*vv
            hn,wn,xn = inner.step(L,h,AB,kc,kc0,w,x)
            for actual,expected in ((hn,href),(wn,wref),(xn,xref)):
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
            checks.append("independent complex reference: read/write/trace equations")
            torch.testing.assert_close(Kr,Kr.transpose(-1,-2),rtol=0,atol=1e-6)
            torch.testing.assert_close(anti,-anti.transpose(-1,-2),rtol=0,atol=0)
            wz = inner.step(L,h,AB,kc,kc0,torch.zeros_like(w),x)[1]
            torch.testing.assert_close((wz+wz.transpose(-1,-2))/2,eta*gain*Kr*w0*agree,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close((wz-wz.transpose(-1,-2))/2,eta*gain*Kr*A*anti,rtol=2e-5,atol=2e-6)
            checks.append("symmetric K; symmetric/antisymmetric write parity")
            wf = inner.step(L,h,AB,kc,kc0,torch.zeros_like(w),torch.zeros_like(x))[1]
            torch.testing.assert_close(wf,eta*gain*Kr*w0*agree,rtol=2e-5,atol=2e-6)
            old_A = L.A_raw.detach().clone()
            L.A_raw.add_(1.0)
            h2,w2,x2 = inner.step(L,h,AB,kc,kc0,w,x)
            L.A_raw.copy_(old_A)
            assert torch.equal(hn,h2) and not torch.equal(wn,w2) and torch.equal(xn,x2)
            checks.append("one-block read delay; fresh first write = eta*tgt")
            prev = F.normalize(torch.randn_like(vv),dim=-1)
            mm = torch.einsum("bthc,bnhc->bhtn",vv,prev)
            delta = vv-prev
            identity = (torch.einsum("bthc,bnhc->bhtn",delta,vv)-
                        torch.einsum("bthc,bnhc->bhtn",vv,delta))
            torch.testing.assert_close(mm-mm.transpose(-1,-2),identity,rtol=2e-5,atol=1e-6)
            checks.append("rho=0 exact finite-difference identity")
            # Explicit finite-history sum and the frozen-parameter bounds.
            rv = torch.tensor([0.2,0.8]).view(1,1,H,1)
            xx = torch.zeros(B,5,H,dh)
            history = []
            ww = torch.zeros(B,H,5,5)
            for k in range(20):
                cur = F.normalize(torch.randn_like(xx),dim=-1)
                expected = sum(((1-rv)*rv**(k-1-j)*history[j] for j in range(k)),torch.zeros_like(xx))
                torch.testing.assert_close(xx,expected,rtol=2e-5,atol=1e-6)
                mm = torch.einsum("bthc,bnhc->bhtn",cur,xx)
                aa = mm-mm.transpose(-1,-2)
                ag = torch.einsum("bthc,bnhc->bhtn",cur,cur)
                kk = torch.einsum("bthc,bnhc->bhtn",cur,cur)
                ww = (1-eta)*ww+eta*gain*kk*(w0*ag+A*aa)
                assert (aa.abs()<=2.0+1e-6).all()
                assert (ww.abs()<=gain*(abs(w0)+2*A)+1e-6).all()
                xx = rv*xx+(1-rv)*cur
                assert (xx.norm(dim=-1)<=1.0+1e-6).all()
                history.append(cur)
            checks.append("normalized exponential history and fixed-parameter bounds")

        batch = dict(inputs=torch.randint(1,11,(B,T),dtype=torch.int32),
                     labels=torch.randint(2,11,(B,T),dtype=torch.int32),
                     puzzle_identifiers=torch.zeros(B,dtype=torch.int32))
        alt = {k:v.clone() for k,v in batch.items()}
        alt["inputs"] = torch.randint(1,11,(B,T),dtype=torch.int32)
        alt["labels"] = torch.randint(2,11,(B,T),dtype=torch.int32)
        with torch.no_grad():
            c0 = base.initial_carry(batch)
            c1,_ = base.model(c0,batch)
            c2,_ = base.model(c1,alt)
            assert torch.equal(c2.current_data["inputs"],batch["inputs"])
            mixed = replace(c1,halted=torch.tensor([True,False]))
            cm,_ = base.model(mixed,alt)
            fresh_alt,_ = base.model(base.initial_carry(alt),alt)
            continued,_ = base.model(c1,alt)
            for name in ("current_hidden","coupling","value_trace"):
                torch.testing.assert_close(getattr(cm,name)[0],getattr(fresh_alt,name)[0])
                torch.testing.assert_close(getattr(cm,name)[1],getattr(continued,name)[1])
            assert cm.steps.tolist()==[1,2]
            checks.append("per-lane h/w/x/data reset; active puzzle retention")
            inner.config.blocks_per_seg = 2
            ca,_ = base.model(base.initial_carry(batch),batch)
            cb,ob = base.model(ca,batch)
            inner.config.blocks_per_seg = 4
            cc,oc = base.model(base.initial_carry(batch),batch)
            for name in ("current_hidden","coupling","value_trace"):
                torch.testing.assert_close(getattr(cb,name),getattr(cc,name),rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(ob["logits"],oc["logits"])
            inner.config.blocks_per_seg = 3
            assert not cb.current_hidden.requires_grad and not cb.coupling.requires_grad and not cb.value_trace.requires_grad
            checks.append("two segments equal one longer unroll in value; boundary detach")

        other = ACTLossHead(LT(dict(cfg,activation_checkpoint=True)),q_weight=cfg["q_weight"])
        other.load_state_dict(base.state_dict())
        for m in (base,other):
            m.zero_grad(set_to_none=True)
            _,loss,_,_,_ = m(carry=m.initial_carry(batch),batch=batch,return_keys=set())
            loss.backward()
            for n,p in m.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),n
            for suffix in ("A_raw","rho_raw"):
                p = next(p for n,p in m.named_parameters() if n.endswith(suffix))
                assert p.grad is not None and p.grad.norm()>0,suffix
        for (n,p),(n2,p2) in zip(base.named_parameters(),other.named_parameters()):
            assert n==n2
            if p.grad is not None:
                torch.testing.assert_close(p.grad,p2.grad,rtol=2e-5,atol=3e-6)
        torch.testing.assert_close(base.model.puzzle_emb.local_weights.grad,
                                   other.model.puzzle_emb.local_weights.grad,rtol=2e-5,atol=3e-6)
        checks.append("finite gradients to A/rho; activation-checkpoint gradient equivalence")
        for fixes in (dict(stdp_A_fixed=0.0,value_trace_rho_fixed=0.0),dict(stdp=False)):
            ablated = ACTLossHead(LT(dict(cfg,**fixes)))
            _,loss,_,_,_ = ablated(carry=ablated.initial_carry(batch),batch=batch,return_keys=set())
            loss.backward()
            assert torch.isfinite(loss)
        checks.append("exact A=0/rho=0 and STDP-off execution")

        sol = np.array([[(3*(r%3)+r//3+c)%9+1 for c in range(9)] for r in range(9)],dtype=np.uint8)
        inp = sol.copy(); inp[::2,::2] = 0
        for seed in range(30):
            prm = _draw_aug_params(np.random.default_rng(seed))
            xx,yy = _apply_aug(inp,*prm),_apply_aug(sol,*prm)
            _check_boards(xx[None],yy[None],"selftest_aug")
            dm,tr,rp,cp = prm
            ref = dm[(inp.T if tr else inp)[np.ix_(rp,cp)]]
            assert np.array_equal(xx,ref)
        args = dict(seed=0,num_aug=10,global_batch_size=2,rank=0,world_size=1,
                    epochs_per_iter=2,start_iter=0,total_iters=2)
        ds = SudokuTrainDataset(np.repeat(inp[None],8,axis=0),np.repeat(sol[None],8,axis=0),**args)
        assert np.array_equal(ds._augmented(0,7)[0],ds._augmented(0,7)[0])
        full = list(ds)
        skipped = list(SudokuTrainDataset(ds.inputs,ds.labels,**args,skip_batches=3))
        assert len(skipped)==len(full)-3
        assert all(torch.equal(a,b) for a,b in zip(full[3][1].values(),skipped[0][1].values()))
        checks.append("augmentation validity, fixed pool, and data-cursor skip")

        # Complete training-state save/reload must give the exact same next CPU update.
        torch.manual_seed(91)
        m = ACTLossHead(LT(cfg),q_weight=cfg["q_weight"])
        opts,lrs = create_optimizers(m,cfg,1)
        ema = EMAHelper(cfg["ema_rate"]); ema.register(m)
        ts = TrainState()
        for j in range(2):
            train_batch(m,m,ts,full[j][1],cfg,opts,lrs,16,0,1,torch.device("cpu"))
            ema.update(m); ts.batch_in_iter += 1
        with tempfile.TemporaryDirectory() as td:
            p = save_checkpoint(td,ts,m,opts,ema,cfg,0,1,torch.device("cpu"))
            rand_expected = torch.rand(4)
            train_batch(m,m,ts,full[2][1],cfg,opts,lrs,16,0,1,torch.device("cpu"))
            ema.update(m); ts.batch_in_iter += 1
            resumed = ACTLossHead(LT(cfg),q_weight=cfg["q_weight"])
            ro,rl = create_optimizers(resumed,cfg,1)
            rema = EMAHelper(cfg["ema_rate"]); rema.register(resumed)
            rs = load_checkpoint(p,resumed,ro,rema,cfg,0,1,torch.device("cpu"))
            assert rs.step==2 and rs.batch_in_iter==2
            assert torch.equal(rand_expected,torch.rand(4))
            train_batch(resumed,resumed,rs,full[2][1],cfg,ro,rl,16,0,1,torch.device("cpu"))
            rema.update(resumed); rs.batch_in_iter += 1
            for n,v in m.state_dict().items():
                assert torch.equal(v,resumed.state_dict()[n]),n
            for n,v in ema.shadow.items():
                assert torch.equal(v,rema.shadow[n]),n
            for name in ("current_hidden","coupling","value_trace"):
                assert torch.equal(getattr(ts.carry,name),getattr(rs.carry,name)),name
        checks.append("checkpoint resume: identical next weights/EMA/h/w/x and RNG on CPU")
        for item in checks:
            print("[selftest] PASS "+item,flush=True)
        print(f"[selftest] {len(checks)}/{len(checks)} groups passed (CPU, synthetic inputs).",flush=True)
        return checks
    finally:
        _restore_rng(rng0,torch.device("cpu"))
        torch.set_num_threads(old_threads)


def _cli():
    ap = argparse.ArgumentParser(description="Self-contained LT value-trace STDP trainer")
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
    runtime = Path(tempfile.mkdtemp(prefix="_lt_value_stdp_", dir=str(work_root)))
    trainer_path = runtime / "train_value_stdp.py"
    config_path = runtime / "config.json"
    trainer_path.write_text(_TRAINER_SOURCE, encoding="utf-8")
    config_path.write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

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
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
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
    if rc:
        raise RuntimeError(f"LT 학습 프로세스가 종료 코드 {rc}로 실패했습니다. 위 오류 로그를 확인하세요.")


if __name__ == "__main__":
    launch_lt_one_cell()
