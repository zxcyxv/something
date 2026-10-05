# -*- coding: utf-8 -*-
# LT KV-STDP — 기존 스도쿠 하네스의 Kaggle 단일 셀 (v1.71/v1.8 경로도 유지)
# 이 파일 전체를 한 셀에 붙여넣고 실행합니다. 설정은 아래 CFG에서 변경합니다.
# 데이터: 기존처럼 sudoku_lt_1k.npz를 Kaggle Input으로 연결하거나 data_npz를 지정합니다.
# 기본 CFG는 KV-STDP를 처음부터 학습한다. 입력 재주입 → 기억 쓰기·읽기 → 쌍선형 FFN → 고정 Φ.
#   G = mean_tokens(V e_Kᵀ − e_V Kᵀ), M ← M + G. Q/K/e_K에 학습형 실수 2D RoPE를 적용한다.
#   각 헤드·feature 쌍의 회전각은 θ행·행좌표 + θ열·열좌표. θ 초기화는 v1.7과 같은 U[-π/2,π/2].
#   헤드·RoPE 쌍별 λ를 학습하며 K/V 흔적을 공유 계수로 누수적분한다. 기억 유지·쓰기 계수는 둘 다 1.
#   kv_memory_update='ema'이면 M ← rho*M + (1-rho)*G. 고정 rho는 kv_memory_rho로 지정한다.
#   kv_memory_update='leaky'이면 기억만 감쇠하고 쓰기 강도는 유지한다: M ← rho*M + G.
#   기본 Q/K/e_K 정규화는 끈다. K/V 흔적과 외적 차감은 원시 활동으로 계산한다.
#   qkv_no_decay=False: Q/K/V와 출력 사영, FFN에 weight_decay를 적용한다.
# 세그먼트마다 앞 8블록은 no-grad, 뒤 8블록만 역전파(nograd_fixed=8).
# 기존 모델: memory_type='address', plastic_select=False(v1.71)/True(v1.8).
# 현재 KV 외적만 누적하는 비교: memory_type='kv_hebbian', stdp=False, use_trace=False.
# 출력은 /kaggle/working/lt_kv_stdp_ng8 에 저장합니다. 같은 모델의 체크포인트에서 재개합니다.
# max_steps는 추가 횟수가 아닌 절대 종료 step입니다.
# 기본: batch128, 16seg x (8 no-grad + 8 grad) blocks, lr1e-4 고정 (warmup 2000).
# milestone: 매10000step, 전체 중 고정512문제, seg128까지 평가하고 표를 화면과 train.log에 출력.
# 기본 사영 정밀도는 BF16. 정규화·흔적·기억·쓰기 차감·읽기는 FP32로 계산합니다.

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
 'hidden_size': 832,
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
 'puzzle_emb_ndim': 832,
 'memory_type': 'kv_stdp',
 'trace_decay_init': 0.1,
 'trace_decay_mode': 'pair',
 'kv_write_reduction': 'mean',
 'kv_memory_update': 'additive',
 'kv_memory_rho': 0.95,
 'kv_trace_activity_detach': False,
 'kv_qk_rmsnorm': False,
 'kv_qk_l2norm': False,
 'kv_projection_fp32': False,
 'rope_type': 'learned_2d',
 'rope_base': 10000.0,
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
 'beta_init_mean': 1.5707963267948966,
 'beta_init_std': 0.0,
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
 'eval_interval': 250,
 'compile': True,
 'inductor_no_persist': True,
 'out_dir': '/kaggle/working/lt_kv_stdp_ng8',
 'resume_from': None,
 'require_resume': False,
 'allow_trace_activity_fork': False,
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
 'address_projection': 'linear',
 'trace_rho_init': 0.5,
 'inj_gate_init': 0.25,
 'gamma_init': 0.1,
 'plastic_select': False,
 'select_g_max': 4.0,
 'late_sup_prob': 0.0,
 'late_sup_min': 16,
 'late_sup_max': 112,
 'nograd_fixed': 8,
 'nograd_every': 0,
 'nograd_start': 0,
 'nograd_max': 16}

# 독립 실행되는 학습 프로그램 전체
_TRAINER_SOURCE = r'''# -*- coding: utf-8 -*-
"""LT Sudoku trainer: KV-STDP fast weights and the existing v1.8 / v1.71 models.

The execution, data, loss, optimizer, checkpoint and launcher harness comes from
the user's existing working Kaggle cell. The model follows v1.71 in this repository:
post MLP, QR-free W_C, original normalized address/value kernels, write then read W.
v1.8 (plastic_select=True) generates the memory keep/write/read gates from the
per-cell state instead of per-head constants; the write window (beta) is unchanged.
KV-STDP uses past-only K/V eligibility traces, real feature-pair RoPE, additive
or EMA channel memory and an affine-free hidden post-norm. One step is one segment;
hidden, memory and traces detach only at segment boundaries.
Activation checkpointing only recomputes blocks. No repository imports are needed.
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

MODEL_ID = "lt-v171-address-trace-linear-v1"          # v1.71 (plastic_select=False)
MODEL_ID_V18 = "lt-v18-selective-plasticity-v1"      # v1.8  (plastic_select=True)
MODEL_ID_KV_STDP = "lt-kv-stdp-fast-weight-v1"
MODEL_ID_KV_HEBBIAN = "lt-kv-hebbian-fast-weight-v1"
IGNORE_LABEL_ID = -100
_STOP_REQUESTED = False


def model_id_of(cfg):
    if cfg.get("memory_type", "address") == "kv_hebbian":
        return MODEL_ID_KV_HEBBIAN
    if cfg.get("memory_type", "address") == "kv_stdp":
        return MODEL_ID_KV_STDP
    return MODEL_ID_V18 if cfg.get("plastic_select") else MODEL_ID


def model_tag_of(cfg):
    if cfg.get("memory_type", "address") == "kv_hebbian":
        return "kv_hebbian"
    if cfg.get("memory_type", "address") == "kv_stdp":
        return "kv_stdp"
    return "v18" if cfg.get("plastic_select") else "v171"

# DEFAULT_CFG is also copied to the top of the one-cell notebook.
DEFAULT_CFG = {'data_npz': None,
 'num_aug': 1000,
 'test_size': 2048,
 'hidden_size': 832,
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
 'puzzle_emb_ndim': 832,
 'memory_type': 'address',
 'trace_decay_init': 0.1,
 'trace_decay_mode': 'head',
 'kv_write_reduction': 'mean',
 'kv_memory_update': 'additive',
 'kv_memory_rho': 0.95,
 'kv_trace_activity_detach': False,
 'kv_qk_rmsnorm': False,
 'kv_qk_l2norm': False,
 'kv_projection_fp32': False,
 'rope_type': 'learned_2d',
 'rope_base': 10000.0,
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
 'beta_init_mean': 1.5707963267948966,
 'beta_init_std': 0.0,
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
 'eval_interval': 250,
 'compile': True,
 'inductor_no_persist': True,
 'out_dir': None,
 'resume_from': None,
 'require_resume': False,
 'allow_trace_activity_fork': False,
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
 'address_projection': 'linear',
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

PRESETS = {                       # v1.71 은 v1.7 의 주소 사영에서 QR 만 제거한다. v1.8 은 v1.71 + 선택적 가소성 게이트
    "v1":    dict(legacy_gauge=True,  block_order="pre",  use_trace=False, address_projection="qr",     plastic_select=False),  # 9/1 원본
    "v1.1":  dict(legacy_gauge=False, block_order="pre",  use_trace=False, address_projection="qr",     plastic_select=False),  # √d 고정 게이지. 학습·측정 완료
    "v2":    dict(legacy_gauge=False, block_order="post", use_trace=False, address_projection="qr",     plastic_select=False),  # v1.1 + post 순서. 아직 학습 안 함
    "v1.7":  dict(legacy_gauge=False, block_order="post", use_trace=True,  address_projection="qr",     plastic_select=False),  # v2 + 주소 흔적 z. 학습 후반 불안정
    "v1.71": dict(legacy_gauge=False, block_order="post", use_trace=True,  address_projection="linear", plastic_select=False),  # v1.7 + 자유 선형 W_C. 204k Kaggle 학습
    "v1.8":  dict(legacy_gauge=False, block_order="post", use_trace=True,  address_projection="linear", plastic_select=True),   # v1.71 + η·g·λ 를 칸 상태에서 생성
}
for _preset in PRESETS.values():
    _preset["memory_type"] = "address"
PRESETS["kv_stdp"] = dict(legacy_gauge=False, block_order="post", use_trace=True,
                          address_projection="linear", plastic_select=False,
                          memory_type="kv_stdp", stdp=True, rope_type="learned_2d",
                          trace_decay_mode="pair",
                          kv_write_reduction="mean",
                          kv_memory_update="additive", kv_memory_rho=0.95,
                          kv_trace_activity_detach=False,
                          qkv_no_decay=False, kv_qk_rmsnorm=False, kv_qk_l2norm=False,
                          kv_projection_fp32=False)
PRESETS["kv_hebbian"] = dict(PRESETS["kv_stdp"], memory_type="kv_hebbian",
                              stdp=False, use_trace=False, kv_trace_activity_detach=False)

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


# 2. v1.71 model: direct address projection, post MLP, rotating Z
# Canonical v1.71 model; importless fragment for the standalone trainer.
# Needs the harness imports, trunc_normal_init_, CastedSparseEmbedding, checkpoint.
# Numerical operations retain original_train.py's precision and ordering.
@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    coupling: Optional[torch.Tensor] = None       # 주소 기억 [B,H,T,T] / KV 기억 [B,H,Dv,Dk]
    fresh: Optional[torch.Tensor] = None          # [B] bool — 이 퍼즐의 w·z 가 아직 초기화 전
    trace: Optional[torch.Tensor] = None          # 주소 흔적 z [B,T,H,p,2] (use_trace 일 때만)
    key_trace: Optional[torch.Tensor] = None      # KV-STDP: 과거 K 활동 [B,H,T,Dk]
    value_trace: Optional[torch.Tensor] = None    # KV-STDP: 과거 V 활동 [B,H,T,Dv]


@dataclass
class LTConfig:
    """모델 설정. 기존 판은 QR 주소 사영을 유지하며, v1.71 만 직접 선형 사영을 쓴다.
      v1   = legacy_gauge=True,  block_order="pre",  use_trace=False   (9/1 원본)
      v1.1 = legacy_gauge=False, block_order="pre",  use_trace=False   (최신 학습판)
      v2   = legacy_gauge=False, block_order="post", use_trace=False   (미학습)
      v1.7 = legacy_gauge=False, block_order="post", use_trace=True
      v1.71 = v1.7 + address_projection="linear". 공유 W_C 와 복소 주소·정규화·흔적은 동일하다.
      v1.8 = v1.71 + plastic_select=True. 쓰기 창(β)·흔적·agree 는 그대로, 헤드별 상수 η·g·λ 대신
             유지 A=exp(−Δ)·쓰기 이득 g·읽기 보간 λ 를 매 블록 칸 상태 q 에서 생성한다.
      address_projection 키가 없는 기존 설정은 "qr", plastic_select 키가 없으면 False 로 읽는다.
    """
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int = 832
    num_heads: int = 8
    loops: int = 16
    grid: int = 9
    blocks_per_seg: int = 8     # 세그먼트당 블록 수 (× num_layers)
    num_layers: int = 1         # 가중치 벌 수. 블록 k → layers[k % num_layers]
    mlp_expansion: float = 4.0
    memory_type: str = "address"   # 기존 칸×칸 기억 / 채널×채널 KV-STDP Fast Weight
    trace_decay_init: float = 0.1  # KV-STDP: 이전 흔적 10%, 현재 활동 90%로 시작 (K/V 공유)
    trace_decay_mode: str = "head"  # head: 헤드별 / pair: 헤드·RoPE 쌍별 (두 채널, K/V 공유)
    kv_write_reduction: str = "mean"  # mean: 칸 합을 T로 나눔 / sum: 나누지 않음. 키 없는 과거 cfg는 mean.
    kv_memory_update: str = "additive"  # additive: M+G / ema: rho*M+(1-rho)*G / leaky: rho*M+G.
    kv_memory_rho: float = 0.95  # EMA/leaky 기억 유지율. 고정값이며 K/V 흔적 lambda와 별개.
    kv_trace_activity_detach: bool = False  # EMA에 들어가는 K/V 원천만 detach. 현재 쓰기와 lambda 미분은 유지.
    kv_qk_rmsnorm: bool = False    # Q 읽기 / (e_K+iK) 쓰기 사본의 affine 없는 RMS. 과거 cfg는 비정규화.
    kv_qk_l2norm: bool = False     # Q/K/eK 각각 norm+eps로 단위화. raw K/V 흔적은 그대로 이월.
    kv_projection_fp32: bool = False  # K/V 사영도 AMP 제외. 키 없는 과거 cfg는 기존 AMP 사영.
    rope_type: str = "learned_2d"  # KV-STDP: 헤드별 θ행·행 + θ열·열. axial_2d는 기존 체크포인트 호환.
    rope_base: float = 10000.0     # axial_2d에서만 사용
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
    address_projection: str = "qr"  # qr: QR(wc_raw.T).Q.T | linear: wc; 둘 다 헤드별 [dh,d]
    psi_zero: bool = False      # ψ ≡ 0 고정 (읽기 커널 대칭)
    stdp: bool = True           # 결합 기억 w ← (1−η)w + η·g·a^β·agree,  읽기 a_eff = (1−λ)a + λw
    stdp_eta_init: float = 0.05
    stdp_gain_init: float = 1.0
    stdp_lam_init: float = 0.25
    stdp_gain_fixed: float = -1.0   # ≥0 이면 g 를 이 값으로 고정
    stdp_lam_fixed: float = -1.0    # ≥0 이면 λ 를 이 값으로 고정
    beta_init_mean: float = 0.0    # rad; 키 없는 과거 cfg는 기존 N(0, 0.5²) 유지
    beta_init_std: float = 0.5     # 0이면 mean에서 정확히 시작; 이후 beta는 계속 학습
    block_order: str = "pre"    # pre: 경계 → 주입 → 스텝(Φ 포함) | post: 주입 → 스텝 → 경계 → Φ
    use_trace: bool = False     # 쓰기 창의 주소를 흔적 z ← μ⊙z + √(1−ρ²)⊙u 로 (μ_j = ρ_j e^{iω_j})
    trace_rho_init: float = 0.5
    plastic_select: bool = False    # v1.8: η·g·λ 를 칸 상태에서 생성 (쓰기 쌍은 대칭 결합)
    select_g_max: float = 4.0       # 생성된 쓰기 이득의 상한 → w 는 g_max·max|G| 로 유계
    nograd_blocks: int = 0          # 세그먼트 앞에서 gradient 없이 돌리는 블록 수 (하네스가 스텝마다 정한다)

    def __post_init__(self):
        if self.memory_type not in ("address", "kv_stdp", "kv_hebbian"):
            raise ValueError("memory_type must be address, kv_stdp or kv_hebbian")
        if self.memory_type in ("kv_stdp", "kv_hebbian"):
            if self.memory_type == "kv_stdp" and self.trace_decay_mode not in ("head", "pair"):
                raise ValueError("trace_decay_mode must be head or pair")
            if self.kv_write_reduction not in ("mean", "sum"):
                raise ValueError("kv_write_reduction must be mean or sum")
            _validate_kv_memory_update(self.kv_memory_update, self.kv_memory_rho)
            if not isinstance(self.kv_trace_activity_detach,bool):
                raise ValueError("kv_trace_activity_detach must be a boolean")
            if not isinstance(self.kv_qk_rmsnorm,bool):
                raise ValueError("kv_qk_rmsnorm must be a boolean")
            if not isinstance(self.kv_qk_l2norm,bool):
                raise ValueError("kv_qk_l2norm must be a boolean")
            if self.kv_qk_rmsnorm and self.kv_qk_l2norm:
                raise ValueError("kv_qk_rmsnorm and kv_qk_l2norm are mutually exclusive")
            if not isinstance(self.kv_projection_fp32,bool):
                raise ValueError("kv_projection_fp32 must be a boolean")
            if (self.kv_qk_rmsnorm or self.kv_qk_l2norm) and (not math.isfinite(self.eps) or self.eps<=0):
                raise ValueError("Q/K normalization requires finite eps>0")
            if self.legacy_gauge or self.block_order != "post" or self.plastic_select:
                raise ValueError("KV memory requires fixed injection/post-norm, block_order='post', plastic_select=False")
            if self.memory_type == "kv_stdp" and (not self.stdp or not self.use_trace):
                raise ValueError("kv_stdp requires stdp=True and use_trace=True")
            if self.memory_type == "kv_hebbian" and (self.stdp or self.use_trace or
                    self.kv_trace_activity_detach or self.kv_qk_rmsnorm or self.kv_qk_l2norm):
                raise ValueError("kv_hebbian requires stdp=False, use_trace=False and no Q/K normalization or trace activity detach")
            if self.memory_type == "kv_stdp" and not 0 < self.trace_decay_init < 1:
                raise ValueError("trace_decay_init must lie in (0, 1)")
            if self.rope_type not in ("learned_2d", "axial_2d"):
                raise ValueError("rope_type must be learned_2d or axial_2d")
            if self.rope_type == "axial_2d" and (not math.isfinite(self.rope_base) or self.rope_base <= 1):
                raise ValueError("rope_base must be finite and greater than 1")
            divisor = 4 if self.rope_type == "axial_2d" else 2
            if self.num_heads < 1 or self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % divisor:
                raise ValueError(f"KV memory needs hidden_size / num_heads divisible by {divisor} for {self.rope_type}")
            if not 0 <= self.puzzle_emb_ndim <= self.hidden_size:
                raise ValueError("puzzle_emb_ndim must lie between 0 and hidden_size")
        if not math.isfinite(self.beta_init_mean) or not math.isfinite(self.beta_init_std) or self.beta_init_std < 0:
            raise ValueError("beta_init_mean/std must be finite and beta_init_std must be nonnegative")
        if self.amp_dtype not in ("auto", "bfloat16", "float32"):
            raise ValueError("amp_dtype must be auto, bfloat16, or float32")
        if self.address_projection not in ("qr", "linear"):
            raise ValueError(f"address_projection must be 'qr' or 'linear', got {self.address_projection!r}")
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
        if d.get("memory_type") == "kv_stdp" and "rope_type" not in d:
            dd["rope_type"] = "axial_2d"                  # 옵션 도입 전 KV 체크포인트의 고정 RoPE
        if "use_trace" not in d and "trace_rho_init" in d:     # v1.6/v1.7 체크포인트 cfg 호환 (use_trace 키가 없던 시절)
            dd["use_trace"] = True
        return cls(**dd)


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LTLayer(nn.Module):
    """가중치 한 벌. QR 판은 기존 wc_raw, 직접 선형 사영 판은 wc 를 학습한다.

    주소 행렬의 모양과 초기 난수 draw 는 같고, 읽기·쓰기에 같은 사영을 공유한다.
    서로 다른 이름으로 저장해 QR raw 값을 실효 선형 가중치로 오인해 로드하지 않는다.
    """

    def __init__(self, config: "LTConfig", H, d, dh, p) -> None:
        super().__init__()
        wc = nn.Parameter(torch.randn(H, dh, d) / math.sqrt(d))
        if config.address_projection == "qr":
            self.wc_raw = wc
        else:
            self.wc = wc
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
            self.beta = nn.Parameter(torch.zeros(H, p))       # 쓰기 창의 위상 (ψ 와 별개)
            with torch.no_grad():
                # Consume the same random draws in every mode: changing beta
                # initialization alone must not change later weights at a fixed seed.
                self.beta.normal_(0.0, 0.5)
                if config.beta_init_std == 0:
                    self.beta.fill_(config.beta_init_mean)
                else:
                    self.beta.mul_(config.beta_init_std / 0.5).add_(config.beta_init_mean)
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
    def W_C(self, L):
        """공유 복소 주소의 실수 적층 [A;B]. qr 은 행직교, linear 는 wc 자체."""
        if self.config.address_projection == "qr":
            Q, _ = torch.linalg.qr(L.wc_raw.transpose(-1, -2))           # [H,d,dh]
            AB = Q.transpose(-1, -2)
        elif self.config.address_projection == "linear":
            AB = L.wc                                                # [H,dh,d], QR 없이 직접 사용
        else:
            raise ValueError(f"Unknown address_projection: {self.config.address_projection!r}")
        return AB[:, :self.p, :], AB[:, self.p:, :]


    def kernel(self, L, psi=None):
        """decay_h [H,T,T], 위상각 A_t = ψ/2 + θ·pos_t (q), B_t = −ψ/2 + θ·pos_t (k) → cos/sin [T,H,p]. psi 를 주면 그 위상차로 (STDP 창 β 용)."""
        psi = L.psi if psi is None else psi
        decay_h = torch.exp(-L.alpha[:, 0, None, None] * self.l1) if self.config.dist_decay else torch.ones_like(self.l1).expand(L.alpha.shape[0], -1, -1)
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w         # [H,p,T]
        A = (ppos + psi[..., None] / 2).permute(2, 0, 1); B = (ppos - psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)


    def attn_xy(self, xy, kc):
        """정규화된 주소에서 커널 kc 로 a 를 만든다 (회전 + 내적 + 감쇠)."""
        x, y = xy; decay_h, cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = torch.einsum('bthj,bnhj->bhtn', qx, kx) + torch.einsum('bthj,bnhj->bhtn', qy, ky)
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
            # no-grad 앞부분에서 캐시된 BF16 가중치가 뒤의 grad/checkpoint 구간에
            # 재사용되면 일부 사영의 그래프가 빠진다. KV 경로는 캐시 없이 캐스팅한다.
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=self.config.memory_type == "address"):
                nc, logits = self._forward(carry, batch)
            return replace(nc, current_hidden=nc.current_hidden.float()), logits.float()
        return self._forward(carry, batch)


    def addr_raw(self, h, AB):
        """정규화 전 복소 주소 u = W_C h 의 (실부, 허부) [B,T,H,p]."""
        A, Bm = AB
        return torch.einsum('btd,hjd->bthj', h, A), torch.einsum('btd,hjd->bthj', h, Bm)

    def _unit(self, x, y):
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.config.eps), y / (nrm + self.config.eps)

    def addr(self, h, AB):
        """정규화된 순간 복소 주소 û (분석 스크립트 호환)."""
        return self._unit(*self.addr_raw(h, AB))

    def trace_step(self, L, ux, uy, ztr, fresh):
        """흔적 z ← μ⊙z + √(1−ρ²)⊙u,  μ_j = ρ_j e^{iω_j}.  풀면 z_j(k) = Σ_m ρ_j^m e^{iω_j m} u_j(k−m)."""
        rho = torch.sigmoid(L.mu_rho_raw)
        cw, sw = torch.cos(L.mu_omega), torch.sin(L.mu_omega)
        gin = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-6))
        if ztr is None:
            zx, zy = gin * ux, gin * uy
        else:
            zx0, zy0 = ztr[..., 0], ztr[..., 1]
            if fresh is not None:
                m = fresh.view(-1, 1, 1, 1)
                zx0 = torch.where(m, torch.zeros_like(zx0), zx0)
                zy0 = torch.where(m, torch.zeros_like(zy0), zy0)
            zx = rho * (cw * zx0 - sw * zy0) + gin * ux
            zy = rho * (sw * zx0 + cw * zy0) + gin * uy
        return zx, zy, torch.stack((zx, zy), dim=-1)

    def step(self, L, h, AB, kc, w=None, fresh=None, kcb=None, ztr=None, apply_phi=True):
        """한 블록의 어텐션+수송.  읽기 a(û, ψ) · 쓰기 창 a^β(û 또는 ẑ, β) × agree · w 누적 · a_eff 로 v 수송."""
        ux, uy = self.addr_raw(h, AB)
        uh = self._unit(ux, uy)                                             # 순간 주소 û
        a = self.attn_xy(uh, kc)                                            # 읽기 커널 (ψ, 거리감쇠)
        v = torch.einsum('btd,hcd->bthc', h, L.w_sh)                        # 값 사영
        ztr_new = None
        if self.stdp:
            if self.use_trace:
                zx, zy, ztr_new = self.trace_step(L, ux, uy, ztr, fresh)
                win = self.attn_xy(self._unit(zx, zy), kcb)                 # 쓰기 창 on 흔적 ẑ
            else:
                win = self.attn_xy(uh, kcb)                                 # 쓰기 창 on 순간 û
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
                w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
                w = keep * w + write * tgt
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

    def block(self, L, h, inj, AB, kc, kcb, w, fresh, ztr):
        # Layer and fresh are explicit arguments so recomputation cannot pick up
        # another loop iteration's layer or reset mask.
        pre = self.config.block_order == "pre"
        if pre:
            h = self.boundary(L, h)
        h = h + self.embed_scale * inj
        h, w, ztr = self.step(L, h, AB, kc, w, fresh, kcb, ztr, apply_phi=pre)
        if not pre:
            h = self.boundary(L, h)
            h = self.phi(h)
        return h, w, ztr

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        ABs = [self.W_C(L) for L in self.layers]
        kcs = [self.kernel(L) for L in self.layers]
        kcbs = [self.kernel(L, L.beta) if self.stdp else None for L in self.layers]
        w = carry.coupling if self.stdp else None; fresh = carry.fresh if self.stdp else None
        ztr = carry.trace if self.use_trace else None
        if self.config.nograd_blocks:
            # TRM 식: 세그먼트 앞부분은 gradient 없이 상태만 진행하고, 뒤의 blocks_per_seg 블록만 역전파한다.
            with torch.no_grad():
                for _ in range(self.config.nograd_blocks):
                    for li, L in enumerate(self.layers):
                        h, w, ztr = self.block(L, h, inj, ABs[li], kcs[li], kcbs[li], w, fresh, ztr)
                        fresh = None
        for _ in range(self.config.blocks_per_seg):
            for li, L in enumerate(self.layers):
                AB, kc, kcb = ABs[li], kcs[li], kcbs[li]
                if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                    h, w, ztr = checkpoint(self.block, L, h, inj, AB, kc, kcb, w, fresh, ztr,
                                           use_reentrant=False, preserve_rng_state=False)
                else:
                    h, w, ztr = self.block(L, h, inj, AB, kc, kcb, w, fresh, ztr)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=(w.detach() if w is not None else None),
                       trace=(ztr.detach() if ztr is not None else None), fresh=None), self.w_cls(h)


class KVSTDPLayer(nn.Module):
    """독립 Q/K/V 사영, 헤드별/공간 회전쌍별 흔적 λ·공간 θ, 기존 쌍선형 FFN."""
    def __init__(self, config):
        super().__init__()
        d, H = config.hidden_size, config.num_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.head_dim = d // H
        if config.memory_type == "kv_stdp":
            logit = math.log(config.trace_decay_init / (1 - config.trace_decay_init))
            self.trace_decay_mode = config.trace_decay_mode
            lam_shape = (H, self.head_dim // 2) if self.trace_decay_mode == "pair" else (H,)
            self.trace_lam_raw = nn.Parameter(torch.full(lam_shape, logit))
        inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.b_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()
        if config.rope_type == "learned_2d":
            # 다른 가중치와 초기 hidden의 난수 draw를 유지하도록 Inner 마지막에 초기화한다.
            self.theta = nn.Parameter(torch.zeros(H, d // H // 2, 2))

    @property
    def trace_decay(self):
        return torch.sigmoid(self.trace_lam_raw)

    @property
    def trace_decay_channels(self):
        lam = self.trace_decay
        if self.trace_decay_mode == "pair":
            return lam.repeat_interleave(2, dim=-1)
        return lam[:, None].expand(-1, self.head_dim)


class KVSTDPInner(LT_Inner):
    """재귀축 STDP를 채널 외적 Fast Weight로 구현한다.

    k=e_K+iK, v=e_V+iV → Im(v kᴴ)=V e_Kᵀ−e_V Kᵀ.
    쓰기는 현재 활동이 들어오기 전 흔적으로 계산한다. 그 뒤 흔적을 갱신한다.
    RoPE는 K와 e_K의 feature 쌍에 동일하게 적용하며 과거/현재 복소축은 섞지 않는다.
    입력/출력, sparse embedding, BF16 문맥, 고정 Φ, 쌍선형 경계는 기존 하네스를 공유한다.
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.memory_decay = config.kv_memory_rho if config.kv_memory_update in ("ema", "leaky") else 1.0
        self.write_scale = 1.0 - self.memory_decay if config.kv_memory_update == "ema" else 1.0
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        if T != g * g or config.num_layers < 1 or config.blocks_per_seg < 1:
            raise ValueError("KV memory requires seq_len=grid**2 and positive layer/block counts")
        self.d, self.H, self.dh = d, H, d // H
        self.stdp = self.use_trace = config.memory_type == "kv_stdp"
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
        # 기존 하네스와 같은 초기 hidden. 입력과 독립적인 상태로 첫 재귀를 시작한다.
        self.init_hidden = nn.Buffer(trunc_normal_init_(
            torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)
        positions = torch.stack((torch.arange(T) // g, torch.arange(T) % g), dim=-1)
        rope_dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        self.register_buffer("pos_u", positions[:, 0].to(rope_dtype), persistent=False)
        self.register_buffer("pos_w", positions[:, 1].to(rope_dtype), persistent=False)
        if config.rope_type == "learned_2d":
            with torch.no_grad():
                self.layers[0].theta.uniform_(-math.pi / 2, math.pi / 2)
                for layer in self.layers[1:]:
                    layer.theta.copy_(self.layers[0].theta)
        else:
            per_axis = self.dh // 2
            inv_freq = config.rope_base ** (-torch.arange(0, per_axis, 2, dtype=rope_dtype) / per_axis)
            angles = (positions.to(rope_dtype)[..., None] * inv_freq).flatten(-2)
            self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
            self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)

    def rope_tables(self, L):
        if self.config.rope_type == "learned_2d":
            angles = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w
            angles = angles.transpose(-1, -2)[None]         # [1,H,T,Dk/2]
            return angles.cos(), angles.sin()
        return self.rope_cos, self.rope_sin

    def apply_rope(self, x, L=None, tables=None):
        """실수 feature 쌍에 공간 회전. 과거/현재를 나타내는 복소축에는 작용하지 않는다."""
        cos, sin = self.rope_tables(self.layers[0] if L is None else L) if tables is None else tables
        pairs = x.reshape(*x.shape[:-1], self.dh // 2, 2)
        x0, x1 = pairs.unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def memory_qk_views(self, q, k, e_k):
        """FP32/64 사용 사본만 정규화. raw K/V 흔적은 그대로 보존한다.

        L2: v1.7과 같은 x/(||x||₂+eps), Q/K/eK에 각각 독립 분모.
        기존 RMS 실험: Q는 자체 RMS, K/eK는 복소 K 전체 실수 성분의 RMS 공유.
        학습 배율/bias는 없다. vector_norm은 zero trace에서도 finite backward를 유지한다.
        """
        if self.config.kv_qk_l2norm:
            return tuple(x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True)
                              + self.config.eps) for x in (q, k, e_k))
        if not self.config.kv_qk_rmsnorm:
            return q, k, e_k
        q_inv = torch.rsqrt(q.square().mean(-1,keepdim=True)+self.config.eps)
        k_inv = torch.rsqrt(0.5*(k.square().mean(-1,keepdim=True)
                                +e_k.square().mean(-1,keepdim=True))+self.config.eps)
        return q*q_inv, k*k_inv, e_k*k_inv

    def update_memory(self, memory, write):
        """기억 갱신만 선택한다. 흔적, 쓰기 대상과 읽기 순서는 유지한다."""
        if self.config.kv_memory_update == "ema":
            return self.memory_decay * memory + self.write_scale * write
        if self.config.kv_memory_update == "leaky":
            return self.memory_decay * memory + write
        return memory + write

    def memory_step(self, L, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
        """누적 기억과 과거 흔적을 읽고 쓴다. 상태/차감/읽기에는 AMP를 적용하지 않는다."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            e_k = torch.zeros_like(k) if e_k is None else e_k.to(dtype)
            e_v = torch.zeros_like(v) if e_v is None else e_v.to(dtype)
            if memory is None:
                memory = v.new_zeros(v.shape[0], self.H, v.shape[-1], k.shape[-1])
            else:
                memory = memory.to(dtype)
            if fresh is not None:
                mask = fresh.view(-1, 1, 1, 1)
                memory = torch.where(mask, torch.zeros_like(memory), memory)
                e_k = torch.where(mask, torch.zeros_like(e_k), e_k)
                e_v = torch.where(mask, torch.zeros_like(e_v), e_v)
            tables = self.rope_tables(L)
            q_read, k_write, past_k_write = self.memory_qk_views(q,k,e_k)
            qr, kr, past_k = (self.apply_rope(x, L, tables) for x in (q_read,k_write,past_k_write))
            # 두 실수 외적의 차 = Im[(e_V+iV)(RoPE(eK_write)+iRoPE(K_write))ᴴ].
            # L2는 K/eK를 각각 단위화, 기존 RMS는 공통 분모. V/eV와 기억은 원시 값이다.
            # 비슷한 활동의 차감이 BF16에서 사라지지 않도록 FP32로 계산한다.
            write = v.transpose(-1, -2) @ past_k - e_v.transpose(-1, -2) @ kr
            if self.config.kv_write_reduction == "mean":
                write = write / k.shape[-2]
            memory = self.update_memory(memory, write)
            read = qr @ memory.transpose(-1, -2)  # 같은 블록에서 갱신된 기억을 읽는다.
            lam = L.trace_decay_channels.to(dtype)[None, :, None, :]
            # 원천 활동만 미분에서 상수 취급한다. 흔적 자체/계수는 미분 가능하며,
            # 현재 블록의 signed write/read에는 원래 K/V 그래프를 사용한다.
            trace_k = k.detach() if self.config.kv_trace_activity_detach else k
            trace_v = v.detach() if self.config.kv_trace_activity_detach else v
            e_k = lam * e_k + (1 - lam) * trace_k
            e_v = lam * e_v + (1 - lam) * trace_v
        return read, memory, e_k, e_v

    def block(self, L, h, inj, memory, e_k, e_v, fresh):
        h = h + self.embed_scale * inj
        B, T, _ = h.shape
        def heads(x):
            return x.reshape(B, T, self.H, self.dh).transpose(1, 2)
        q = heads(L.q_proj(h))
        if self.config.kv_projection_fp32:
            # 외적 차감 전에 사영에서 반올림된 활동은 나중에 float()해도 복구되지 않는다.
            # K/V만 처음부터 FP32로 계산한다. FP64 검산 경로는 그대로 유지한다.
            with torch.autocast(device_type=h.device.type, enabled=False):
                projection_h = h if h.dtype == torch.float64 else h.float()
                k, v = heads(L.k_proj(projection_h)), heads(L.v_proj(projection_h))
        else:
            k, v = heads(L.k_proj(h)), heads(L.v_proj(h))
        read, memory, e_k, e_v = self.memory_step(L, q, k, v, memory, e_k, e_v, fresh)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(h.dtype)
        h = h + L.out_proj(read)
        h = self.phi(self.boundary(L, h))
        return h, memory, e_k, e_v

    def _forward(self, carry, batch):
        h, inj = carry.current_hidden, self.injection(batch)
        memory, e_k, e_v = carry.coupling, carry.key_trace, carry.value_trace
        fresh = carry.fresh
        if self.config.nograd_blocks:
            with torch.no_grad():
                for _ in range(self.config.nograd_blocks):
                    for L in self.layers:
                        h, memory, e_k, e_v = self.block(L, h, inj, memory, e_k, e_v, fresh)
                        fresh = None
        for _ in range(self.config.blocks_per_seg):
            for L in self.layers:
                if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                    h, memory, e_k, e_v = checkpoint(self.block, L, h, inj, memory, e_k, e_v, fresh,
                                                    use_reentrant=False, preserve_rng_state=False)
                else:
                    h, memory, e_k, e_v = self.block(L, h, inj, memory, e_k, e_v, fresh)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=memory.detach(), trace=None,
                       key_trace=(e_k.detach() if e_k is not None else None),
                       value_trace=(e_v.detach() if e_v is not None else None), fresh=None), self.w_cls(h)


class KVHebbianInner(KVSTDPInner):
    """현재 K/V 외적만 재귀 블록마다 누적한다. Eligibility 흔적과 차감항은 없다."""
    def memory_step(self, L, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            if memory is None:
                memory = v.new_zeros(v.shape[0], self.H, v.shape[-1], k.shape[-1])
            else:
                memory = memory.to(dtype)
            if fresh is not None:
                mask = fresh.view(-1, 1, 1, 1)
                memory = torch.where(mask, torch.zeros_like(memory), memory)
            tables = self.rope_tables(L)
            qr, kr = (self.apply_rope(x, L, tables) for x in (q, k))
            write = v.transpose(-1, -2) @ kr
            if self.config.kv_write_reduction == "mean":
                write = write / k.shape[-2]
            memory = self.update_memory(memory, write)
            read = qr @ memory.transpose(-1, -2)
        return read, memory, None, None


class LT(nn.Module):
    """URM 하네스 인터페이스. ACT 없음 (halted = steps ≥ loops). q 로짓은 상수."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
        inner_type = {"kv_stdp": KVSTDPInner, "kv_hebbian": KVHebbianInner}.get(self.config.memory_type, LT_Inner)
        self.inner = inner_type(self.config)

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


_RESUME_KEYS = tuple(f.name for f in fields(LTConfig) if f.name not in
                     ("batch_size","amp","amp_dtype","activation_checkpoint",
                      "beta_init_mean","beta_init_std","nograd_blocks")) + (
    "global_batch_size","epochs","eval_interval","num_aug","seed","grad_accum_steps",
    "lr","lr_min_ratio","lr_warmup_steps","weight_decay","qkv_no_decay","beta1","beta2","puzzle_emb_lr",
    "puzzle_emb_weight_decay","q_weight","ema","ema_rate","data_fingerprint",
    "lr_rewarm_start","lr_rewarm_steps","lr_rewarm_from_ratio","late_sup_prob","late_sup_min","late_sup_max",
    "nograd_fixed","nograd_every","nograd_start","nograd_max")
# 이 키들이 생기기 전의 체크포인트(v1.71 Kaggle 런 등)는 아래 값으로 학습된 것이다.
_LEGACY_DEFAULTS = dict(plastic_select=False,select_g_max=4.0,lr_rewarm_start=None,lr_rewarm_steps=0,
                        lr_rewarm_from_ratio=1.0,late_sup_prob=0.0,late_sup_min=16,late_sup_max=112,
                        nograd_fixed=0,nograd_every=0,nograd_start=0,nograd_max=16,
                        memory_type="address",trace_decay_init=0.9,trace_decay_mode="head",
                        rope_type="axial_2d",rope_base=10000.0,
                        kv_write_reduction="mean",
                        kv_memory_update="additive",kv_memory_rho=0.95,
                        kv_trace_activity_detach=False,
                        qkv_no_decay=False,kv_qk_rmsnorm=False,kv_qk_l2norm=False,kv_projection_fp32=False)


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
    if r["memory_type"] == "address":
        r["trace_decay_init"] = r["rope_type"] = r["rope_base"] = None
        r["trace_decay_mode"] = None
        r["kv_write_reduction"] = None
        r["kv_memory_update"] = r["kv_memory_rho"] = None
        r["kv_trace_activity_detach"] = None
        r["qkv_no_decay"] = None
        r["kv_qk_rmsnorm"] = None
        r["kv_qk_l2norm"] = None
        r["kv_projection_fp32"] = None
    else:
        if r["kv_memory_update"] == "additive":
            r["kv_memory_rho"] = None
        if r["memory_type"] == "kv_hebbian":
            r["trace_decay_init"] = r["trace_decay_mode"] = r["kv_trace_activity_detach"] = None
        if r["rope_type"] == "learned_2d":
            r["rope_base"] = None
        # KV-STDP에서는 기존 주소 커널·읽기 보간·가소성 계수를 사용하지 않는다.
        if not (r["kv_qk_rmsnorm"] or r["kv_qk_l2norm"]):
            r["eps"] = None
        for key in ("alpha_init", "dist_decay", "inj_gate_init", "gamma_init",
                    "address_projection", "psi_zero", "stdp_eta_init", "stdp_gain_init",
                    "stdp_lam_init", "stdp_gain_fixed", "stdp_lam_fixed", "trace_rho_init"):
            r[key] = None
    return r


def load_training_checkpoint(path,base,optimizers,ema,cfg,rank,ws,device):
    # Only load checkpoints you trust: weights_only=False is needed for RNG/optimizer objects.
    ck = torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("model_id") != model_id_of(cfg):
        raise ValueError(f"Checkpoint model_id={ck.get('model_id')!r} but this run is {model_id_of(cfg)!r}. "
                         "Resume needs the same architecture; use init_from to start v1.8 from v1.71 weights.")
    old = ck["cfg"]
    was,now = _effective_recipe(old),_effective_recipe(cfg)
    changed = {k:(was[k],now[k]) for k in _RESUME_KEYS if was[k]!=now[k]}
    trace_activity_fork = (cfg.get("allow_trace_activity_fork",False)
                           and changed.get("kv_trace_activity_detach") == (False,True))
    if trace_activity_fork:
        # Explicit controlled branch: restore all training state and change this
        # backward policy only. Every other recipe difference still fails below.
        changed.pop("kv_trace_activity_detach")
    if changed:
        raise ValueError(f"Resume config/data mismatch (start a new out_dir for a new experiment): {changed}")
    # Initialization is not reapplied on resume. Preserve the actual experiment's
    # initializer in subsequent config/checkpoints, including pre-option runs.
    for key,default in (("beta_init_mean",0.0),("beta_init_std",0.5)):
        cfg[key] = old.get(key,default)
        setattr(base.model.config,key,cfg[key])
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
    if trace_activity_fork and rank==0:
        print(f"[LT] TRACE_ACTIVITY_GRADIENT_FORK step={ts.step}: False -> True; "
              "raw weights/optimizer/EMA/carry/RNG/data cursor restored; "
              "only K/V activity sources in eligibility EMA are detached",flush=True)
    return ts


# 6b. Warm start: weights (raw + EMA) from another run, fresh optimizer ----------
# 같은 데이터 흐름을 이어가려면 구조·데이터 규약이 같아야 한다. 가소성 파라미터화만 달라도 된다.
_INIT_KEYS = ("vocab_size","puzzle_emb_ndim","hidden_size","num_heads","loops","grid","blocks_per_seg",
              "num_layers","mlp_expansion","legacy_gauge","dist_decay","eps","forward_dtype","address_projection",
              "psi_zero","stdp","stdp_gain_fixed","stdp_lam_fixed","block_order","use_trace",
              "global_batch_size","epochs","eval_interval","num_aug","seed","data_fingerprint")


def convert_plastic_state(sd,g_max):
    """v1.71 의 헤드별 η·g·λ (eta_raw, gain_raw, lam_raw) → v1.8 게이트. sel_w=0 이므로 함수가 같다:
    A = exp(−softplus(dt_bias)) = 1−η,  g_max·σ(gsel_bias) = g,  σ(lsel_bias) = λ.  나머지 텐서(β 포함)는 그대로."""
    out = {}
    for name,value in sd.items():
        leaf = name.rsplit(".",1)[-1]
        if leaf in ("gain_raw","lam_raw"):
            continue
        if leaf!="eta_raw":
            out[name] = value
            continue
        pre = name[:-len("eta_raw")]
        eta = torch.sigmoid(value.double()).flatten()
        gain = F.softplus(sd[pre+"gain_raw"].double()).flatten()
        if not bool((gain<g_max).all()):
            raise ValueError(f"{pre}gain {gain.max().item():.4f} >= select_g_max {g_max}; raise select_g_max.")
        dt = -torch.log1p(-eta)
        out[pre+"dt_bias"] = (dt+torch.log(-torch.expm1(-dt))).to(value.dtype)
        out[pre+"gsel_bias"] = torch.logit(gain/g_max).to(value.dtype)
        out[pre+"lsel_bias"] = sd[pre+"lam_raw"].flatten().clone()
        out[pre+"sel_w"] = torch.zeros(3,eta.numel(),sd[pre+"w_sh"].shape[-1],dtype=value.dtype)
    return out


def init_from_checkpoint(path,base,ema,cfg,device):
    """raw·EMA 가중치만 가져오고 optimizer 는 새로 만든다. step 과 데이터 커서는 원본을 이어받는다
    (원본 런의 같은 구간과 같은 배치를 본다). carry 는 새로 시작한다."""
    ck = torch.load(path,map_location="cpu",weights_only=False)
    src = ck.get("model_id")
    if cfg.get("memory_type", "address") in ("kv_stdp", "kv_hebbian") or src in (MODEL_ID_KV_STDP, MODEL_ID_KV_HEBBIAN):
        raise ValueError("KV-STDP starts fresh or resumes its own checkpoint; init_from conversion is not defined.")
    if src not in (MODEL_ID,MODEL_ID_V18):
        raise ValueError(f"init_from expects a v1.71 or v1.8 checkpoint, got model_id={src!r}")
    if src==MODEL_ID_V18 and not cfg.get("plastic_select"):
        raise ValueError("v1.8 -> v1.71 conversion is not defined.")
    old = ck["cfg"]
    changed = {k:(old.get(k),cfg.get(k)) for k in _INIT_KEYS if old.get(k)!=cfg.get(k)}
    if changed:
        raise ValueError(f"init_from architecture/data protocol mismatch: {changed}")
    convert = ((lambda sd: convert_plastic_state(sd,cfg["select_g_max"]))
               if cfg.get("plastic_select") and src==MODEL_ID else (lambda sd: sd))
    base.load_state_dict(convert(ck["raw_model_state_dict"]),strict=True,assign=False)
    if ema is not None:
        shadow = ck.get("ema_shadow")
        if shadow is None:
            raise ValueError("init_from checkpoint has no EMA shadow; set ema=False or use another checkpoint.")
        shadow = convert(shadow)
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
        if cfg.get("memory_type", "address") == "address" and cfg.get("research_arch") != "urm_full_bptt":
            print(f"[LT] beta initialization: mean={cfg['beta_init_mean']:.8f} rad, "
                  f"std={cfg['beta_init_std']:.8f}; "
                  + ("resumed learned beta from checkpoint" if path else
                     "learned beta from init_from" if cfg.get("init_from") else "fresh, learnable beta"),flush=True)
        print(f"[LT] data={cfg['data_npz']} train={len(tr_x)} test={len(te_x)}",flush=True)
        print(f"[LT] planned steps={planned_steps}; actual steps={actual_steps}; "
              f"1 step={cfg['blocks_per_seg']} blocks; loops={cfg['loops']} segments",flush=True)
        if cfg.get("memory_type", "address") == "kv_stdp":
            rope_description = ("learned real joint 2D RoPE theta=[head,pair,row/col] "
                                "init=U[-pi/2,pi/2]" if cfg["rope_type"] == "learned_2d" else
                                f"fixed real axial 2D RoPE base={cfg['rope_base']}")
            print(f"[LT] KV-STDP: heads={cfg['num_heads']} head_dim={cfg['hidden_size']//cfg['num_heads']} "
                  f"trace_lambda_init={cfg['trace_decay_init']} "
                  f"(learned {cfg.get('trace_decay_mode','head')}, shared K/V; "
                  f"lambda_shape={tuple(base.model.inner.layers[0].trace_lam_raw.shape)}); "
                  f"memory_update={base.model.config.kv_memory_update} "
                  f"memory_decay={base.model.inner.memory_decay:g} write_scale={base.model.inner.write_scale:g} "
                  f"token_reduction={cfg.get('kv_write_reduction','mean')}; {rope_description}; "
                  "inject -> memory -> bilinear FFN -> fixed Phi; FP32 memory/traces",flush=True)
            kv_precision = "FP32 (autocast disabled)" if cfg.get("kv_projection_fp32",False) else "AMP"
            print(f"[LT] KV-STDP projection precision: K/V={kv_precision}; "
                  "Q/out_proj/FFN follow configured AMP",flush=True)
            print(f"[LT] KV-STDP trace activity detach={cfg.get('kv_trace_activity_detach',False)}; "
                  "eligibility values unchanged; lambda explicit gradient retained; current signed write/read unchanged",flush=True)
            qkv_decay = 0.0 if cfg.get("qkv_no_decay",False) else cfg["weight_decay"]
            print(f"[LT] KV-STDP optimizer: Q/K/V weight_decay={qkv_decay:g}; "
                  f"out_proj/FFN weight_decay={cfg['weight_decay']:g}; lambda/theta weight_decay=0",flush=True)
            if cfg.get("kv_qk_l2norm",False):
                print(f"[LT] KV-STDP L2: Q/K/eK normalized independently per token/head; "
                      f"denominator=norm(x)+eps, eps={cfg['eps']:g}; no learned gain/bias; "
                      "raw K/V eligibility traces; raw V/eV and memory",flush=True)
            elif cfg.get("kv_qk_rmsnorm",False):
                print(f"[LT] KV-STDP RMS: Q read normalized per token/head; "
                      f"past/current K write share sqrt(mean([eK,K]^2)+eps), eps={cfg['eps']:g}; "
                      "no learned gain/bias; raw K/V traces; raw V and memory",flush=True)
        elif cfg.get("memory_type") == "kv_hebbian":
            rope_description = ("learned real joint 2D RoPE theta=[head,pair,row/col] "
                                "init=U[-pi/2,pi/2]" if cfg["rope_type"] == "learned_2d" else
                                f"fixed real axial 2D RoPE base={cfg['rope_base']}")
            print(f"[LT] KV-HEBBIAN: heads={cfg['num_heads']} head_dim={cfg['hidden_size']//cfg['num_heads']} "
                  f"memory_update={base.model.config.kv_memory_update} "
                  f"memory_decay={base.model.inner.memory_decay:g} write_scale={base.model.inner.write_scale:g} "
                  f"token_reduction={cfg.get('kv_write_reduction','mean')}; "
                  "write=V.T @ RoPE(K); read=RoPE(Q) @ M.T; no K/V traces, lambda or subtraction; "
                  f"{rope_description}; inject -> memory -> bilinear FFN -> fixed Phi; FP32 memory",flush=True)
            kv_precision = "FP32 (autocast disabled)" if cfg.get("kv_projection_fp32",False) else "AMP"
            qkv_decay = 0.0 if cfg.get("qkv_no_decay",False) else cfg["weight_decay"]
            print(f"[LT] KV-HEBBIAN: no Q/K normalization; K/V={kv_precision}; "
                  "Q/out_proj/FFN follow configured AMP; autocast cache disabled",flush=True)
            print(f"[LT] KV-HEBBIAN optimizer: Q/K/V weight_decay={qkv_decay:g}; "
                  f"out_proj/FFN weight_decay={cfg['weight_decay']:g}; theta weight_decay=0",flush=True)
        elif cfg.get("research_arch") == "urm_full_bptt":
            print(f"[URM] original softmax attention + {cfg.get('urm_ffn','convswiglu')} + 1D RoPE; "
                  "two post-residual RMSNorms; full within-segment gradients; "
                  "segment-boundary detach; ACT disabled",flush=True)
        else:
            print(f"[LT] {'v1.8' if cfg.get('plastic_select') else 'v1.71'}: projection={cfg['address_projection']} "
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
    """CPU-only checks of v1.71/v1.8 equations, recurrent training, and exact resume."""
    import tempfile
    from unittest import mock
    device = torch.device("cpu")
    old_threads, rng0 = torch.get_num_threads(), _rng_state(device)
    torch.set_num_threads(min(2, old_threads))
    checks = []
    try:
        torch.manual_seed(17)
        cfg = dict(DEFAULT_CFG, hidden_size=16, num_heads=2, puzzle_emb_ndim=16,
                   global_batch_size=2, batch_size=2, seq_len=81, num_puzzle_identifiers=1,
                   blocks_per_seg=3, loops=3, amp=False, amp_dtype="float32",
                   activation_checkpoint=False, compile=False, epochs=4, eval_interval=2,
                   lr_warmup_steps=0, lr=1e-3, puzzle_emb_lr=1e-3, ema_rate=0.9,
                   num_aug=10, run_selftests=False, dataloader_workers=0,
                   plastic_select=False, lr_rewarm_start=None, lr_min_ratio=1.0, late_sup_prob=0.0)
        with mock.patch.object(torch.linalg, "qr", side_effect=AssertionError("v1.71 called QR")):
            base = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
            inner, L = base.model.inner, base.model.inner.layers[0]
            assert hasattr(L, "wc") and not hasattr(L, "wc_raw")
            with torch.no_grad():
                L.b_down.weight.normal_(0, 0.03)
                torch.testing.assert_close(torch.cat(inner.W_C(L), 1), L.wc, rtol=0, atol=0)
            checks.append("direct learned address matrix; no QR")
            with torch.no_grad():
                rho, omega = torch.sigmoid(L.mu_rho_raw), L.mu_omega
                mu = torch.polar(rho, omega)
                gin = torch.sqrt(torch.clamp(1-rho.square(), min=1e-6))
                history, trace = [], None
                for k in range(5):
                    u = torch.complex(torch.randn(2, 81, 2, 4), torch.randn(2, 81, 2, 4))
                    history.append(u)
                    zx, zy, trace = inner.trace_step(L, u.real, u.imag, trace, None)
                    expected = gin*sum(mu**(k-j)*v for j, v in enumerate(history))
                    torch.testing.assert_close(torch.complex(zx, zy), expected, rtol=2e-5, atol=2e-6)
                zx, zy, _ = inner.trace_step(L, u.real, u.imag, trace, torch.ones(2, dtype=torch.bool))
                torch.testing.assert_close(torch.complex(zx, zy), gin*u, rtol=2e-5, atol=2e-6)
                checks.append("complex address trace matches finite-history sum and fresh reset")
                h, w = torch.randn(2, 81, 16), torch.randn(2, 2, 81, 81)*0.03
                AB, kc, kcb = inner.W_C(L), inner.kernel(L), inner.kernel(L, L.beta)
                for old_w, old_z, fresh in ((None, None, None), (w, trace, torch.zeros(2, dtype=torch.bool)),
                                            (w, trace, torch.ones(2, dtype=torch.bool))):
                    ux, uy = inner.addr_raw(h, AB)
                    zx, zy, _ = inner.trace_step(L, ux, uy, old_z, fresh)
                    v = torch.einsum("btd,hcd->bthc", h, L.w_sh)
                    vv = v/(v.norm(dim=-1, keepdim=True)+cfg["eps"])
                    tgt = F.softplus(L.gain_raw)*inner.attn_xy(inner._unit(zx, zy), kcb)*torch.einsum("bthc,bnhc->bhtn", vv, vv)
                    eta, lam = torch.sigmoid(L.eta_raw), torch.sigmoid(L.lam_raw)
                    expected_w = tgt if old_w is None or bool(fresh.all()) else (1-eta)*old_w+eta*tgt
                    attention = (1-lam)*inner.attn_xy(inner._unit(ux, uy), kc)+lam*expected_w
                    message = torch.einsum("bhtn,bnhc->bthc", attention, v)
                    expected_h = h+torch.einsum("bthc,hcd->btd", message, L.w_sh)
                    hn, wn, _ = inner.step(L, h, AB, kc, w=old_w, fresh=fresh, kcb=kcb, ztr=old_z, apply_phi=False)
                    torch.testing.assert_close(wn, expected_w, rtol=2e-5, atol=3e-6)
                    torch.testing.assert_close(hn, expected_h, rtol=2e-5, atol=3e-6)
                checks.append("fresh W=tgt; recurrent W update; same-block read uses updated W")
            batch = dict(inputs=torch.randint(1, 11, (2, 81)), labels=torch.randint(2, 11, (2, 81)),
                         puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
            alt = {k: v.clone() for k, v in batch.items()}
            alt["inputs"], alt["labels"] = torch.randint(1, 11, (2, 81)), torch.randint(2, 11, (2, 81))
            with torch.no_grad():
                c1, _ = base.model(base.initial_carry(batch), batch)
                c2, _ = base.model(c1, alt)
                assert torch.equal(c2.current_data["inputs"], batch["inputs"])
                cm, _ = base.model(replace(c1, halted=torch.tensor([True, False])), alt)
                ca, _ = base.model(base.initial_carry(alt), alt)
                for name in ("current_hidden", "coupling", "trace"):
                    torch.testing.assert_close(getattr(cm, name)[0], getattr(ca, name)[0])
                    torch.testing.assert_close(getattr(cm, name)[1], getattr(c2, name)[1])
                    assert not getattr(c2, name).requires_grad
                assert cm.steps.tolist() == [1, 2]
                ca, oa = base.model(c1, batch)
                inner.config.blocks_per_seg = 6
                cb, ob = base.model(base.initial_carry(batch), batch)
                inner.config.blocks_per_seg = 3
                for name in ("current_hidden", "coupling", "trace"):
                    torch.testing.assert_close(getattr(ca, name), getattr(cb, name), rtol=2e-5, atol=3e-6)
                torch.testing.assert_close(oa["logits"], ob["logits"], rtol=2e-5, atol=3e-6)
                checks.append("per-lane reset/data retention; two segments match longer unroll; boundary detach")
            other = ACTLossHead(LT(dict(cfg, activation_checkpoint=True)), q_weight=cfg["q_weight"])
            other.load_state_dict(base.state_dict())
            for model in (base, other):
                model.zero_grad(set_to_none=True)
                _, loss, _, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=set())
                loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), name
                for suffix in ("wc", "beta", "mu_rho_raw", "mu_omega"):
                    p = next(p for n, p in model.named_parameters() if n.endswith("."+suffix))
                    assert p.grad is not None and p.grad.norm()>0, suffix
            for (name, p), (other_name, q) in zip(base.named_parameters(), other.named_parameters()):
                assert name == other_name and (p.grad is None) == (q.grad is None)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=3e-6)
            torch.testing.assert_close(base.model.puzzle_emb.local_weights.grad,
                                       other.model.puzzle_emb.local_weights.grad, rtol=2e-5, atol=3e-6)
            checks.append("nonzero finite wc/beta/mu gradients; checkpoint on/off gradient equality")
        sol = np.array([[(3*(r%3)+r//3+c)%9+1 for c in range(9)] for r in range(9)], dtype=np.uint8)
        inp = sol.copy(); inp[::2, ::2] = 0
        for seed in range(20):
            prm = _draw_aug_params(np.random.default_rng(seed))
            xx, yy = _apply_aug(inp, *prm), _apply_aug(sol, *prm)
            _check_boards(xx[None], yy[None], "selftest_aug")
            dm, tr, rp, cp = prm
            assert np.array_equal(xx, dm[(inp.T if tr else inp)[np.ix_(rp, cp)]])
        args = dict(seed=0, num_aug=10, global_batch_size=2, rank=0, world_size=1,
                    epochs_per_iter=2, start_iter=0, total_iters=2)
        ds = SudokuTrainDataset(np.repeat(inp[None], 8, 0), np.repeat(sol[None], 8, 0), **args)
        full = list(ds); skipped = list(SudokuTrainDataset(ds.inputs, ds.labels, **args, skip_batches=3))
        assert len(skipped) == len(full)-3 and np.array_equal(ds._augmented(0, 7)[0], ds._augmented(0, 7)[0])
        for expected, actual in zip(full[3:], skipped):
            assert expected[0] == actual[0] and all(torch.equal(expected[1][k], actual[1][k]) for k in expected[1])
        checks.append("valid augmentation, fixed pool, and resumed data cursor")
        torch.manual_seed(91)
        model = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
        opts, lrs = create_optimizers(model, cfg, 1)
        ema = EMAHelper(cfg["ema_rate"]); ema.register(model)
        ts = TrainState()
        def update(net, state, optimizers, rates, shadow, data):
            train_batch(net, net, state, data, cfg, optimizers, rates, 16, 0, 1, device)
            shadow.update(net); state.batch_in_iter += 1
        for j in range(2): update(model, ts, opts, lrs, ema, full[j][1])
        with tempfile.TemporaryDirectory() as directory:
            path = save_training_checkpoint(directory, ts, model, opts, ema, cfg, 0, 1, device)
            expected_rng = (torch.rand(4), np.random.random(4), random.random())
            update(model, ts, opts, lrs, ema, full[2][1])
            resumed = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
            ro, rl = create_optimizers(resumed, cfg, 1)
            rema = EMAHelper(cfg["ema_rate"]); rema.register(resumed)
            rs = load_training_checkpoint(path, resumed, ro, rema, cfg, 0, 1, device)
            assert rs.step == 2 and rs.batch_in_iter == 2
            assert torch.equal(expected_rng[0], torch.rand(4)) and np.array_equal(expected_rng[1], np.random.random(4)) and expected_rng[2] == random.random()
            update(resumed, rs, ro, rl, rema, full[2][1])
            for name, value in model.state_dict().items(): assert torch.equal(value, resumed.state_dict()[name]), name
            for name, value in ema.shadow.items(): assert torch.equal(value, rema.shadow[name]), name
            for name in ("current_hidden", "coupling", "trace", "steps", "halted"):
                assert torch.equal(getattr(ts.carry, name), getattr(rs.carry, name)), name
            for name, value in ts.carry.current_data.items():
                assert torch.equal(value, rs.carry.current_data[name]), name
        checks.append("resume reproduces next weights/EMA/h/W/Z and Torch/NumPy/Python RNG")
        _selftest_v18(cfg, full, checks, device)
        _selftest_nograd(cfg, full, checks)
        _selftest_kv_stdp(cfg, full, checks, device)
        for item in checks: print("[selftest] PASS "+item, flush=True)
        print(f"[selftest] {len(checks)}/{len(checks)} groups passed (CPU, synthetic inputs).", flush=True)
        return checks
    finally:
        _restore_rng(rng0, device)
        torch.set_num_threads(old_threads)



def _selftest_kv_stdp(cfg, full, checks, device):
    """단일 셀에서도 새 기억의 복소 항등식·상태 이월·실제 학습 재개를 검산한다."""
    import tempfile
    cfg = dict(cfg, **PRESETS["kv_stdp"])
    torch.manual_seed(117)
    base = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
    inner, L = base.model.inner, base.model.inner.layers[0]
    shape = (2, cfg["num_heads"], cfg["seq_len"], cfg["hidden_size"] // cfg["num_heads"])
    q, k, v, e_k, e_v = (torch.randn(shape) for _ in range(5))
    memory = torch.randn(2, cfg["num_heads"], shape[-1], shape[-1])
    with torch.no_grad():
        q_read, k_write, past_k_write = q, k, e_k
        if cfg.get("kv_qk_l2norm",False):
            q_read, k_write, past_k_write = (
                x / (x.norm(dim=-1,keepdim=True)+cfg["eps"]) for x in (q,k,e_k))
        elif cfg.get("kv_qk_rmsnorm",False):
            q_read = q / torch.sqrt(q.square().mean(-1,keepdim=True)+cfg["eps"])
            key_scale = torch.sqrt(torch.cat((e_k,k),dim=-1).square().mean(-1,keepdim=True)+cfg["eps"])
            k_write, past_k_write = k/key_scale, e_k/key_scale
        ck = torch.complex(inner.apply_rope(past_k_write), inner.apply_rope(k_write))
        cv = torch.complex(e_v, v)
        expected_write = (cv.transpose(-1, -2) @ ck.conj()).imag
        if cfg.get("kv_write_reduction","mean") == "mean":
            expected_write = expected_write / shape[-2]
        expected = memory + expected_write
        read, actual, kn, vn = inner.memory_step(L, q, k, v, memory, e_k, e_v)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(read, inner.apply_rope(q_read) @ expected.transpose(-1, -2), rtol=2e-5, atol=2e-6)
        lam = (L.trace_decay.repeat_interleave(2,dim=-1)[None,:,None,:]
               if cfg.get("trace_decay_mode","head")=="pair" else L.trace_decay.view(1,-1,1,1))
        torch.testing.assert_close(kn, lam * e_k + (1 - lam) * k)
        torch.testing.assert_close(vn, lam * e_v + (1 - lam) * v)
    checks.append("KV-STDP: real outer-product difference matches complex imaginary write; past-only traces")
    batch = full[0][1]
    _, loss, _, _, _ = base(carry=base.initial_carry(batch), batch=batch, return_keys=set())
    loss.backward()
    assert L.trace_lam_raw.grad is not None and L.trace_lam_raw.grad.norm() > 0
    assert L.theta.grad is not None and L.theta.grad.norm() > 0
    assert _is_no_decay("inner.layers.0.theta", L.theta)
    for name, param in base.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
    opts, lrs = create_optimizers(base, cfg, 1)
    ema = EMAHelper(cfg["ema_rate"]); ema.register(base)
    ts = TrainState()
    def update(model, state, optimizers, rates, shadow, data):
        train_batch(model, model, state, data, cfg, optimizers, rates, 16, 0, 1, device)
        shadow.update(model); state.batch_in_iter += 1
    for j in range(2):
        update(base, ts, opts, lrs, ema, full[j][1])
    for name in ("current_hidden", "coupling", "key_trace", "value_trace"):
        assert not getattr(ts.carry, name).requires_grad
        assert torch.isfinite(getattr(ts.carry, name)).all()
    assert ts.carry.trace is None
    assert ts.carry.current_hidden.norm(dim=-1).max() <= math.sqrt(cfg["hidden_size"]) + 1e-5
    with tempfile.TemporaryDirectory() as directory:
        path = save_training_checkpoint(directory, ts, base, opts, ema, cfg, 0, 1, device)
        update(base, ts, opts, lrs, ema, full[2][1])
        resumed = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
        ro, rl = create_optimizers(resumed, cfg, 1)
        rema = EMAHelper(cfg["ema_rate"]); rema.register(resumed)
        rs = load_training_checkpoint(path, resumed, ro, rema, cfg, 0, 1, device)
        update(resumed, rs, ro, rl, rema, full[2][1])
        for name, value in base.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[name]), name
        for name, value in ema.shadow.items():
            assert torch.equal(value, rema.shadow[name]), name
        for name in ("current_hidden", "coupling", "key_trace", "value_trace", "steps", "halted"):
            assert torch.equal(getattr(ts.carry, name), getattr(rs.carry, name)), name
    checks.append("KV-STDP: finite lambda/theta gradients; h/M/K/V boundary detach; exact optimizer/EMA/carry resume")


def _selftest_v18(cfg, full, checks, device):
    """v1.8: 0 게이트 = v1.71, 쓰기 게이트 대칭, 후반 감독·가소성 lr 그룹·감쇠 스케줄의 정확한 재개."""
    import tempfile
    torch.manual_seed(5)
    old = ACTLossHead(LT(cfg), q_weight=cfg["q_weight"])
    with torch.no_grad():
        for n, p in old.named_parameters():
            if n.endswith(("eta_raw", "lam_raw", "gain_raw")):
                p.add_(torch.randn_like(p) * 0.5)
            if n.endswith("b_down.weight"):
                p.normal_(0, 0.03)
    c18 = dict(cfg, plastic_select=True)
    new = ACTLossHead(LT(c18), q_weight=cfg["q_weight"])
    new.load_state_dict(convert_plastic_state(old.state_dict(), c18["select_g_max"]), strict=True)
    L = new.model.inner.layers[0]
    assert hasattr(L, "beta") and hasattr(L, "sel_w") and not hasattr(L, "eta_raw")
    torch.testing.assert_close(L.beta, old.model.inner.layers[0].beta, rtol=0, atol=0)
    batch = full[0][1]
    with torch.no_grad():
        ca, cb = old.model.initial_carry(batch), new.model.initial_carry(batch)
        for _ in range(2):
            ca, oa = old.model(ca, batch); cb, ob = new.model(cb, batch)
            for name in ("current_hidden", "coupling", "trace"):
                torch.testing.assert_close(getattr(cb, name), getattr(ca, name), rtol=1e-4, atol=2e-5)
            torch.testing.assert_close(ob["logits"], oa["logits"], rtol=1e-4, atol=2e-5)
    checks.append("v1.8 with zero gate weights reproduces converted v1.71 (beta kept)")
    with torch.no_grad():
        L.sel_w.normal_(0, 0.3); L.beta.zero_()
        inner = new.model.inner
        keep, write, gain, lam = inner.select_gates(L, torch.randn(2, 81, cfg["hidden_size"]))
        assert bool(((keep > 0) & (keep < 1)).all()) and torch.allclose(keep + write, torch.ones_like(keep))
        assert bool((gain < c18["select_g_max"]).all()) and lam.shape[-1] == 1 and lam.std() > 0
        for x in (keep, gain):
            torch.testing.assert_close(x, x.transpose(-1, -2), rtol=0, atol=0)
        c, _ = new.model(new.model.initial_carry(batch), batch)
        w = c.coupling
        torch.testing.assert_close(w, w.transpose(-1, -2), rtol=1e-5, atol=1e-6)
    new.zero_grad(set_to_none=True)
    _, loss, _, _, _ = new(carry=new.initial_carry(batch), batch=batch, return_keys=set())
    loss.backward()
    for suffix in ("sel_w", "dt_bias", "gsel_bias", "lsel_bias", "beta"):
        g = getattr(L, suffix).grad
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, suffix
    checks.append("v1.8 write gates symmetric (beta=0 keeps W symmetric); read gate per cell; gate gradients")
    rc = dict(c18, loops=3, late_sup_prob=1.0, late_sup_min=2, late_sup_max=2,
              lr_rewarm_start=1, lr_rewarm_steps=4, lr_rewarm_from_ratio=0.1)
    torch.manual_seed(92)
    model = ACTLossHead(LT(rc), q_weight=rc["q_weight"])
    opts, lrs = create_optimizers(model, rc, 1)
    assert lr_at(0, 1.0, rc, 99) == 1.0 and abs(lr_at(1, 1.0, rc, 99) - 0.1) < 1e-12
    assert abs(lr_at(3, 1.0, rc, 99) - 0.55) < 1e-12 and lr_at(5, 1.0, rc, 99) == lr_at(50, 1.0, rc, 99) == 1.0
    ema = EMAHelper(rc["ema_rate"]); ema.register(model)
    ts = TrainState()
    def update(net, state, optimizers, rates, shadow, data):
        train_batch(net, net, state, data, rc, optimizers, rates, 16, 0, 1, device)
        shadow.update(net); state.batch_in_iter += 1
    for j in range(3): update(model, ts, opts, lrs, ema, full[j][1])
    assert ts.carry.steps.tolist() == [5, 5] and not bool(ts.carry.halted.any())      # 3 supervised + 2 extra
    with tempfile.TemporaryDirectory() as directory:
        path = save_training_checkpoint(directory, ts, model, opts, ema, rc, 0, 1, device)
        update(model, ts, opts, lrs, ema, full[3][1])
        assert ts.carry.steps.tolist() == [6, 6] and bool(ts.carry.halted.all())     # supervised seg 6
        resumed = ACTLossHead(LT(rc), q_weight=rc["q_weight"])
        ro, rl = create_optimizers(resumed, rc, 1)
        rema = EMAHelper(rc["ema_rate"]); rema.register(resumed)
        rs = load_training_checkpoint(path, resumed, ro, rema, rc, 0, 1, device)
        update(resumed, rs, ro, rl, rema, full[3][1])
        for name, value in model.state_dict().items(): assert torch.equal(value, resumed.state_dict()[name]), name
        for name in ("current_hidden", "coupling", "trace", "steps", "halted"):
            assert torch.equal(getattr(ts.carry, name), getattr(rs.carry, name)), name
    checks.append("v1.8 late supervision and lr rewarm schedule; exact resume")


def _selftest_nograd(cfg, full, checks):
    """no-grad 선행 블록: 값은 더 긴 unroll 과 같고, gradient 는 선행 상태를 detach 한 뒤 역전파한 것과 같다."""
    torch.manual_seed(7)
    c = dict(cfg, plastic_select=True, blocks_per_seg=2)
    m = ACTLossHead(LT(c), q_weight=c["q_weight"])
    inner, lt = m.model.inner, m.model
    with torch.no_grad():
        inner.layers[0].sel_w.normal_(0, 0.1)
        inner.layers[0].b_down.weight.normal_(0, 0.03)
    batch = full[0][1]
    def run(nograd, blocks, carry=None, grad=True):
        inner.config.nograd_blocks, inner.config.blocks_per_seg = nograd, blocks
        with torch.set_grad_enabled(grad):
            return lt(lt.initial_carry(batch) if carry is None else carry, batch)
    with torch.no_grad():
        ca, oa = run(3, 2); cb, ob = run(0, 5)
    for name in ("current_hidden", "coupling", "trace"):
        torch.testing.assert_close(getattr(ca, name), getattr(cb, name), rtol=0, atol=0)
    torch.testing.assert_close(oa["logits"], ob["logits"], rtol=0, atol=0)
    probe = torch.randn_like(oa["logits"])
    def grads(fn):
        m.zero_grad(set_to_none=True); lt.puzzle_emb.local_weights.grad = None
        (fn()["logits"] * probe).sum().backward()
        out = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        out["puzzle_emb"] = lt.puzzle_emb.local_weights.grad.clone()
        return out
    g1 = grads(lambda: run(3, 2)[1])
    g2 = grads(lambda: run(0, 2, carry=run(0, 3, grad=False)[0])[1])
    assert set(g1) == set(g2)
    for name in g1:
        torch.testing.assert_close(g1[name], g2[name], rtol=2e-5, atol=3e-6, msg=name)
    inner.config.nograd_blocks, inner.config.blocks_per_seg = 0, cfg["blocks_per_seg"]
    checks.append("no-grad pre-blocks: same values as a longer unroll; gradients equal a detached prefix")


# Compatibility APIs for existing local weight-conversion/probe utilities.
def strip_prefix(sd: dict) -> dict:
    """저장소 체크포인트(`_orig_mod.model.inner...`)도 읽을 수 있게 접두를 정리한다."""
    out = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", ):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out

def save_checkpoint(out_dir: str, step: int, base: nn.Module, optimizers, ema: Optional[EMAHelper],
                    iter_id: int, batch_in_iter: int, cfg: dict, keep_last: int):
    os.makedirs(out_dir, exist_ok=True)
    raw = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    if ema is not None:
        with _EMASwap(base, ema):
            ema_sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    else:
        ema_sd = raw
    state = {
        "step": step,
        "iter_id": iter_id,
        "batch_in_iter": batch_in_iter,
        "model_state_dict": ema_sd,                 # EMA (저장소 규약)
        "raw_model_state_dict": raw,                # 재개용 원시
        "ema_shadow": ({k: v.detach().cpu().clone() for k, v in ema.state_dict().items()} if ema is not None else None),
        "optimizer_states": [o.state_dict() for o in optimizers],
        "rng_state": torch.random.get_rng_state(),
        "cfg": cfg,
    }
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

    tmp = os.path.join(out_dir, f".step_{step}.pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(out_dir, f"step_{step}.pt"))     # 중간에 끊겨도 반쪽 파일이 안 남는다

    # 용량 관리 — 최신 keep_last 개만 남긴다 (/kaggle/working 는 ~20GB)
    # [2026-09-04] `step_*.pt` 글롭은 `step_final.pt` 같은 이름도 잡는데 정규식은 숫자만 받는다.
    # 무방비로 .group(1) 을 부르면 AttributeError 로 **학습이 통째로 죽는다** (재현 확인).
    # find_latest_checkpoint 는 `if m:` 로 막고 있는데 여기만 빠져 있었다.
    files = []
    for p in glob.glob(os.path.join(out_dir, "step_*.pt")):
        m = _CKPT_RE.search(os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    for _, p in files[:-keep_last] if keep_last > 0 else []:
        try:
            os.remove(p)
        except OSError:
            pass
    return os.path.join(out_dir, f"step_{step}.pt")

def load_checkpoint(path: str, base: nn.Module, optimizers, device, load_optimizer: bool = True):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("raw_model_state_dict") or ck["model_state_dict"]
    sd = strip_prefix(sd)
    saved_memory = (ck.get("cfg") or {}).get("memory_type", "address")
    model_memory = getattr(base, "model", base).config.memory_type
    if saved_memory != model_memory:
        raise ValueError(f"Checkpoint memory_type={saved_memory!r} does not match model memory_type={model_memory!r}.")
    if model_memory in ("kv_stdp", "kv_hebbian"):
        saved_update = (ck.get("cfg") or {}).get("kv_memory_update","additive")
        model_cfg = getattr(base,"model",base).config
        if saved_update != model_cfg.kv_memory_update:
            raise ValueError(f"Checkpoint kv_memory_update={saved_update!r} does not match model kv_memory_update={model_cfg.kv_memory_update!r}.")
        if saved_update in ("ema", "leaky"):
            saved_rho = (ck.get("cfg") or {}).get("kv_memory_rho",0.95)
            if saved_rho != model_cfg.kv_memory_rho:
                raise ValueError(f"Checkpoint kv_memory_rho={saved_rho!r} does not match model kv_memory_rho={model_cfg.kv_memory_rho!r}.")
        if model_memory == "kv_stdp":
            saved_activity_detach = (ck.get("cfg") or {}).get("kv_trace_activity_detach",False)
            model_activity_detach = getattr(base,"model",base).config.kv_trace_activity_detach
            if saved_activity_detach != model_activity_detach:
                raise ValueError(f"Checkpoint kv_trace_activity_detach={saved_activity_detach!r} does not match model kv_trace_activity_detach={model_activity_detach!r}.")
        saved_reduction = (ck.get("cfg") or {}).get("kv_write_reduction","mean")
        model_reduction = getattr(base,"model",base).config.kv_write_reduction
        if saved_reduction != model_reduction:
            raise ValueError(f"Checkpoint kv_write_reduction={saved_reduction!r} does not match model kv_write_reduction={model_reduction!r}.")
        if model_memory == "kv_stdp":
            saved_mode = (ck.get("cfg") or {}).get("trace_decay_mode","head")
            model_mode = getattr(base,"model",base).config.trace_decay_mode
            if saved_mode != model_mode:
                raise ValueError(f"Checkpoint trace_decay_mode={saved_mode!r} does not match model trace_decay_mode={model_mode!r}.")
        saved_norm = (ck.get("cfg") or {}).get("kv_qk_rmsnorm",False)
        model_norm = getattr(base,"model",base).config.kv_qk_rmsnorm
        if saved_norm != model_norm:
            raise ValueError(f"Checkpoint kv_qk_rmsnorm={saved_norm!r} does not match model kv_qk_rmsnorm={model_norm!r}.")
        saved_l2 = (ck.get("cfg") or {}).get("kv_qk_l2norm",False)
        model_l2 = getattr(base,"model",base).config.kv_qk_l2norm
        if saved_l2 != model_l2:
            raise ValueError(f"Checkpoint kv_qk_l2norm={saved_l2!r} does not match model kv_qk_l2norm={model_l2!r}.")
        saved_fp32 = (ck.get("cfg") or {}).get("kv_projection_fp32",False)
        model_fp32 = getattr(base,"model",base).config.kv_projection_fp32
        if saved_fp32 != model_fp32:
            raise ValueError(f"Checkpoint kv_projection_fp32={saved_fp32!r} does not match model kv_projection_fp32={model_fp32!r}.")
    saved_projection = (ck.get("cfg") or {}).get("address_projection", "qr")
    model_projection = getattr(base, "model", base).config.address_projection
    conversion_hint = (
        "Use an explicitly converted checkpoint when changing the address projection. "
        "For qr -> linear, copy each effective QR(wc_raw.T).Q.T into wc; "
        "renaming wc_raw alone is not a conversion."
    )
    if saved_projection != model_projection:
        raise ValueError(
            f"Checkpoint address_projection={saved_projection!r} does not match "
            f"model address_projection={model_projection!r}. {conversion_hint}"
        )
    # config 오기재도 조용히 통과시키지 않는다. strict=True 는 빠진/추가된 다른 키도 검사한다.
    incompatible_suffix = ".wc_raw" if model_projection == "linear" else ".wc"
    incompatible_keys = [name for name in sd if name.endswith(incompatible_suffix)]
    if incompatible_keys:
        raise ValueError(
            f"Checkpoint keys {incompatible_keys} conflict with "
            f"address_projection={model_projection!r}. {conversion_hint}"
        )
    base.load_state_dict(sd, strict=True, assign=False)   # assign=False 필수 (위 주석)
    if load_optimizer and ck.get("optimizer_states") is not None:
        if len(ck["optimizer_states"]) == len(optimizers):
            for o, s in zip(optimizers, ck["optimizer_states"]):
                o.load_state_dict(s)
        else:
            print("[LT] 옵티마이저 개수 불일치 — 상태 로드 생략", flush=True)
    if ck.get("rng_state") is not None:
        torch.random.set_rng_state(torch.as_tensor(ck["rng_state"], device="cpu").to(torch.uint8))
    return ck

def _cli():
    ap = argparse.ArgumentParser(description="Self-contained LT KV-STDP / v1.8 / v1.71 trainer")
    ap.add_argument("--config",help="JSON overrides for DEFAULT_CFG")
    ap.add_argument("--preset", choices=PRESETS, help="Select kv_stdp or an existing architecture before JSON overrides")
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
    if a.preset:
        cfg.update(PRESETS[a.preset])
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
    tag = CFG["memory_type"] if CFG.get("memory_type") in ("kv_stdp", "kv_hebbian") else ("v18" if CFG.get("plastic_select") else "v171")
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
    # Explicit shell arguments also allow `python lt/train.py --selftest`.
    if "__file__" in globals() and "ipykernel" not in sys.modules and len(sys.argv) > 1:
        exec(compile(_TRAINER_SOURCE, __file__ + "::trainer", "exec"), globals())
    else:
        launch_lt_one_cell()
else:
    # Keep LT/model helpers importable for the existing research scripts.
    exec(compile(_TRAINER_SOURCE, globals().get("__file__", "train.py") + "::trainer", "exec"), globals())
