"""Sudoku adaptation of 새코드3.txt; the uploaded source remains unchanged.

Architecture: shared symbol embeddings, axial attention, MoE, and the original
windowed signed DeltaMemory. Runtime fixes cover device allocation, expert masks,
and routing statistics under activation checkpointing. No TxT-STDP added here.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Set, Tuple
from torch.optim.optimizer import Optimizer


@dataclass
class LTv5Config:
    batch_size: int
    seq_len: int
    grid: int
    vocab_size: int
    num_puzzle_identifiers: int = 0
    puzzle_emb_ndim: int = 0

    hidden_size: int = 256
    num_heads: int = 8
    head_dim: int = 32
    num_heads_t: int = 4
    head_dim_t: int = 32

    loops: int = 1
    blocks_per_seg: int = 8
    num_layers: int = 1

    num_experts: int = 64
    num_active_experts: int = 64   # 라우터가 고를 수 있는 expert 수(앞에서부터). 학습 중 스케줄로 키운다.
    top_k: int = 1
    expert_schedule: str = "routed"  # or sequential_then_routed: E1..EN, then token-wise top-1
    expert_intermediate: int = 64
    shared_intermediate: int = 512
    moe_aux_weight: float = 0.01

    delta_memory: bool = True
    delta_alpha_init: float = 0.9

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "float32"
    amp: bool = True
    ckpt_blocks: bool = False
    dropout: float = 0.0
    block_supervision: str = "last"
    # [2026-09-23 사용자] 세그먼트 안 블록별 채점: "last" = 마지막 블록 출력만(기존), "all" = 블록마다 답을 내 loss 평균

    @classmethod
    def from_dict(cls, d: dict) -> "LTv5Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __post_init__(self):
        if (self.grid, self.seq_len, self.vocab_size) != (9, 81, 11):
            raise ValueError("Sudoku requires grid=9, seq_len=81, vocab_size=11.")
        if self.batch_size < 2:
            raise ValueError("Batch size 1 is disabled for this experiment.")
        if self.puzzle_emb_ndim != 0:
            raise ValueError("Keep puzzle_emb_ndim=0 to preserve digit permutation equivariance.")
        if self.num_layers != 1:
            raise ValueError("This baseline uses one shared layer and one recurrent memory.")
        if min(self.loops, self.blocks_per_seg, self.hidden_size, self.num_heads,
               self.head_dim, self.num_heads_t, self.head_dim_t,
               self.expert_intermediate, self.shared_intermediate) < 1:
            raise ValueError("Model widths and repetition counts must be positive.")
        if self.head_dim % 4:
            raise ValueError("2D rotary head_dim must be divisible by four.")
        if not 1 <= self.top_k <= self.num_active_experts <= self.num_experts:
            raise ValueError("Require 1 <= top_k <= active experts <= all experts.")
        if self.expert_schedule not in ("routed", "sequential_then_routed"):
            raise ValueError("expert_schedule must be routed or sequential_then_routed.")
        if self.expert_schedule == "sequential_then_routed":
            if self.num_active_experts != self.num_experts or self.top_k != 1:
                raise ValueError("Sequential experts require all experts active and top_k=1.")
            if self.blocks_per_seg != self.num_experts + 1:
                raise ValueError("Sequential experts require blocks_per_seg = num_experts + 1.")
        if self.forward_dtype != "float32" or self.dropout != 0:
            raise ValueError("Keep FP32 parameters/hidden carry and dropout=0; amp selects BF16 operations.")
        if self.block_supervision not in ("last", "all"):
            raise ValueError("block_supervision must be last or all.")
        if not -1 < self.delta_alpha_init < 1 or self.rms_norm_eps <= 0:
            raise ValueError("Invalid memory initialization or normalization epsilon.")


def rms_norm(x, eps=1e-5):
    dt = x.dtype
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(dt)


def trunc_normal_init_(t, std=1.0):
    if std == 0:
        t.zero_()
        return t
    with torch.no_grad():
        t.normal_(0, std).clamp_(-2 * std, 2 * std)
    return t


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class CastedLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_f, in_f), std=1.0 / math.sqrt(in_f)))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.num_embeddings = num_embeddings
        self.weights = nn.Buffer(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std), persistent=True)
        self.local_weights = nn.Buffer(torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int64), persistent=False)

    def forward(self, inputs):
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params, world_size, lr=1e-3, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            local_weights_grad = local_ids = weights = None
            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
            assert local_ids is not None and weights is not None
            if local_weights_grad is not None:
                _sparse_emb_signsgd(local_weights_grad, local_ids, weights,
                                    lr=group["lr"], weight_decay=group["weight_decay"],
                                    world_size=group["world_size"])


def _sparse_emb_signsgd(local_grad, local_ids, weights, lr, weight_decay, world_size):
    N, D = local_grad.shape
    all_grad, all_ids = local_grad, local_ids
    if world_size > 1:
        all_grad = torch.empty(world_size * N, D, dtype=local_grad.dtype, device=local_grad.device)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)
        dist.all_gather_into_tensor(all_grad, local_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)
    grad_ids, inv = all_ids.unique(return_inverse=True)
    inv = inv.to(torch.int64)
    grad_ids = grad_ids.to(torch.int64)
    grad = torch.zeros(grad_ids.shape[0], D, dtype=all_grad.dtype, device=all_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_grad)
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p


# ─── Rotary Embedding 2D ───

class RotaryEmbedding2d(nn.Module):
    def __init__(self, dim, max_pos, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 4, dtype=torch.float32) / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        w = int(max_pos ** 0.5)
        fr = torch.outer(t // w, inv_freq)
        fc = torch.outer(t % w, inv_freq)
        emb = torch.cat((fr, fc, fr, fc), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


def _rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    dt = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    qr = q * cos.unsqueeze(-2) + _rotate_half(q) * sin.unsqueeze(-2)
    kr = k * cos.unsqueeze(-2) + _rotate_half(k) * sin.unsqueeze(-2)
    return qr.to(dt), kr.to(dt)


# ─── Delta-rule Memory (DeltaNet family, chunk-wise) ───
#
# Chunk-wise Gated DeltaNet (Schlag et al. 2021; Yang et al. 2024 DeltaNet; Yang et al.
# 2025 Gated DeltaNet): the delta rule applied one block iteration at a time with the
# whole grid as the chunk, all T tokens written simultaneously. The update coincides
# with a single mini-batch SGD step on the associative-recall loss
# sum_t beta_t ||k_t^T S - v_t^T||^2 — the same *form* as TTT-Linear (Sun et al. 2024) /
# Titans (Behrouz et al. 2024) inner steps — but this is NOT the full TTT-Linear method:
# no separate theta_K/V/Q views (attention q/k/v are reused), no inner LayerNorm+residual,
# reads are non-causal within the chunk (all tokens read the post-update S), and there is
# a per-head decay alpha. Cite it as chunk-wise Gated DeltaNet.
#
#     S <- (alpha I - (1/T) K^T diag(beta) K) S + (1/T) K^T diag(beta) V,    o_t = q_t S
#
# with S [B,H,d,d] persisting across block iterations and segments (the recurrent
# axis), keys/queries L2-normalised (standard in DeltaNet; makes the update
# contractive for alpha, beta in (0,1)), per-token write strength beta_t from a
# linear gate (DeltaNet) and a per-head decay alpha (Gated DeltaNet). Compared to a
# plain Hebbian write, the (1/T) K^T diag(beta) K S term erases only along the key
# directions being rewritten. Keys/values/queries are symbol-averaged before the
# write so the state stays permutation-equivariant over the K axis.

class DeltaMemory(nn.Module):
    """청크식 Gated DeltaNet + CKDA 확장 (arXiv 2609.24797) — 2D 윈도우 풀링판 (2026-09-23 최종).

    목적: 재귀(블록 반복) 축의 메모리 전이가 반사·회전을 표현할 수 있게 — signed 채널 decay α∈[-1,1],
    쓰기강도 β∈[0,2], rank-1 지우기의 CKDA 3요소. 실측 이력:
      · v1 평균 M: T=900 전체 평균 → 세기 희석(≈0.08), 회전 미발동.
      · v2 rank-1 풀링(전체): λmax=β̄ 정확해졌으나 희석 동일.
      · v3 행(30토큰) 서브청크 순차: 행이 일관되면 발동 — но 행은 격자의 임의 1D 절단.
      · v4 (사용자 제안, 이 버전): **2D 깊이별 conv 풀링** — 격자 [g×g] 위 3×3 stride-3 depthwise conv (ARC 의 최소 국소 단위 = 인접 이웃; 창이 작을수록 일관성 게이트가 잘 열림 — 무작위 N개 평균 세기 ~1/√N)
        (평균 풀링으로 초기화, 학습 가능)로 100개 윈도우의 지우기 방향 m_c 를 얻고, β̄_c=‖m_c‖(2 에서 캡),
        k̄_c=m_c/β̄_c 로 순차 rank-1 지우기. ARC 의 국소 객체/블롭 단위로 일관성이 모이면 그 자리서
        반사(β̄>1)가 켜진다. 한 바퀴 전이 = Π_c (I−β̄_c k̄_c k̄_cᵀ)·Diag(α) — generalized Householder 곱.
    쓰기는 전체 T 합(1/T 스케일) 유지 — 다방향 연관기억 성질 보존. 비확장성: |α|≤1, β̄≤2(캡), 단위 k̄.
    주의: 논문 정리의 문자적 이식이 아니라 블록 반복 축 전이에 같은 구조를 준 것."""

    def __init__(self, hidden_size, num_heads, head_dim, alpha_init=0.9, grid=30, window=3):
        super().__init__()
        # β spread 초기화 (공식 ComplexKDA 관례): bias ±log3 → 헤드 절반은 β≈1.5(반사 영역),
        # 절반은 ≈0.5 에서 출발 — β>1 동역학이 초기부터 탐색 가능하게.
        self.beta_proj = CastedLinear(hidden_size, num_heads, bias=True)
        with torch.no_grad():
            _sg = torch.ones(num_heads); _sg[: num_heads // 2] = -1.0
            self.beta_proj.bias.copy_(_sg[torch.randperm(num_heads)] * math.log(3.0))
        # tanh(u/2) = 2σ(u)-1 이지만 u≈0 근처 파국적 상쇄가 없는 스펠링 (공식 구현 관례).
        # 초기값: tanh(_a0/2) = alpha_init  (2·atanh — 기존 식과 동일)
        _a0 = math.log((1 + alpha_init) / (1 - alpha_init))
        self.alpha_raw = nn.Parameter(torch.full((num_heads, head_dim, 1), _a0))
        self.grid, self.window = grid, window
        assert grid % window == 0, (grid, window)
        ch = num_heads * head_dim
        self.pool = nn.Conv2d(ch, ch, kernel_size=window, stride=window, groups=ch, bias=False)
        with torch.no_grad():
            self.pool.weight.fill_(1.0 / (window * window))   # 평균 풀링에서 출발

    @property
    def alpha(self):
        return torch.tanh(0.5 * self.alpha_raw)   # ≡ 2σ(u)-1, u≈0 상쇄 없는 스펠링. [H,d,1] ∈ (-1,1)

    def update(self, x_mean, k, v, S, fresh, eps=1e-6):
        # x_mean [B,T,D]; k, v [B,H,T,d] symbol-averaged, k L2-normalised; S [B,H,d,d] or None
        B, H, T, d = k.shape
        g, w = self.grid, self.window
        beta = 2.0 * torch.sigmoid(self.beta_proj(x_mean)).transpose(1, 2).unsqueeze(-1).to(k.dtype)  # [B,H,T,1] ∈ (0,2)
        kb = k * beta
        # 2D 윈도우 풀링: [B,H,T,d] → [B,H·d,g,g] → conv → [B,H,d,Nw] → 윈도우별 지우기 벡터
        kb_grid = kb.permute(0, 1, 3, 2).reshape(B, H * d, g, g)
        m = self.pool(kb_grid.to(self.pool.weight.dtype)).to(k.dtype)      # [B,H·d,g/w,g/w]
        m = m.reshape(B, H, d, -1).transpose(2, 3)                          # [B,H,Nw,d]
        beta_bar = m.norm(dim=-1, keepdim=True).clamp(max=2.0)              # [B,H,Nw,1]
        k_bar = m / m.norm(dim=-1, keepdim=True).clamp_min(eps)             # [B,H,Nw,d]
        if S is None:
            S = torch.zeros(B, H, d, v.shape[-1], dtype=k.dtype, device=k.device)
        elif fresh is not None:
            S = torch.where(fresh.view(-1, 1, 1, 1), torch.zeros_like(S), S)
        S = self.alpha.to(S.dtype) * S                     # Diag(α)⊙S — 바퀴당 decay 한 번
        for c in range(k_bar.shape[2]):                     # 윈도우 순차 (raster)
            kc, bc = k_bar[:, :, c], beta_bar[:, :, c]      # [B,H,d], [B,H,1]
            kTs = torch.einsum('bhd,bhde->bhe', kc, S)
            S = S - (bc.unsqueeze(-1) * kc.unsqueeze(-1)) * kTs.unsqueeze(-2)
        return S + torch.einsum('bhtd,bhte->bhde', kb, v) / T   # 쓰기 (총량 1/T)

    def read(self, q, S):
        # q: [B,K,H,T,d] L2-normalised, S: [B,H,d,d] -> [B,K,H,T,d]
        return torch.einsum('bkhtd,bhde->bkhte', q, S.to(q.dtype))


# ─── Axial Attention ───

class PositionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H = config.num_heads
        self.hd = config.head_dim
        self.out_dim = self.H * self.hd
        self.qkv = CastedLinear(config.hidden_size, 3 * self.H * self.hd)
        self.o_proj = CastedLinear(self.out_dim, config.hidden_size)

    def forward(self, x, cos_sin, mem=None, S=None, fresh=None):
        B, K, T, D = x.shape
        qkv = self.qkv(x).view(B * K, T, 3, self.H, self.hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = _apply_rotary(q, k, cos[:T], sin[:T])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        vt = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, vt)

        if mem is not None:
            q4 = F.normalize(q.view(B, K, self.H, T, self.hd), dim=-1)
            k4 = F.normalize(k.view(B, K, self.H, T, self.hd).mean(1), dim=-1)
            v4 = vt.view(B, K, self.H, T, self.hd).mean(1)
            S = mem.update(x.mean(1), k4, v4, S, fresh)
            out = out + mem.read(q4, S).reshape(B * K, self.H, T, self.hd)

        out = out.transpose(1, 2).reshape(B, K, T, self.out_dim)
        return self.o_proj(out), S


class SymbolAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.H = config.num_heads_t
        self.hd = config.head_dim_t
        self.out_dim = self.H * self.hd
        self.qkv = CastedLinear(config.hidden_size, 3 * self.H * self.hd)
        self.o_proj = CastedLinear(self.out_dim, config.hidden_size)

    def forward(self, x):
        B, K, T, D = x.shape
        xt = x.transpose(1, 2)
        qkv = self.qkv(xt).view(B * T, K, 3, self.H, self.hd)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        out = out.reshape(B, T, K, self.out_dim)
        return self.o_proj(out).transpose(1, 2)


# ─── MoE FFN ───

class SwiGLU(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.gate_up = CastedLinear(dim, inter * 2)
        self.down = CastedLinear(inter, dim)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class MoERouter(nn.Module):
    # Auxiliary-loss-free load balancing (Wang et al. 2024, arXiv 2408.15664; DeepSeek-V3 §2.1.2):
    # expert별 bias b_e 를 top-k *선택* 점수에만 더하고(게이팅 가중치엔 안 씀), 스텝마다 gradient 없이
    # 과부하 expert 는 b_e -= gamma, 굶은 expert 는 b_e += gamma. 선택 카운트는 forward 에서 누적하고
    # train_batch 가 optimizer step 뒤에 balance_step() 으로 (rank 간 all_reduce 한 뒤) 갱신한다.
    def __init__(self, dim, num_experts, top_k=1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = CastedLinear(dim, num_experts)
        self.bias = nn.Buffer(torch.zeros(num_experts), persistent=True)
        self.acc_counts = nn.Buffer(torch.zeros(num_experts), persistent=False)

    def forward(self, x, n_active=None):
        logits = self.gate(x)
        if n_active is not None and n_active < self.num_experts:
            # 앞 n_active개만 선택 가능 — 나머지는 -inf로 막아 softmax 확률 0, top-k에서 제외
            logits = logits.masked_fill(
                torch.arange(self.num_experts, device=logits.device) >= n_active, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        # 선택은 (0,1) 범위의 확률에 bias 를 더해서 — 논문/V3 가 bias 를 sigmoid 점수(0~1)에 더하는 것과 같은 스케일.
        # 로짓(범위 무제한)에 더하면 gamma=0.001 이 상대적으로 너무 약해 쏠림을 못 잡는다 (2026-09-22 실측).
        scores = probs + self.bias.to(probs.dtype)
        if n_active is not None and n_active < self.num_experts:
            scores = scores.masked_fill(
                torch.arange(self.num_experts, device=scores.device) >= n_active, float("-inf"))
        _, idx = torch.topk(scores, self.top_k, dim=-1)
        vals = torch.gather(probs, -1, idx)
        if self.top_k > 1:
            vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-9)
        # top-1 은 정규화하지 않는다 (Switch Transformer): p/(p+eps)=1 로 만들면 게이트에 gradient 가 0 이 되어
        # 라우터가 학습을 못 한다 — 원래 확률 p 를 expert 출력에 곱해야 라우팅이 손실로부터 배운다.
        # Counting happens in LTv5Inner outside checkpointed blocks, exactly once.
        return vals, idx, logits

    @torch.no_grad()
    def pop_counts(self, world_size=1):
        # 이번 스텝의 expert별 선택 수(전체 rank 합). 누적기는 비운다. bias 갱신과 학습 중 라우팅 기록 둘 다 이걸 쓴다.
        counts = self.acc_counts.clone()
        if world_size > 1:
            dist.all_reduce(counts)
        self.acc_counts.zero_()
        return counts

    @torch.no_grad()
    def apply_balance(self, gamma, n_active, counts):
        active = counts[:n_active]
        self.bias[:n_active] += gamma * torch.sign(active.mean() - active)

    @torch.no_grad()
    def unlock(self, old_n, new_n):
        # 새로 열리는 expert 는 기존 활성 expert 의 평균 bias 로 시작 — 0 으로 두면 (다른 bias 가 음수로
        # 내려가 있을 때) 열리는 순간 토큰을 독식한다. 굶은 상태라 이후 +gamma 로 천천히 편입된다.
        if 0 < old_n < new_n:
            self.bias[old_n:new_n] = self.bias[:old_n].mean()


class MoEExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate):
        super().__init__()
        self.n = num_experts
        self.gate_up = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden_size))
        self.down = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate))
        for i in range(num_experts):
            trunc_normal_init_(self.gate_up[i], std=1.0 / math.sqrt(hidden_size))
            trunc_normal_init_(self.down[i], std=1.0 / math.sqrt(intermediate))

    def forward_single(self, x, expert_index):
        """Apply one expert to every token, without routing or dispatch/scatter."""
        g, u = F.linear(x, self.gate_up[expert_index]).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down[expert_index])

    def forward(self, x, idx, vals):
        # One host transfer for all group sizes, instead of a GPU sync per expert.
        flat_idx = idx.reshape(-1)
        order = torch.argsort(flat_idx, stable=True)
        counts = torch.bincount(flat_idx, minlength=self.n).cpu().tolist()
        token_ids = torch.div(order, idx.shape[-1], rounding_mode="floor")
        weights = vals.reshape(-1)[order].unsqueeze(-1)
        dispatched = x[token_ids]
        pieces, start = [], 0
        for e, count in enumerate(counts):
            if count == 0:
                continue
            h = dispatched[start:start+count]
            g, u = F.linear(h, self.gate_up[e]).chunk(2, dim=-1)
            pieces.append(F.linear(F.silu(g) * u, self.down[e]) * weights[start:start+count])
            start += count
        out = torch.zeros_like(x)
        out.index_add_(0, token_ids, torch.cat(pieces, dim=0).to(out.dtype))
        return out


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        D = config.hidden_size
        self.router = MoERouter(D, config.num_experts, config.top_k)
        self.experts = MoEExperts(config.num_experts, D, config.expert_intermediate)
        self.shared = SwiGLU(D, config.shared_intermediate)
        self.shared_gate = nn.Linear(D, 1, bias=False)
        self.num_experts = config.num_experts
        self.config = config

    def forward(self, x, expert_index=None):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if expert_index is None:
            vals, idx, logits = self.router(x2, self.config.num_active_experts)
            expert_out = self.experts(x2, idx, vals)
            routing = (logits, idx)
        else:
            # The prescribed pass has unit weight; only the final routed pass
            # uses the router probability and contributes routing statistics/loss.
            expert_out = self.experts.forward_single(x2, expert_index)
            routing = None
        shared_out = self.shared(x2) * torch.sigmoid(self.shared_gate(x2))
        return (expert_out + shared_out).view(shape), routing


# ─── Block ───

class LTv5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_attn = PositionAttention(config)
        self.sym_attn = SymbolAttention(config)
        self.moe = MoEBlock(config)
        self.eps = config.rms_norm_eps
        self.mem = DeltaMemory(config.hidden_size, config.num_heads, config.head_dim, config.delta_alpha_init, grid=config.grid) if config.delta_memory else None

    def forward(self, x, cos_sin, w=None, fresh=None, expert_index=None):
        attn_out, w = self.pos_attn(x, cos_sin, self.mem, w, fresh)
        x = rms_norm(x + attn_out, self.eps)
        x = rms_norm(x + self.sym_attn(x), self.eps)
        moe_out, router_logits = self.moe(x, expert_index)
        x = rms_norm(x + moe_out, self.eps)
        return x, w, router_logits


# ─── Inner Model ───

class LTv5Inner(nn.Module):
    def __init__(self, config: LTv5Config):
        super().__init__()
        self.config = config
        self.fwd_dtype = getattr(torch, config.forward_dtype)
        K, D, T = config.vocab_size, config.hidden_size, config.seq_len

        self.embed_scale = math.sqrt(K)
        self.embed_special = nn.Parameter(trunc_normal_init_(torch.empty(2, D, dtype=self.fwd_dtype), std=1.0))
        self.embed_common = nn.Parameter(trunc_normal_init_(torch.empty(1, D, dtype=self.fwd_dtype), std=1.0))

        self.rotary = RotaryEmbedding2d(config.head_dim, T, config.rope_theta)

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                config.batch_size, init_std=0, cast_to=self.fwd_dtype)

        self.layers = nn.ModuleList([LTv5Block(config) for _ in range(config.num_layers)])

        self.lm_head = CastedLinear(D, 1)

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(D, dtype=self.fwd_dtype), std=1.0), persistent=True)

    def embed(self, inputs, puzzle_ids=None):
        B, T = inputs.shape
        K, D = self.config.vocab_size, self.config.hidden_size

        onehot = F.one_hot(inputs.long(), K).to(self.fwd_dtype)
        sym = torch.cat([self.embed_special, self.embed_common.expand(K - 2, -1)], 0)
        emb = onehot.transpose(1, 2).unsqueeze(-1) * sym.unsqueeze(0).unsqueeze(2) * self.embed_scale

        if self.config.puzzle_emb_ndim > 0 and puzzle_ids is not None:
            pe = self.puzzle_emb(puzzle_ids).to(emb.dtype)
            pe_global = pe[:, :D].view(B, 1, 1, D)
            pe_color = pe[:, D:D + K].view(B, K, 1, 1)
            emb = emb + pe_global + pe_color

        return emb

    def readout(self, h):
        return self.lm_head(h).squeeze(-1).transpose(1, 2)

    def forward(self, hidden, coupling, batch, fresh=None):
        inj = self.embed(batch["inputs"], batch.get("puzzle_identifiers"))
        cos_sin = self.rotary()

        h = hidden + inj
        w = coupling
        all_router_logits = []
        block_logits = []

        use_ckpt = self.config.ckpt_blocks and torch.is_grad_enabled()
        for block_index in range(self.config.blocks_per_seg):
            expert_index = (block_index
                if self.config.expert_schedule == "sequential_then_routed"
                and block_index < self.config.num_experts else None)
            for layer in self.layers:
                if use_ckpt:
                    from torch.utils.checkpoint import checkpoint
                    h, w, rl = checkpoint(layer, h, cos_sin, w, fresh, expert_index,
                                          use_reentrant=False)
                else:
                    h, w, rl = layer(h, cos_sin, w, fresh, expert_index)
                if rl is not None:
                    all_router_logits.append(rl)
                    if self.training:
                        with torch.no_grad():
                            layer.moe.router.acc_counts.add_(torch.bincount(
                                rl[1].reshape(-1), minlength=layer.moe.router.num_experts
                            ).to(layer.moe.router.acc_counts.dtype))
                fresh = None
            if self.config.block_supervision == "all":
                block_logits.append(self.readout(h))

        logits = block_logits[-1] if block_logits else self.readout(h)
        return h.detach(), (w.detach() if w is not None else None), logits, all_router_logits, block_logits


# ─── Carry ───

@dataclass
class LTv5Carry:
    hidden: torch.Tensor
    coupling: Optional[torch.Tensor]
    steps: torch.Tensor
    halted: torch.Tensor
    data: Dict[str, torch.Tensor]


# ─── Outer Model ───

class LTv5(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTv5Config.from_dict(config_dict)
        self.inner = LTv5Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        D = self.config.hidden_size
        K, T = self.config.vocab_size, self.config.seq_len
        fwd = getattr(torch, self.config.forward_dtype)
        device = batch["inputs"].device
        return LTv5Carry(
            hidden=torch.zeros(B, K, T, D, dtype=fwd, device=device),
            coupling=None,
            steps=torch.zeros(B, dtype=torch.int32, device=device),
            halted=torch.ones(B, dtype=torch.bool, device=device),
            data={k: torch.zeros_like(v) for k, v in batch.items()},
        )

    def forward(self, carry, batch, compute_target_q=False):
        inner = self.inner
        hidden = torch.where(carry.halted.view(-1, 1, 1, 1), inner.init_hidden, carry.hidden)
        fresh_flag = carry.halted.clone()

        coupling = carry.coupling
        if coupling is not None:
            coupling = torch.where(fresh_flag.view(-1, 1, 1, 1), torch.zeros_like(coupling), coupling)

        steps = torch.where(carry.halted, 0, carry.steps)
        data = {}
        for k, v in carry.data.items():
            bk = batch[k]
            ndim_extra = bk.ndim - 1
            shape = (-1,) + (1,) * ndim_extra
            data[k] = torch.where(carry.halted.view(shape), bk, v)

        if self.config.amp and hidden.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                hidden, coupling, logits, router_logits, block_logits = inner(hidden, coupling, data, fresh_flag)
            logits = logits.float()
            block_logits = [bl.float() for bl in block_logits]
        else:
            hidden, coupling, logits, router_logits, block_logits = inner(hidden, coupling, data, fresh_flag)

        q = torch.full((logits.shape[0],), -5.0, device=logits.device, dtype=torch.float32)
        outputs = {"logits": logits, "q_halt_logits": q, "q_continue_logits": q, "router_logits": router_logits}
        if block_logits:
            outputs["block_logits"] = block_logits

        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.config.loops

        new_carry = LTv5Carry(hidden=hidden, coupling=coupling, steps=steps, halted=halted, data=data)
        return new_carry, outputs


# ─── Loss ───

IGNORE_LABEL_ID = -100


def _s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    sx = _s(x)
    return torch.log(sx / torch.sum(sx, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    valid = labels != ignore_index
    tl = torch.where(valid, labels, 0)
    p = torch.gather(logprobs, index=tl.long().unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid, p, 0)


def softmax_cross_entropy(logits, labels, ignore_index=-100):
    # readout 이 심볼축 transpose 를 거쳐 non-contiguous 라 view 불가 → reshape (2026-09-23, 잠복 버그 수정)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.long().reshape(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def moe_load_balancing_loss(router_logits_list, n_active):
    # Switch-Transformer 균형 손실: n_active · Σ_e freq_e · avg_prob_e. 텐서 크기는 로짓의 전체 expert 수,
    # 스케일은 지금 활성인 expert 수 — 마스킹된(-inf) expert 는 freq=prob=0 이라 합에 안 들어가고,
    # 활성 expert 끼리 균등하면 손실 = 1.0 (스케줄 단계와 무관한 바닥).
    # 입력은 (logits, idx) 쌍의 리스트 — freq 는 bias 가 반영된 *실제* 선택(idx), prob 는 원 점수.
    if not router_logits_list:
        return 0.0
    total = 0.0
    for rl, idx in router_logits_list:
        E = rl.shape[-1]
        avg_probs = F.softmax(rl, dim=-1).mean(0)
        freq = torch.bincount(idx.reshape(-1), minlength=E).to(avg_probs.dtype) / idx.shape[0]
        total = total + n_active * (freq * avg_probs).sum()
    return total / len(router_logits_list)


class ACTLossHead(nn.Module):
    def __init__(self, model, loss_type="stablemax_cross_entropy", q_weight=0.5, focal_gamma=0.0, moe_aux_weight=0.01):
        super().__init__()
        self.model = model
        loss_fns = {"stablemax_cross_entropy": stablemax_cross_entropy, "softmax_cross_entropy": softmax_cross_entropy}
        self.loss_fn = loss_fns[loss_type]
        self.q_weight = q_weight
        self.focal_gamma = focal_gamma
        self.moe_aux_weight = moe_aux_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, return_keys: Set[str], **model_kwargs):
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        def _lm(lg):
            ce = self.loss_fn(lg, labels, ignore_index=IGNORE_LABEL_ID)
            if self.focal_gamma > 0:
                ce = ce * (1.0 - torch.exp(-ce)) ** self.focal_gamma
            return (ce / loss_divisor).sum()
        if outputs.get("block_logits"):     # block_supervision="all": 블록마다 채점, 평균
            lm_loss = torch.stack([_lm(bl) for bl in outputs["block_logits"]]).mean()
        else:
            lm_loss = _lm(outputs["logits"])

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        aux_loss = 0.0
        if self.moe_aux_weight > 0 and outputs.get("router_logits"):
            # 균형 손실은 지금 활성인 expert 수 기준 (비활성 expert는 확률 0이라 합에 안 들어감 →
            # 활성 expert끼리 균등하면 손실 = 1.0 로 스케줄 단계와 무관하게 바닥이 같다)
            num_exp = self.model.config.num_active_experts
            aux_loss = self.moe_aux_weight * moe_load_balancing_loss(outputs["router_logits"], num_exp)

        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})
        if isinstance(aux_loss, torch.Tensor):
            metrics["moe_aux_loss"] = aux_loss.detach()

        if outputs.get("router_logits"):
            with torch.no_grad():
                n_act = self.model.config.num_active_experts
                idx = torch.cat([ix.reshape(-1) for _, ix in outputs["router_logits"]])
                counts = torch.bincount(idx, minlength=self.model.config.num_experts).float()[:n_act]
                metrics["moe_used"] = (counts > 0).sum().float()
                metrics["moe_cv"] = counts.std() / counts.mean().clamp_min(1e-6)

        returned_outputs = {}
        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        total_loss = lm_loss + self.q_weight * q_halt_loss + aux_loss
        return new_carry, total_loss, metrics, returned_outputs, new_carry.halted.all()
