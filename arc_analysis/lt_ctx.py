"""In-context LT: 시범 쌍 + 질의를 슬롯으로 나란히 둔 한 시퀀스. 설계는 protocol_context.md.

토큰 = (슬롯 s, 칸 (u,w)),  T = n_slots × S².  슬롯 순서: in₁ out₁ … in_K out_K in_q out_q.
LT_Inner / D4Inner 에서 바뀌는 것 세 곳:
  · 위치: pos_u, pos_w, ‖Δ‖₁ 를 슬롯 안 칸 좌표로만 계산 (다른 슬롯의 같은 칸 = 거리 0)
  · 커널: A_t = ψ/2 + θ·pos_t + σ[slot_t], B_n = −ψ/2 + θ·pos_n + σ[slot_n].  σ = psi_slot (분리형 → attn_xy 그대로)
    D4 판은 σ 를 헤드 간 공유 (공간 방향이 없으므로 등변성 유지)
  · 주입: embed(x_t) + E_role[slot_t] + 퍼즐 ID 임베딩(원본과 같이 모든 토큰에).
step / _forward / boundary / phi / W_C 는 부모 것을 그대로 쓴다.
"""
from dataclasses import dataclass, fields
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LT, LT_Inner, LTConfig, LTLayer, trunc_normal_init_        # noqa: E402
from lt_d4 import D4Inner, D4Layer, G                                         # noqa: E402

ROLE_DEMO_IN, ROLE_DEMO_OUT, ROLE_QUERY_IN, ROLE_QUERY_OUT = 0, 1, 2, 3
N_ROLES = 4


@dataclass
class CtxConfig(LTConfig):
    n_slots: int = 6            # 2 (K+1), K = 시범 쌍 수
    use_puzzle_id: bool = True  # False 면 퍼즐 ID 임베딩 없이 (시범만으로 규칙을 읽어야 함). puzzle_emb_ndim 은 하네스용으로만 남는다

    @classmethod
    def from_dict(cls, d: dict) -> "CtxConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def slot_roles(n_slots):
    """슬롯 s 의 역할. 마지막 두 슬롯이 질의."""
    roles = [ROLE_DEMO_IN if s % 2 == 0 else ROLE_DEMO_OUT for s in range(n_slots - 2)]
    return torch.tensor(roles + [ROLE_QUERY_IN, ROLE_QUERY_OUT], dtype=torch.long)


class _CtxMixin:
    """두 Inner 가 공유하는 부분. `_init_ctx` 는 부모 __init__ 뒤에 호출한다."""

    def _init_ctx(self, config: CtxConfig, role_dim: int, sigma_heads: int):
        S, n = config.grid, config.n_slots
        assert n >= 4 and n % 2 == 0 and config.seq_len == n * S * S
        cell = torch.arange(S * S).float()
        u, w = (cell // S).repeat(n), (cell % S).repeat(n)                    # 슬롯 타일
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.register_buffer("slot", torch.arange(n).repeat_interleave(S * S), persistent=False)
        self.register_buffer("role", slot_roles(n).repeat_interleave(S * S), persistent=False)
        self.psi_slot = nn.Parameter(torch.rand(sigma_heads, self.p, n) * 2 * math.pi - math.pi)   # ψ 와 같이 [−π, π] 균등. 이름에 psi → wd 제외
        self.role_emb = nn.Embedding(N_ROLES, role_dim)
        with torch.no_grad():
            trunc_normal_init_(self.role_emb.weight, std=1.0 / self.embed_scale)

    def kernel(self, L, psi=None):
        psi = L.psi if psi is None else psi
        decay_h = torch.exp(-L.alpha[:, 0, None, None] * self.l1) if self.config.dist_decay else torch.ones_like(self.l1).expand(L.alpha.shape[0], -1, -1)
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w          # [H,p,T]
        sig = self.psi_slot[..., self.slot]                                                     # [H|1,p,T]
        A = (ppos + psi[..., None] / 2 + sig).permute(2, 0, 1); B = (ppos - psi[..., None] / 2 + sig).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def _role_injection(self):
        return self.role_emb(self.role)                                                         # [T, role_dim]


class CtxInner(_CtxMixin, LT_Inner):
    def __init__(self, config: CtxConfig):
        LT_Inner.__init__(self, _square_alias(config))       # 부모는 T == grid² 를 요구 → 임시 grid 로 통과
        self.config = config
        self._init_ctx(config, role_dim=self.d, sigma_heads=self.H)

    def injection(self, batch):
        inj = self.embed(batch["inputs"].to(torch.long)) + self._role_injection().unsqueeze(0)
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.d - self.puzzle_emb_ndim
            if pad > 0: pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj


class D4CtxInner(_CtxMixin, D4Inner):
    def __init__(self, config: CtxConfig):
        D4Inner.__init__(self, _square_alias(config))
        self.config = config
        self._init_ctx(config, role_dim=self.dh, sigma_heads=1)      # σ·역할 = 자명표현 (헤드/슬라이스 공유)

    def injection(self, batch):
        inj = (self.embed(batch["inputs"].to(torch.long)) + self._role_injection().unsqueeze(0)).repeat(1, 1, G)
        if self.puzzle_emb_ndim > 0:
            inj = inj + self.puzzle_emb(batch["puzzle_identifiers"]).to(inj.dtype).unsqueeze(1)   # 정칙표현 [B,8·dh]
        return inj


def _square_alias(config: CtxConfig) -> CtxConfig:
    """부모 __init__ 의 assert T == grid² 를 통과시키는 임시 cfg. 위치 버퍼는 _init_ctx 가 진짜 T 로 덮어쓴다."""
    g = math.isqrt(config.seq_len)
    g += g * g != config.seq_len
    return CtxConfig(**{**{f.name: getattr(config, f.name) for f in fields(config)}, "grid": g, "seq_len": g * g})


class LTCtx(LT):
    """train.LT 인터페이스, inner = CtxInner (기존 LT 커널) 또는 D4CtxInner."""

    inner_cls = CtxInner

    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = CtxConfig.from_dict(config_dict)
        inner_cfg = self.config if self.config.use_puzzle_id else CtxConfig.from_dict(dict(config_dict, puzzle_emb_ndim=0))
        self.inner = self.inner_cls(inner_cfg)
        assert self.inner.l1.shape[0] == self.config.seq_len

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb if self.config.use_puzzle_id else None


class D4LTCtx(LTCtx):
    inner_cls = D4CtxInner
