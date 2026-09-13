"""색 슬롯 LT — 칸 상태에 색 축을 두어 색 순열(1..9)에 등변인 판. 사용자 스펙(2026-09-13)을 다음과 같이 구체화했다.

상태  h_t ∈ R^{12·dk}: 슬롯 = [PAD, EOS, 색0(검정), 색1 … 색9], 슬롯당 dk.  토큰 id = 슬롯 번호.
1. 경계   h_{t,k} ← h_{t,k} + W_d · SwiGLU([h_{t,k} ; Σ_{k'} h_{t,k'}])         슬롯 공유 MLP, 칸 안
2. 주입   h_t ← h_t + √d · inj_t        inj = 종류 임베딩(pad/eos/black/color 4개, 색 1..9 공유)을 토큰의 슬롯에 + ID 임베딩(12 슬롯 전체)
3. 주소   hin_t = [h_pad ; h_eos ; h_black ; Σ_{k=1..9} h_k]  (색 순열 불변, 256)
          û_t = unit(W_C hin_t)   읽기 바늘 H×p = 8×24 = 192,   û^β_t = unit(W_C^β hin_t)  쓰기 바늘 192   (헤드별 QR 행직교)
4. 결합   a_tn = e^{−α‖Δ‖₁} Σ_j cos(φ_t,j − φ_n,j + θ_j·Δ + ψ_j),  a^β 는 쓰기 바늘과 β 로          (LT_Inner.kernel/attn_xy 그대로)
5. 값     v_{t,k} = W_v h_{t,k} (슬롯 공유, [H, dv, dk], dv = dk/H) → 헤드별 값 = 12 슬롯 이어붙임 (12·dv),  agree = cos
6. 기억   w ← (1−η) w + η g a^β agree
7. 수송   a_eff = (1−λ) a + λ w,   h_{t,k} ← Φ(h_{t,k} + Σ_h W_v^{(h)T} Σ_n a_eff v_{n,k}),  Φ 는 전체 d 노름
판독      logit_pad = w_pad·h_pad, logit_eos = w_eos·h_eos, logit_black = w_black·h_black, logit_색k = w_color·h_k (공유)
검정을 따로 두는 이유: URM 증강이 검정을 고정하므로 등변성은 색 1..9 의 S₉ 로 충분하고, 배경을 구분할 수 있어야 한다.
"""
from dataclasses import dataclass, fields, replace
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lt"))
from train import LT, LT_Inner, LTConfig, CastedSparseEmbedding, inv_softplus, trunc_normal_init_   # noqa: E402

N_SLOTS = 12
KIND_OF_TOKEN = [0, 1, 2] + [3] * 9          # pad, eos, black, color


@dataclass
class ColorConfig(LTConfig):
    slot_dim: int = 64
    mlp_inter: int = 1024
    addr_p: int = 24                        # 헤드당 바늘 수 (총 8×24 = 192)

    @classmethod
    def from_dict(cls, d: dict) -> "ColorConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class ColorLayer(nn.Module):
    def __init__(self, cfg: ColorConfig, H, dk, p, hin, dv):
        super().__init__()
        self.wc_raw = nn.Parameter(torch.randn(H, 2 * p, hin) / math.sqrt(hin))
        self.wcb_raw = nn.Parameter(torch.randn(H, 2 * p, hin) / math.sqrt(hin))
        self.psi = nn.Parameter(torch.rand(H, p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(cfg.alpha_init)))
        w_v = torch.zeros(H, dv, dk)
        for m in range(H):
            w_v[m, :, m * dv:(m + 1) * dv] = torch.eye(dv)
        self.w_v = nn.Parameter(w_v + 0.01 * torch.randn(H, dv, dk) / math.sqrt(dk))
        lg = lambda x: math.log(x / (1 - x))
        self.eta_raw = nn.Parameter(torch.full((H, 1, 1), lg(cfg.stdp_eta_init)))
        self.lam_raw = nn.Parameter(torch.full((H, 1, 1), lg(cfg.stdp_lam_init)))
        self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(cfg.stdp_gain_init)))
        self.beta = nn.Parameter(torch.randn(H, p) * 0.5)
        self.b_gate_up = nn.Linear(2 * dk, 2 * cfg.mlp_inter, bias=False)
        self.b_down = nn.Linear(cfg.mlp_inter, dk, bias=False)
        with torch.no_grad():
            self.b_down.weight.zero_()

    @property
    def alpha(self): return F.softplus(self.alpha_raw)


class ColorInner(LT_Inner):
    def __init__(self, config: ColorConfig):
        nn.Module.__init__(self)
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, H, dk = config.seq_len, config.grid, config.num_heads, config.slot_dim
        d = N_SLOTS * dk
        assert T == g * g and config.hidden_size == d and dk % H == 0 and config.vocab_size == N_SLOTS
        assert config.stdp and not config.use_trace and config.num_layers == 1 and not config.legacy_gauge
        self.hin_dim = 4 * dk
        self.d, self.H, self.dk, self.dv, self.p = d, H, dk, dk // H, min(config.addr_p, self.hin_dim // 2)   # 헤드별 행직교 조건 2p ≤ hin
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.register_buffer("kind", torch.tensor(KIND_OF_TOKEN), persistent=False)
        self.embed = nn.Embedding(4, dk)                                    # 종류 임베딩
        self.gamma = 1.0 / d
        self.embed_scale = math.sqrt(d)
        with torch.no_grad():
            trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
        self.out_w = nn.Parameter(torch.randn(4, dk) / math.sqrt(dk))        # 판독 (pad, eos, black, color 공유)
        self.out_b = nn.Parameter(torch.zeros(4))
        self.stdp, self.use_trace = True, False
        self.layers = nn.ModuleList([ColorLayer(config, H, dk, self.p, self.hin_dim, self.dv)])
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            assert config.puzzle_emb_ndim == d
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, d, batch_size=config.batch_size,
                                                    init_std=0, cast_to=self.forward_dtype)
        h0 = trunc_normal_init_(torch.empty(4, dk, dtype=self.forward_dtype), std=1.0)
        self.init_hidden = nn.Buffer(h0[self.kind].reshape(d), persistent=True)   # 종류별 → 색 1..9 슬롯 동일 (순열 불변)

    # ---- 부품
    def _qr(self, raw):
        Q, _ = torch.linalg.qr(raw.transpose(-1, -2))                        # [H, hin, 2p]
        AB = Q.transpose(-1, -2)
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def W_C(self, L): return self._qr(L.wc_raw)
    def W_Cb(self, L): return self._qr(L.wcb_raw)

    def hin(self, h):
        hs = h.view(*h.shape[:-1], N_SLOTS, self.dk)
        return torch.cat([hs[..., 0, :], hs[..., 1, :], hs[..., 2, :], hs[..., 3:, :].sum(-2)], dim=-1)

    def addr_raw(self, hin, AB):
        A, Bm = AB
        return torch.einsum('bti,hji->bthj', hin, A), torch.einsum('bti,hji->bthj', hin, Bm)

    def injection(self, batch):
        tok = batch["inputs"].to(torch.long)                                  # [B,T] = 슬롯 번호
        e = self.embed(self.kind[tok])                                        # [B,T,dk]
        inj = F.one_hot(tok, N_SLOTS).to(e.dtype).unsqueeze(-1) * e.unsqueeze(-2)   # [B,T,12,dk]
        inj = inj.reshape(*tok.shape, self.d)
        if self.puzzle_emb_ndim > 0:
            inj = inj + self.puzzle_emb(batch["puzzle_identifiers"]).to(inj.dtype).unsqueeze(1)
        return inj

    def boundary(self, L, h):
        hs = h.view(*h.shape[:-1], N_SLOTS, self.dk)
        x = torch.cat([hs, hs.sum(-2, keepdim=True).expand_as(hs)], dim=-1)  # [.., 12, 2dk]
        g, u = L.b_gate_up(x).chunk(2, dim=-1)
        return (hs + L.b_down(F.silu(g) * u)).reshape(h.shape)

    def w_cls(self, h):
        hs = h.view(*h.shape[:-1], N_SLOTS, self.dk)
        special = torch.einsum('btkd,kd->btk', hs[..., :3, :], self.out_w[:3]) + self.out_b[:3]
        colors = torch.einsum('btkd,d->btk', hs[..., 3:, :], self.out_w[3]) + self.out_b[3]
        return torch.cat([special, colors], dim=-1)                          # [B,T,12] = 토큰 순서

    def step(self, L, h, AB, kc, w=None, fresh=None, kcb=None, ABb=None, apply_phi=True):
        hin = self.hin(h)
        uh = self._unit(*self.addr_raw(hin, AB))
        a = self.attn_xy(uh, kc)
        B, T = h.shape[:2]
        hs = h.view(B, T, N_SLOTS, self.dk)
        v = torch.einsum('btkd,hcd->bthkc', hs, L.w_v).reshape(B, T, self.H, N_SLOTS * self.dv)
        win = self.attn_xy(self._unit(*self.addr_raw(hin, ABb)), kcb)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.config.eps)
        agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
        tgt = F.softplus(L.gain_raw) * win * agree
        eta, lam = torch.sigmoid(L.eta_raw), torch.sigmoid(L.lam_raw)
        if w is None:
            w = tgt
        else:
            w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
            w = (1 - eta) * w + eta * tgt
        a = (1 - lam) * a + lam * w
        o = torch.einsum('bhtn,bnhc->bthc', a, v).reshape(B, T, self.H, N_SLOTS, self.dv)
        f = torch.einsum('bthkc,hcd->btkd', o, L.w_v).reshape(B, T, self.d)
        hout = self.phi(h + f) if apply_phi else (h + f)
        return hout, w

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        L = self.layers[0]
        AB, ABb, kc, kcb = self.W_C(L), self.W_Cb(L), self.kernel(L), self.kernel(L, L.beta)
        w, fresh = carry.coupling, carry.fresh
        for _ in range(self.config.blocks_per_seg):
            h = self.boundary(L, h)
            h = h + self.embed_scale * inj
            h, w = self.step(L, h, AB, kc, w, fresh, kcb, ABb)
            fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=w.detach(), trace=None, fresh=None), self.w_cls(h)


class ColorLT(LT):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = ColorConfig.from_dict(config_dict)
        self.inner = ColorInner(self.config)
