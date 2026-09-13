"""사용자가 준 풀버전 train.py(2026-09-13)의 모델 부분을 그대로 옮긴 것. LTConfig · LTLayer · LT_Inner · LT.
공유 부품(LTCarry, CastedSparseEmbedding, trunc_normal_init_, inv_softplus)은 lt/train.py 에서 가져온다.
판:  v1.1-se = sym_equiv, sym_dk=64            v3 = sym_equiv, sym_dk=128, swiglu, addr_p=192, write_addr (2.83M 파라미터)
"""
import math
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lt"))
from train import LTCarry, CastedSparseEmbedding, trunc_normal_init_, inv_softplus   # noqa: E402

PRESETS_V3 = {
    "v1.1-se": dict(legacy_gauge=False, block_order="pre", use_trace=False, sym_equiv=True, sym_dk=64),
    "v3":      dict(legacy_gauge=False, block_order="pre", use_trace=False, sym_equiv=True, sym_dk=128,
                    boundary_act="swiglu", addr_p=192, write_addr=True),
}


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
    alpha_init: float = 0.1
    legacy_gauge: bool = False
    inj_gate_init: float = 0.25
    gamma_init: float = 0.1
    dist_decay: bool = True
    eps: float = 1e-4
    amp: bool = True
    forward_dtype: str = "float32"
    psi_zero: bool = False
    stdp: bool = True
    stdp_eta_init: float = 0.05
    stdp_gain_init: float = 1.0
    stdp_lam_init: float = 0.25
    stdp_gain_fixed: float = -1.0
    stdp_lam_fixed: float = -1.0
    block_order: str = "pre"
    use_trace: bool = False
    trace_rho_init: float = 0.5
    ckpt_blocks: bool = False
    # ---- 기호 등변 (SE-RRM 의 색 순열 등변성). 셀 상태 h_t ∈ R^d 를 [K=vocab, d_k] 슬롯으로 본다.
    sym_equiv: bool = False
    sym_dk: int = 64
    boundary_act: str = "none"
    addr_p: int = 0
    write_addr: bool = False
    n_mem: int = 1
    # ---- sheaf 확장 (2026-09-13, arc_analysis). 기본값은 원본과 수치 동일.
    transport: str = "adj"      # adj: h += Σ a·FᵀF h_n (원본) | laplacian: h += Σ a·FᵀF (h_n − h_t)  (Dirichlet 에너지의 경사 하강 형태)
    sheaf_cond: str = "none"    # none | scale: 제한사상 F_h(e) = diag(1+s_h(e))·W_sh,h, s 는 퍼즐 임베딩 뒤쪽 H·c 차원 (0 에서 시작)

    @classmethod
    def from_dict(cls, d: dict) -> "LTConfig":
        known = {f.name for f in fields(cls)}
        dd = {k: v for k, v in d.items() if k in known}
        if "use_trace" not in d and "trace_rho_init" in d:
            dd["use_trace"] = True
        return cls(**dd)


class LTLayer(nn.Module):
    """가중치 한 벌. 파라미터 이름은 모든 기존 체크포인트와 동일하다."""

    def __init__(self, config: "LTConfig", H, d, dh, p) -> None:
        super().__init__()
        self.sym = config.sym_equiv
        self.M = config.n_mem
        if self.sym:
            K, dk = config.vocab_size, config.sym_dk
            da = 3 * dk                                   # 주소 입력: [h_pad ; h_eos ; Σ_색 h_k]
        else:
            da = d
        assert 2 * p <= da, f"주소 성분 2p={2*p} 가 주소 입력 폭 {da} 를 넘는다 (행직교 불가)"
        self.wc_raw = nn.Parameter(torch.randn(H, 2 * p, da) / math.sqrt(da))       # 원본: 2p == dh
        if config.write_addr:
            self.wc_raw_b = nn.Parameter(torch.randn(H, 2 * p, da) / math.sqrt(da))  # 쓰기 창 전용 주소
        if config.psi_zero:
            self.register_buffer("psi", torch.zeros(H, p), persistent=False)
        else:
            self.psi = nn.Parameter(torch.rand(H, p) * 2 * math.pi - math.pi)
        self.theta = nn.Parameter((torch.rand(H, p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_raw = nn.Parameter(torch.full((H, 1), inv_softplus(config.alpha_init)))
        if self.sym:
            c = dk // H                                   # 슬롯별 헤드 값 폭. K·c = dh
            w_sh = torch.zeros(H, c, dk)
            for m in range(H):
                w_sh[m, :, m * c:(m + 1) * c] = torch.eye(c)
            self.w_sh = nn.Parameter(w_sh + 0.01 * torch.randn(H, c, dk) / math.sqrt(dk))
        else:
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
            if self.M == 1:
                self.eta_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_eta_init)))
                self.lam_raw = nn.Parameter(torch.full((H, 1, 1), lg(config.stdp_lam_init)))
                self.gain_raw = nn.Parameter(torch.full((H, 1, 1), inv_softplus(config.stdp_gain_init)))
            else:
                self.eta_raw = nn.Parameter(torch.tensor([[[[lg(config.stdp_eta_init / 10 ** mi)]]] * H for mi in range(self.M)]))
                self.lam_raw = nn.Parameter(torch.full((self.M, H, 1, 1), lg(config.stdp_lam_init)))
                self.gain_raw = nn.Parameter(torch.full((self.M, H, 1, 1), inv_softplus(config.stdp_gain_init)))
            self.beta = nn.Parameter(torch.zeros(H, p))
            self.beta.data.normal_(0.0, 0.5)
        if self.sym:                                      # 슬롯 공유 경계: 입력 [h_k ; Σ_k' h_k'] (2·d_k) → d_k
            inter = int(config.mlp_expansion * 2 * dk * 2 / 3 + 63) // 64 * 64
            self.b_gate_up = nn.Linear(2 * dk, 2 * inter, bias=False)
            self.b_down = nn.Linear(inter, dk, bias=False)
        else:
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
        self.p = config.addr_p if config.addr_p > 0 else self.dh // 2
        self.M = config.n_mem
        self.write_addr = config.write_addr
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.sym = config.sym_equiv
        if self.sym:
            assert not config.legacy_gauge, "sym_equiv 는 √d 고정 게이지에서만"
            self.K, self.dk = config.vocab_size, config.sym_dk
            assert d == self.K * self.dk, f"sym_equiv: hidden_size {d} != vocab {self.K} · sym_dk {self.dk}"
            assert self.dk % H == 0 and self.dh <= 3 * self.dk, (self.dk, H, self.dh)
            self.n_scale = H * (self.dk // H) if config.sheaf_cond == "scale" else 0
            assert config.puzzle_emb_ndim in (0, self.dk + self.K + self.n_scale), f"sym_equiv: puzzle_emb_ndim 은 sym_dk+vocab(+H·c)={self.dk + self.K + self.n_scale} (받은 값 {config.puzzle_emb_ndim})"
            self.gamma = 1.0 / d
            self.embed_scale = math.sqrt(d)
            self.embed_sym = nn.Parameter(trunc_normal_init_(torch.empty(3, self.dk), std=1.0 / math.sqrt(self.dk)))
            self.w_cls = nn.Linear(self.dk, 1)
        else:
            assert config.sheaf_cond == "none", "sheaf_cond 는 sym_equiv 판에서만 구현"
            self.n_scale = 0
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
        assert config.block_order in ("pre", "post")
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
        if self.sym:
            self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(self.dk, dtype=self.forward_dtype), std=1.0).repeat(self.K), persistent=True)
        else:
            self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=1.0), persistent=True)

    def W_C(self, L, write: bool = False):
        raw = L.wc_raw_b if write else L.wc_raw
        Q, _ = torch.linalg.qr(raw.transpose(-1, -2))
        AB = Q.transpose(-1, -2)
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def kernel(self, L, psi=None):
        psi = L.psi if psi is None else psi
        decay_h = torch.exp(-L.alpha[:, 0, None, None] * self.l1) if self.config.dist_decay else torch.ones_like(self.l1).expand(L.alpha.shape[0], -1, -1)
        ppos = L.theta[..., 0, None] * self.pos_u + L.theta[..., 1, None] * self.pos_w
        A = (ppos + psi[..., None] / 2).permute(2, 0, 1); B = (ppos - psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def attn_xy(self, xy, kc):
        x, y = xy; decay_h, cosA, sinA, cosB, sinB = kc
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB
        a = torch.einsum('bthj,bnhj->bhtn', qx, kx) + torch.einsum('bthj,bnhj->bhtn', qy, ky)
        return a * decay_h.unsqueeze(0)

    def phi(self, h):
        g = F.softplus(self.gamma_raw) if hasattr(self, "gamma_raw") else self.gamma
        return h / torch.sqrt(1.0 + g * h.pow(2).sum(-1, keepdim=True))

    def injection(self, batch):
        x = batch["inputs"].to(torch.long)
        if self.sym:
            onehot = F.one_hot(x, self.K).to(self.embed_sym.dtype)
            e = self.embed_sym[torch.clamp(x, max=2)]
            inj = onehot.unsqueeze(-1) * e.unsqueeze(2)
            self._fscale = None
            if self.puzzle_emb_ndim > 0:
                pe = self.puzzle_emb(batch["puzzle_identifiers"]).to(inj.dtype)      # [B, dk+K(+H·c)]
                inj = inj + pe[:, None, None, :self.dk] + pe[:, None, self.dk:self.dk + self.K, None]
                if self.n_scale:                                                      # 제한사상 채널 스케일 1+s [B,H,c]
                    self._fscale = 1.0 + pe[:, self.dk + self.K:].reshape(pe.shape[0], self.H, self.dk // self.H)
            return inj.reshape(x.shape[0], x.shape[1], self.d)
        inj = self.embed(x)
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.d - self.puzzle_emb_ndim
            if pad > 0: pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def addr_in(self, h):
        if not self.sym:
            return h
        hk = h.view(*h.shape[:2], self.K, self.dk)
        return torch.cat([hk[:, :, 0], hk[:, :, 1], hk[:, :, 2:].sum(2)], dim=-1)

    def values(self, L, h):
        if not self.sym:
            return torch.einsum('btd,hcd->bthc', h, L.w_sh)
        B, T = h.shape[:2]
        hk = h.view(B, T, self.K, self.dk)
        v = torch.einsum('btkd,hcd->bthkc', hk, L.w_sh)
        if getattr(self, "_fscale", None) is not None:
            v = v * self._fscale[:, None, :, None, :].to(v.dtype)
        return v.reshape(B, T, self.H, self.dh)

    def transport_back(self, L, o):
        if not self.sym:
            return torch.einsum('bthc,hcd->btd', o, L.w_sh)
        B, T = o.shape[:2]
        ok = o.view(B, T, self.H, self.K, self.dk // self.H)
        if getattr(self, "_fscale", None) is not None:
            ok = ok * self._fscale[:, None, :, None, :].to(ok.dtype)
        return torch.einsum('bthkc,hcd->btkd', ok, L.w_sh).reshape(B, T, self.d)

    def readout(self, h):
        if not self.sym:
            return self.w_cls(h)
        return self.w_cls(h.view(*h.shape[:2], self.K, self.dk)).squeeze(-1)

    def empty_carry(self, batch_size):
        return LTCarry(current_hidden=torch.empty(batch_size, self.config.seq_len, self.d, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag, carry):
        return replace(carry, current_hidden=torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def forward(self, carry, batch):
        if self.config.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nc, logits = self._forward(carry, batch)
            return replace(nc, current_hidden=nc.current_hidden.float()), logits.float()
        return self._forward(carry, batch)

    def addr_raw(self, h, AB):
        A, Bm = AB
        return torch.einsum('btd,hjd->bthj', h, A), torch.einsum('btd,hjd->bthj', h, Bm)

    def _unit(self, x, y):
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        return x / (nrm + self.config.eps), y / (nrm + self.config.eps)

    def addr(self, h, AB):
        return self._unit(*self.addr_raw(self.addr_in(h), AB))

    def trace_step(self, L, ux, uy, ztr, fresh):
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

    def step(self, L, h, AB, kc, w=None, fresh=None, kcb=None, ztr=None, apply_phi=True, ABb=None):
        hin = self.addr_in(h)
        ux, uy = self.addr_raw(hin, AB)
        uh = self._unit(ux, uy)
        a = self.attn_xy(uh, kc)
        v = self.values(L, h)
        ztr_new = None
        if self.stdp:
            if ABb is not None:
                ux, uy = self.addr_raw(hin, ABb); uhb = self._unit(ux, uy)
            else:
                uhb = uh
            if self.use_trace:
                zx, zy, ztr_new = self.trace_step(L, ux, uy, ztr, fresh)
                win = self.attn_xy(self._unit(zx, zy), kcb)
            else:
                win = self.attn_xy(uhb, kcb)
            vv = v / (v.norm(dim=-1, keepdim=True) + self.config.eps)
            agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
            G = win * agree
            gain = F.softplus(L.gain_raw) if self.config.stdp_gain_fixed < 0 else float(self.config.stdp_gain_fixed)
            eta = torch.sigmoid(L.eta_raw)
            lam = torch.sigmoid(L.lam_raw) if self.config.stdp_lam_fixed < 0 else torch.full_like(L.lam_raw, float(self.config.stdp_lam_fixed))
            if self.M == 1:
                tgt = gain * G
                if w is None:
                    w = tgt
                else:
                    w = torch.where(fresh.view(-1, 1, 1, 1), tgt, w) if fresh is not None else w
                    w = (1 - eta) * w + eta * tgt
                a = (1 - lam) * a + lam * w
            else:
                tgt = gain.unsqueeze(0) * G.unsqueeze(1) if torch.is_tensor(gain) else gain * G.unsqueeze(1).expand(-1, self.M, -1, -1, -1)
                if w is None:
                    w = tgt
                else:
                    w = torch.where(fresh.view(-1, 1, 1, 1, 1), tgt, w) if fresh is not None else w
                    w = (1 - eta.unsqueeze(0)) * w + eta.unsqueeze(0) * tgt
                lam = lam / self.M
                a = (1 - lam.sum(0)) * a + (lam.unsqueeze(0) * w).sum(1)
        o = torch.einsum('bhtn,bnhc->bthc', a, v)
        if self.config.transport == "laplacian":
            o = o - a.sum(-1).transpose(1, 2).unsqueeze(-1) * v             # − (Σ_n a_tn) v_t  → Σ_n a_tn (v_n − v_t)
        # Dirichlet 에너지 E = ½ Σ_h Σ_{t,n} a⁺_tn ‖v_t − v_n‖²  (a 의 양의 부분. 진단용, 그래프 밖)
        with torch.no_grad():
            ap = a.clamp_min(0).float(); vf = v.float()
            sq = vf.pow(2).sum(-1).transpose(1, 2)                              # [B,H,T]
            cross = torch.einsum('bhtn,bthc,bnhc->bh', ap, vf, vf)
            self.last_dirichlet = 0.5 * ((ap.sum(-1) * sq).sum(-1) + (ap.sum(-2) * sq).sum(-1) - 2 * cross).sum(-1)   # [B]
        f = self.transport_back(L, o)
        hout = self.phi(h + f) if apply_phi else (h + f)
        return hout, w, ztr_new

    def _gate(self, g, u):
        if self.config.boundary_act == "swiglu":
            return F.silu(g) * u
        return 0.5 * g * u

    def boundary(self, L, h):
        if self.sym:
            hk = h.view(*h.shape[:2], self.K, self.dk)
            ctx = hk.sum(2, keepdim=True).expand_as(hk)
            g, u = L.b_gate_up(torch.cat([hk, ctx], dim=-1)).chunk(2, dim=-1)
            return (hk + L.b_down(self._gate(g, u))).reshape(h.shape)
        g, u = L.b_gate_up(h).chunk(2, dim=-1)
        return h + L.b_down(self._gate(g, u))

    def _block(self, L, h, w, ztr, inj, AB, kc, kcb, fresh, pre, ABb=None):
        if pre:
            h = self.boundary(L, h)
        h = h + self.embed_scale * inj
        h, w, ztr = self.step(L, h, AB, kc, w, fresh, kcb, ztr, apply_phi=pre, ABb=ABb)
        if not pre:
            h = self.boundary(L, h)
            h = self.phi(h)
        return h, w, ztr

    def _forward(self, carry, batch):
        h = carry.current_hidden; inj = self.injection(batch)
        ABs = [self.W_C(L) for L in self.layers]
        ABbs = [self.W_C(L, write=True) if (self.stdp and self.write_addr) else None for L in self.layers]
        kcs = [self.kernel(L) for L in self.layers]
        kcbs = [self.kernel(L, L.beta) if self.stdp else None for L in self.layers]
        w = carry.coupling if self.stdp else None; fresh = carry.fresh if self.stdp else None
        ztr = carry.trace if self.use_trace else None
        pre = self.config.block_order == "pre"
        use_ckpt = self.config.ckpt_blocks and torch.is_grad_enabled()
        for _ in range(self.config.blocks_per_seg):
            for li, L in enumerate(self.layers):
                AB, kc, kcb, ABb = ABs[li], kcs[li], kcbs[li], ABbs[li]
                if use_ckpt:
                    from torch.utils.checkpoint import checkpoint
                    h, w, ztr = checkpoint(self._block, L, h, w, ztr, inj, AB, kc, kcb, fresh, pre, ABb, use_reentrant=False)
                else:
                    h, w, ztr = self._block(L, h, w, ztr, inj, AB, kc, kcb, fresh, pre, ABb)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=(w.detach() if w is not None else None),
                       trace=(ztr.detach() if ztr is not None else None), fresh=None), self.readout(h)


class LT(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig.from_dict(config_dict)
        self.inner = LT_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        return LTCarry(current_hidden=self.inner.empty_carry(B).current_hidden,
                       steps=torch.zeros((B,), dtype=torch.int32), halted=torch.ones((B,), dtype=torch.bool),
                       current_data={k: torch.empty_like(v) for k, v in batch.items()})

    def forward(self, carry, batch, compute_target_q: bool = False):
        inner = self.inner.reset_carry(carry.halted, carry)
        inner = replace(inner, fresh=carry.halted.clone())
        steps = torch.where(carry.halted, 0, carry.steps)
        data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        inner, logits = self.inner(inner, data)
        q = torch.full((logits.shape[0],), -5.0, device=logits.device, dtype=torch.float32)
        outputs = {"logits": logits, "q_halt_logits": q, "q_continue_logits": q,
                   "dirichlet": getattr(self.inner, "last_dirichlet", None)}
        with torch.no_grad():
            steps = steps + 1; halted = steps >= self.config.loops
        return LTCarry(current_hidden=inner.current_hidden, steps=steps, halted=halted, current_data=data,
                       coupling=inner.coupling, trace=inner.trace), outputs
