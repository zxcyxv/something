"""D4 등변 LT — 헤드 축 = D4 정칙표현.

원본 `lt/train.py` 의 `LT_Inner.step / _forward` 는 그대로 쓰고, 파라미터를 D4 G-합성곱 구조로 묶는다.
  · 은닉 상태 h_t ∈ R^{8·dh} 를 8 슬라이스(군 원소 k)로 본다: (g·h)_x[k] = h_{g⁻¹x}[g⁻¹k]
  · 헤드 m 의 주소·값 사영 = G-합성곱의 출력 슬라이스 m  (블록(k,k') = base[k⁻¹k'])
  · 헤드 m 의 위치 주파수 θ^{(m)} = M_m θ,  ψ·β·α·η·λ·g 는 헤드 간 공유
  · 경계 MLP 두 선형층 = 슬라이스 축 G-합성곱, 게이트 곱은 슬라이스 안에서
  · 입력 임베딩 = 자명표현(8 슬라이스에 동일 타일)
  · 퍼즐 임베딩 = 정칙표현. 평평한 d 벡터를 [8,dh] 로 읽으면 되므로 원본 CastedSparseEmbedding 을 그대로 쓴다.
    회전한 과제 = 같은 행의 슬라이스 순열이지만, URM 처럼 증강마다 행을 따로 두면 그 순열을 쓸 일이 없다
    (한 행이 한 방향에서만 학습되므로 순열은 행의 재명명일 뿐). 그래서 학습·평가 코드에 방향 정보가 필요 없다.
  · 판독 = 슬라이스 합 → 선형 (불변)
파라미터 수는 원본의 약 1/8 (헤드 8개가 묶임). 상태·w 크기·어텐션 비용은 동일.
EOS 를 4변 테두리로 바꾼 대칭 부호화(train_arc_s.py --encoding frame)를 전제로 한다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LT, LT_Inner, LTConfig, CastedSparseEmbedding, inv_softplus, trunc_normal_init_   # noqa: E402
import d4                                                                                             # noqa: E402

G = d4.ORDER


def _gconv_weight(base, idx):
    """base [8, out', in'] → 전체 [8·out', 8·in'],  block(k,k') = base[idx[k,k']]."""
    out_, in_ = base.shape[1:]
    w = base[idx]                                   # [8(k), 8(k'), out', in']
    return w.permute(0, 2, 1, 3).reshape(G * out_, G * in_)


class D4Layer(nn.Module):
    """LTLayer 와 같은 이름·모양의 텐서를 property 로 내놓되, 실제 파라미터는 D4 로 묶인 base 다."""

    def __init__(self, config: LTConfig, H, d, dh, p):
        super().__init__()
        assert H == G, "헤드 수 = |D4| = 8"
        self.H, self.d, self.dh, self.p = H, d, dh, p
        self.register_buffer("idx", d4.conv_index(), persistent=False)
        self.register_buffer("mats", torch.tensor(np.array(d4.MATS), dtype=torch.float32), persistent=False)   # [8,2,2]
        self.wc_base = nn.Parameter(torch.randn(G, dh, dh) / math.sqrt(d))       # 주소 G-conv base (출력 dh=2p)
        self.psi_base = nn.Parameter(torch.rand(p) * 2 * math.pi - math.pi)
        self.theta_base = nn.Parameter((torch.rand(p, 2) * 2 - 1) * (math.pi / 2))
        self.alpha_base = nn.Parameter(torch.full((1,), inv_softplus(config.alpha_init)))
        sh = torch.zeros(G, dh, dh); sh[0] = torch.eye(dh)                       # 원본의 블록대각 항등 init 과 같은 역할
        self.sh_base = nn.Parameter(sh + 0.01 * torch.randn(G, dh, dh) / math.sqrt(d))
        lg = lambda x: math.log(x / (1 - x))
        self.eta_base = nn.Parameter(torch.full((1,), lg(config.stdp_eta_init)))
        self.lam_base = nn.Parameter(torch.full((1,), lg(config.stdp_lam_init)))
        self.gain_base = nn.Parameter(torch.full((1,), inv_softplus(config.stdp_gain_init)))
        self.beta_base = nn.Parameter(torch.randn(p) * 0.5)
        inter = int(config.mlp_expansion * dh * 2 / 3 + 31) // 32 * 32          # 슬라이스당 중간폭 (d=832 → 288, 합계 2304 = 원본과 동일)
        self.inter = inter
        self.gu_base = nn.Parameter(torch.randn(G, 2 * inter, dh) / math.sqrt(d))
        self.down_base = nn.Parameter(torch.zeros(G, dh, inter))

    # ---- LTLayer 호환 property (LT_Inner.kernel / step 이 그대로 읽는다)
    @property
    def theta(self):                                                             # [H,p,2] : θ^{(m)} = M_m θ
        return torch.einsum('mab,pb->mpa', self.mats, self.theta_base)
    @property
    def psi(self): return self.psi_base.expand(G, -1)
    @property
    def beta(self): return self.beta_base.expand(G, -1)
    @property
    def alpha(self): return F.softplus(self.alpha_base).expand(G, 1)
    @property
    def eta_raw(self): return self.eta_base.view(1, 1, 1).expand(G, 1, 1)
    @property
    def lam_raw(self): return self.lam_base.view(1, 1, 1).expand(G, 1, 1)
    @property
    def gain_raw(self): return self.gain_base.view(1, 1, 1).expand(G, 1, 1)
    @property
    def w_sh(self):                                                              # [H,dh,d] : 헤드 m 행 = G-conv 출력 슬라이스 m
        return _gconv_weight(self.sh_base, self.idx).reshape(G, self.dh, G * self.dh)
    def wc_full(self):
        return _gconv_weight(self.wc_base, self.idx)                             # [8·dh, d]


class D4Inner(LT_Inner):
    def __init__(self, config: LTConfig):
        nn.Module.__init__(self)
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g, d, H = config.seq_len, config.grid, config.hidden_size, config.num_heads
        assert T == g * g and H == G and d % H == 0 and (d // H) % 2 == 0
        assert not config.use_trace and config.num_layers == 1 and not config.legacy_gauge
        self.d, self.H = d, H
        self.dh = d // H; self.p = self.dh // 2
        u = torch.arange(T).float() // g; w = torch.arange(T).float() % g
        self.register_buffer("pos_u", u, persistent=False); self.register_buffer("pos_w", w, persistent=False)
        self.register_buffer("l1", (u[:, None] - u[None]).abs() + (w[:, None] - w[None]).abs(), persistent=False)
        self.embed = nn.Embedding(config.vocab_size, self.dh)                    # 자명표현: 슬라이스에 타일
        self.gamma = 1.0 / d
        self.embed_scale = math.sqrt(d)
        with torch.no_grad():
            trunc_normal_init_(self.embed.weight, std=1.0 / self.embed_scale)
        self.cls = nn.Linear(self.dh, config.vocab_size)                         # 불변 판독 (슬라이스 합)
        self.stdp, self.use_trace = config.stdp, False
        self.layers = nn.ModuleList([D4Layer(config, H, d, self.dh, self.p)])
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            assert config.puzzle_emb_ndim == d, "정칙표현 퍼즐 임베딩은 폭 d 전체를 쓴다"
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, d, batch_size=config.batch_size,
                                                    init_std=0, cast_to=self.forward_dtype)
        self.init_hidden = nn.Buffer(trunc_normal_init_(torch.empty(self.dh, dtype=self.forward_dtype), std=1.0).repeat(G),
                                     persistent=True)

    def W_C(self, L):
        """헤드 0 의 [dh,d] 를 QR 로 행직교화한 뒤, 헤드 m 은 열 슬라이스 순열 (Q_m[slice k'] = Q_0[slice m⁻¹k'])."""
        We = L.wc_full()[: self.dh]
        Q, _ = torch.linalg.qr(We.t())                                           # [d, dh]
        Qs = Q.reshape(G, self.dh, self.dh)                                      # [k', c, j]
        perm = torch.tensor([[d4.MUL[d4.INV[m], k] for k in range(G)] for m in range(G)], device=Q.device)   # [m, k'] = m⁻¹k'
        Qm = Qs[perm].reshape(G, G * self.dh, self.dh)                           # [m, d, dh]
        AB = Qm.transpose(-1, -2)                                                # [H, dh, d]
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def injection(self, batch):
        inj = self.embed(batch["inputs"].to(torch.long)).repeat(1, 1, G)         # [B,T,d]
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])                    # [B,d] = [B,8,dh] 정칙표현
            inj = inj + pe.to(inj.dtype).unsqueeze(1)
        return inj

    def boundary(self, L, h):
        gu = F.linear(h, _gconv_weight(L.gu_base, L.idx)).reshape(*h.shape[:-1], G, 2 * L.inter)
        g_, u_ = gu.chunk(2, dim=-1)
        mid = (0.5 * g_ * u_).reshape(*h.shape[:-1], G * L.inter)
        return h + F.linear(mid, _gconv_weight(L.down_base, L.idx))

    def w_cls(self, h):
        return self.cls(h.reshape(*h.shape[:-1], G, self.dh).sum(-2))


class D4LT(LT):
    """train.LT 와 같은 하네스 인터페이스, inner 만 D4Inner."""

    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = LTConfig.from_dict(config_dict)
        self.inner = D4Inner(self.config)
