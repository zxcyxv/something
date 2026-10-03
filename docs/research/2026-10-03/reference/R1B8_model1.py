"""
Model 1  —  "클로드 대화내용.txt" §1(개선된 순전파) + §2(양방향화) 사양 구현.

핵심 요약
  h_t^(0) = Embed(x_t)                                     (§1.2)
  z_t^(m) = W_C^(m) h_t,   ẑ = z / (||z||_2 + eps)         (§1.3) 진폭/위상 분리
  q̂ = e^{+iψ/2} ⊙ ẑ,  k̂ = e^{-iψ/2} ⊙ ẑ,  v^(m) = P_m h
  U_jj(t←n) = exp(-α_j ||Δ||_1 - i(θ_j^u Δu + θ_j^w Δw))   (§2.5) 2D 감쇠 Gabor 커널
  a_tn^(m)  = Σ_j Re[ conj(q̂_tj) U_jj(t←n) k̂_nj ]  ∈ [-1,1]
  f_t = Λ h_t + Σ_m W_OV^(m) Σ_n a_tn^(m) v_n^(m) + (Σ_m d_t^(m)) b     (§1.5)
  Strang 분할: Φ_{1/2R} ∘ (I + f/R) ∘ Φ_{1/2R},  Φ_τ(h)=h/√(1+2γ||h||²τ)  (§1.6)

양방향성은 §2.2의 두 스캔(S→ + S← − 자기항 중복)과 대수적으로 동일한
명시적 T×T 행렬로 계산한다 (Δ가 부호를 가지므로 exp(-α|Δ|₁ - iθ·Δ)가
전방/후방 상태합을 한 번에 준다; t=n은 정확히 한 번만 계산됨).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Model1(nn.Module):
    def __init__(self, d=64, H=4, R=8, grid=7, patch=4, n_classes=10,
                 eps=1e-4, gamma_init=0.1, alpha_init=0.1,
                 use_ov=True, lam_mode="full", orth_wc=True,
                 use_bias_v=True, learn_gamma=True, psi_zero=False, freeze_A=False,
                 positions=None, vocab=None, pool=True, boundary_wo=False, wo_mode='plain', sheaf=False,
                 alpha_per_head=False):
        super().__init__()
        assert d % H == 0
        self.d, self.H, self.R = d, H, R
        self.dh = d // H
        assert self.dh % 2 == 0
        self.p = self.dh // 2
        self.grid, self.patch, self.eps = grid, patch, eps
        self.use_ov, self.lam_mode, self.orth_wc = use_ov, lam_mode, orth_wc
        self.use_bias_v = use_bias_v
        # freeze_A: A 를 h^(0) 에서 한 번만 계산해 모든 재귀 깊이에서 재사용.
        # 그러면 f 가 h 에 대해 선형(+반경 감쇠)이 되어 카오스가 원리적으로 불가능하다.
        # 입력에 대한 비선형성은 A(x) 를 통해 그대로 남는다 (fast-weight 기계).
        self.freeze_A = freeze_A
        self.pool = pool          # False 면 토큰별 로짓 (셀룰러 오토마타처럼 조밀 지도용)
        # boundary_wo: τ 블록 경계(R스텝마다)에서만 채널 혼합 W_O 적용.
        # 블록 내부는 순수 noov 흐름 유지 → 확산 분해·에너지 해석이 블록 단위로 보존된다.
        # 항등 초기화 → 초기엔 기본 모델과 동일하게 시작.
        self.boundary_wo = boundary_wo
        self.wo_mode = wo_mode
        if boundary_wo:
            if wo_mode == 'plain':          # 비잔차 · 항등 init · 자유
                self.w_bo = nn.Parameter(torch.eye(d) + 0.01*torch.randn(d, d)/math.sqrt(d))
            elif wo_mode == 'residual':     # h ← h + W_O h · 영 init (초기 항등 사상)
                self.w_bo = nn.Parameter(0.01*torch.randn(d, d)/math.sqrt(d))
            elif wo_mode == 'orth':         # W_O = exp(S−Sᵀ) : 직교 → 흡수구 증명 보존
                self.w_bo_skew = nn.Parameter(0.01*torch.randn(d, d)/math.sqrt(d))
            elif wo_mode == 'contract':     # ρ·I init, ρ=e^{−λ̂}≈0.18 (λ̂≈1.7, 블록시간 1)
                self.w_bo = nn.Parameter(0.18*torch.eye(d) + 0.01*torch.randn(d, d)/math.sqrt(d))
            elif wo_mode == 'perhead':      # 헤드별 블록대각 d_h×d_h — 파라미터 1/H
                self.w_bo_blk = nn.Parameter(torch.eye(self.dh).repeat(H,1,1)
                                             + 0.01*torch.randn(H, self.dh, self.dh)/math.sqrt(self.dh))
            else: raise ValueError(wo_mode)
        T = grid * grid if positions is None else positions.shape[0]
        self.T = T

        # --- 임베딩 (§1.2) ---------------------------------------------------
        # vocab 이 주어지면 토큰 시퀀스 과제(1D), 아니면 MNIST 패치(2D).
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d) if vocab is not None else nn.Linear(patch * patch, d)

        # --- W_C^(m): C^{p×d}, 행-직교(Stiefel) 파라미터화 (§1.1/(d)) ------
        # [A;B] ∈ R^{d_h×d} 를 exp(skew) 의 상단 d_h 행으로 둔다 (추가 제약 불필요).
        if orth_wc:
            self.wc_skew = nn.Parameter(torch.randn(H, d, d) * (1.0 / math.sqrt(d)))
        else:
            self.wc_raw = nn.Parameter(torch.randn(H, self.dh, d) / math.sqrt(d))

        # --- 위상/주파수/감쇠 ----------------------------------------------
        # ψ init U(-π,π) : 0 근처 초기화 금지 (대칭 함정, §2.4)
        # psi_zero=True 면 ψ≡0 으로 고정 → A 대칭 → 실고유값 → gradient flow.
        # §2.4 의 "ψ 는 소산계/진동계를 가르는 유일한 파라미터" 주장의 직접 검증.
        self.psi_zero = psi_zero
        if psi_zero:
            self.register_buffer("psi", torch.zeros(H, self.p))
        else:
            self.psi = nn.Parameter(torch.rand(H, self.p) * 2 * math.pi - math.pi)
        # θ : 2D 파수 벡터 (§2.5). 나이퀴스트(≈π) 내부로 초기화.
        self.theta = nn.Parameter((torch.rand(H, self.p, 2) * 2 - 1) * (math.pi / 2))
        # α = softplus(α̃) > 0, init 0.1 (넓은 수용장)
        # alpha_per_head: 감쇠를 헤드 단위로 묶음 → e^{−α_h‖Δ‖} 가 [H,T,T] 마스크로 인수분해되어
        # 고속 경로(attn_fast: RoPE 화 + matmul)가 가능. θ 는 채널별 유지 (파수 다양성 보존).
        self.alpha_per_head = alpha_per_head
        self.alpha_raw = nn.Parameter(torch.full((H, 1 if alpha_per_head else self.p),
                                                 inv_softplus(alpha_init)))

        # --- 값 경로 -------------------------------------------------------
        if use_ov:
            self.w_ov = nn.Parameter(torch.randn(H, d, self.dh) / math.sqrt(self.dh))
        else:
            # W_OV 없음: 헤드 슬라이스를 제자리로 되돌리는 고정 블록 항등사상
            # (채널 믹싱 없음 → §2 "순수 좌곱셈" 판정용)
            self.register_buffer("w_ov_fixed", torch.eye(d).view(H, self.dh, d).transpose(1, 2).contiguous()
                                 if False else self._block_identity(d, H, self.dh))
        # sheaf 값 경로: v = W^(m)h, 출력 = W^(m)ᵀ Σ_n a v_n → 수송 = W^(m)ᵀW^(m) (대칭 PSD).
        # 블록 항등 init → 시작점이 정확히 noov. 학습이 '종(species) 기저'를 회전시키고
        # 헤드 간 기저 불일치가 채널 혼합을 만든다. 간선별 에너지 귀속이 혼합 아래서도 보존.
        self.sheaf = sheaf
        if sheaf:
            self.w_sh = nn.Parameter(self._block_identity(d, H, self.dh).transpose(1, 2).contiguous()
                                     + 0.01 * torch.randn(H, self.dh, d) / math.sqrt(d))
        # value bias b : 우함수(2차) 항 생성 (§1.5 마지막 항)
        self.b = nn.Parameter(torch.zeros(d)) if use_bias_v else None

        # --- 선형 흥분 Λ ---------------------------------------------------
        if lam_mode == "full":
            self.lam = nn.Parameter(torch.randn(d, d) * (0.5 / math.sqrt(d)))
        elif lam_mode == "diag":
            self.lam = nn.Parameter(torch.zeros(d))
        elif lam_mode == "none":
            self.lam = None
        else:
            raise ValueError(lam_mode)

        # --- 4차 소산 γ ----------------------------------------------------
        g0 = torch.tensor(inv_softplus(gamma_init))
        if learn_gamma:
            self.gamma_raw = nn.Parameter(g0)
        else:
            self.register_buffer("gamma_raw", g0)

        # --- 분류 헤드: 토큰별 로짓 후 평균 (§2.7 후자) --------------------
        self.w_cls = nn.Linear(d, n_classes)

        # --- 패치 좌표 & Δ 격자 (§2.5) --------------------------------------
        if positions is None:
            u = torch.arange(grid).repeat_interleave(grid).float()
            w = torch.arange(grid).repeat(grid).float()
        else:
            u, w = positions[:, 0].float(), positions[:, 1].float()
        self.register_buffer("du", u[:, None] - u[None, :])   # [T,T]
        self.register_buffer("dw", w[:, None] - w[None, :])
        self.register_buffer("l1", (u[:, None] - u[None, :]).abs() + (w[:, None] - w[None, :]).abs())
        self.register_buffer("pos_u", u)                       # [T] 원좌표 (고속 경로 위상용)
        self.register_buffer("pos_w", w)

    @staticmethod
    def _block_identity(d, H, dh):
        M = torch.zeros(H, d, dh)
        for m in range(H):
            M[m, m * dh:(m + 1) * dh, :] = torch.eye(dh)
        return M

    # ---------------------------------------------------------------- utils
    @property
    def gamma(self):
        return F.softplus(self.gamma_raw)

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def W_C(self):
        """returns A, B  (each [H,p,d]) with [A;B] row-orthonormal if orth_wc."""
        if self.orth_wc:
            S = self.wc_skew - self.wc_skew.transpose(-1, -2)
            Q = torch.matrix_exp(S)                    # [H,d,d] orthogonal
            AB = Q[:, :self.dh, :]                     # [H,dh,d] row-orthonormal
        else:
            AB = self.wc_raw
        return AB[:, :self.p, :], AB[:, self.p:, :]

    def OV(self):
        return self.w_ov if self.use_ov else self.w_ov_fixed

    def kernel(self):
        """U_jj(t←n) 의 감쇠/위상 부분.  decay,[H,p,T,T] ; Theta,[H,p,T,T]"""
        a = self.alpha[..., None, None]                       # [H,p,1,1]
        decay = torch.exp(-a * self.l1)                       # [H,p,T,T]
        Theta = (self.theta[..., 0, None, None] * self.du
                 + self.theta[..., 1, None, None] * self.dw)  # θ·Δ
        Theta = Theta + self.psi[..., None, None]             # ψ_j + θ·Δ
        return decay, Theta

    def kernel_fast(self):
        """고속 경로 사전량 (alpha_per_head 전용).
        decay_h [H,T,T] = e^{−α_h‖Δ‖₁},  위상각 A/B [T,H,p]:
          A_t = +ψ/2 + θ·pos_t  (q 쪽),   B_t = −ψ/2 + θ·pos_t  (k 쪽)
        유도: conj(q̂_t)·U(t←n)·k̂_n = conj(ẑ_t e^{iA_t}) · (ẑ_n e^{iB_n}) · e^{−α‖Δ‖}"""
        assert self.alpha_per_head
        decay_h = torch.exp(-self.alpha[:, 0, None, None] * self.l1)          # [H,T,T]
        ppos = (self.theta[..., 0, None] * self.pos_u                        # [H,p,T] = θ·pos_t
                + self.theta[..., 1, None] * self.pos_w)
        A = (ppos + self.psi[..., None] / 2).permute(2, 0, 1)                # [T,H,p]
        B = (ppos - self.psi[..., None] / 2).permute(2, 0, 1)
        return decay_h, torch.cos(A), torch.sin(A), torch.cos(B), torch.sin(B)

    def attn_fast(self, h, decay_h, cosA, sinA, cosB, sinB, AB=None):
        """a = decay_h ⊙ ( Q̃ₓK̃ₓᵀ + Q̃ᵧK̃ᵧᵀ ) — 거대 [B,T,H,p,T] 중간텐서 없음.
        Q̃ = ẑ 를 각 A_t 로 회전, K̃ = ẑ 를 각 B_t 로 회전.  느린 경로와 수학적으로 동일
        (alpha_per_head 제약 하에서), 수치 검증: tests 참조."""
        A, Bm = self.W_C() if AB is None else AB
        x = torch.einsum('btd,hjd->bthj', h, A)
        y = torch.einsum('btd,hjd->bthj', h, Bm)
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()
        x = x / (nrm + self.eps); y = y / (nrm + self.eps)
        qx = x * cosA - y * sinA; qy = x * sinA + y * cosA                    # e^{+iA_t} ẑ_t
        kx = x * cosB - y * sinB; ky = x * sinB + y * cosB                    # e^{+iB_n} ẑ_n
        # Re[conj(q̃_t) k̃_n] = qx_t·kx_n + qy_t·ky_n  → 헤드별 bmm 2회
        a = (torch.einsum('bthj,bnhj->bhtn', qx, kx)
             + torch.einsum('bthj,bnhj->bhtn', qy, ky))
        a = a * decay_h.unsqueeze(0)
        return a, x, y, nrm.squeeze(-1)

    # -------------------------------------------------------------- forward
    def attn(self, h, decay, Theta, AB=None):
        """a^(m)_{tn} : [B,H,T,T] , ẑ 성분도 반환 (해석가능성용).
        AB 를 주면 W_C 를 다시 만들지 않는다 — matrix_exp 는 h 에 무관하므로 forward 당 1회면 된다."""
        A, Bm = self.W_C() if AB is None else AB              # [H,p,d]
        x = torch.einsum('btd,hjd->bthj', h, A)               # Re z
        y = torch.einsum('btd,hjd->bthj', h, Bm)              # Im z
        nrm = (x.pow(2) + y.pow(2)).sum(-1, keepdim=True).sqrt()   # ||z||_2 [B,T,H,1]
        x = x / (nrm + self.eps)
        y = y / (nrm + self.eps)
        cosT, sinT = torch.cos(Theta), torch.sin(Theta)       # [H,p,T,T]
        # a_tn = Σ_j D_j[(x_t x_n + y_t y_n) cosΘ + (x_t y_n − y_t x_n) sinΘ]
        Dc, Ds = decay * cosT, decay * sinT
        a = (torch.einsum('bthj,bnhj,hjtn->bhtn', x, x, Dc)
             + torch.einsum('bthj,bnhj,hjtn->bhtn', y, y, Dc)
             + torch.einsum('bthj,bnhj,hjtn->bhtn', x, y, Ds)
             - torch.einsum('bthj,bnhj,hjtn->bhtn', y, x, Ds))
        return a, x, y, nrm.squeeze(-1)

    def field(self, h, decay, Theta, a_fixed=None, AB=None, fast_ctx=None):
        """벡터장 f_t (§1.5) 와 진단량.  a_fixed 를 주면 A 를 h 에서 다시 계산하지 않는다.
        fast_ctx = kernel_fast() 결과를 주면 고속 attn (alpha_per_head 전용)."""
        B, T, d = h.shape
        if a_fixed is None and fast_ctx is not None:
            a, zx, zy, znorm = self.attn_fast(h, *fast_ctx, AB=AB)
        elif a_fixed is None:
            a, zx, zy, znorm = self.attn(h, decay, Theta, AB)  # [B,H,T,T]
        else:
            a = a_fixed
            zx = zy = torch.zeros(B, T, self.H, self.p, device=h.device)
            znorm = torch.zeros(B, T, self.H, device=h.device)
        if getattr(self, 'sheaf', False):
            v = torch.einsum('btd,hcd->bthc', h, self.w_sh)    # v = W^(m) h
            o = torch.einsum('bhtn,bnhc->bthc', a, v)
            f = torch.einsum('bthc,hcd->btd', o, self.w_sh)    # 출력 = W^(m)ᵀ o
        else:
            v = h.view(B, T, self.H, self.dh)                  # P_m = 슬라이스
            o = torch.einsum('bhtn,bnhc->bthc', a, v)
            f = torch.einsum('bthc,hdc->btd', o, self.OV())
        dt = a.sum(-1)                                         # d_t^(m)  [B,H,T]
        if self.b is not None:
            f = f + dt.sum(1).unsqueeze(-1) * self.b
        if self.lam_mode == "full":
            f = f + h @ self.lam.t()
        elif self.lam_mode == "diag":
            f = f + h * self.lam
        return f, a, dt, zx, zy, znorm

    def phi(self, h, tau):
        """반응(4차 소산) 반스텝의 해석적 해 Φ_τ(h)=h/√(1+2γ||h||²τ)  (§1.6A)"""
        n2 = h.pow(2).sum(-1, keepdim=True)
        return h / torch.sqrt(1.0 + 2.0 * self.gamma * n2 * tau)

    def forward(self, x, R=None, steps=None, record=False, record_fields=None):
        """x: [B,1,28,28] → logits.

        R      : 적분 스텝 크기를 결정 (dt = 1/R). 학습 시의 값을 유지해야 한다.
        steps  : 재귀 횟수. 기본 = R (총 시간 τ=1). steps>R 이면 τ=steps/R 로
                 **시간 외삽**이 된다 (§3.4 의 R 외삽은 이쪽이다).
        """
        R = self.R if R is None else R
        steps = R if steps is None else steps
        dt = 1.0 / R
        h = self.embed_patches(x)
        decay, Theta = self.kernel()
        AB = self.W_C()                                         # 재귀 전에 1회
        a_fix = self.attn(h, decay, Theta, AB)[0] if self.freeze_A else None
        keys = ("h", "a", "d", "zphase", "znorm", "f") if record_fields is None else tuple(record_fields)
        traj = {k: [] for k in keys} if record else None
        if record and "h" in keys:
            traj["h"].append(h.detach())
        for r in range(steps):
            h = self.phi(h, dt / 2)
            f, a, dtm, zx, zy, znorm = self.field(h, decay, Theta, a_fix, AB)
            if record:
                if "a" in keys: traj["a"].append(a.detach())
                if "d" in keys: traj["d"].append(dtm.detach())
                if "zphase" in keys: traj["zphase"].append(torch.atan2(zy, zx).detach())
                if "znorm" in keys: traj["znorm"].append(znorm.detach())
                if "f" in keys: traj["f"].append(f.detach())
            h = h + dt * f
            h = self.phi(h, dt / 2)
            if self.boundary_wo and (r + 1) % R == 0 and (r + 1) < steps:
                if self.wo_mode == 'residual':
                    h = h + h @ self.w_bo.t()
                elif self.wo_mode == 'orth':
                    Q = torch.matrix_exp(self.w_bo_skew - self.w_bo_skew.t())
                    h = h @ Q.t()
                elif self.wo_mode == 'perhead':
                    B_, T_, _ = h.shape
                    h = torch.einsum('bthc,hdc->bthd', h.view(B_, T_, self.H, self.dh),
                                     self.w_bo_blk).reshape(B_, T_, self.d)
                else:
                    h = h @ self.w_bo.t()                   # plain / contract
            if record and "h" in keys:
                traj["h"].append(h.detach())
        logits = self.w_cls(h)
        logits = logits.mean(1) if self.pool else logits          # §2.7
        return (logits, traj) if record else logits

    def embed_patches(self, x):
        if self.vocab is not None:
            return self.embed(x)                      # x: [B,T] 정수 토큰
        B = x.shape[0]
        p, g = self.patch, self.grid
        xp = x.view(B, 1, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p)
        return self.embed(xp)

    # ------------------------------------------------------- 이론적 진단량
    @torch.no_grad()
    def kappa(self):
        """κ = max_m ||W_OV^(m)||_2 · Σ_Δ e^{-α_min||Δ||_1}   (§1.7 / §2.6)"""
        ov = self.OV()
        s = max(torch.linalg.matrix_norm(ov[m], ord=2).item() for m in range(self.H))
        amin = self.alpha.min().item()
        row = torch.exp(-amin * self.l1).sum(-1).max().item()   # max_t Σ_n e^{-α_min|Δ|}
        return s * row

    @torch.no_grad()
    def absorbing_radius(self):
        """M ≤ max(M(0), √((λmax(Λsym)+2κ)/γ))  (§1.7)"""
        if self.lam_mode == "full":
            L = 0.5 * (self.lam + self.lam.t())
            lmax = torch.linalg.eigvalsh(L).max().item()
        elif self.lam_mode == "diag":
            lmax = self.lam.max().item()
        else:
            lmax = 0.0
        k = self.kappa()
        g = self.gamma.item()
        return math.sqrt(max(lmax + 2 * k, 0.0) / g), lmax, k, g
