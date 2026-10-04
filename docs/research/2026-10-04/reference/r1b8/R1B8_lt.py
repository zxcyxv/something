"""LT — LinearTuring 코어의 URM 하네스 어댑터.

구조 (SUDOKU.md 채택 표):
  세그먼트(하네스 loop) = [ 경계 재부호화 W_O(contract) → 경계 입력 주입 → R 마이크로스텝 ]
  세그먼트 사이 detach (하네스가 수행) → 그래디언트 노출 = 블록 1개 (e^{λτ} 문제의 구조적 해법)
  ACT 제거: halted = (steps >= loops) 결정론. q_head 는 손실 호환용 보조로만 존치.
  삼중 루프(loops×H×L) → 이중(loops × R) 축소. H_cycles 워밍업 없음.
  puzzle_emb: 토큰 앞붙임 대신 주입 시 브로드캐스트 가산 (T=81·2D 위치 보존; 스도쿠는 식별자 1개라 전역 학습 바이어스)
"""
import sys
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import pydantic

sys.path.insert(0, "/workspace/LinearTuring")
from model1 import Model1                                   # noqa: E402
from models.common import trunc_normal_init_
from models.layers import rms_norm                # noqa: E402
from models.sparse_embedding import CastedSparseEmbedding   # noqa: E402


@dataclass
class LTCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class LTConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")   # causal 등 하네스 주입 키 허용
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int = 0
    hidden_size: int
    num_heads: int
    R: int = 8                 # dt = 1/R (이산화 해상도)
    seg_steps: int = 0         # 세그먼트당 마이크로스텝 수 (0 = R, 즉 τ=1; 2R 이면 τ=2)
    loops: int = 16
    grid: int = 9
    ckpt: bool = True          # 마이크로스텝 gradient checkpointing (메모리 ÷R, 연산 ×~1.5)
    boundary_mlp: bool = False # 경계를 W_O 대신 g·h + SwiGLU(h) 로 — 내부 흐름은 순수 유지 (땜빵 검증용)
    sub_boundaries: int = 1    # [깊이연구] 세그먼트 내 접기 밀도 M: R/M 스텝마다 SwiGLU 경계 삽입 (τ 총량 불변)
    sub_norm: bool = False     # [깊이연구] 서브경계 직후 RMSNorm — 야코비안 이득을 O(1)로 고정 (URM 성분)
    sub_carry_init: float = -1.0  # [깊이연구] 서브경계 carry 초기값. <0 이면 0.6**(1/M) (세그먼트당 이득 보존)
    distinct_boundaries: bool = False  # [접기 다양성] 블록마다 자기 SwiGLU 경계 (비공유). 블록1=기존 b_gate_up/b_down,
                               # 블록 2..K = bb_gu/bb_dn[k]. down 영init → init 항등 (공유판과 비트 동치 출발)
    boundary_layers: int = 1   # [접기 다양성] 경계 SwiGLU 스택 층수 N. 층 2..N = 비공유·잔차(계수1)·down 영init
    b_carry_init: float = 1.0  # [규약 2026-08-23] 잔차 계수 = 1 고정. 경계 = h + SwiGLU(h) (init 항등).
                               # 0.6 은 CA 이월 관성값으로 폐기 — 경계당 ×c 는 K블록에서 그래디언트 지수손실 c^{K-1}
    inj_gate_init: float = 0.25  # [주입 결합] 주입 게이트 β init. 0.25 = carry-익사 보정 이월값 (세그당 1회 주입 기준).
                               # URM 은 게이트 없이 1.0 + embed_scale — 블록 주입 구조에선 재보정 대상.
    cont_inj: float = 0.0      # [지속주입] >0 이면 흐름에 상수 강제항 ḣ += β_c·e_x 추가 (β_c 학습, 이 값으로 init).
                               # 0차 항(b 와 동류)이라 세그 내 자율성·분석 보존. 깊이·BPTT 창 불변. 경계 주입은 그대로.
    lin_shared: bool = False   # [2026-08-24] 선형 경계 공유 1벌 (SwiGLU 공유 규약과 동형). 폭으로 파라미터 정합.
    lin_boundaries: bool = False  # [활성화 절제 2026-08-24] 경계 = 잔차 선형 h←h+W_k h (W_k 영init, 블록별 비공유).
                               # 원칙: 선형 파라미터는 A(h) 재계산 사이에만 산다 (인접 선형은 접혀 무효).
                               # 깊이 = 재주소화 횟수, 파라미터 = B·d² (깊이축 비례). 활성화 함수 0개.
    blocks_per_seg: int = 1    # [삼중루프] 세그당 [경계→τ=1 흐름] 블록 수 M. 전방 τ=M/세그, BPTT 창 M·R.
                               # 경계는 세그 시작 경계와 동일 연산자·가중치 공유 → M=1 = 현행과 동일, 추가 파라미터 0
    distinct_fields: bool = False  # [K-깊이] blocks_per_seg 블록마다 자기 (ψ,θ,α,W_sh,Λ) — "레이어 수 = 블록 수"
    block_inj: bool = False    # [주입 밀도] 블록마다 재주입 (URM 의 cycle-주입과 동형; 밀도가 K 에 불변)
    k_fields: int = 1          # [연산깊이] 구간별 벡터장 수 K: τ=1 을 K구간으로, 구간마다 자기 (ψ,θ,α,W_sh).
                               # 상태 연속·순수 흐름 보존·접기 무추가. 전 장 동일 init → init 에서 K=1 과 동치.
    mlp_expansion: float = 4.0
    forward_dtype: str = "float32"
    bilinear: bool = False     # [2026-08-26] 경계 게이트 silu 제거: h + W_d[(W_g h)⊙(W_u h)]·½ — 활성화 0개, 채널곱 유지.
                               # 근거: R1B8 추론에서 silu→x/2 치환 0.673/0.6995 (gate_swap.py). ½ 은 silu'(0) 정합(하이퍼 이월용)
    gate_quad: bool = False   # [2026-08-26] 게이트 = g/2 + g²/4 (silu 테일러 2차, 활성화 0개, 홀수 3차항 포함). 쌍선형(1차)은 벽(0.628@7812) — 우함수뿐
    amp: bool = False          # [2026-08-26] 세그먼트 forward 를 bf16 autocast (추론 정합 검증: 0.6996 vs 0.6995). 로짓은 fp32 로 반환


class LT_Inner(nn.Module):
    def __init__(self, config: LTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        T, g = config.seq_len, config.grid
        assert T == g * g
        pos = torch.stack([torch.arange(T).float() // g, torch.arange(T).float() % g], 1)
        # 채택 레시피: sheaf(W,Wᵀ) + Λ full + 경계 W_O(수축 init) + b 유지 + 커널 lr ×1(하네스 단일 lr)
        self.core = Model1(d=config.hidden_size, H=config.num_heads, R=config.R,
                           n_classes=config.vocab_size, positions=pos, vocab=config.vocab_size,
                           pool=False, sheaf=True, lam_mode="full",
                           boundary_wo=True, wo_mode="contract",   # w_bo 파라미터 확보용 — 적용은 여기서 직접
                           alpha_per_head=True)                    # 감쇠 헤드 단위 → 고속 경로 (등록된 편차)
        self.puzzle_emb_ndim = config.puzzle_emb_ndim
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0,
                                                    cast_to=self.forward_dtype)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        # 주입 게이트 — carry 익사 수선 (step 15624 프로브: ‖W_O h‖/‖inj‖=0.47 → β=0.25 로 ~1.9 반전)
        self.inj_gate = nn.Parameter(torch.tensor(float(config.inj_gate_init)))
        if config.boundary_mlp:
            # 경계 SwiGLU: down_proj 영 초기화 → 시작 시 항등성 유지(carry 익사 3회차 방지), 학습으로 성장
            d = config.hidden_size
            inter = int(config.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
            self.b_gate_up = nn.Linear(d, 2 * inter, bias=False)
            self.b_down = nn.Linear(inter, d, bias=False)
            with torch.no_grad():
                self.b_down.weight.zero_()
            self.b_carry = nn.Parameter(torch.tensor(float(config.b_carry_init)))
            if config.boundary_layers > 1:
                self.b_gu2 = nn.ModuleList([nn.Linear(d, 2 * inter, bias=False)
                                            for _ in range(config.boundary_layers - 1)])
                self.b_dn2 = nn.ModuleList([nn.Linear(inter, d, bias=False)
                                            for _ in range(config.boundary_layers - 1)])
                with torch.no_grad():
                    for m_ in self.b_dn2:
                        m_.weight.zero_()
        if config.lin_boundaries:
            assert not config.boundary_mlp and not config.distinct_boundaries, "축 혼합 금지"
            d_ = config.hidden_size
            self.lb_w = nn.ParameterList([nn.Parameter(torch.zeros(d_, d_))
                                          for _ in range(1 if config.lin_shared else config.blocks_per_seg)])
        if config.sub_boundaries > 1:
            assert config.boundary_mlp and config.R % config.sub_boundaries == 0, "sub_boundaries 는 boundary_mlp 필요, R 의 약수"
            ci = config.sub_carry_init if config.sub_carry_init > 0 else 0.6 ** (1.0 / config.sub_boundaries)
            self.sub_carry = nn.Parameter(torch.tensor(float(ci)))
        if config.cont_inj > 0:
            self.cinj_gate = nn.Parameter(torch.tensor(float(config.cont_inj)))
        if config.blocks_per_seg > 1:
            assert config.sub_boundaries == 1 and config.k_fields == 1, "축 혼합 금지"
        if config.distinct_boundaries:
            assert config.blocks_per_seg > 1 and config.boundary_mlp
            d_, inter_ = config.hidden_size, int(config.mlp_expansion * config.hidden_size * 2 / 3 + 255) // 256 * 256
            self.bb_gu = nn.ModuleList([nn.Linear(d_, 2 * inter_, bias=False)
                                        for _ in range(config.blocks_per_seg - 1)])
            self.bb_dn = nn.ModuleList([nn.Linear(inter_, d_, bias=False)
                                        for _ in range(config.blocks_per_seg - 1)])
            with torch.no_grad():
                for m_ in self.bb_dn:
                    m_.weight.zero_()
        if config.distinct_fields:
            assert config.blocks_per_seg > 1
            m0 = self.core
            mk = lambda t: nn.ParameterList([nn.Parameter(t.detach().clone())
                                             for _ in range(config.blocks_per_seg - 1)])
            # 이름에 wd 제외 키(psi/theta/alpha_raw)를 보존 — 옵티마이저 그룹 일관성
            self.bf_psi, self.bf_theta, self.bf_alpha_raw = mk(m0.psi), mk(m0.theta), mk(m0.alpha_raw)
            self.bf_wsh, self.bf_lam = mk(m0.w_sh), mk(m0.lam)
        if config.k_fields > 1:
            assert config.sub_boundaries == 1, "축 혼합 금지 (k_fields 는 서브경계와 별도 검증)"
            assert config.R % config.k_fields == 0
            m0 = self.core
            mk = lambda t: nn.ParameterList([nn.Parameter(t.detach().clone()) for _ in range(config.k_fields)])
            self.f_psi, self.f_theta = mk(m0.psi), mk(m0.theta)
            self.f_alpha, self.f_wsh = mk(m0.alpha_raw), mk(m0.w_sh)
        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True)

    def empty_carry(self, batch_size: int) -> LTCarry:
        return LTCarry(current_hidden=torch.empty(
            batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype))

    def reset_carry(self, reset_flag: torch.Tensor, carry: LTCarry) -> LTCarry:
        return replace(carry, current_hidden=torch.where(
            reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def _act(self, g):
        if self.config.gate_quad: return 0.5 * g + 0.25 * g * g
        return 0.5 * g if self.config.bilinear else F.silu(g)

    def _boundary(self, h):
        gate, up = self.b_gate_up(h).chunk(2, dim=-1)
        h = self.b_carry * h + self.b_down(self._act(gate) * up)
        if self.config.boundary_layers > 1:
            for gu, dn in zip(self.b_gu2, self.b_dn2):      # 층 2..N: h + SwiGLU_i(h)
                g2, u2 = gu(h).chunk(2, dim=-1)
                h = h + dn(self._act(g2) * u2)
        return h

    def _injection(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inj = self.core.embed(batch["inputs"].to(torch.long))            # [B,T,d]
        if self.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])            # [B,ndim]
            pad = self.config.hidden_size - self.puzzle_emb_ndim
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            inj = inj + pe.to(inj.dtype).unsqueeze(1)                    # 브로드캐스트 가산
        return inj

    def forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor]):
        if self.config.amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                new_carry, logits, q = self._forward(carry, batch)
            return replace(new_carry, current_hidden=new_carry.current_hidden.float()), logits.float(), (q[0].float(), q[1].float())
        return self._forward(carry, batch)

    def _forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor]):
        m = self.core
        h = carry.current_hidden
        # ── 경계: 재부호화 + 입력 주입 ─────────────────────────────
        if self.config.lin_boundaries:
            h = h + h @ self.lb_w[0].t()
        elif self.config.boundary_mlp:
            h = self._boundary(h)
        else:
            h = h @ m.w_bo.t()
        inj = self._injection(batch)
        h = h + self.inj_gate * inj
        # ── 내부 R 재귀 — 고속 경로: RoPE 화 위상 + 헤드 감쇠 마스크 (동치 검증됨) ──
        fc = m.kernel_fast(); AB = m.W_C(); dt = 1.0 / m.R
        nstep = self.config.seg_steps or self.config.R          # τ = nstep/R

        cg = getattr(self, "cinj_gate", None)

        def micro(hh):
            hh = m.phi(hh, dt / 2)
            f, *_ = m.field(hh, None, None, None, AB, fast_ctx=fc)
            hh = hh + dt * f if cg is None else hh + dt * (f + cg * inj)
            return m.phi(hh, dt / 2)

        K = self.config.k_fields
        if K > 1:
            # [연산깊이] 구간 k 에 장 k 바인딩. 상태는 연속 — 흐름을 자르지 않는다.
            orig = (m.psi, m.theta, m.alpha_raw, m.w_sh)
            try:
                assert nstep % K == 0
                for k in range(K):
                    m.psi, m.theta, m.alpha_raw, m.w_sh =                         self.f_psi[k], self.f_theta[k], self.f_alpha[k], self.f_wsh[k]
                    fck = m.kernel_fast()
                    def micro_k(hh, _fc=fck):
                        hh = m.phi(hh, dt / 2)
                        f, *_ = m.field(hh, None, None, None, AB, fast_ctx=_fc)
                        hh = hh + dt * f
                        return m.phi(hh, dt / 2)
                    for _ in range(nstep // K):
                        if self.config.ckpt and self.training and torch.is_grad_enabled():
                            h = checkpoint(micro_k, h, use_reentrant=False)
                        else:
                            h = micro_k(h)
            finally:
                m.psi, m.theta, m.alpha_raw, m.w_sh = orig   # named_parameters 안정 (EMA·저장)
        else:
            M = self.config.sub_boundaries
            for _i in range(nstep):
                if self.config.ckpt and self.training and torch.is_grad_enabled():
                    h = checkpoint(micro, h, use_reentrant=False)
                else:
                    h = micro(h)
                # [깊이연구] 서브경계: R/M 스텝마다 접기. 마지막 스텝 뒤는 제외 — 실전 1k 에서 사망 확인된 축
                if M > 1 and (_i + 1) % (self.config.R // M) == 0 and (_i + 1) < nstep:
                    gate, up = self.b_gate_up(h).chunk(2, dim=-1)
                    h = self.sub_carry * h + self.b_down(F.silu(gate) * up)
                    if self.config.sub_norm:
                        h = rms_norm(h, 1e-5)
        # [삼중루프] 블록 2..M: [경계 → (주입) → τ=1 흐름]. distinct_fields 면 블록별 장 스왑
        DF = self.config.distinct_fields
        if DF:
            _orig = (m.psi, m.theta, m.alpha_raw, m.w_sh, m.lam)
        try:
            for _blk in range(1, self.config.blocks_per_seg):
                if self.config.lin_boundaries:
                    h = h + h @ self.lb_w[0 if self.config.lin_shared else _blk].t()
                elif self.config.distinct_boundaries:   # 블록별 자기 접기 (계수 1 규약)
                    _g, _u = self.bb_gu[_blk - 1](h).chunk(2, dim=-1)
                    h = self.b_carry * h + self.bb_dn[_blk - 1](F.silu(_g) * _u)
                elif self.config.boundary_mlp:
                    h = self._boundary(h)
                else:
                    h = h @ m.w_bo.t()
                if self.config.block_inj:
                    h = h + self.inj_gate * inj
                if DF:
                    j = _blk - 1
                    m.psi, m.theta, m.alpha_raw, m.w_sh, m.lam = (
                        self.bf_psi[j], self.bf_theta[j], self.bf_alpha_raw[j],
                        self.bf_wsh[j], self.bf_lam[j])
                    _fcb = m.kernel_fast()
                    def stepfn(hh, _fc=_fcb):
                        hh = m.phi(hh, dt / 2)
                        f, *_ = m.field(hh, None, None, None, AB, fast_ctx=_fc)
                        hh = hh + dt * f if cg is None else hh + dt * (f + cg * inj)
                        return m.phi(hh, dt / 2)
                else:
                    stepfn = micro
                for _ in range(nstep):
                    if self.config.ckpt and self.training and torch.is_grad_enabled():
                        h = checkpoint(stepfn, h, use_reentrant=False)
                    else:
                        h = stepfn(h)
        finally:
            if DF:
                m.psi, m.theta, m.alpha_raw, m.w_sh, m.lam = _orig   # named_parameters 안정
        new_carry = replace(carry, current_hidden=h.detach())
        logits = m.w_cls(h)                                              # [B,T,vocab]
        q = self.q_head(h.mean(1)).to(torch.float32)
        return new_carry, logits, (q[..., 0], q[..., 1])


class LT(nn.Module):
    """URM 클래스 미러 — ACT 제거판 (halted = steps>=loops 결정론)."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LTConfig(**config_dict)
        self.inner = LT_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> LTCarry:
        batch_size = batch["inputs"].shape[0]
        return LTCarry(
            current_hidden=self.inner.empty_carry(batch_size).current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: LTCarry, batch: Dict[str, torch.Tensor], compute_target_q: bool = False):
        new_inner = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        new_inner, logits, (q_halt, q_cont) = self.inner(new_inner, new_current_data)
        outputs = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_cont}
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = new_steps >= self.config.loops        # ACT 없음 — 결정론적 소진
        return (
            LTCarry(current_hidden=new_inner.current_hidden, steps=new_steps,
                    halted=halted, current_data=new_current_data),
            outputs,
        )
