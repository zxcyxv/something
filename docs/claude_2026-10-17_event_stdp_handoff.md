# 2026-10-17 인수인계: 변화량 STDP 비판 검토와 Event-STDP 대조군

Claude가 이날 사용자와 진행한 검토·구현·학습 착수의 기록이다. 다음 세션의 AI가 이 문서와
[claude_2026-10-15_memory_and_rule110.md](claude_2026-10-15_memory_and_rule110.md)(직전 인수인계, 작업 방식 §1 포함)만 읽고 이어받을 수 있게 썼다.
모든 수치는 단일 seed다.

---

## 0. 한눈에 보기

**하루의 흐름**
1. 다른 AI의 비판: "k, v를 외적하는 것이 아니라 Δk, Δv를 외적해야 정확한 STDP다." 검토 결과, 표준 STDP 정의로는 틀린 주장이다(§1).
2. 그 AI의 자세한 근거인 상태 창과 변화 창의 쌍대성을 검증했다. 수학은 맞고, 지연 c·sin φ를 포함해도 성립한다(§2). Claude가 처음 "O에만 적용된다"고 한 것은 틀렸고 철회했다.
3. 그 AI가 제안한 대조군 **Event-STDP**(토닉 Hebb + 사건 STDP)를 `lt/`에 구현하고 테스트했다(§3).
4. B-EMA와 맞춘 설정으로 학습을 시작했다. 이 인스턴스(L4)에서는 357 ms/step으로 느려서, 사용자가 RTX 4090 서버를 새로 열어 **거기서 처음부터 다시 학습**하기로 했다(§4).

**다음 세션이 할 일**
1. 4090에서 `configs/b_ema_event.json`으로 학습한다(§4 명령). 결과는 약 1만 스텝 단위와 오류가 났을 때만 보고한다.
2. 결과를 B-EMA 8블록과 비교한다(§5 표가 기준).
3. 끝나면 `docs/evidence/run_b_ema_event/`에 로그를 보관하고, results.md와 architecture.md를 갱신한다(AGENTS.md 규칙).

---

## 1. 비판 1: "Δk ⊗ Δv가 정확한 STDP"

- 표준 쌍 기반 STDP(Bi & Poo 1998, Song 2000)는 사건 쌍에 `W(t_post − t_pre)`를 곱한다. 발화율 형태(Kempter 1999)도 발화율 자체의 상관이다. 변화량끼리 곱하라는 정의는 없다.
- 맞는 부분: 홀 창 STDP에 천천히 변하는 신호를 넣으면 1차 근사로 `x_pre · ẋ_post`가 된다(Rao & Sejnowski 2001, TD 형태). 이 모델에서는 이미 `O ≈ γ(Δv k̃ᵀ − v Δk̃ᵀ)`, γ = 0.662로 정리되어 있다(architecture §6). 여기서도 변화량은 한쪽뿐이다.
- 같은 시각의 Δv Δkᵀ(Kosko의 differential Hebbian)는 시간을 거꾸로 돌려도 값이 같아 순서를 표현하지 못한다. 단, 블록 간 지연 창을 건 변화량 쌍은 순서를 표현한다. 사용자와의 대화 중에 이 구분을 정정했다.

## 2. 비판 2: 상태 창과 변화 창의 쌍대성

**내용.** 고정 창 L(s−t)의 쌍선형 규칙에 부분합을 두 번 적용하면 다음이 성립한다.

    Σ L(s−t) v_s k_tᵀ = Σ G(s−t) Δv_s Δk_tᵀ,   L = −D²G,   G(g) = Σ_{Δ≥1} min(Δ, g) L(Δ)

- 이것은 **같은 규칙을 두 좌표로 쓴 항등식**이다. 어느 좌표가 "진짜 피연산자"인지는 말해 주지 않는다.
- 이 모델의 홀 창에서 G = 0.271, 0.459, 0.569, 0.625, 0.660, 0.664 (g = 1, 2, 3, 4, 6, 8)이다. G는 감쇠하지 않고 γ = 0.662(= m₁/2, m₁ = 1.325)에 머문다. 그 AI가 이 값을 "κ"라고 불렀는데, 저장소의 κ(1.221)와는 다른 양이다.
- **지연 포함**: `sin(ωΔ + ψ_V(s) − ψ_K(t)) = Im[e^{iωΔ} e^{iψ_V(s)} e^{−iψ_K(t)}]`이다. 창은 고정이고 지연은 흐름 쪽으로 들어가므로, 변환은 Im S 전체에 성립한다. 이때 변화 좌표의 "변화"는 Δ(v e^{iψ_V})이고, 값의 변화와 위상의 변화가 섞여 있다.
- **고정점에서 L = κ·SB**: 변화 좌표로 보면, 퍼즐 시작 사건(0에서 값이 생기는 순간)이 감쇠하지 않는 짝 창을 통해 자기 자신과 짝지어진 값이다. 정확도 경로가 변화 좌표에서는 가장 비국소적인 항인 셈이다.
- 검증 스크립트: [evidence/analysis_2026-10-17/duality/](evidence/analysis_2026-10-17/duality/)
  - `dual.py`: 창 값, G, A의 블록별 펄스
  - `dual2.py`: c = 0 쌍대성. 궤적 끝에서 창이 다 지나갈 때까지 멈춰 있어야 정확히 성립한다
  - `dual_c.py`: c = 0.14 지연 포함. 12자리까지 일치

**정리된 결론 (사용자와 합의한 표현)**

| 규칙 | 상태 표기 | 변화 표기 | 성질 |
|---|---|---|---|
| 지금 (A, B-EMA) | 상태 × L | 변화 × G | 인과의 방향(누가 먼저)은 있고, 근접성은 없다(G 포화). 표류에 반응하고, 고정점 기억 L이 있다 |
| Event-STDP | 상태 × (−D²L) | 변화 × L | 변화 간격에 국소적이다. 매끄러운 표류에는 반응하지 않고, 고정점에서 사건 기억이 0이다 |

- 두 행은 서로 다른 규칙이다. 등호는 각 행 안에서만 성립한다.
- 같은 계단 입력(K가 블록 20, V가 20+g에 켜짐)에서 지금 규칙은 G(g) = 0.27 … 0.66, Event 규칙은 L(g) = 0.083, 0.079, …, 0.001을 낸다.
- A는 현재 블록의 Im S만 읽으므로, 펄스의 정점 Σ_{Δ≤g} L도 g ≈ 4 이상에서 0.27로 포화한다. B-EMA는 약 8블록 창이라 G에 더 가깝다.
- Claude의 의견(미검증): 이 모델에서 원인이 상태처럼 계속 작용하는지(G가 합리적), 사건처럼 한 번 작용하는지(L이 합리적)는 이론으로 정해지지 않는다. 학습해서 비교해야 한다. 그래서 대조군을 돌린다.

## 3. Event-STDP 구현

명세는 다른 AI가 작성했다(사용자가 대화에 붙여 넣음). `event_stdp=True`일 때 메모리 단계만 바뀐다.

```text
e_x = x − x_prev  (x ∈ q̂, k̂, v; RoPE 전),  event_theta > 0이면 e = d·r²/(r² + θ²),  퍼즐 첫 블록은 e = 0
P   = Im S(e_V, e_K), τ = n·ω (지연 없음)       [+ γ_h Re S, event_even]
W   = (1−η)W + ηP  (또는 ρW + P, 또는 P)
read = α_h RoPE(e_Q) Wᵀ + q̃ (Σ_t v k̃ᵀ)ᵀ         (사건 읽기 + 상태의 토닉 Hebb 읽기)
```

| 위치 | 변경 |
|---|---|
| `lt/model.py` | `ModelConfig.event_stdp / event_theta / event_even`. 검증: `tag_bound == 0` 필수, Triton 커널 불가. `Carry.prev`(직전 q̂, k̂, v, [B,H,T,3D]) 추가. 메모리 상태 튜플이 5개로 늘었다 `(clock, E_K, E_V, W, prev)`. `LT.event_memory_step`, `LT.accumulate`(W 누적 공용). `stdp_step(..., real=True)`이면 Re S도 반환. n은 clock 텐서의 마지막 채널을 쓴다. 새 파라미터 `event_theta_raw` [2,H], `gamma_even` [H]는 난수를 쓰지 않는다 |
| `lt/train.py` | DEFAULTS와 RECIPE_KEYS에 세 키 추가. `event_theta_raw`는 weight decay에서 제외 |
| `tests/test_event.py` | α=0이면 Hebb-only와 FP64로 같음(초기 가중치 동일 포함). 변화가 없으면 사건 읽기 0, W는 감쇠만. 첫 블록 사건 0. 블록별 P의 합 = 변화 쌍의 명시적 합(Re 포함). 설정 검증. 새 파라미터에 기울기 도달 |
| `configs/b_ema_event.json` | 학습 설정(§4) |
| `kaggle/lt_cell.py` | 재생성 |

- 사건 STDP의 쌍 합은 기존 `stdp_step`에 (e_K, e_V, τ = nω)를 넣어 계산한다. 명세 6~11단계와 식이 같다.
- 기존 모드는 바뀌지 않았다(test_model 그대로 통과). 전체 21개 테스트가 통과했다(CPU FP64와 CUDA 커널 검사).
- 해석 도구의 probe(`OrderLatencyProbe` 등)는 Event 모드를 지원하지 않는다(`NotImplementedError`). `lt.analysis.extrapolate`는 일반 forward라 동작할 것이지만, 아직 실행해 보지 않았다.

## 4. 학습

**설정** (`configs/b_ema_event.json`, Claude가 B-EMA와 맞춰 고른 값)
- event_theta 0 (선형 사건, 파라미터 없음), event_even false (홀 창만), tag_bound 0
- outer_eta 0.125 (B-EMA와 같은 W 누적), alpha_init 1
- 세그먼트당 8블록, 50,800스텝, 1만 스텝마다 마일스톤과 128세그먼트 외삽, keep_last 6
- 파라미터 8,538,835개 (B-EMA와 같음)

**4090에서 실행**

```bash
git clone <repo> && cd something        # 또는 git pull
source /venv/main/bin/activate
python -m unittest discover tests       # 21개 통과 확인
mkdir -p runs/b_ema_event
nohup setsid python -m lt.train --config configs/b_ema_event.json --out runs/b_ema_event \
  > runs/b_ema_event/stdout.log 2>&1 < /dev/null &
grep -E "^\[(EVAL|EXTRAP)\]" runs/b_ema_event/stdout.log
```

- 같은 `--out`으로 다시 실행하면 재개한다. 학습을 멈추려면 SIGTERM을 보낸다(저장한 뒤 종료).
- AGENTS.md: 새 실험 전에 직전 실험의 `step_*.pt`를 지운다(새 서버라면 없다). 체크포인트와 `runs/`는 커밋하지 않는다.

**L4 인스턴스(이 세션)의 상태**
- `runs/b_ema_event_20261017/`에서 같은 설정으로 학습을 시작했다(357 ms/step, torch.compile 켬). 4090에서 다시 돌리므로 이 런은 버린다. 인스턴스가 사라지면 함께 없어진다.
- 초반 학습 손실(50스텝 평균; 배치 내 적응이 섞여 비교 근거는 약함): step 100 2.200 (B-EMA 2.279), 200 1.747 (1.787), 300 1.447 (1.464), 400 1.336 (1.335).
- 속도 차이는 GPU 차이로 본다(사용자 판단, 환경 비교 측정은 하지 않기로 함). 메모리 단계가 FP32(autocast 끔)로 돌고, L4는 FP32 약 30 TFLOPS, 300 GB/s, 4090은 82.6 TFLOPS, 1,008 GB/s다.
- 주의: 이 세션에서 L4의 B-EMA 속도를 재려다 학습을 일시정지하지 못한 채(SIGSTOP이 setsid 프로세스 그룹이 아닌 셸에 감) 두 학습이 겹쳐 돌았다. 그때 나온 858 ms/step은 무효다. 4090에서 Event-STDP의 ms/step을 B-EMA의 125 ms/step과 비교하면 된다.

## 5. 비교 기준: B-EMA 8블록 (η 0.125, 4090, 125 ms/step)

| step | B-EMA | A |
| ---: | ---: | ---: |
| 1,953 | 53.8% / 0 | 54.3% / 0 |
| 3,906 | 65.5% / 108 | 65.5% / 109 |
| 5,859 | 67.7% / 268 | 67.1% / 222 |
| 9,765 | 68.9% / 357 | 69.4% / 341 |
| 19,530 | 72.9% / 441 | 72.7% / 438 |
| 25,389 | 77.8% / 707 | 78.0% / 722 |
| 39,060 | 81.6% / 1,007 | 82.3% / 1,038 |
| 50,778 | 83.1% / 1,090 | 83.4% / 1,087 |

외삽(앞 512문제, seg16 → 128세그먼트까지 최고): 10k 104 → 106, 20k 137 → 145, 30k 212 → 227, 40k 249 → 274, 50k 259 → 287 (11.1%).
원 로그는 `evidence/run_b_ema/eval_lines.log`, 학습 손실은 `evidence/run_b_ema/train.jsonl.gz`에 있다. 참고로 Hebb를 뺀 런(`evidence/run_a_no_hebb`)은 9,765스텝에 66.6% / 184였다.

**결과를 읽는 법 (사전에 정해 둔 것)**
- Event-STDP가 B-EMA와 같거나 나으면: 상태 STDP의 비국소 창 G와 고정점 기억 L이 필수는 아니다. 사건 쪽 설계를 계속 검토한다.
- 낮으면: 상태 창이 하는 일(표류 반응 O, 고정점 기억 L)이 쓸모 있다는 증거다. 단, 단일 seed이고 제거 실험과 같은 종류의 결과이므로 "그것과 등가인 것이 무엇인지"까지 설명해야 한다(사용자 기준).
- 외삽은 따로 본다. 사건 읽기는 고정점에서 0이므로, 늦은 국면에는 Hebb만 남는다.

## 6. 미결

1. 그 AI가 처음 제안한 대조군인 **차분 항** `α(v_n k_{n−1}ᵀ − v_{n−1} k_nᵀ)`(O의 1차 근사)는 아직 구현하지 않았다. Event-STDP와는 다른 대조군이다.
2. 학습 없이 할 수 있는 측정: B-EMA 50k에서 블록마다 O와 γ(Δv k̃ᵀ − v Δk̃ᵀ)의 코사인과 노름비. O가 실제 궤적에서도 1차 근사로 맞는지 확인한다.
3. Event 변형: event_theta > 0(소프트 문턱), event_even(공변화 창), outer_eta 0(누적 없음).
4. 직전 인수인계 §7의 후보(Rule 110 고정점 학습, Kaggle B-EMA 16블록 결과, 상태 의존 η)는 그대로 유효하다.
