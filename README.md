# LinearTuring

활성화 함수 없는 재귀 모델로 스도쿠를 푼다. 토큰(칸) 사이의 결합은 **위상차 커널**로 정하고,
그 결합을 블록 반복에 걸쳐 누적하는 **결합 기억 `w`** 가 있다. 학습은 16 세그먼트(= 128 블록)까지만 하고,
추론에서 세그먼트를 더 돌리면 정답률이 계속 오른다. 이 외삽이 이 모델이 보이려는 것이다.

```
                      학습 지평 seg16      seg128 까지 돌리면       오차 감소율
v1    (310k)          240 / 512  46.9%     386 / 512  75.4%          53.7%
v1.1  (160k, 최신)    234 / 512  45.7%     338 / 512  66.0%          37.4%   (130k 시점엔 53.4%)
```
held-out 512 퍼즐. 오차 감소율 = (seg16 오차 − seg128 최고 오차) / seg16 오차.

## 판(preset)의 계보

| preset | 재주입 게이지 | 블록 순서 | 주소 흔적 z | 상태 |
|---|---|---|---|---|
| `v1` | 학습 스칼라 (실효 ≈7) | pre | 없음 | 9/1 원본. 외삽 최고. 후반에 단서를 바꾼 오답이 생김 |
| `v1.1` | √d 고정 | pre | 없음 | **최신 학습판.** 단서 불변, 같은 스텝에서 v1 보다 빠르게 정답에 도달 |
| `v2` | √d 고정 | post | 없음 | **아직 학습하지 않았다.** v1.1 에 v1.7 의 블록 순서만 적용한 것 |
| `v1.7` | √d 고정 | post | 있음 | v2 + 흔적. 학습 후반에 불안정 (아래 결과) |

숫자 이름은 만든 순서가 아니다. v1.7 은 v1.1 보다 먼저 만들었고, v2 는 v1.7 에서 흔적을 뺀 것을 v1.1 다음 실험으로 정의한 것이다.
세 판의 차이는 `lt/train.py` 의 `PRESETS` 세 플래그가 전부다.

## 모델

칸 `t = 1..81`, 상태 `h_t ∈ R^832`, 헤드 8개, 헤드마다 복소 주소 성분 `j = 1..52`. 모든 블록이 같은 가중치를 쓴다.
한 블록(pre 순서):

```
(B) 경계   h_t ← h_t + W_d[ ½ (W_g h_t) ⊙ (W_u h_t) ]         칸 안에서만 도는 쌍선형 항. 이 모델의 유일한 비선형 혼합
(I) 주입   h_t ← h_t + s · E(x_t)                              입력 칸 x_t 를 매 블록 다시 넣는다.  s = √d (v1.1/v2/v1.7), 학습 스칼라 (v1)
(S) 스텝
    u_t = W_C h_t ∈ C^52 ,   û_t = u_t / ‖u_t‖                  복소 주소. W_C 는 행직교. 크기를 버리고 방향(위상)만 쓴다
    a_tn   = D(Δ_tn) · Σ_j Re[ û_t,j conj(û_n,j) · e^{i(θ_j·Δ_tn + ψ_j)} ]      읽기 커널: 위상차 + 위치 회전 + 상수 위상 ψ
    a^β_tn = 같은 식, ψ 대신 β                                    쓰기 창 (v1.7 만 û 대신 흔적 ẑ)
    v_t = W_sh h_t ,   agree_tn = ⟨v̂_t, v̂_n⟩                     값 사영과 값공간 코사인
    w ← (1−η) w + η · g · a^β ⊙ agree                            결합 기억. 블록·세그먼트를 넘어 이월된다
    a_eff = (1−λ) a + λ w
    h_t ← Φ( h_t + W_shᵀ Σ_n a_eff,tn v_n ) ,   Φ(h) = h / √(1 + γ‖h‖²)
```
`Δ_tn` 은 두 칸의 격자 위치 차, `D(Δ) = e^{−α‖Δ‖₁}` 는 거리 감쇠, `η·λ·g·α` 는 헤드별 학습 스칼라, `ψ_j·β_j·θ_j` 는 성분별 학습.
`a` 는 부호가 있다 — 같은 행의 두 칸은 밀고, 서로 정보를 줄 칸은 당긴다. softmax 도 정규화도 없다.

post 순서(v2·v1.7)는 주입 → 스텝(Φ 없이) → 경계 → Φ. 같은 연산 집합에서 위상만 다르다.

`w` 가 이 모델의 요점이다. 매 블록 새로 계산되는 순간 결합 `a` 와 달리 `w` 는 `a^β ⊙ agree` 를 시정수 `1/η` (학습 결과 수십 블록)로
누적한다. 부호가 바뀌는 쓰기는 EMA에서 감쇠되지만, 위상차의 방향 반전만으로 상쇄되는 것은 아니다.
고정 η의 EMA가 기억하는 유효 시간 범위도 무한히 늘어나지는 않는다. 외삽 개선은 이 기억과 활동이 함께
변하는 과정에 대한 가설이다. `docs/stdp_fourier.md`는 주파수 표현의 항등식과 추가 가정을 구분한다.
위상과 내용의 분리 가능한 집단 활동을 가정하면 `agree`까지 하나의 상관으로 구성할 수 있으며,
그 유도와 연구용 후보는 아래 문서에 있다.

### 2026-09-13 이론 연구

- [위상 집단 활동과 가소성 창](docs/phase_population_v11.md): 활동 모델의 가정, 커널 유도, 검산.
- [짝수 가소성 창 후보](docs/even_plasticity_candidate.md): ψ와 두 사영을 유지한 구조, 학습 좌표의 가설, 작은 비교.
- [실제 가소성 되먹임](docs/plastic_feedback_derivative_v11.md): 활동→쓰기→다음 활동의 국소 미분.

연구용 구현은 `lt/even_plasticity.py`, 학습 진입점은 `lt/train_even.py`다.
기본 모델은 계속 `lt/train.py`의 v1.1이며, 후보의 본 규모 재학습 우위는 아직 검증하지 않았다.

학습 하네스는 URM(arXiv 2512.14693)을 옮긴 것이다: 세그먼트마다 `h, w` 를 detach, 세그먼트마다 감독, AdamATan2, EMA 0.999,
배치 128, lr 1e-4 상수, 데이터는 Sudoku-Extreme 1k 퍼즐 × 증강 1000.

## 결과

### 오차 감소율 (held-out 512, seg128, EMA 가중치)

| step | v1 | v1.1 | v1.7 |
|---|---|---|---|
| 40k | — | 16.6 | — |
| 60k | 13.8 | 18.0 | — |
| 100k | — | 49.0 | 8.9 |
| 120k | 32.9 | 39.0 | ≈20 |
| 130k | 36.1 | **53.4** | 21.4 |
| 140k | 33.2 | 46.4 | 31.1 |
| 150k | 34.8 | 44.4 | — |
| 160k | 47.2 | 37.4 | 8.2 |
| 180k | — | — | 26.4 |
| 200k | — | — | **45.9** |
| 300k | — | — | 30.3 |
| 310k | **53.7** | — | — |
| 378k | — | — | 18.9 |

v1 의 120k~160k 는 재현 런의 120k 체크포인트에서 재개해 얻었다. v1.1 은 160k 에서 멈췄다. v1.7 은 200k 에서 정점을 찍고 내려온다.

### 같은 스텝에서 v1 과 v1.1

seg128 에서 퍼즐을 세 상태로 나눈다 — C 정답, V 무모순인데 오답(행·열·박스 제약은 다 맞지만 정답과 다름), I 모순.
I→C 는 세그먼트당 모순→정답으로 넘어가는 확률.

| step | 판 | C / V / I | I→C |
|---|---|---|---|
| 130k | v1.1 | 377 / 0 / 135 | 0.0143 |
| 130k | v1 | 308 / 0 / 204 | 0.0095 |
| 140k | v1.1 | 356 / 0 / 156 | 0.0129 |
| 140k | v1 | 311 / 1 / 200 | 0.0095 |
| 150k | v1.1 | 349 / 0 / 163 | 0.0122 |
| 150k | v1 | 326 / 1 / 185 | 0.0105 |
| 310k | v1 | 386 / 107 / 19 | 0.0210 |

- v1 은 후반에 **단서 하나를 바꾼 무모순 격자**로 정착하는 오답(V)이 자란다 (310k 에서 107/512, 그중 106 이 단서 1개 변경).
  V 는 종착 상태다 — V 에서 C 로 돌아간 경우는 5,799 세그먼트 중 0 이었고, 정답은 전부 I 에서 직접 온다.
- √d 고정 게이지(v1.1·v1.7)에서는 V 가 0 이다. 재주입이 세서 단서를 덮어쓸 수 없다.
- v1.1 은 같은 스텝에서 v1 보다 I→C 가 1.2~1.5배 높다. 이 전이는 단서를 지킨 채 일어난다 (v1 의 I→C 375회 중 직전 세그먼트에 단서 위반 0).

### v1.7 의 불안정

| | v1.1 (120k~160k) | v1.7 (105k~380k) |
|---|---|---|
| seg16 정확도(/2048) 변동계수 | 3.6% | 20.4% |
| 급락 | 없음 | 7회 (328k 에 118/2048) |
| 풀었던 퍼즐을 다시 깨는 전이 C→I | 0 | 380k 에서 세그당 0.0054 |

흔적 `z` 자체는 노름·기울기 면에서 정상이다 (‖z‖/‖u‖ ≈ 1.25 로 포화, 흔적 파라미터의 기울기는 전체의 1~2%). 불안정의 원인은 특정하지 못했다.
v2 는 이 질문을 위한 것이다 — v1.7 에서 흔적만 뺀 판이 v1.1 처럼 안정적이면 원인은 흔적이고, 아니면 블록 순서다.

## 재현

```bash
pip install torch numpy                      # CUDA GPU 한 장. A4000 에서 약 6 it/s, 160k 스텝에 ~7 시간

python lt/train.py --preset v1.1             # runs/v1_1/ 에 저장. 10k 스텝마다 마일스톤 + seg128 외삽 (milestones/extrap_step_N.txt)
python lt/train.py --preset v2               # 미학습 판
python lt/train.py --preset v1.1 --resume_from runs/v1_1/step_160000.pt

LT_CKPT=checkpoints/v1.1_step160000.npz python lt/extrapolate.py         # 세그먼트별 정확도 · 완답 · churn · 감소율
LT_CKPT=checkpoints/v1.1_step160000.npz python lt/count_valid_grids.py   # 세그먼트별 C / V / I
LT_CKPT=checkpoints/v1.1_step160000.npz LT_SAVE=traj.npz python lt/dump_trajectory.py
python lt/state_transitions.py traj.npz      # C/V/I 전이 확률, seg16→seg128, V 경유 여부
python lt/violation_trajectory.py traj.npz   # I 퍼즐의 위반 수 궤적, 후반 churn, 동결률
python lt/phase_oscillation.py && python lt/phase_oscillation_vs_difficulty.py    # 칸 위상 진동 vs 난이도 (v1)
```

`lt/train.py --selftest` 는 GPU 없이 증강 규칙을 원본과 대조한다. 데이터는 `data/prep_dataset.py` 로 다시 만들 수 있다.

새코드3의 스도쿠 학습기는 `python -m lt.train_new3 --config configs/new3_sudoku.json`으로 실행한다.
기본값은 **1세그먼트 × 8블록**, 배치 128이며 기존 데이터·증강을 유지한다. 설정, 재개 및 외삽 평가는 [학습 안내](docs/new3_training.md)를 참고한다.
Expert 8개를 모든 토큰에 순서대로 적용하고 마지막에 토큰별 expert를 선택하는 실험은
`python -m lt.train_new3 --config configs/new3_sudoku_sequential_experts.json`으로 실행한다.
이 설정은 **1세그먼트 × 9블록**, 배치 128, 20,000스텝이며 별도 출력 경로를 사용한다.

## 체크포인트

| 파일 | 판 | 내용 |
|---|---|---|
| `checkpoints/v1_step310527.npz` | v1 | 9/1 원본. raw + EMA, fp32 |
| `checkpoints/v1_repro_step120000.npz` | v1 | 재현 런 120k. raw + EMA, fp32. 위 v1 120k~160k 행의 출발점 |
| `checkpoints/v1.1_step160000.npz` | v1.1 | EMA, fp16 (완답 수 ±1% 흔들림) |
| `checkpoints/v1.7_step200000.npz` | v1.7 | EMA, fp16 |

`lt/ckpt_npz.py` 가 `.pt ↔ .npz` 변환과 로딩(`load_lt`)을 맡는다.

## 파일

```
lt/train.py                         모델 + 학습 하네스 (하나의 파일)
lt/extrapolate.py                   외삽 표
lt/count_valid_grids.py             무모순 격자 수
lt/dump_trajectory.py               세그먼트별 예측 저장
lt/state_transitions.py             C/V/I 전이
lt/violation_trajectory.py          위반 수 궤적
lt/phase_oscillation.py, _vs_difficulty.py   칸 위상 진동
lt/sudoku_solver.py                 제약전파 솔버 (난이도 기준)
lt/ckpt_npz.py                      체크포인트 입출력
data/                               데이터와 빌드 스크립트
docs/stdp_fourier.md                이론 문서 하나
```

라이선스는 아직 정하지 않았다.
