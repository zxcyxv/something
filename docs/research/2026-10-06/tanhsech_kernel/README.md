# 정확한 tanh·sech STDP 창의 fused 커널

2026-10-06, RTX 3090 (PyTorch 2.11.0+cu128, Triton 3.6.0).

자유위상 모델의 쓰기 `G[i,j] = mean_n V[n,i] K[n,j] L(phiV[n,i] - phiK[n,j])`에서
창을 근사 없이 `L(d) = tanh(d) sech(d)`로 두고 직접 계산한다. L은 홀함수이며
원점 기울기 1, peak 1/2(d=asinh 1), 꼬리 `2exp(-|d|)`다. 부호 판정이 없다.
양쪽으로 감쇠하는 창은 유한 분리 rank로 정확히 표현되지 않으므로 사인 특징
대신 pair 합을 [T,D,D] 텐서 없이 커널 안에서 누적한다.

## 수식 트릭

`r = e^d`, `p = r^2`이면 `L = 2r(p-1)/(p+1)^2`이고 r은 `e^{phiV} e^{-phiK}`로 분리된다.
토큰·채널별로 `A = V e^{phiV}`, `P = e^{2phiV}`, `B = K e^{-phiK}`, `Q = e^{-2phiK}`를
미리 계산하면 `V K L = A B h(PQ)`, `h(p) = 2(p-1)/(p+1)^2`, `h'(p) = 2(3-p)/(p+1)^3`이다.
pair마다 exp 없이 역수 1회(`rcp.approx`)와 FMA 몇 번이면 된다. 커널은 A, P, B, Q의
gradient를 내고 phase/K/V까지는 autograd가 O(TD) 사전 계산을 통해 연결한다.
phase는 모델에서 `(pi/2)tanh`로 제한되므로 `p+1`은 [1, e^{2pi}+1] 범위다.

- forward: (i,j) 타일에서 토큰 n을 순회하며 외적 누적. 타일 16x128, 1 warp.
- backward: 행/열 두 방향을 각각 한 커널로. [D,T] 전치 레이아웃의 (채널, 토큰)
  타일에서 상대편 채널을 순회하며 누적한다. atomic과 스레드 간 reduction이 없다.
  타일 32x32, 1 warp.
- `torch.library.triton_op`으로 등록해 `torch.compile(fullgraph)`에서 그래프가 끊기지
  않는다. compile은 `num_warps` 실행 인자를 무시하므로 단일 Config autotune으로 고정한다.

구현: [lt/tanhsech_phase_window.py](../../../../lt/tanhsech_phase_window.py),
모델 연결: `lt.experiment_free_phase_windows --window tanhsech`.

## 정확도

FP64 직접 pair 계산 대비 상대 L2 오차(값, K/V/위상 gradient 전체): 모델 shape
(81x104), 불규칙 shape(9x17), D=130(채널 타일 3개)에서 모두 4e-7 이하.
compile과 eager 결과 일치. 테스트 9개 통과([tests.log](tests.log)): 위 커널 검사,
유리식·도함수·홀함수·peak, 그리고 기존 자유위상 모델 테스트(초기화 동일성,
실제 모델 경로의 전체 파라미터 gradient, 두 residual RMSNorm 위치·수식)에 tanhsech 추가.

## 비용

연산 단위: batch128, 8heads, T81, D104, write+read forward+전체 backward,
compile fullgraph, FP32 matmul highest ([benchmark_rtx3090.json](benchmark_rtx3090.json)).

| 연산 | ms | 추가 peak MiB | FP64 대비 오차 |
| --- | ---: | ---: | ---: |
| 고정 지수 창 (공유 위상) | 2.13 | 249 | - |
| 사인 4성분 eps0.5 FP32 (학습한 모델) | 8.02 | 834 | 4.5e-7 (근사 창 대비) |
| 사인 4성분 BF16 특징 | 4.83 | 759 | 2.2e-3 (근사 창 대비) |
| tanh·sech 직접 pair (torch.compile) | 20.31 | 3,662 | 2.3e-7 |
| **tanh·sech fused Triton** | **7.24** | **536** | **3.7e-7** |

학습 스텝: 실제 trainer의 batch128, 8블록, BF16 본체/FP32 위상, compile,
activation checkpoint, diagonal 위상 생성기, 40스텝 사전 검증의 warmup 3스텝 제외
중앙값(preflight_*.json). 모든 위상 파라미터 gradient finite nonzero.

| 모델 | 초/step | peak MiB |
| --- | ---: | ---: |
| 고정 지수 창 | 0.1357 | 1,303 |
| **tanh·sech 정확, 자유위상** | **0.1890** | **2,701** |
| 사인 4성분 FP32, 자유위상 | 0.2017 | 7,331 |
| 사인 4성분 BF16 특징, 자유위상 | 0.1850 | 5,542 |

정확한 창이 학습한 FP32 사인 모델보다 6% 빠르고 BF16 근사와 2% 이내다.
메모리는 사인 FP32의 37%다. 같은 3090에서 고정 창보다는 39% 느리다.
이 3090의 절대 시간은 어제 RTX 4090 기록과 직접 비교하지 않는다.

## 학습 전에 정할 것

- 창 크기: tanh·sech peak는 1/2, 사인 대조군은 약 1.03. 정규화하지 않았다.
- 창 폭: peak 위치 0.88rad(사인 대조군 목표 창은 0.57rad).
- 위 비용은 연산·스텝 시간이며 학습 성능은 아직 측정하지 않았다.

재현:

```bash
python -m unittest lt.test_tanhsech_phase_window lt.test_free_phase_windows
python -m lt.benchmark_tanhsech_window --out /tmp/tanhsech_benchmark.json
python -m lt.experiment_free_phase_windows --window tanhsech --generator diagonal \
  --preflight --steps 40 --out runs/preflight_tanhsech
```

참고: 수식 분리는 Mamba-3의 데이터 의존 RoPE와 같은 원리(위상차의 지수를 양쪽으로
분리), 타일 누적과 backward 재계산은 FlashAttention, 수식 정의 행렬 축약은 KeOps,
초월함수 유닛 병목 대응은 FlashAttention-4의 FMA 에뮬레이션을 참고했다.

## 3,008스텝 학습 대조 (RTX 3090, seed0, diagonal 생성기)

훈련 지표는 2,512–3,008스텝의 16세그먼트 종료 배치 32개 평균이다. 사인 행은
2026-10-05 RTX 4090 기록(`continuation_audit/train_terminal_metrics.csv`, eval 이력)이다.
×1 런은 전체 일정으로 시작해 3,025스텝에서 SIGTERM으로 멈췄고, EMA/raw는
3,000스텝 체크포인트를 `lt.evaluate_free_phase_raw_ema`로 평가했다
(`runs/free_tanhsech_diag_full_20261006/diagnostics/`). ×2 런은 `--steps 3008`의
종료 평가다. 평가는 모두 held-out 2,048문제, 16segments×8blocks.

| 모델 | 훈련 칸 | 훈련 완전 | 훈련 loss | EMA@1953 칸 | EMA@3000/3008 칸 | EMA 완전 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 사인 4성분 eps0.5 | 92.57% | 15.67% | 0.3625 | 52.88% | 60.91% | 45/2048 |
| tanh·sech ×1 (peak 1/2, 기울기 1) | 91.21% | 13.35% | 0.3984 | 53.43% | 61.46% | 46/2048 |
| tanh·sech ×2 (peak 1, 기울기 2) | 91.61% | 12.23% | 0.3836 | 53.55% | 61.65% | 40/2048 |

×1의 3,000스텝 raw 평가는 칸 63.00%, 완전 39/2048이다. 단일 seed의 초기 구간이며
EMA(0.999)는 이 시점의 가중치를 늦게 반영하므로 세 창의 일반화 차이를 판정하지 않는다.
