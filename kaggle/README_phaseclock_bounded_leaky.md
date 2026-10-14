# 위상 시계 STDP 모델 B: Kaggle 단일 셀

파일: [phaseclock_bounded_leaky_cell.py](phaseclock_bounded_leaky_cell.py). 이 파일 전체를 Kaggle 노트북 셀 하나에 붙여넣고 실행한다.
저장소 import는 없다. 하네스는 `lt/train.py`(데이터·증강 풀·stablemax 손실·AdamATan2·EMA·체크포인트·평가·milestone·torchrun 실행기)를
그대로 옮겼다. 모델은 `lt/research_phase_clock_stdp.py`의 3런(`--clock displacement`, torch 경로)에 아래 세 가지를 바꾼 것이다.

## B의 정의

3런과 같은 부분: 한 층 공유, 블록마다 `h ← h + √d·inj`, `q, k, v = heads(W h)`, `q̂, k̂ = x/(‖x‖₂+eps)`, 학습형 2D RoPE,
V는 원시값, 토큰 합 쓰기, `h ← RMS(h + W_o read)`, `h ← RMS(u + D(silu(g)·v))`(SwiGLU), α_h 학습(초기 1).

```text
φ_r  = 3런 변위 태그: φ_r ← wrap(φ_r + wrap(θ_r(지금) − θ_r(직전)))     θ_K = atan2(k̂ 쌍, RoPE 전), θ_V = atan2(v 쌍)
n    = 이 레인의 퍼즐이 시작된 뒤 돈 블록 수 (첫 블록 0, 세그먼트를 넘어 유지, 새 퍼즐이면 0)
1. 태그:   tag_r = c·sin(φ_r) + n·ωT            c = 0.14, ωT = 0.34, r = K, V
2. 흔적:   λ_K = λ_V = 0.5 고정 (clock_lam_raw 동결, 기울기 없음)
           E_K ← λE_K + (1−λ)·k̂·e^{−i tag_K};  D_V = λE_V;  E_V ← D_V + (1−λ)·v·e^{+i tag_V}
           Im S_b = Im Σ_n [ (v e^{i tag_V})ᵀ RoPE(E_K) + D_Vᵀ (kr e^{−i tag_K}) ]      (3런의 phase_clock_step 그대로)
3. 누설 적분: W_b = ρ·W_{b−1} + Im S_b,  ρ = 0.5
           read = qr · (Σ_n v krᵀ + α_h W_b)ᵀ        (3런: Σ_n v krᵀ + α_h Im S_b)
```

같은 블록 쌍은 지연차 `c(sin φ_V − sin φ_K)`(|·| ≤ 2c)만 보고, m블록 떨어진 쌍은 여기에 `±m·ωT`가 더해진다.
W는 블록과 세그먼트를 넘어 이월되고(세그먼트 경계에서 detach), 새 퍼즐 레인에서는 0이다. 세그먼트 안에서는 누적을 통해 역전파한다.

carry (모두 FP32, 새 퍼즐이면 0, 세그먼트 경계에서 detach, 학습 체크포인트에 저장):

| 슬롯 | 내용 | 모양 |
| --- | --- | --- |
| `coupling` | `cat(φ_K, φ_V, z_K_prev, z_V_prev, n)` — 3런 시계 상태 + 블록 카운터 1채널 | [B,H,T,6P+1] |
| `key_trace`, `value_trace` | `cat(Re E, Im E)` (K는 RoPE 전) | [B,H,T,2D] |
| `stdp_w` (LTCarry에 새로 추가한 필드) | W | [B,H,D,D] |

파라미터 8,541,747개(3런과 같다. 학습 대상은 그중 `clock_lam_raw` 832개를 뺀 8,540,915개). 같은 seed에서 초기 가중치는
3런과 비트 단위로 같고 `clock_lam_raw`만 다르다(λ 0.1 → 0.5). 상수 c, ωT, λ, ρ는 맨 위 `CFG`의 `b_tag_bound`, `b_block_omega`,
`b_trace_lambda`, `b_outer_rho`이며 재개 검사 대상이다(값이 다른 체크포인트에서는 재개를 거부한다).

## Kaggle에서 실행

1. 노트북에 `sudoku_lt_1k.npz` 데이터셋을 Input으로 연결한다. 기본 경로는
   `/kaggle/input/datasets/jrjinwoo/sudoku-lt-1k/sudoku_lt_1k.npz`이고, 없으면 `/kaggle/input/**/sudoku_lt_1k.npz`,
   `./data/`, 현재 폴더를 차례로 찾는다(로컬에서는 저장소의 `data/sudoku_lt_1k.npz`).
2. 파일 전체를 셀 하나에 붙여넣고 실행한다. 학습은 별도 프로세스에서 돈다(GPU가 둘이면 torchrun, 배치 128을 rank당 64로 나눈다).
3. 출력은 `/kaggle/working/lt_phaseclock_bounded_leaky`. 같은 세션에서 셀을 다시 실행하면 그 폴더의 마지막 `step_*.pt`에서 이어 간다.
   세션을 바꿀 때는 이전 출력을 Input으로 연결하고 `CFG['resume_from']`에 그 체크포인트 파일(또는 폴더)을 지정한다.
   `max_hours=11.5`에 도달하면 완료된 스텝 경계에서 저장하고 멈춘다. `max_steps=50800`은 절대 종료 스텝이다.
4. 자체검사만 하려면 `CFG['selftest_only'] = True`. 로컬에서는 `python kaggle/phaseclock_bounded_leaky_cell.py --selftest`,
   학습은 `--config run.json`(CFG 키를 덮어쓰는 JSON).

| 설정 | 값 |
| --- | --- |
| 학습 | 배치 128, 16세그먼트 × 8블록, 블록 전부 역전파(no-grad 0), activation checkpoint 없음 |
| 최적화 | AdamATan2, lr 1e-4(warmup 2,000 후 상수), weight decay 1.0, β = (0.9, 0.95), EMA 0.999 |
| 평가 | 250에폭(1,953스텝)마다 held-out 2,048, EMA, 새 carry 16세그먼트 |
| milestone | 10,000스텝마다 체크포인트 보관 + 앞 512문제 128세그먼트 외삽표 |
| 저장 | 2,000스텝마다 + 평가 경계, 최근 2개 유지(milestone은 별도 보관) |
| 정밀도 | `amp_dtype='auto'`: native BF16 GPU(Ampere 이상)는 BF16 autocast, T4·P100은 FP32. FP16은 쓰지 않는다. 시계·흔적·W·읽기는 항상 FP32 |
| compile | `'auto'`: CUDA capability ≥ 7.0이면 `torch.compile`(T4 예), P100은 eager |

## 출력

- `train.log`: 셀 출력 전체. 시작 시 자체검사 6/6, 모델·상수 요약(λ = 0.500000 frozen, α_h), `[LT] step N lm_loss … peak …GiB`(250스텝마다),
  `[EVAL] step N … acc … exact k/2048`, milestone 외삽표, 종료 시 peak 메모리.
- `train.jsonl`: 스텝마다 한 줄(`step, seconds, elapsed, segment, lm_loss, accuracy, exact_accuracy, lr, …`). 연구 실행기와 같은 형식이다.
- `config.json`, `step_*.pt`(raw·EMA 가중치, optimizer, EMA, rank별 carry(W·카운터 포함)·RNG, 데이터 위치; 하나에 약 0.5 GB),
  `milestones/step_*.pt`, `milestones/extrap_step_*.txt|.json`, `milestones/extrap_results.jsonl`.

## 자체검사 (CPU, FP64, 합성 입력; 학습 전에 자동 실행, 약 1초)

- (a) 태그 = c·sin(리셋 이후 wrap된 쌍 각도 변위의 명시적 합) + n·ωT (K, V), 중간에 한 레인 리셋, |tag − n·ωT| ≤ c, 카운터 채널 = n+1.
- (b) 메모리 경로 전체를 명시적 합과 비교: Im S = 모든 (post, pre) 사건 쌍의 합(λ_K ≠ λ_V로 일반화), W = 리셋 이후 Im S 증분의
  기하급수 합(ρ = 0.5와 0), read = RoPE(q̂)(Hebb + αW)ᵀ, 중간 리셋 포함. 1e-9 상대오차.
- (c) 3블록 세그먼트 2개, no-grad 3 + 3블록 = 6블록 한 번(hidden, 시계 상태와 카운터, 두 흔적, W, logits).
- (d) 레인별 리셋: 새 레인 = 새 실행(카운터·시계·흔적·W 0), 이어지는 레인은 영향 없음.
- (e) α와 모든 학습 파라미터의 유한·비영 기울기, 쓰지 않는 3런 슬롯(theta_k_raw, theta_v_raw, trace_lam_raw)과 λ는 기울기 없음,
  λ = 0.5가 optimizer·EMA 밖. 5블록 세그먼트에서 α 방향과 전체 파라미터 무작위 방향의 방향미분 = 중앙차분(1e-6) — 블록 사이에서 W를
  detach하거나 V를 detach하면 이 검사가 실패한다(변이 검사로 확인).
- (f) 저장 → 로드: carry(h, 시계·카운터, 흔적, W) 동일, 다음 스텝 loss·정확도·가중치·EMA·carry 동일. B 상수가 다른 설정은 재개 거부.

추가 확인(파일에 넣지 않은 일회성 검사): ρ = 0일 때 B는 저장소의 독립 구현 `model_class(clock='bounded', lambda_init=0.5,
freeze_lambda=True, kernel='torch')`와 6세그먼트(레인 리셋 포함) 동안 logits 3e-6, 기울기 상대 1e-6 이내로 같다(FP32 CPU).
변이 검사: spacing 제거, W 리셋 누락, 카운터 정지, W 이월 누락, read에 Im S 사용, W detach, V detach 모두 실패로 잡힌다.

## GPU 스모크 테스트 (RTX 4090, 다른 학습 런과 GPU 공유 중 — 그 런이 GPU 사용률 98~100%)

셀 실행기 경로 그대로 12스텝 → 재개 12스텝(BF16 auto), `--config`로 FP32 8스텝. 출력은 삭제했다.

| 구성 | ms/step (컴파일 뒤 중앙값) | peak allocated / reserved |
| --- | ---: | ---: |
| BF16 autocast (4090의 auto), batch 128 | 265 | 10.0 / 11.0 GiB |
| FP32 (T4·P100이 auto에서 쓰는 경로), batch 128 | 470 | 11.8 / 12.9 GiB |

- 공유 GPU라 속도는 상한이 아니다. 같은 GPU를 단독으로 쓴 3런 torch 경로(checkpoint 켬)는 141 ms였다(10-13 §1).
- 첫 두 스텝은 컴파일(약 125 s + 65 s, carry가 None인 그래프와 텐서인 그래프 두 번)이다. 재개할 때마다 다시 컴파일한다.
- FP32 batch 128 peak 12.9 GiB(reserved)는 16 GB T4·P100에 들어간다. T4 ×2는 rank당 64라 더 작다. T4·P100의 실제 속도는 재지 않았다.
  4090 FP32보다 몇 배 느릴 것이므로 50,800스텝은 12시간 세션 여러 개에 걸쳐 재개해야 할 가능성이 크다.
- 초기 loss 2.506(3런 기록 2.494, B-only 2.508). 평가·milestone 외삽·저장·재개 경로가 모두 오류 없이 돌았다.

## 하네스·3런과의 차이

- 정밀도: `lt/train.py` CFG와 3런은 `amp_dtype='bfloat16'`. 여기서는 `'auto'`(같은 하네스 규칙: native BF16이면 BF16, 아니면 FP32).
  4090·A100에서는 3런과 같고, T4·P100에서는 사영·FFN도 FP32다.
- `compile='auto'`: P100(sm 6.0)은 Inductor/Triton이 지원하지 않아 eager로 돈다. 하네스는 `True` 고정.
- activation checkpoint 옵션을 없앴다(3런과 같다). 셀에는 address/v1.71/v1.8 모델, `init_from` 변환, 옛 체크포인트 API를 넣지 않았다.
- `n·ωT`는 `[−π, π)`로 wrap해서 더한다. 위상자는 같고(저장소 `bounded` 시계는 wrap하지 않음, 차이는 FP32 반올림 수준),
  128세그먼트 외삽(n ≤ 1023)에서 FP32 정밀도를 지킨다.
- RMSNorm(`phi`)은 FP64 입력을 FP64로 계산한다(3런 클래스는 `.float()`로 FP32를 거친다). 자체검사 경로에만 해당하고 GPU 경로는 같다.
- `LTCarry`에 `stdp_w` 필드를 추가했다. 체크포인트 carry는 하네스의 `_carry_dict`/`_restore_carry`가 필드 전체를 저장·복원하므로 W도
  자동으로 포함된다. 재개 검사 키에 B 상수가 들어가고, model_id는
  `lt-kv-stdp-fast-weight-v1:research-phaseclock_disp_bounded_leakyW_lfrozen_swiglu_torch_qkl2_sum`이다.
- 로그 추가: 스텝별 `train.jsonl`(연구 실행기와 같은 형식), 로그 줄과 종료 시 peak GPU 메모리. 학습 수치에는 영향이 없다.
- Kaggle T4 ×2에서는 `num_processes='auto'`가 2 rank × 64로 나눈다(하네스 기본). 기울기는 SUM all-reduce라 수식은 단일 GPU와 같지만
  비트 단위로 같지는 않다. 3런과 같은 단일 프로세스로 돌리려면 `num_processes=1`.
- 3런은 융합 Triton 커널(BF16 dot)로 학습했다. 이 셀은 torch 경로(FP32 dot)만 쓴다.
