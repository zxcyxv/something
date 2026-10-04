# 2026-10-04 연구 기록

오늘의 중심 질문은 헤비안 외적을 채널 뉴런의 STDP 관계로 바꾸면서,
상태에 따라 위상차가 변해도 현재 G를 효율적으로 계산할 수 있는가였다.
학습 하네스와 지표 집계를 먼저 감사했고, 고정 위상의 sine·지수창을 거쳐
상태 의존 위상의 부호 있는 가우시안 창을 구현하고 처음부터 학습을 시작했다.
현재 후보의 성능 우위와 반복 외삽 개선은 아직 확인하지 않았다.

## 목표와 시간의 정의

프로젝트의 최종 목표는 학습 범위를 넘는 반복에서 성능이 개선되고,
추론 중의 비지도 가소성이 그 개선을 만드는 것이다. 이번 단계에서는
FFN이 관계를 표현하고 STDP 창이 관계의 방향을 읽도록 한다.
재귀 시점 자체를 뉴런의 발화 시각으로 간주하던 과거 KV 차감식에서 벗어나,
채널별 위상차를 공통 주파수에 대한 시차 좌표로 사용한다.

토큰은 관측을 모으는 축이고 채널이 뉴런이다. 한 헤드 안의 Value 채널 i와
Key 채널 j를 연결한다. 공통 carrier `exp(i omega t)`는 켤레 곱에서 소거되므로
실제로 시계나 oscillator 상태를 굴리지 않는다. 위상의 상태 의존성은 carrier의
공통 주파수 가정과 양립한다. 물리 시차를 정하려면 omega의 단위도 별도로 정해야 한다.

Fast Weight는 선형 어텐션을 연상 기억으로 읽는 중간 다리로 참고했다.
논문의 토큰 시간축이나 별도 내부 반복 s를 우리 재귀축에 그대로 가져오는 것은
이번 설계가 아니다. 이번 구현에는 추가적인 fast-weight settling 반복이 없다.

## 하네스 감사와 loss 역전 해석의 수정

각 optimizer step은 한 segment다. segment 안에는 공유 블록 재귀 8회가 있고,
같은 샘플은 16 segment 동안 유지된다. 내부 8회는 모두 미분하며,
segment 끝에서 backward/update하고 다음 segment로 넘길 carry를 detach한다.
Activation checkpointing은 내부 그래프의 재계산이며 이 detach와 구분한다.

`train.jsonl`의 `lm_loss`는 모든 step의 현재 loss다. 셀/완답 정확도는
`_count_raw > 0`인 종료 segment에만 집계된다. console은 현재 loss와 최근 종료
segment의 정확도를 함께 출력한다. 출력 간격 250은 여러 segment 위치를 샘플링하고,
16은 종료 segment만 샘플링한다. 과거 console끼리 loss와 정확도를 무조건 같은
시점의 값으로 비교하면 안 된다.

현재 B-only에서 관측했던 종료 segment loss 상승은 실제 수치지만,
전체 segment 평균 loss의 상승을 뜻하지 않았다. 아래 구간은 원본 스칼라 로그로
다시 집계한 값이다. 구간은 `(start, end]`이고 loss는 `lm_loss`다.

| raw B-only 구간 | 전체 segment 평균 loss | 종료 segment 평균 loss |
|---|---:|---:|
| 2000–4000 | 0.71955 | 0.32912 |
| 4000–6000 | 0.70820 | 0.35467 |
| 6000–7400 | 0.69857 | 0.36929 |

따라서 이것을 모델 전체의 학습 역전이나 weight decay의 치명적 오류로 설명했던
해석은 철회한다. Weight decay와 Q/K 크기는 검토할 변수지만 원인으로 입증되지 않았다.
같은 B-only 모델을 구/신 하네스에 넣은 32-step CPU FP32 비교와 16-step checkpoint
비교에서 loss, gradient, 모델, carry, optimizer의 최대 차이는 0이었다.
EMA에는 약 1e-7 이하의 반올림 차이가 있었다. 이 범위의 감사는 장기 GPU 학습의
모든 버그가 없다는 증명은 아니다.

근거: [집계 감사](evidence/loss_aggregation_audit_20261004.json),
[하네스 감사](evidence/harness_audit_20261004/results.json),
[관련 commit diff](evidence/harness_commit_history/index.json).

## 공통 블록 규칙과 B-only

앞으로 새로 설계하거나 변경하는 아키텍처는 다음 두 번의 정규화를 기본으로 한다.
이는 구조 규칙이며 성능 개선이나 과거 실패 원인의 증명과는 별개다.

```text
u = RMSNorm(h + Attention(h))
h_next = RMSNorm(u + FFN(u))
RMSNorm(x) = x * rsqrt(mean_hidden(x.float()**2) + 1e-5)
```

토큰별 전체 hidden 채널을 FP32로 계산하고 입력 dtype으로 반환한다.
학습 가능한 scale/bias는 없다. Q/K 정규화로 이를 대체하지 않는다.
실제 forward 위치와 수식은 테스트했다. 규칙은 [AGENTS.md](../../../AGENTS.md)에 있다.
과거 실험을 재현하는 옵션에는 당시 정규화 구조를 유지했다.

B-only는 현재의 signed real K,V 외적 `B = mean_tokens(V K.T)`를 실수 Q로 읽는다.
복소 trace, G 누적, 읽기 보간이 없다. R1B8과는 Q/K의 사영 분리,
Value/output 가중치 공유, 재주입의 정규화, FFN/어텐션 순서 등에서 다르다.
비교에 사용한 R1B8 코드·설정·로그는 [reference/r1b8](reference/r1b8/manifest.json)에 남겼다.

오늘 raw B-only, v1.7식 Q/K L2, 두 residual RMSNorm의 B-only를 각각 학습했다.
이 비교의 초기 곡선만으로 B-only가 v1.7이나 다른 구조를 전반적으로 이긴다고
결론내리지는 않는다. 전체 로그와 모델별 설정을 보존했다.

## URM 비교에서 고친 부분

URM을 옮긴 최초 두 런은 `loops=1`로 실행했다. 이것은 기존 B/phase 런의
`loops=16`과 샘플 유지/update 조건이 다르므로 구조만의 대조군으로 사용할 수 없다.
당시에도 ACT는 이미 꺼져 있었다. 매 segment 샘플이 바뀐 이유는 학습된
halting이 아니라 `loops=1`이었다. ACT 때문이라고 설명했던 부분을 바로잡았다.

수정한 유효 비교는 `urm_swiglu_loops16_20261004`다. URM의 서로 다른 layer 2개를
각 재귀에서 적용하고 segment당 8회 반복한다. 즉 segment당 layer 적용은 16회다.
기존 B/phase는 layer 1개를 8회 적용하므로 계산 깊이와 파라미터 수는 다르다.
ACT와 내부 no_grad/TBPTT는 없고 ConvSwiGLU 대신 일반 SwiGLU를 사용한다.
segment 사이 detach/update, 샘플 16 segment 유지, 데이터/loss/optimizer는 공통 하네스다.
외부 16 segment를 한 번에 관통하는 BPTT는 아니다.

v1.7도 과거 초기화와 아키텍처를 유지한 채 현재 하네스에서 6000 step 재현했다.
신규 RMSNorm을 넣지 않은 것은 과거 실험 재현 예외다.
아래는 `(4000, 6000]`의 같은 step·종료 segment를 맞춘 비교다.

| 모델 | 전체 segment loss | 종료 segment loss | 종료 셀 정확도 | 종료 완답률 |
|---|---:|---:|---:|---:|
| URM 2-layer, SwiGLU, loops16 | 0.69249 | 0.39178 | 90.029% | 28.656% |
| 고정 위상 지수형 G | 0.70168 | 0.33713 | 92.894% | 27.181% |
| v1.7 현재 하네스 재현 | 0.73814 | 0.53919 | 81.168% | 5.094% |

이것은 학습 퍼즐의 online 지표다. held-out EMA 평가와 섞지 않는다.
초기 학습 능력 확인용이며 외삽 성능이나 최종 일반화의 순위는 아니다.
상세 조건은 [URM 비교 문서](urm_full_bptt.md),
원본 확인은 [loops 감사](evidence/urm_loops_audit_20261004.json)에 있다.

## 작은 연상 기억 실험

초기 간섭 측정은 최신 association을 정답으로 정했기 때문에 current-G에 유리했다.
이 결과를 Hopfield 기억 용량이나 누적 자체의 열등성으로 해석할 수 없다.
실제 Hopfield attractor/recall 조건의 용량과 행렬 rank만의 상한도 구분해야 한다.
[측정 원본](evidence/kv_interference_20261004/results.json)에 이 제한을 기록했다.

이후 32차원 작은 모델에서 동일한 reader와 미학습 패턴 bank를 사용해
헤비안, 공통 trace의 balanced STDP, 진폭/감쇠를 분리한 general STDP를 비교했다.
3 seed의 frozen-memory 미관측 bit 정확도는 각각 88.84%, 69.39%, 85.37%였다.
general 형태는 balanced 형태보다 나았지만 이 학습 실험에서 헤비안을 넘지는 못했다.
일반 창에서 진폭과 감쇠를 분리하는 것은 통상 pair-STDP의 자유도를 되돌린 것이다.

별도로 발견 단계에서 고른 비대칭 trace 창을 고정하고 새 bank 3072개로 검증했다.
공통 배경이 있는 조건에서 헤비안 대비 4패턴 +3.02%p, 8패턴 +2.19%p였다.
깨끗한 settling 조건에서는 각각 -0.19%p, -0.48%p였다. 특정 고정 활동/창의 결과이며,
학습된 모델의 보편적 STDP 우위나 도메인 부적합을 증명하지 않는다.
현재 구현의 제약과 학습 효율을 계속 검토해야 한다.

근거: [완료된 작은 실험](evidence/associative_stdp_20261004_v2/analysis.json),
[독립 holdout](evidence/associative_stdp_background_holdout_20261004/results.json).
정규화, write scale, validation 선택 조건도 같은 폴더의 protocol에 보존했다.

## 위상 창과 계산 비용의 연구

sine만 읽으면 `delta -> 0`에서 창의 크기도 0으로 간다. 부호 있는 지수 STDP 창은
작은 양/음 시차에서 각각 +1/-1에 가깝고, 이번 convention에서는 정확히 0만 0이다.
공통 carrier나 Euler 공식 자체가 실수 지수의 거리 감쇠를 만들어주지는 않는다.
고정 위상 지수창은 명시적인 pairwise 커널로 구현했다.

상태 의존 위상에서는 창을 token 평균 뒤로 꺼낼 수 없다. Mamba-3의 회전 좌표
변환은 참고할 수 있지만 우리의 선후관계 분기까지 자동으로 제거하지 않는다.
지수 항을 뉴런별로 분리해 계산할 수 있어도 순서 마스크는 남는다.
Q별 prefix read는 잘못 구현하면 token 제곱 비용이 생긴다. 공유 G를 먼저 쓰고
모든 Q가 그것을 읽도록 하면 Q 개수에 대해서는 선형이다.

정렬 순서를 공유하는 Triton 구현도 만들었지만 측정상 더 느렸다. 이 실패 기록은
같은 최적화를 되풀이하지 않도록 남겼다. 아래 값은 RTX A5000, batch128,
heads8, head_dim104, FP32/TF32 off의 G 쓰기+읽기 순전파/역전파 median ms다.
compile 시간은 제외하며, QKV/위상 사영·FFN·RMSNorm·optimizer가 없는 연산자 측정이다.

| 독립 측정 세션/구현 | 81 tokens | 900 tokens |
|---|---:|---:|
| 동적 지수창 분리형 / 최초 측정 | 16.105 | 170.451 |
| 공유 정렬 세션의 지수창 분리형 기준 | 15.791 | 168.824 |
| 공유 정렬 + G | 37.834 | 366.286 |
| 가우시안 세션의 지수창 분리형 기준 | 15.723 | 168.233 |
| 단위 위상 부호 + 가우시안 / 현재 학습 구현 | 34.111 | 324.630 |
| 같은 가우시안의 직접 시차 부호 대조군 | 24.024 | 235.621 |
| sin/cos를 뉴런별로 미리 계산하는 구현 | 52.553 | 469.780 |

최초 900-token 고정 위상 기준은 20.834ms였고 동적 지수창은 그 8.18배였다.
이 배수는 전체 모델 학습 시간이나 실제 ARC 학습 결과가 아니다.
900-token은 향후 ARC 30x30의 비용 확인이며 실제 ARC 모델 학습은 하지 않았다.
직접 부호 대조군은 `(-pi,pi)`에서 `sign(sin delta)=sign(delta)`인 동치 계산이다.
현재 학습은 실제 `sign(sin delta)` 커널을 사용한다.

[위상 수학 조사](phase_stdp_math.html), [동적 지수창 비용](dynamic_phase_900_benchmark.html),
[단위 위상 가우시안 검증/비용](unit_phase_gaussian_benchmark.html)과
[원시 수치](evidence/)를 남겼다. Mamba-3·Complex KDA·phase-coded STDP의
논문은 수학적 형태를 참고했으며 다른 논문의 아키텍처 전체를 도입하지 않았다.
위상 수학 HTML은 당시 조사한 후보들을 포함한 중간 기록이다.
그 안의 geometric/jitter 후보를 현재 모델로 채택한 것은 아니다.

## 현재 학습 중인 후보

```text
x = hidden + sqrt(hidden_size) * input_injection
phiK = (pi/2 - 1e-4) * tanh(W_phaseK x + thetaK)
phiV = (pi/2 - 1e-4) * tanh(W_phaseV x + thetaV)
delta[t,i,j] = phiV[t,i] - phiK[t,j]
A = sign(sin(delta))
L = A * exp(-delta**2 / tau), tau=1 rad^2
G[i,j] = mean_tokens(V[t,i] * RoPE(K)[t,j] * L[t,i,j])
read = RoPE(Q) @ G.T
u = RMSNorm(x + W_output read)
next_hidden = RMSNorm(u + bilinear_FFN(u))
```

K/V의 활동 사영과 위상 사영은 독립이다. 위상은 `[B,H,T,D]`이며 입력과
재귀 상태에 따라 바뀐다. 이는 signed real 활동과 위상의 polar factorization이다.
독립 Cartesian 실수부/허수부 사영에서 `atan2`로 각도를 추출하는 구현은 아니다.
G는 매번 새로 계산하고 누적하지 않는다. carry의 G는 진단용이며 다음 읽기에 쓰이지 않는다.
Q는 실수이고 trace/읽기 보간은 없다.

`A`는 sine의 크기를 버리고 부호만 취한다. 별도의 unit normalization 나눗셈은 없다.
양의 크기 정규화는 복소 관계의 허수부 부호를 바꾸지 않는다.
정확한 위상차 0에서는 A=G 기여=0, 작은 양/음 차이에서는 강도 크기가 1에 가깝다.
Gaussian은 지수형 STDP와 다른 거리 창이다. tau의 단위도 rad에서 rad^2로 바뀐다.
`sign` 미분은 0이며 위상은 Gaussian envelope의 일반 미분으로 학습한다.
그 위상 gradient는 0에 가까워질수록 작아진다. surrogate나 smoothing은 추가하지 않았다.
signed 활동값 때문에 L의 LTP/LTD 부호와 개별 G 성분의 부호는 다를 수 있다.

위상 제한은 FP32 tanh 포화에서도 모든 pairwise 차이가 `(-pi,pi)` 안에 있도록 한다.
phase 사영은 FP32, 기존 활동 사영/FFN은 BF16 AMP를 유지했다.
QKV/output/FFN/phase 사영의 weight decay는 1, theta offset은 0이다.
hidden832, heads8, layer1, 재귀8, loops16, batch128, seed0,
lr1e-4/warmup2000, EMA.999이며 파라미터는 9,925,355개다.
이전 고정 위상 지수형의 8,540,907개에서 phase 사영 두 개가 추가됐다.
고정 지수형과 비교하면 위상의 상태 의존성과 창 형태가 함께 바뀌므로 단일 변수 ablation은 아니다.

실행 파일: [모델](../../../lt/kv_stability.py),
[가우시안 Triton 커널](../../../lt/unit_phase_stdp.py),
[실험 설정](../../../configs/kv_unit_phase_gaussian_research.json).

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m lt.research_kv_collapse \
  --variant phase_unit_gaussian_current_only \
  --config configs/kv_unit_phase_gaussian_research.json \
  --out runs/kv_unit_phase_gaussian_new --steps 10000 --save-every 1000 --keep-last 3
```

모델/블록 출력·기울기·두 RMSNorm 위치를 검증했고 실제 compiled optimizer step이
진행되는 것을 확인했다. K/V phase 사영과 offset 모두 finite nonzero gradient가 있다.
현재 런은 계속 실행 중이다. 이 기록에 묶은 로그는 1382 step까지의 스냅샷이며
마지막 종료 segment는 1376이다. [같은 구간의 곡선](training_curves.png)과
[집계 수치](training_curves.json)는 이 고정 스냅샷 기준이다.

## 보관 및 재현

[archive_manifest.json](archive_manifest.json)에 원본 경로, 보관 경로, 크기,
압축 전 SHA256와 캡처 시간을 기록했다. `logs/`는 완료 실험 9개와 현재 런 1개의
전체 step 로그·console·진단 스칼라·설정이다. JSONL과 console은 gzip으로 보관했다.
`evidence/`에는 작은 실험, 하네스 감사, 수학 검산, CUDA 정확성 검사, 성능 측정의
원시 숫자가 있다. 체크포인트·중복 source snapshot·PID·대기용 queue 스크립트·
URM 전체 데이터 복제본은 기록에서 제외했다. 실행에 필요한 URM vendor 코드와
원본 학습 코드/설정의 출처는 [vendor 출처](../../../lt/urm_vendor/README.md)와
[원본 provenance](reference/urm/IMPORT_PROVENANCE.json)에 남겼다.
보관 파일의 해시를 확인한 뒤 완료 런의 체크포인트와 중복 복제본을 정리했다.
현재 학습 중인 런은 유지했다. 삭제 내역과 확보한 약 7.36GB는
[cleanup.json](cleanup.json)에, 51개 테스트와 집계 검산은 [validation.json](validation.json)에 기록했다.

아래 도구는 체크포인트와 GPU 없이 압축 로그에서 곡선을 다시 만든다.
`_count_raw > 0`인 같은 step의 loss/정확도를 함께 집계하며 loops=1을 거부한다.

```bash
python -m lt.summarize_training_curves \
  --run Gaussian=docs/research/2026-10-04/logs/kv_unit_phase_gaussian_dynamic_20261004 \
  --run Exp=docs/research/2026-10-04/logs/kv_phase_exp_current_20261004 \
  --run URM=docs/research/2026-10-04/logs/urm_swiglu_loops16_20261004 \
  --out runs/research_day_curve/training_curves.json
```

현재 단계에서 남은 질문은 상태 의존 위상과 Gaussian 창의 학습 효율,
새 창에서 G를 누적했을 때의 효과, 반복 외삽에서의 개선이다.
초기 학습 loss만으로 이 질문들의 답을 정하지 않는다.
