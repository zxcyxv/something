# KV-STDP 연구 기록 — 2026-10-03

현재 판단: 재귀 시간을 기준으로 한 KV 차감식은 pair-STDP에서 정확히
도출된다. 실수 Query로 읽는 것도 성립한다. 구현 감사에서 이를 뒤집을
부호·축·흔적 갱신 순서 오류는 찾지 못했다. 그러나 누적 M 모델의 학습
역전 원인은 아직 확정하지 못했다. G-only는 고정 가중치로 재귀를 실행하면
많은 문제를 끝까지 풀므로, 이 구조가 계산 자체를 못 한다고 결론낼 수 없다.

이 문서는 현재 결론과 증거의 입구다. 상세한 조사 과정은
[전체 연구 노트](../../kv_collapse_research.md)에 있다.
긴 학습은 모두 종료했고, 현재 추가 작업은 이론 도출과 제한된 진단이다.

## 모델 이름과 실제 계산

채널을 뉴런, r을 재귀 블록 시점, p를 토큰으로 둔다. 아래 식에서 K는
공간 RoPE가 적용된 Key이고, 외적 방향은 Value x Key이다.

```text
B_r = mean_p[V_rp K_rp.T]
G_r = mean_p[V_rp eK_(r-1),p.T - eV_(r-1),p K_rp.T]
M_r = M_(r-1) + G_r
read = read_operator @ q_r

eK_r = lambda eK_(r-1) + (1-lambda) K_r
eV_r = lambda eV_(r-1) + (1-lambda) V_r
```

이 표기는 가중치와 좌표계가 고정된 경우다. 실제 원본은 회전 전 K 흔적을
저장하고 읽을 때 현재 theta로 회전한다. 느린 가중치가 갱신되는 세그먼트
경계에서는 과거 시점에 회전한 K를 저장하는 대조군과 달라진다.
코드는 행벡터를 사용하므로 읽기는 `q @ read_operator.T`이다.

| 실행 폴더 | 실제 읽기 연산 | 마지막 optimizer step |
|---|---|---:|
| baseline | M, 8 no-grad + 8 grad의 초기 재현 | 4500 |
| baseline_ng0 | M | 6000 |
| v17_no_address_norm | v1.7의 현재 QK/누적 W 보간, 주소 정규화 제거 | 6000 |
| v17_no_norm_memory_read | 위 v1.7에서 W만 읽기 | 2385 |
| kv_interpolated_read_ng0 | `(1-alpha) B + alpha M`, 헤드별 alpha | 2875 |
| kv_current_only_ng0 | B만 읽기 | 6000 |
| kv_complex_current_only_ng0 | G만 읽기 | 6000 |
| kv_current_plus_stdp_ng0 | B+G, M 누적 없음 | 3000 |
| kv_historical_key_trace_ng0 | M, 과거 시점에 회전한 K 흔적 사용 | 6000 |
| kv_read_gain_quarter_ng0 | 0.25 M | 3000 |

명칭 교정: 기존 “KV 보간판”은 B/M 보간이었다. 순수 G/M 보간의 완료 기록은
없다. 따라서 “G만 읽기와 G/M 보간 모두 실패했다”는 비교는 성립하지 않는다.
정규화 KV 역시 준비만 했고 학습하지 않았다. 초기 요청과 다르게 만들어졌던
B-only 비교를 복소 STDP 실험으로 설명해서는 안 된다.

## 같은 스텝에서의 학습 비교

주 비교는 no-grad 0, gradient block 8, segment 16, batch 128, d832/H8,
seed 0이다. AdamATan2, lr 1e-4, warmup 2000, weight decay 1을 사용했다.
이 표의 각 값은 `(endpoint-256, endpoint]` 안의 **종료 세그먼트 16개 평균**이다.
loss는 lm_loss, 정확도는 셀 정확도다. EMA나 테스트 정확도를 섞지 않았다.
3000 바로 전 마지막 종료 세그먼트는 2992이다.

| 구조 | 2992 loss | 2992 정확도 | 6000 loss | 6000 정확도 |
|---|---:|---:|---:|---:|
| 원본 M | 1.04130 | 54.84% | 0.84316 | 61.84% |
| v1.7 주소 정규화 제거 | 0.47737 | 84.28% | 0.48880 | 83.23% |
| 현재 B만 | 0.32060 | 94.24% | 0.33403 | 92.75% |
| 현재 G만 | 0.70270 | 72.74% | 0.57112 | 78.32% |
| B+G | 0.39972 | 90.58% | — | — |
| 과거 회전 K 흔적의 M | 1.02749 | 55.40% | 0.79979 | 63.99% |
| 0.25 M | 1.04397 | 56.80% | — | — |

2000에서 v1.7 정규화 제거 보간은 90.05%, W-only는 56.64%였다.
B/M 보간은 종료 직전 2864 구간 평균 88.50%였다. 이 둘의 종료 시점이 달라
2992 표에 억지로 넣지 않았다. B-only도 2992에서 6000 사이 정확도가 낮아져,
차감 항이 없으면 모든 역전이 사라진다고 단정할 수 없다.

역사적 R1B8과 v1.7의 6000 단일 종료 배치 정확도는 각각 80.84%, 81.66%이고,
새 원본 M은 60.51%이다. 이 수치는 위의 256스텝 평균과 다른 지표다.
R1B8은 성공한 signed attention + bilinear의 기제 대조군이지만 초기화와
학습 파이프라인까지 완전히 같은 paired run은 아니다.

모든 구간의 원시 값은 [training_summary.json](training_summary.json),
종료 배치 전체는 [terminal_training.csv](terminal_training.csv),
각 스텝의 원본 스칼라 로그 전체는 [logs](logs)에 보존했다.

## 이론적으로 확인한 것

공통 lambda와 0 초기 흔적에서는 다음 식이 정확하다.

```text
M_R = mean_p sum_{s<r<=R} (1-lambda)*lambda**(r-s-1)
      * [V_rp K_sp.T - V_sp K_rp.T]
```

과거 pre/current post와 과거 post/current pre가 반대 부호를 갖는
balanced pair-STDP다. 실제 pair별 lambda는 양·음 lag의 시간 상수가
시냅스 성분에 따라 다른 L_ij를 만든다. 이것도 STDP이지만 하나의 공통된
홀수 함수 L로 표현되는 특별한 경우와는 구분해야 한다.

```text
Kc = eK + i*K
Vc = eV + i*V
Vc @ conj(Kc).T = (eV eK.T + V K.T) + i*(V eK.T - eV K.T)
Im(Vc @ conj(Kc).T) = G
```

이 복소수 구성의 실수부와 허수부는 독립적으로 학습하는 두 사영이 아니다.
각각 과거 흔적과 현재 활동이다. Im을 취하면 정확히 G가 나온다.
전체 복소 외적을 읽거나 실수부 일부를 삭제하면 다른 규칙이 된다.
STDP 자체가 현재 B 추가, 복소 Query, Query 흔적을 요구하지 않는다.

실수 Query 읽기는 다음 signed associative read와 같다.

```text
extended_keys   = [eK; K]
extended_values = [V; -eV]
Q @ G.T = Q @ extended_keys.T @ extended_values
```

K와 V가 독립 사영이므로 실제 채널 M이 항상 반대칭이라는 주장은 틀리다.
반대칭 temporal operator와 채널 공간의 M을 혼동하면 안 된다.
고정 활동에서는 공통 lambda의 G가 0으로 가지만, 그것만으로 문제 풀이가
불가능하다는 결론도 나오지 않는다. 상태가 답을 보존하거나 시간에 따라
다른 활동을 사용할 수 있다. 구성적 반례와 실제 모델의 결과를 모두 확인했다.

누적 M은 현재 B의 단순한 배수도 아니다. 흔적 경로의 순서에 의존한다.
완전한 주기 상태가 존재하려면 한 주기 동안 `sum G = 0`이어야 한다.
공통 lambda의 정상 2주기는 이를 만족하지만, 다른 주기나 성분별 lambda에서는
M이 계속 변할 수 있다. 이는 필요 조건이며 붕괴 원인이나 안정성 증명이 아니다.

## 이번 추가 연구의 관찰과 반례

6000 체크포인트의 동일한 저장 학습 문제 128개를 사용했다. 세 모델의
입력·라벨·ID가 바이트 단위로 같다. raw 가중치를 고정하고 상태를 새로
초기화해 128블록을 실행했다. BF16 사영/FP32 상태를 유지했다.

| 구조 | 저장 online carry 셀 / exact | 새 고정 가중치 재귀 셀 / exact |
|---|---:|---:|
| 원본 누적 M | 60.54% / 0% | 56.51% / 0% |
| G-only | 81.33% / 10.94% | 90.10% / 53.91% |
| B-only | 91.28% / 22.66% | 93.13% / 25.78% |

G-only exact는 32블록의 7.03%에서 128블록의 53.91%까지 증가했다.
이는 학습 문제를 고정 가중치로 재생한 결과이며 새로운 테스트 성능이 아니다.
저장 carry 재채점은 마지막 가중치를 사용하므로 원본 로그 수치와 조금 다르다.

복사한 체크포인트에 같은 배치를 16번 더 갱신하는 짧은 진단도 했다.
G-only의 online exact는 85.16%로 높아졌다. 따라서 “문제 풀이 중 optimizer
갱신이 항상 계산을 망가뜨린다”는 간단한 설명은 지지되지 않는다.
기존 학습 경로 전체에서 영향이 없다는 의미는 아니다. 추가 장기 학습이나
새 체크포인트 저장은 하지 않았다.

성분별 lambda 때문에 생기는 2주기 평균 쓰기를 도출하고 실제 활동으로
검증했다. 하지만 lambda를 헤드 평균으로 바꾼 고정 재생은 56.53%에서
56.55%로만 변했고 exact는 둘 다 0이었다. 이것을 해결책이라고 볼 근거는 없다.

부호/축/trace 순서/미분/상태 초기화/activation checkpoint를 독립 계산과
대조했다. 관련 unit test 30개가 통과했다. 이론 검산은 17개 그룹으로 구성되며,
FP64 항등식 오차는 약 1e-14 이하다. 이는 식과 구현의 대응을 확인한 결과다.
학습 성공이나 붕괴 원인을 검증한 결과로 확대 해석하지 않는다.

## 다음에 구분해야 할 질문

1. G-only는 고정 가중치에서 잘 계산하는데 누적 M은 왜 재귀 계산이 거의
   진행되지 않는가? M에 남는 경로 정보가 현재 Query와 어떻게 결합되는가?
2. 누적의 기여만 바꾸는 순수 G/M 보간은 실제로 아직 비교되지 않았다.
   식은 `(1-alpha)G_r + alpha M_r = G_r + alpha M_(r-1)`이다.
   0.25 M 실험은 현재 G와 과거 M을 함께 줄였으므로 이 비교를 대신하지 못한다.
3. 6000에서의 국소 진단을 초기 학습 역전의 원인으로 오인하지 않고,
   도출된 가설에 맞는 비교를 먼저 정해야 한다.

## 증거와 재현

- [archive_manifest.json](archive_manifest.json): 원본/압축 파일 SHA256,
  기준 코드 커밋, 기록 시점. 체크포인트와 데이터셋은 포함하지 않았다.
- [protocols](protocols): 각 실행의 실제 config, protocol, 종료 상태.
- [reference](reference): LinearTuring의 역사적 소스·로그와 출처 manifest.
- [audit](audit): 구현 감사, matched comparison, 이전 진단과 실패 기록.
- [theory_1h](theory_1h): 수식 검산, frozen/online episode, 시간창 분석 결과.
- [검산 코드](../../../lt/analyze_kv_stdp_theory.py),
  [기록 추출 코드](../../../lt/export_kv_research.py).

저장소 루트에서 실행한다. 테스트는 CUDA 학습을 새로 시작하지 않는다.

```sh
python -m unittest lt.test_kv_stability lt.test_kv_stdp_reference lt.test_v17_read_control -v
python -m lt.analyze_kv_stdp_theory --out /tmp/kv_stdp_algebra.json
```

새 no-grad 0 학습을 재현할 때는 원본 데이터셋을 준비한 뒤 별도 폴더를 쓴다.
아래 명령은 예시이며 이 문서를 만드는 동안 새로 실행하지 않았다.

```sh
python -m lt.research_kv_collapse --config configs/kv_stdp_ng0_research.json --variant original --steps 6000 --save-every 6000 --keep-last 1 --out runs/reproduction_m
python -m lt.research_v17_normalization --source docs/research/2026-10-03/reference/train_v17.py --steps 6000 --out runs/reproduction_v17
```

원본 `lt/train.py` SHA256은
`65221608b193b1c86893bc1bfe973a6c400fe960d2fefed74905aedcb9c71e2f`로 유지했다.
아카이브는 스칼라 기록과 소스를 제공한다. 체크포인트가 필요한 진단까지
새 clone만으로 바로 재생할 수 있다는 뜻은 아니다.

이론의 참고 원문은 [pair-STDP와 eligibility trace](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html),
[Fast Weights](https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past.pdf),
[독립 Q/K/V fast-weight 해석](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf),
[differential Hebb 도출](https://www.cs.cmu.edu/Groups/NIPS/NIPS99/99papers-pub-on-web/Named/XieSeung.pdf)이다.
일반 STDP 식과 실제 모델 사이의 대응 및 반례는 위 연구 코드에서 별도로 검산했다.
