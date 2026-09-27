# 재귀 LT의 체스 학습: 작은 탐색으로 목표를 만들고, 그 탐색을 모델에 압축하기

2026-09-27. 연구 설계 문서. **체스 모델의 학습·대국 성능은 아직 측정하지 않았다.**
아래에서 논문 결과, 현재 코드에서 확인한 사실, 새로 제안하는 실험을 구분한다.
수식은 CLI에서 읽을 수 있도록 일반 텍스트로 적었다.

## 1. 내가 우선 실행할 설계

**전문가 데이터로 policy/value를 먼저 학습하고, 작은 MCTS로 개선한 정책을
다시 학습시키는 방식을 우선한다.** 처음부터 대규모 자기대국을 시작하지 않는다.
고정 국면에서 탐색이 실제로 수의 질을 높이는지, 그 개선을 학습한 모델이
탐색 없이도 더 강해지는지부터 확인한다. 두 조건을 통과하면 자기대국으로 확장한다.

우리 모델은 한 국면을 1개 공유 블록으로 8회 처리한다. 스도쿠의 16세그먼트 학습은
체스의 기본 설정에서 사용하지 않는다. 탐색의 각 새 국면도 이 8회 처리를 한 번 받는다.

이 선택의 근거는 “MCTS가 GRPO보다 언제나 싸다”가 아니다. 우리가 원하는 것은
비싼 평가 한 번에서 얻은 정보를 최대한 학습에 쓰는 것이다. 탐색은 후보별 가치와
상대 응수를 비교하고, 학습기는 그 결과로 만들어진 정책 분포를 직접 배울 수 있다.
이 역할 분담은 Expert Iteration의 관점과 맞는다. ExIt의 원 논문 실험은 Hex이며,
우리 체스 모델이나 GRPO와의 우위를 입증한 결과는 아니다.
[Expert Iteration 원 논문](https://arxiv.org/abs/1705.08439)

우선순위는 다음과 같다.

1. 정책·가치 지도학습과 탐색 없는 성능 기준선.
2. 가중치를 고정한 작은 탐색으로 정책 개선 여부 측정.
3. 저장된 국면의 탐색 결과를 재학습하는 오프라인 반복.
4. 개선이 확인되면 자기대국으로 방문 국면을 넓히기.
5. 동일 예산의 GRPO, 내부 반복 수, 기억 유무 비교.

MCTS는 데이터를 만드는 탐색법이고 GRPO는 정책을 갱신하는 방법이다.
원리상 함께 사용할 수도 있다. 여기서는 먼저 **탐색 목표에 대한 지도학습**을 택한다.

## 2. 내부 반복과 실제 수 탐색은 서로 다른 축이다

```text
t: 실제 체스의 ply. 한쪽이 수를 한 번 두면 1 증가.
k: 고정된 국면 하나를 처리하는 내부 반복. 기본 1..8.
n: MCTS의 simulation 번호.

s_t            = 규칙과 실제 기보 이력을 포함한 체스 상태
e_t            = 모델이 관찰할 수 있도록 인코딩한 s_t
M_(t,0)        = Init(e_t)
M_(t,k)        = F_theta(M_(t,k-1), e_t)
M              = (h, W, Z)
(p_(t,k),v_(t,k)) = Heads_theta(h_(t,k))

MCTS(s_t, p_(.,8), v_(.,8)) -> 개선 정책 pi_t, 실제 선택 수 a_t
s_(t+1)        = Rules.step(s_t,a_t)
```

8회 모두 같은 theta를 쓴다. 같은 정책 모델을 서로 다른 내부 상태에서 읽는 것이다.
현재 구조에서는 k번째 출력에서 샘플링한 체스 수를 k+1번째 내부 상태에 넣지 않는다.
따라서 8개 출력은 “수를 두고 상대 응수를 본 8개 가지”가 아니다.
MCTS는 Rules.step으로 실제 다음 국면을 만들기 때문에 그 기능을 추가한다.

처음 구현에서는 **새 국면마다 h/W/Z를 초기화**한다. 동일 국면의 8회 계산 안에서는
그 상태가 계속 이어진다. 부모 국면의 W/Z를 자식이나 형제에게 그대로 넘기면
같은 국면의 평가가 탐색 경로에 의존하게 되며, 기존 모델과 다른 설계가 된다.
그 변경을 기본안에 섞지 않는다.

MCTS 노드에는 규칙 상태, p/v, 방문 통계만 보관하면 된다. 8회 계산이 끝난 뒤
모든 노드의 h/W/Z를 계속 저장할 필요는 없다. 나중에 내부 계산을 중간부터 이어가는
설계를 시험할 때만 해당 노드의 내부 상태를 별도 보관한다.

## 3. 모델 크기와 체스 입출력

기본 후보는 v1.71의 공유 선형 주소 사영, Z 흔적, W 쓰기·읽기, MLP 순서를 유지한다.
체스에서의 유효성은 새로 학습해 검증한다. 스도쿠 가중치를 그대로 사용할 근거는 없다.

- 64개 칸 토큰, 8헤드, 1개 공유 블록, 한 국면당 8회 반복.
- 퍼즐 ID 임베딩은 제거한다.
- 입력은 기물·색과 차례, 캐슬링 권리, 앙파상, halfmove clock 등의 국면 정보.
- 각 출발 칸의 hidden에서 73개 이동 유형을 출력하여 전체 64×73 정책을 만든다.
  합법수만 남긴 뒤 전체 합법수에 대해 softmax한다. 칸별 softmax가 아니다.
- pooled hidden에서 scalar value를 출력한다. `v=tanh(value_logit)`이며 현재 차례
  플레이어 기준 기대 결과를 [-1,1]로 나타낸다. 무승부 결과는 0이다.

64×73 이동 표현은 AlphaZero의 체스 표현을 따른다. 나이트 이동·직선 이동·언더프로모션을
포함하며, 캐슬링·승격·앙파상은 규칙 엔진과 인코더의 왕복 일치로 검증해야 한다.
[AlphaZero 원 논문, Methods](https://arxiv.org/pdf/1712.01815)

현재 로컬 `LTLayer`를 CPU에서 생성해 센 파라미터 수는 다음과 같다.
주소 사영 linear, trace 사용, MLP expansion 4, 헤드 8을 적용했다.

| hidden d | 반올림된 MLP 폭 | 공유 블록 | 기물 임베딩+정책·가치 head 포함 소계 |
|---:|---:|---:|---:|
| 288 | 768 | 830,336 | 855,466 |
| 304 | 1,024 | 1,119,664 | 1,146,186 |
| 320 | 1,024 | 1,188,832 | 1,216,746 |

소계는 `블록 + 13*d + 73*(d+1) + (d+1)`이다. 부가 국면 정보의 인코더는
아직 구현 전이므로 이 숫자에 포함하지 않았다. MLP 폭은 코드에서 256의 배수로
반올림하므로 d를 조금 바꿔도 파라미터 수가 크게 달라질 수 있다.
**d=304를 약 112만 블록의 출발점으로 잡되, d=320과 실제 처리량도 비교한다.**

ChessBench의 작은 대조군은 약 9M, 8층, d=256, 8헤드다.
그 논문의 주요 action-value 모델은 `(국면, 후보 수)`를 입력으로 받아 후보마다 평가한다.
우리 policy/value 모델의 국면당 한 번 호출과 그 추론 비용을 그대로 등치하면 안 된다.
주요 구조 비교에는 같은 입출력·policy/value 손실을 쓰는 8층 Transformer도 둔다.
9M/8로 파라미터를 맞추는 것은 출발점이며 FLOPs·처리량 동등성의 증명이 아니다.
[ChessBench 논문, 2.3 및 A.2](https://arxiv.org/html/2402.04494v2)

규칙 엔진은 실제 기보를 포함한 상태를 유지한다. 반면 FEN만 있는 데이터는 반복 국면의
전체 과거를 복원할 수 없다. 없는 이력을 “반복 없음”이라고 라벨링하지 않는다.
초기 네트워크를 FEN 기반으로 학습하면 history에 관해서는 부분 관찰 가치 추정기가 된다.
탐색의 반복·무승부 판정은 완전한 규칙 상태로 수행하고, history 입력 추가는 별도 비교한다.
같은 배치나 캐시에서 이력이 다른 상태를 무조건 같은 상태로 취급하지 않는다.

## 4. 먼저 전문가의 정책과 가치를 배운다

기존 ChessBench에서 이용 가능한 국면·최선 수·state-value를 먼저 사용한다.
데이터 전체를 내려받는 것을 전제로 하지 않고 작은 고정 부분집합부터 시작한다.
게임 단위로 train/validation/test를 나누고 동일 국면 누출도 별도 집계한다.
[공식 데이터·코드](https://github.com/google-deepmind/searchless_chess)

```text
L_warmup = CE(pi_teacher, p_8) + c_v * (v_8 - V_teacher)^2
```

최선 수만 있으면 pi_teacher는 해당 수의 one-hot이다. 모든 합법수의 Q 라벨이 있으면
그 값에 온도를 적용한 soft target도 비교할 수 있다. 이는 새 설계 선택이다.
일부 수의 Q만 있을 때 나머지를 0점으로 채워 완전한 정책 목표인 것처럼 만들지 않는다.

데이터의 값 범위가 [0,1]이면 모델 target으로 `2*q-1`을 사용한다. 다만 제공된
Stockfish 점수 변환값은 실제 자기대국 승·무·패 빈도와 동일한 확률이라고 가정하지 않는다.
자기대국으로 전환할 때 가치의 의미와 calibration이 달라질 수 있다.
또한 child state를 엔진으로 평가했다면 차례가 바뀌므로 부모 관점으로 부호를 뒤집는다.

초기 손실은 **8번째 출력에만** 적용하고, 8회 내부 계산 전체에 역전파한다.
중간에 detach하거나 반복마다 optimizer step을 하지 않는다.
초기 출력에도 같은 정답을 주는 deep supervision은 추가 실험으로 분리한다.
중간 표현이 일시적으로 나쁜 수를 출력해야 최종 계산이 좋아질 가능성까지 미리 제거하지 않는다.

## 5. 작은 탐색이 정확히 하는 일

먼저 PUCT를 구현하여 해석 가능한 기준으로 삼는다. 노드 s의 각 합법수 a에
방문 수 N, 부모 플레이어 관점의 평균 가치 Q, prior P를 저장한다.

```text
score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(1 + sum_b N(s,b)) / (1 + N(s,a))
```

한 simulation은 다음 순서다.

1. 점수가 높은 간선을 따라가며 규칙 엔진으로 국면을 전이한다.
2. 아직 평가하지 않은 비종료 국면에서 모델을 8회 실행해 p/v를 얻는다.
3. 종료 국면은 정확한 승·무·패 값을 사용하며 네트워크를 호출하지 않는다.
4. 얻은 값을 경로에 되돌려 주면서 ply마다 부호를 뒤집고 N/Q를 갱신한다.

예를 들어 어떤 수 뒤의 child가 상대에게 `v=-0.6`이면 그 수의 부모 평가 기여는 +0.6이다.
터미널 결과를 leaf value와 별도 transition reward에 동시에 넣어 두 번 더하지 않는다.
이 설계는 중간 보상 0, terminal leaf의 정확한 결과, ply별 부호 반전을 사용한다.

PUCT 기준의 정책 목표는 `pi_search(a) = N(s,a) / sum_b N(s,b)`로 한다.
실제로 수를 선택하는 온도는 별도 설정으로 둔다. 미방문 간선의 초기 Q는 부모 v로
두는 것을 파일럿 기본값으로 기록하고, c_puct와 함께 고정해 비교한다.
이는 우리 구현 선택이며 AlphaZero 전체 설정의 재현이라고 부르지 않는다.

AlphaZero는 p/v로 탐색하고, 탐색 정책과 최종 경기 결과로 네트워크를 학습한다.
우리는 이 분업을 가져온다. 원 논문의 대규모 자기대국 예산을 가져오는 것은 아니다.
[AlphaZero의 목적함수와 탐색](https://arxiv.org/pdf/1712.01815)

### 작은 예산에서 Gumbel을 비교할 이유

우선 후보 설정은 root 후보 8개, simulation 32회다. 여기에 16회를 비교한다.
Gumbel-Top-k로 후보를 중복 없이 뽑고 Sequential Halving으로 유망한 후보에
남은 예산을 배분한다. “8번 내부 반복이 서로 다른 수를 내기를 기다리는 것”보다
탐색 후보의 다양성을 명시적으로 제어할 수 있다.

Gumbel 방식의 학습 목표는 단순 방문 비율과 다르다. 방문한 수의 Q와 미방문 수의
completed Q를 사용해 prior를 개선한 분포를 만든다. 개념적으로는 다음과 같다.

```text
pi_improved = softmax(prior_logits + transformed_completed_Q)
```

`transformed_completed_Q`는 raw Q를 임의로 더한 값이 아니다. 값 보완·척도 조정을
포함한 해당 알고리즘의 정의를 사용한다. 구현 시 공개 Mctx의 `action_weights`와
대조한다. 선택한 실제 수, 방문 수, 학습 target은 각각 따로 저장한다.
[Mctx 공식 구현](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)

정확한 Q 아래의 정책 개선 성질을, 오차가 있는 신경망 가치로 돌리는 체스 전체의
성능 보장으로 옮기지 않는다. Gumbel 논문의 체스 주 실험은 400회 simulation이다.
매우 작은 예산의 결과는 주로 9×9 Go 실험에 제시된다. **체스 16·32회는 우리가
검증할 제안이지 이미 입증된 충분 예산이 아니다.**
[Gumbel AlphaZero 논문, Sections 3–6](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)

## 6. 자기대국 전에 탐색의 학습 가치를 판정한다

첫 반복은 게임 종료까지 기다릴 필요가 없다. 지도학습에 사용하지 않은 검증 국면에서
모델을 고정하고 탐색 유무에 따른 선택 수를 비교한다. 평가용 전문가 Q는 학습용 탐색의
leaf 평가에 넣지 않는다. 자기 자신의 가치 오류를 이용해 자신이 좋아졌다고 판정하는
순환을 피하기 위해 별도의 고정 평가기가 필요하다.

```text
regret(s,a) = max_b Q_eval(s,b) - Q_eval(s,a)
improvement(s) = regret(s,a_prior) - regret(s,a_search)
```

동일 국면의 paired improvement가 양수인지 확인한다. Q_eval도 유한 예산 엔진의
추정치라는 한계가 있으므로, 전술 정답과 실제 대국 결과를 함께 본다.
8회 탐색 없는 모델에 더 많은 내부 계산을 준 경우도 같은 시간 예산에서 비교한다.

검색이 유용하면 **학습 split의 국면**을 다시 탐색하여 다음을 저장한다.

```text
(state, pi_search, existing_teacher_value, actor_version, R, search_budget)
```

그 데이터의 policy target으로 student를 학습하고 value는 기존 전문가 라벨로 유지한다.
이 단계는 탐색 기반 policy distillation이며, 완전한 AlphaZero 자기대국 학습은 아니다.
탐색 target은 고정 데이터로 취급하며 MCTS 내부에 미분하지 않는다.
가중치를 갱신한 모델이 탐색 없이 개선됐는지 별도 holdout에서 측정한다.

초기에는 각 minibatch의 절반을 원래 전문가 데이터로 유지하는 혼합을 제안한다.
이는 약한 자기 탐색 target으로 전문가 지식을 잃는지 확인할 제어 장치다.
혼합률 50%는 검증할 시작값이며, 결과를 보고 사후에 성공 기준을 바꾸지 않는다.
고정 분포에서의 distillation을 통과한 뒤 learner가 방문한 새 국면을 추가한다.

## 7. 그 다음에 자기대국을 붙이는 방법

actor의 가중치를 한 묶음으로 고정하여 여러 게임을 병렬 생성한다. 각 수의 root에서
탐색한 정책과 실제 선택 수를 저장한다. 게임 종료 후 그 국면의 현재 플레이어 기준으로
승 +1, 무 0, 패 -1을 붙인다.

```text
replay item = (state, legal_mask, pi_search, outcome, actor_version, search_config)
L_selfplay = CE(stopgrad(pi_search), p_8) + c_v * (v_8 - outcome)^2
```

초기 c_v는 1로 두되 두 손실의 크기와 trunk에 전달되는 gradient norm을 함께 기록한다.
regularization, optimizer, clipping은 모든 학습법 비교에서 동일하게 둔다.
새 알고리즘별 최적 learning rate를 비교할 경우 각 방법에 같은 튜닝 예산을 준다.

replay에는 최근 국면과 전문가 anchor를 섞는다. 특정 모델 버전의 게임만 과도하게
반복해서 쓰지 않도록 버전·중복률·게임 단위 샘플링 비율을 기록한다.
한 게임/탐색 도중 actor 가중치를 바꾸지 않는다. learner의 최신 raw 또는 EMA 중
하나를 명시적으로 선택해 새 actor snapshot을 만들고, 두 가중치를 한 tree 안에 섞지 않는다.

게임 길이 제한으로 잘린 trajectory를 정확한 무승부로 라벨링하지 않는다.
규칙상 draw와 인위적 timeout을 구별하고, 후자는 초기 구현에서 outcome loss에서 제외한다.
draw claim 처리, 반복 판정, 시간 제한은 학습과 평가에서 같은 명시적 환경 규약을 쓴다.

첫 구현에서는 각 실제 수마다 새 tree를 만든다. 이후 tree 재사용은 정확한 게임 상태와
actor 버전이 일치할 때만 추가한다. neural cache key에도 history/관찰 표현, 모델 버전,
내부 반복 수를 포함한다. MCTS 방문 통계까지 일반적인 신경망 출력 캐시처럼 합치지 않는다.

이 단계의 이점은 사람이 제공한 국면 밖에서 생기는 실수를 새 데이터로 수집한다는 것이다.
DAgger가 learner 방문 분포에 전문가 지도를 붙이려는 취지와 연결되지만, 여기의 정책
전문가는 모델과 탐색으로 구성된다. 데이터 합치기만으로 DAgger의 모든 보장이 생기지는 않는다.

## 8. 8개 중간 출력은 어떻게 활용할 것인가

첫 버전에서는 탐색이 만든 target을 최종 p_8에만 적용한다. 최종 loss의 gradient가
h/W/Z와 1..8회 계산을 지나므로, 앞선 반복도 이미 학습되고 있다.

그 다음 유망한 추가 실험은 **중간 출력의 다양성 강제보다, 더 빠른 정책·가치 추정**이다.

```text
L_aux = average over k in {2,4} of
        [CE(pi_search, p_k) + c_v * (v_k - value_target)^2]
L = L_final + lambda_aux * L_aux
```

lambda_aux=0과 0.25를 우선 비교한다. 목표는 충분히 학습된 k=2/4 출력을 싼 leaf
평가기로 쓸 수 있게 만드는 것이다. 같은 target을 반복에 주는 것이 후반 계산을
방해하는지도 최종 p_8 성능으로 판정한다. 이 계수와 출력 위치는 제안값이다.

초기에는 한 search 전체의 leaf 반복 수 R을 고정한다. 더 나중에는 싼 평가로
후보를 거른 뒤 중요한 leaf만 R=8로 계산할 수 있지만 다음 문제가 생긴다.

- R=2와 R=8의 value calibration이 다를 수 있다.
- 같은 leaf를 정밀 평가했다고 두 번의 독립 증거처럼 backup하면 통계가 왜곡된다.
- 이미 방문한 노드의 prior를 바꾸면 이전 방문 분포와 새 prior의 관계가 달라진다.

따라서 가변 R을 기본 성능의 근거로 삼기 전에, 고정 R별 탐색을 먼저 비교한다.
같은 국면에서 이어 계산하는 경우에는 모델·입력·내부 상태가 같아야 prefix를 재사용할 수 있다.

중간 출력의 순위가 계속 좋아지는지도 관찰할 수 있다. 다만 p_k의 변화만으로
실제 수순 탐색이나 STDP의 선후 강화 규칙을 증명할 수는 없다.

## 9. GRPO와 비교할 때 바로잡아야 할 점

**같은 p_8에서 수 8개를 샘플링하는 데 8번의 모델 순전파가 필요한 것은 아니다.**
한 번 8회 unroll하여 p_8을 얻은 뒤 categorical sampling만 8번 하면 된다.
추가 비용은 주로 후보 수의 전문가 평가다. 중복 수는 Q를 재사용할 수 있다.
그러므로 작은 모델이라고 해서 MCTS가 이 방식보다 계산상 더 싸다고 단정할 수 없다.

반복마다 한 수를 뽑는 안은 별도다.

```text
a_k ~ p_(theta_old,k)(.|s), k=1..8
r_k = Q_expert(s,a_k)
ratio_k = p_(theta,k)(a_k|s) / p_(theta_old,k)(a_k|s)
```

여기서 `p_(theta,k)`는 theta로 초기 상태부터 k회 재계산한 분포다.
old theta의 hidden을 고정한 채 head만 재계산하면 전체 재귀 정책의 비율과 다르다.
clipped policy objective와 reference KL을 쓸 수 있지만, 원 GRPO의 group normalization을
우리 상황에 적용하는 효과는 별도로 확인해야 한다.
[GRPO 원 논문, DeepSeekMath 4.1](https://arxiv.org/html/2402.03300v2)

수 샘플을 다음 반복에 넣지 않고, 고정된 내부 trajectory에서 독립적으로 뽑는 경우
다른 표본의 보상 평균은 현재 샘플과 독립인 baseline으로 사용할 수 있다.
하지만 앞서 논의한 leave-one-out과 전체 group 평균은 고정 K에서 다음 관계다.

```text
r_k - mean(r_except_k) = K/(K-1) * (r_k - mean(r_all))
K=8이면 배율은 8/7.
```

따라서 이 변경만으로 새 방향의 gradient나 큰 효율 이득이 생기는 것은 아니다.
독립 on-policy 표본의 비정규화·비클립 REINFORCE에서 전체 평균을 빼면 자기 포함에
따른 (K-1)/K 축소가 생기며, leave-one-out은 그 축소를 제거한다. std로 나누거나
PPO clipping을 붙인 목적함수의 성질까지 이것으로 증명하지 않는다.

또한 1..8회 출력 모두에 보상을 주면 평균 중간 성능을 최적화하는 목적이 추가된다.
최종 8회 출력만 배포할 경우 그것이 유리하다는 보장은 없다.
**공정한 첫 GRPO 대조군은 p_8에서 8개를 뽑는 방식**으로 두고,
반복별 표본 방식은 중간 출력 활용의 별도 ablation으로 둔다.

모든 합법수의 전문가 Q가 이미 제공되는 국면에서는 sampling 기반 학습 전에
그 Q로 만든 정책 target을 직접 학습하는 기준선을 반드시 둔다.
MCTS와 GRPO 모두 이 싼 기준선을 이길 이유를 실험으로 보여야 한다.
Stockfish 보상 자체에도 lookahead가 들어 있으므로 “GRPO에는 탐색이 없다”는 설명은
이 비교에 맞지 않는다. 차이는 lookahead를 수행하는 주체·예산과 결과를 학습하는 방식이다.

## 10. 비용을 어디까지 셀 것인가

```text
C_R = 한 국면의 R회 내부 반복 + 입출력 head 비용
N_new = root 외에 실제로 새로 평가한 비종료 국면 수

MCTS의 모델 비용 ~= C_R(root) + N_new * C_R(leaf)
전체 데이터 생성 비용 = 위 비용 + 규칙/트리 CPU 비용 + 필요한 전문가 평가 비용
learner 비용 = R회 forward/backward * 학습 batch 수
```

simulation 수는 새 neural evaluation 수와 같지 않을 수 있다. terminal, cache hit,
중복 leaf 요청을 따로 세어야 한다. 총비용에는 warmup, 재분석, replay의 반복 사용,
평가 시간, peak memory도 기록한다.

예를 들어 root까지 모두 R=8이고 새 국면 32개를 평가하면 블록 호출은 약 264회다.
8회 내부 반복을 MCTS의 8회 simulation으로 세거나, 이 탐색 비용을 learner의
optimizer step 비용에 숨기면 안 된다.

| 고정 R | 새 leaf 평가 수 예시 | root 포함 블록 호출 근사 |
|---:|---:|---:|
| 2 | 32 | 66 |
| 4 | 16 | 68 |
| 8 | 8 | 72 |

이 표는 예산 분할의 출발점이다. 동등한 wall time을 보장하지 않으므로 실제 측정으로
맞춘다. 작은 R의 policy/value를 학습하지 않은 모델을 조기 종료시켜 놓고 R=8과의
공정한 모델 비교라고 부르지 않는다. 앞 절의 auxiliary 학습 여부를 고정해야 한다.

GPU batching은 **여러 독립 tree에서 leaf를 하나씩 모으는 방식**부터 구현한다.
한 tree에서 여러 미완료 simulation을 동시에 예약하는 복잡성은 뒤로 미룬다.
CPU 규칙 처리, inference queue 대기, GPU 이용률을 나눠 측정한다.
Mctx는 JAX 구현이므로 현재 PyTorch 모델과 그대로 연결해 GPU 복사 비용을 숨기지 않는다.
알고리즘 참조로 사용하고, 첫 실행기는 CPU 규칙 엔진과 PyTorch batched inference로 구성한다.
[Mctx 공식 프로젝트](https://github.com/google-deepmind/mctx)

## 11. 적은 실험으로 판단하는 순서와 중단 기준

아래 예산은 파일럿 제안이다. 실행 전 config에 고정하며, 결과를 본 뒤
기준을 바꿀 때에는 새 실험으로 구분한다. 처음부터 모든 조합을 전수 실행하지 않는다.

| 단계 | 먼저 실행할 내용 | 다음 단계로 갈 근거 | 실패했을 때의 조치 |
|---|---|---|---|
| A | 인코더, 합법수 왕복, mate/draw 및 value 부호 검산 | 규칙과 backup의 불일치 0 | 학습 전 구현 수정 |
| B | 고정 부분집합 지도학습, R=8, 재귀 모델과 8층 기준선 | holdout 정책·가치가 학습되고 finite gradient 유지 | 입력·target·최적화 진단 |
| C | 고정 checkpoint, holdout 2,048국면, 탐색 0/16/32 | 고정 Q 평가의 paired regret 감소 | value/후보 커버리지 점검 후 예산 재결정 |
| D | C에서 선택한 한 설정으로 학습 국면 10,000개 재분석 | 재학습 뒤 탐색 없는 holdout 개선 | target 오류 또는 증류 실패 진단 |
| E | 개선된 설정만 자기대국, GRPO 및 직접 Q 증류 비교 | 같은 전체 예산에서 대국 점수 개선 | 더 싼 기준선 채택 |

단계 C에서는 PUCT-32와 Gumbel-32를 먼저 비교하고, 가능성이 있는 쪽만 16회로 줄인다.
2,048국면은 개막·중반·종반과 전술 국면을 구분해 집계한다. 선택 수의 regret 외에도
정책 prior가 좋은 수를 후보에 포함했는지, leaf value 오차가 어떤 수를 과대평가했는지 본다.
탐색이 실패했을 때 무조건 simulation 수만 늘리지 않는다.

단계 C의 예산 선택용 validation과 최종 test를 분리한다. 개발용 validation에서는
paired bootstrap 95% 구간과 층별 결과를 보고 다음 실행을 정하되, 확정한 한 설정의
개선 주장은 미사용 test에서 재확인한다. 가능성이 없으면 정한 파일럿 예산에서 중단한다.

후보가 남으면 같은 시작 국면을 양쪽 색으로 교환하는 대국을 최소 200쌍 진행한다.
고정 opponent와 고정 수당 시간에서 승/무/패 및 game score를 보고하며, 쌍 단위
bootstrap 구간을 사용한다. 간격이 넓으면 우위 미확정으로 기록하고 추가 대국 예산을
별도로 잡는다. Elo 숫자를 서로 다른 사이트·상대 풀 사이에서 직접 비교하지 않는다.

최종 훈련법 비교는 같은 warmup checkpoint, 같은 신규 GPU 시간·CPU/엔진 예산에서
각 3개 seed로 확인한다. 학습 step 수만 같게 하는 비교와 총시간을 같게 하는 비교를
혼동하지 않는다. 기존 라벨 비용은 공통 sunk cost로 별도 표기하고, 신규 라벨 비용은 포함한다.

반드시 분리해서 답할 질문은 다음과 같다.

- 탐색 자체가 나은 수를 선택했나? 고정 모델의 탐색 유무 비교.
- 그 결과를 학습해 모델이 나아졌나? 학습 전후의 **탐색 없는** 평가.
- 외부 탐색 아래에서도 재귀 구조가 유리한가? 같은 search와 비용의 Transformer 비교.
- W/Z의 이점인가? 같은 훈련·탐색 예산의 기억 없는 재학습 대조군.
- 내부 계산을 늘린 효과인가? 학습 조건을 명시한 R별 정확도·시간 곡선.

학습된 모델에서 W를 갑자기 0으로 만드는 실험은 의존성을 보여준다.
그것만으로 해당 구조가 다른 구조보다 효율적으로 학습된다는 결론은 내리지 않는다.
체스 성능 개선이나 beta 분포만으로 STDP 원리가 입증됐다고 주장하지 않는다.

## 12. 구현할 때 남겨야 할 기록

각 실행의 config에 다음을 남긴다.

```text
data split/hash, state encoding, history availability, legal action encoding
model parameter count, R, memory configuration, raw/EMA actor version
warmup checkpoint, search algorithm, root candidates, simulation budget
N_new, cache/terminal counts, Q initialization, Q transform, action temperature
policy target definition, value target source/perspective, loss coefficients
actor GPU time, rule/tree CPU time, oracle CPU time, learner GPU time
replay age/reuse, finite-gradient checks, policy/value calibration
no-search metrics, search metrics, paired games, seeds, confidence intervals
```

현재 결정은 **약 1.15M급 공유 모델의 8회 계산 → 전문가 warmup → 소규모 탐색 검증
→ 탐색 정책 증류**다. 자기대국과 가변 내부 반복은 앞 단계가 실제 효용을 보일 때 붙인다.
검증할 연구 주장은 “명시적 탐색으로 만든 개선을 적은 파라미터의 내부 반복이
얼마나 효율적으로 흡수하는가”이며, 아직 체스에서 달성된 결과는 없다.
