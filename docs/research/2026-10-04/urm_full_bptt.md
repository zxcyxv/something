# URM 구조의 공통 하네스 비교

현재 유효한 비교는 `urm_swiglu_loops16_20261004`다. 학습은 6025 step에서
종료했으며 비교 구간은 최대 6000 step까지 맞춘다. 설정은
`configs/urm_swiglu_2layer_8iter_loops16.json`이고 기본 runner도 이 설정과
일반 SwiGLU를 선택한다.

원본은 UbiquantAI/URM commit `c14e55f5f9227873617015cf60a239126b55adcd`다.
실행에 필요한 네 파일은 `lt/urm_vendor`에, 원본 학습 코드와 YAML 및 해시는
[reference/urm](reference/urm/IMPORT_PROVENANCE.json)에 보관했다.
[변경/출처 설명](../../../lt/urm_vendor/README.md)도 남겼다.

## 실제 forward와 학습 단위

한 segment에서 다음을 8번 실행한다. layer는 서로 다른 가중치 두 벌이다.

```text
hidden = hidden + input_embedding
hidden = URM_layer_0(hidden)
hidden = URM_layer_1(hidden)
```

각 URM layer는 softmax attention residual 직후 RMSNorm, FFN residual 직후
RMSNorm을 적용한다. 토큰별 전체 hidden 채널, FP32 제곱 평균, eps=1e-5,
입력 dtype 반환, affine scale/bias 없음이다. Q/K의 별도 정규화는 없다.
일반 SwiGLU는 `down_proj(SiLU(gate) * up)`이고 convolution은 없다.

segment당 내부 재귀 8회, layer 적용 16회다. 내부 no_grad/detach가 없고
마지막 logits의 loss로 한 번 backward/update한다. Activation checkpointing은
그 그래프를 재계산한다. segment 경계에서는 hidden을 detach하고 optimizer를
갱신하므로, 외부 16 segment 전체를 관통하는 full BPTT는 아니다.

ACT는 꺼져 있다. 공통 fixed-loop wrapper의 q_logits=-5이며 원본 q_head는
사용하지 않는다. 같은 샘플을 16 segment 유지하고 17번째 호출에서 교체한다.
테스트에서 q_head bias를 +100으로 바꾸고 다른 입력을 매번 공급해도 이 조건을
유지했다. 데이터/loss/AdamATan2/EMA와 샘플 유지/update 프로토콜은 B/phase와 같다.
URM은 prefix puzzle token, 1D RoPE, 원본 초기화/재주입을 유지하며,
B/phase보다 layer가 한 개 더 많아 계산 깊이와 파라미터 수는 다르다.

hidden832, heads8, batch128, seed0, lr1e-4, warmup2000, weight_decay1,
EMA=.999, Sudoku 1k/augmentation1000이다. QKV와 FFN 모두 weight decay 대상이다.
사용자 요청으로 내부 반복 수를 직접 8회로 정했다. 원본 YAML 기본값인
num_layers=8, H_cycles=4, L_cycles=3, loops=16을 4-layer/42회로 설명하지 않는다.
원본은 내부 no_grad 구간 및 ACT가 있어 이 비교와 다르다.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m lt.research_urm_full_bptt \
  --config configs/urm_swiglu_2layer_8iter_loops16.json --ffn swiglu \
  --out runs/urm_swiglu_loops16_new --steps 6000
```

## 최초 loops=1 런의 제한

`urm_2layer_8iter_full_bptt_20261004`는 ConvSwiGLU로 2050 step,
`urm_swiglu_2layer_8iter_20261004`는 일반 SwiGLU로 850 step 실행했다.
두 런 모두 ACT는 이미 꺼져 있었지만 outer loops=1로 만들어 새 입력을
매 optimizer step에 받았다. 따라서 기존 B/phase와 구조만의 곡선 비교가 아니다.
샘플 교체를 ACT 때문이라고 했던 설명은 잘못이었다.

이 오류를 확인한 뒤 loops=16으로 처음부터 다시 학습했다. 두 이전 설정은
재사용 기본값에서 제거했으며 당시 실제 config와 전체 스칼라 로그는
`logs/urm_2layer_8iter_full_bptt_20261004`와
`logs/urm_swiglu_2layer_8iter_20261004`에 보존했다.
필요하면 그 config를 명시하고 `--ffn convswiglu` 또는 `--ffn swiglu`로 재현한다.
원본 감사는 [urm_loops_audit](evidence/urm_loops_audit_20261004.json)에 있다.

## 검증과 지표

16개 layer 적용 모두 finite nonzero gradient, activation checkpoint 사용 여부의
출력/gradient 일치, 두 RMSNorm 위치, ACT 비활성화, loops16 샘플 유지/교체,
segment 경계 detach를 검증했다. 관련 전체 검사 51개가 통과했다.

`train.jsonl`의 `_count_raw > 0`인 종료 segment의 loss와 정확도만 같이 비교한다.
전체 segment 평균 loss는 별도 지표다. 최초 loops1의 학습 곡선과 과거 console의
혼합 segment loss를 이 비교에 합치지 않는다. 숫자는 [오늘 연구 기록](README.md)과
[보존한 학습 로그](logs/urm_swiglu_loops16_20261004/config.json)에 있다.
