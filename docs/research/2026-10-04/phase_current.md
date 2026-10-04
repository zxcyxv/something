# 현재 위상 G 읽기

실행 옵션: `phase_current_only`. 기존 B-only 및 과거 실험 옵션은 유지한다.

```bash
python -m lt.research_kv_collapse --variant phase_current_only \
  --config configs/kv_phase_current_research.json \
  --out runs/kv_phase_current_new --steps 20000
```

행 벡터 구현은 `RoPE(Q) @ G.T`, 열 벡터 표기로는 `Gq`다.
`G_ij = mean_tokens(V_i * RoPE(K)_j * sin(phiV_i - phiK_j))`.
복소수 `zV zK^H`의 허수부를 두 실수 행렬곱으로 계산한다.
공통 carrier `exp(i omega t)`는 소거되므로 시계 상태를 만들지 않는다.
이 G는 일반적으로 반대칭 행렬이 아니다.

## 구현에서 선택한 세부 사항

- K/V 각각 층별 `[heads, head_dim]` 독립 위상. hidden=832면 각각 832개다.
  모든 토큰과 재귀 반복이 같은 위상 파라미터를 사용한다.
- 위상은 slow gradient 학습 대상이며 추론 중에는 고정된다.
  `phi = (pi/2) tanh(theta_raw)`; 초기 실제 각도는 독립 uniform
  `[-pi/4, pi/4]`. 동일 위상 초기화에 따른 전체 G=0을 피한다.
- 기존 optimizer의 theta 제외 규칙으로 위상 weight decay는 0이다.
  Q/K/V와 output/FFN의 기존 weight decay 설정은 유지한다.
- 위상차 범위를 [-pi, pi] 안에 두지만 sine의 비단조성은 남는다.
  이는 지수형 STDP 창이나 모든 앨리어싱 문제를 해결한 구현이 아니다.
- Q는 실수이며 별도 뉴런 위상을 붙이지 않는다. 공간 RoPE는 Q/K에
  기존대로 적용한 뒤 K 채널에 뉴런 위상을 붙인다. V는 공간 회전하지 않는다.
- G 누적, 과거 K/V trace, 읽기 보간은 없다. 반환 carry의 G는 진단용으로만
  남고 다음 반복에서 무시한다. 기존 초기화 비교를 위해 미사용 trace 파라미터는
  남아 있으나 gradient나 읽기에 참여하지 않는다.
- 활동은 signed real projection이며 실제 발화 시각을 검출하지 않는다.
  sine 위상차 창만 구현한다. tau, 지수 감쇠, 고조파, B 항, 별도 gain은 추가하지 않았다.
- 어텐션 residual과 FFN residual 직후 각각 URM식 FP32 RMSNorm을 적용한다.

검증: 복소수 외적 및 직접 sin 위상차 수식과 일치, 공통 carrier 소거,
NaN 과거 carry 무시, 위상/QKV gradient, 위상 범위와 decay 제외,
state_dict 복원, 블록 내 두 번의 정규화 및 optimizer 위상 갱신.
