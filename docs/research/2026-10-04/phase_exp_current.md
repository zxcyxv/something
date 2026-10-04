# 고정 위상차의 정확한 지수형 창

variant: `phase_exp_current_only`; `ExponentialPhaseCurrentReadInner`.

`delta[h,i,j] = phiV[h,i] - phiK[h,j]`
`L(delta) = sign(delta) * exp(-abs(delta)/1.0)`
`G = mean_tokens(V.T @ RoPE(K)) * L(delta)`
`read = RoPE(Q) @ G.T` (열 벡터에서는 Gq).

위상, 초기화, 공간 RoPE, raw QKV, 두 post-residual RMSNorm, token mean,
optimizer 및 데이터는 이전 sine 모델과 같다. 현재 G만 사용하며 carry와 trace는
무시한다. 실수 활동의 부호는 위상차 창과 별개로 외적에 남는다.
단일 복소 외적의 허수부가 아닌 명시적인 채널 쌍별 위상차 커널이다.
고정 위상이므로 커널을 token 합 밖에서 곱할 수 있다.
공통 carrier는 위상차에서 소거되며 oscillator를 실제로 계산하지 않는다.

선택: tau=1 rad 고정 (물리 시간이라면 omega*tau_time), A+=A-=1 고정,
위상차 wrap 없음, 정확히 0에서는 L=0. 양쪽 극한은 +1/-1이다.
진짜 불연속 창을 사용하며 smoothing/surrogate gradient는 없다.
sign의 gradient는 0; 0 밖에서는 지수 envelope를 통해 위상이 학습된다.
이는 0 교차의 불연속을 gradient가 직접 예측하지 못한다는 한계가 있다.
읽기 크기를 sine 모델에 맞추는 추가 gain은 두지 않았으므로 성능 차이는
창 형태와 읽기 크기 변화 양쪽을 포함한다.

실행:
```bash
python -m lt.research_kv_collapse --variant phase_exp_current_only \
  --config configs/kv_phase_current_research.json \
  --out runs/kv_phase_exp_current_new --steps 10000
```
이 지수형 실험은 step 10025에서 종료했다. 위 명령은 새로운 출력 폴더를 사용한다.
원본 로그와 실제 설정은 `logs/kv_phase_exp_current_20261004`에 보존했다.

이전 sine 실험은 사용자 요청으로 step 7475에서 정상 종료하고 저장했다.
새 실험은 이전 가중치를 로드하지 않고 동일 seed로 처음부터 학습한다.
