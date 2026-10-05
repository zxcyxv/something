# 자유위상 사인 4성분 모델의 전체 일정 재개

사용자 요청에 따라 2026-10-05 08:04:11 UTC에 3,008스텝 checkpoint에서
같은 모델의 학습을 재개했다. 연구 대조군의 원본 로그·설정은
[연구 보고서 자료](2026-10-05/free_phase_window_research/REPORT.md)에 보관되어 있다.

- 실행 디렉터리: `runs/research_free_phase_free_fourier4_e05_20261005`
- 모델: 자유로운 순서, 토큰별 채널 gain, 사인 4성분, epsilon=0.5
- 정밀도: 기존 BF16 본체 / FP32 위상·특징·메모리
- 설정: seed0, batch128, hidden832, 8heads, 8blocks/segment, 16segments
- 전체 일정: 기존 50,000epochs, 실제 종료 예정 optimizer step 390,600
- 별도 step/time 제한: 없음 (`max_steps=null`, `max_hours=null`)
- 저장: 1,000스텝 간격 및 기존 평가 경계, 최근 3개만 유지
- 최초 재개 PID: 23333

원래 trainer의 strict resume 경로로 raw weights, optimizer, EMA, carry, RNG,
데이터 cursor를 복원했다. raw weights와 EMA는 실제 복원된 전체 tensor를 원본
checkpoint와 비교하여 정확히 일치함을 추가 확인했다. 학습률은 1e-4로 이어지며
warmup을 다시 시작하거나 모델 구조·창·연산 정밀도를 바꾸지 않았다.

재개 직후 확인 시점은 3,788스텝이었다. 새 780개
스텝의 기록된 손실이 모두 finite였고, 최근 128스텝 중앙값은
0.12205초/스텝이었다. 마지막 16개 종료 배치의
평균 훈련 칸 정확도는 93.41%, 완전 정답률은
23.10%였다. 진행 중인 학습의 시점 기록이며
전체 일정 완료나 새로운 test 평가를 뜻하지 않는다.

재개 진입점은 [resume_free_phase_windows.py](../../lt/resume_free_phase_windows.py)다.
이 명령은 해당 디렉터리의 최신 checkpoint와 저장된 protocol을 사용한다.
로그/체크포인트 step이 다르면 자동으로 진행 기록을 덮어쓰지 않고 중단한다.

```bash
python -u -m lt.resume_free_phase_windows \
  --run runs/research_free_phase_free_fourier4_e05_20261005 \
  --steps 0 --save-every 1000 --keep-last 3
```

위 명령은 이미 실행 중이다. 동시에 재실행하지 않는다. 진입점의 파일 잠금이
같은 런의 중복 재개를 차단한다. 콘솔은 `continue.log`, step별 손실·속도·종료 배치
정확도는 `train.jsonl`, 시작 및 종료 상태는 `resume_status.json`에 기록한다.
이전 설정·종료 기록과 재개 코드 snapshot은 런의 `continuations/`에 보존한다.
학습 checkpoint는 Git에 포함하지 않는다.
