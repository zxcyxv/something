# LinearTuring

활성화 함수 없는 공유 블록 재귀 모델로 스도쿠를 푼다. 칸 사이 결합은 매 블록 새로 쓰는 KV 행렬 하나다.
그 쓰기는 블록 축의 위상 시계로 시각을 매긴 사건 쌍의 STDP 적분이다:

    G_b = Σ_t v k̃ᵀ  +  α_h · Im S_b,     τ = n·ωT + c·sin φ

Hebb 운반자에 STDP 창(블록 순서로 LTP/LTD 부호가 정해지는 감쇠 사인 λ^k sin(kωT))을 더한 것이다.
설계와 근거는 [docs/architecture.md](docs/architecture.md)에 있다. 결과, 부정 결과, 미결 문제는 [docs/results.md](docs/results.md)에 있다.
A가 어떤 일반 형태의 특수한 경우인지 해체한 분석은 [docs/claude_analysis_expressivity.md](docs/claude_analysis_expressivity.md)에 있다(Claude의 주장, 실험 미검증).

| 모델 | 상태 | held-out (칸 / 완답, 2048; A는 50.8k) | 외삽 (seg16 → 128세그먼트까지 최고, 512문제) |
| --- | --- | --- | --- |
| A: ρ = 0 | 2026-10-14 학습 | 83.4% / 1,087 | 267 → 288 (오차 감소 8.6%) |
| B: W ← 0.5W + Im S | 2026-10-15, 40k에서 중단 | 39k 81.2% / 954 (A 82.3% / 1,038) | 40k 220 → 240 (6.8%, A 41k 8.6%) |

학습된 STDP 항은 두 성분으로 나뉜다.
- 순서부 O: 풀이의 속도와 안정을 맡는 TD 델타 규칙이다. 고정점에서 0이 되므로 라벨 없는 수렴 검출기로도 쓴다(97%).
- 지연부 L: 정확도를 맡는다. 고정점에서 κ·SB와 같다(측정 1.218, 예측 1.221).

## 구조

```text
lt/model.py           모델: 시계, 위상 태그 흔적, 쓰기·읽기, 블록, 세그먼트(URM 레인 교체)
lt/kernel.py          융합 Triton 블록(순전파·역전파, ρ = 0), RTX 4090 약 90 ms/step
lt/data.py            데이터 검사, 고정 증강 풀, 배치 스트림
lt/train.py           하네스: stablemax CE, AdamATan2, EMA, 평가, 마일스톤 외삽, 정확한 재개, torchrun
lt/analysis/          해석 도구
  order_latency.py      Im S = O + L 분해, 고정점 검사(κ), 라벨 없는 수렴 검출
  interventions.py      O/L 제거·반전, ωT 반전, 정착 뒤 반전, 논리적 깊이 대비 결정 블록
  depth.py              L0/L1 동기식 제약 전파 깊이
  extrapolate.py        학습 지평 너머 세그먼트, 학습 퍼즐 대 held-out
tests/                쌍 합 항등식, 시계, B의 W, O/L 정리, 커널 대 참조, 재개 정확성
configs/              a_bounded.json (런 A), b_leaky.json (모델 B)
kaggle/               build_cell.py → lt_cell.py (셀 하나에 붙여넣어 학습)
data/                 sudoku_lt_1k.npz와 그 생성 스크립트
docs/                 architecture.md, results.md, claude_analysis_expressivity.md, evidence/run_a/
```

## 실행

```bash
python -m unittest discover tests                                    # CPU 검사 + CUDA가 있으면 커널 검사
python -m lt.train --config configs/a_bounded.json --out runs/a      # 같은 --out으로 다시 실행하면 재개
python -m lt.train --config configs/b_leaky.json --out runs/b --set max_steps=20000
torchrun --nproc_per_node=2 -m lt.train --config configs/a_bounded.json --out runs/a2

python -m lt.analysis.extrapolate runs/a/step_50800.pt
python -m lt.analysis.order_latency runs/a/step_50800.pt --out ol.json
python -m lt.analysis.interventions runs/a/step_50800.pt --out iv.json
python kaggle/build_cell.py                                          # lt/를 바꾼 뒤 Kaggle 셀 다시 생성
```

해석 도구는 리팩토링 전의 체크포인트도 읽는다. 런 A(`protocol.json`이 있는 런 폴더의 `step_*.pt`)와 Kaggle 모델 B 셀의 체크포인트가 해당된다.

## 기록

- 리팩토링 전의 전체 코드는 브랜치 `archive/pre-refactor-2026-10-15`에 있다. 날짜별 연구 일지(`docs/research/`), 이전 모델(v1~v1.8, KV-STDP 변형들), 체크포인트가 여기에 포함된다.
- 이 저장소의 모델 코드는 런 A와 비트 단위로 같은 초기화와 순전파를 낸다. CPU, CUDA torch 경로, Triton BF16 경로에서 4세그먼트를 비교해 확인했다.
- 학습 궤적은 GPU 비결정성 범위 안에서 같다.
