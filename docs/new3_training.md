# 새코드3 스도쿠 학습

모델은 [`lt/new3.py`](../lt/new3.py), 학습기는 [`lt/train_new3.py`](../lt/train_new3.py), 기본 설정은 [`configs/new3_sudoku.json`](../configs/new3_sudoku.json)이다. 업로드한 `새코드3.txt`의 심볼별 상태, 공간/심볼 attention, MoE, 윈도우 DeltaMemory를 스도쿠에 맞췄다. TxT 시냅스 적분 ablation은 아직 포함하지 않았다.

## 기본 설정

| 항목 | 값 |
|---|---|
| 학습 반복 | **1세그먼트 × 8블록**, 8블록 전체 BPTT |
| 상태 경계 | 한 배치에서 optimizer 1회 갱신. 다음 배치에서 hidden과 memory 초기화 |
| 입력 | 9×9, 11심볼, D=256, 숫자별 파라미터 없이 공유 |
| 배치 | 학습 128, 평가 128; batch 1 금지 |
| 데이터 | 기존 `data/sudoku_lt_1k.npz`, 학습 원본 1,000개 / 평가 2,048개 |
| 증강 | 기존 숫자·공간 변환 모두 유지, `num_aug=1000`, seed=0 |
| 장치 | 단일 GPU 자동 선택, BF16 autocast, activation checkpoint 사용 |
| optimizer | AdamATan2, lr=1e-4, 처음 2,000스텝 warmup 이후 상수, EMA=0.999 |
| 학습 종료 | 기본 160,000 optimizer steps |
| 정기 평가 | 1,953스텝마다 EMA seg1, 2,048문제 |
| 마일스톤 | 10,000스텝마다 체크포인트 보존 및 EMA seg128 외삽, 고정 첫 512문제 |

`loops`가 퍼즐당 세그먼트 수, `blocks_per_seg`가 세그먼트 안 공유 블록 반복 수다. `loops=1`에서도 블록은 8번 실행한다. `loops>1`로 설정하면 동일 퍼즐의 상태를 세그먼트 사이에 이월하되 gradient는 경계에서 detach한다. 입력은 원본 새코드3처럼 세그먼트 시작 시 한 번 주입한다. `ckpt_blocks=true`는 메모리 절약을 위한 재계산이며 no_grad가 아니다.

## Expert 8개 순차 적용 후 토큰별 선택

[`configs/new3_sudoku_sequential_experts.json`](../configs/new3_sudoku_sequential_experts.json)은 `expert_schedule="sequential_then_routed"`를 사용한다. 각 세그먼트에서 모든 토큰이 **E1 → E2 → … → E8 → 토큰별로 선택한 expert**를 순서대로 거친다. 한 반복의 출력 hidden/memory가 다음 반복의 입력이며, 매 반복 공간/심볼 attention, DeltaMemory 갱신, shared FFN을 함께 수행한다. 마지막 선택 단위는 **문제 × 숫자 × 칸**이고, 앞의 8회가 끝난 현재 상태로 expert를 선택한다.

앞의 8회는 지정 expert를 가중치 1로 직접 적용하며 토큰 dispatch가 없다. 9회째는 기존 top-1 확률 게이트를 사용하고, 라우터의 균형 손실과 선택 빈도 집계는 이 마지막 회에만 적용한다. Expert 순서는 세그먼트마다 E1부터 다시 시작한다. 전체 9회에 gradient가 흐르고, 입력 재주입은 세그먼트 시작 시 한 번이다. 숫자에 따라 expert 번호를 고정하지 않는다.

설정은 expert 8개, 9블록, **1세그먼트**, batch 128, 총 파라미터 **1,187,592개**다. 기본 세그먼트 수 1과 기존 데이터/증강/optimizer 조건을 유지하고, 종료 스텝은 20,000으로 둔다. 세그먼트 수를 늘려 학습할 경우 `loops`를 별도로 변경한다.

```bash
python -m lt.train_new3 --config configs/new3_sudoku_sequential_experts.json
```

새 출력 경로 `runs/new3_sudoku_sequential_experts/checkpoints/`에서 처음부터 학습한다. 실행 명령을 호출해야 학습이 시작된다. 기존 64-expert 모델의 optimizer/가중치를 이어받는 설정이 아니다. 이전 체크포인트에 `expert_schedule` 키가 없으면 기존 `routed` 방식으로 읽으며, 다른 방식으로 재개하거나 평가하는 요청은 거부한다. 새 방식의 세그먼트 1회는 9블록이므로 기존 8블록과 계산량이 동일하지 않다.

## 데이터 sampling

현재 풀은 원본당 1,000개의 무작위 숫자·공간 조합과 원본 1개를 사용한다. 기존 로더의 그룹별 sampling을 그대로 사용하며 전체 1,001,000개를 매 epoch 순회하는 방식은 아니다. `loops=1`은 매 update 새 배치를 처리하므로 `loops=16` 실험과 같은 optimizer step에서도 실제 처리한 새 퍼즐 수가 다르다.

## 실행과 재개

저장소 루트에서 실행한다. `torch`, `numpy`와 해당 GPU를 지원하는 PyTorch CUDA 환경이 필요하다.

```bash
python -m lt.train_new3 --config configs/new3_sudoku.json
```

출력은 `runs/new3_sudoku/checkpoints/`에 저장한다. 같은 명령은 해당 디렉터리의 마지막 체크포인트에서 자동 재개한다. 다른 실험은 JSON 또는 `--out-dir`로 새 출력 경로를 지정한다.

```bash
python -m lt.train_new3 --resume runs/new3_sudoku/checkpoints
```

명시적인 `--resume`은 체크포인트의 설정도 읽는다. raw/EMA 가중치, optimizer, router bias, hidden/memory, RNG, 데이터 cursor를 복원한다. 데이터와 학습 조건을 바꾸는 재개는 거부하며 `max_steps`, 로그·저장 간격 등 실행 제어 설정은 변경할 수 있다. v1/v1.1/새코드1 체크포인트와는 호환되지 않으므로 최초 학습은 새로 초기화한다.

`SIGINT`/`SIGTERM`은 진행 중인 optimizer update가 완료된 뒤 저장하고 종료한다. 비유한 loss/gradient는 update 전에 중단하며 마지막 정상 체크포인트를 보존한다.

`lr_min_ratio=1.0`이 warmup 이후 상수 LR을 지정한다. 더 작은 값으로 설정하면 cosine decay를 사용하며, horizon은 `epochs × 원본 수 / batch`다. `max_steps`는 실행 종료 조건이다.

## 외삽 평가

```bash
python -m lt.train_new3 --resume runs/new3_sudoku/checkpoints --eval-only --eval-segs 128
python -m lt.train_new3 --resume runs/new3_sudoku/checkpoints --eval-only --eval-segs 256 --eval-n 2048
```

`--eval-n` 생략 시 체크포인트 설정의 전체 평가 데이터(기본 2,048문제)를 사용한다. 학습 기본값이 seg1이어도 외삽에서는 같은 hidden/memory를 계속 이월하며 매 세그먼트를 기록한다. 채점 시 힌트를 강제로 덮어쓰거나 답을 보정하지 않는다.

JSON에는 각 세그먼트의 완답 수, cell accuracy, 힌트 위반, 스도쿠 제약 위반, 예측 변화율과 다음 값을 저장한다.

`error_reduction = (segN 완답 수 − seg1 완답 수) / (평가 문제 수 − seg1 완답 수)`

여기서 N은 요청한 마지막 세그먼트이며 중간 최고값을 선택하지 않는다. `loops>1` 실험은 seg1 대신 해당 학습 지평을 기준으로 한다. 중단된 부분 평가는 `partial=true`로 표시하고 감소율을 산출하지 않는다.

## 저장 파일과 검증

- `step_N.pt`: 정기 체크포인트, 최근 2개 보존.
- `milestones/step_N.pt`: 10k 간격 영구 체크포인트.
- `training.jsonl`: 매 update loss, LR, 완답률, 전체 gradient norm.
- `gradients.jsonl`: 매 update 파라미터별 gradient norm·실제 weight 변화량 및 hidden/memory RMS.
- `evaluations/*.json`, `milestones/eval_*.json`: 정기/외삽 평가 결과.

```bash
python -m unittest lt.test_train_new3 -v
```

14개 검사는 원본과의 forward/backward 일치, 숫자 치환 등변성, checkpoint 재계산 시 라우팅 중복 집계 방지, 비활성 expert 제외, MoE dispatch, 중간 세그먼트 재개, 평가 후 상태 복원 및 **seg1 이후 새 배치 상태 초기화**를 다룬다. 순차 방식은 모든 토큰의 E1→E8 순서와 hidden/memory 이월, 마지막 한 번의 라우팅, 보조 손실 없이 expert 8개와 라우터에 전달되는 정답 손실 gradient, 숫자 치환, checkpoint 재계산 및 정확한 재개를 추가로 검사한다. 이전 체크포인트의 기본 방식 복원과 방식 변경 거부도 확인한다. 원본과의 일치 검사는 `새코드3.txt`가 있을 때 실행한다.

아키텍처와 대칭성의 자세한 분석은 [적용 설계](new3_sudoku_adaptation.md)를 참고한다.
