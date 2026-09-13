# LT v1.1을 URM ARC-AGI-1 인터페이스로 학습하기

`lt/train_arc.py`는 기존 `lt/train.py`의 LT 모델을 **새로 초기화해 ARC에서 학습**하는 단일 장치 CLI다.
Sudoku 학습 파일과 체크포인트 로더는 수정하지 않았다. 모델 기본값은 v1.1의 고정 게이지,
pre 순서, 흔적 없음, 폭 832, 8헤드, 세그먼트당 8블록, 16세그먼트다.

## 출처와 기준

공식 [URM 논문](https://arxiv.org/abs/2512.14693)이 연결하는 저장소는
[UbiquantAI/URM](https://github.com/UbiquantAI/URM)이며, 확인한 commit은
`c14e55f5f9227873617015cf60a239126b55adcd`다.

| 원본 | 가져온 인터페이스 |
|---|---|
| [data/build_arc_dataset.py](https://github.com/UbiquantAI/URM/blob/c14e55f5f9227873617015cf60a239126b55adcd/data/build_arc_dataset.py) | 데이터 분할, 토큰, 증강, 식별자, `.npy` 포맷 |
| [puzzle_dataset.py](https://github.com/UbiquantAI/URM/blob/c14e55f5f9227873617015cf60a239126b55adcd/puzzle_dataset.py) | 그룹→증강 퍼즐→예제 샘플링, 패딩, ignore 라벨 |
| [evaluators/arc.py](https://github.com/UbiquantAI/URM/blob/c14e55f5f9227873617015cf60a239126b55adcd/evaluators/arc.py) | 예측 격자 복원, 역증강, 빈도 투표, task별 pass@K |
| [scripts/URM_arcagi1.sh](https://github.com/UbiquantAI/URM/blob/c14e55f5f9227873617015cf60a239126b55adcd/scripts/URM_arcagi1.sh), [config/cfg_pretrain.yaml](https://github.com/UbiquantAI/URM/blob/c14e55f5f9227873617015cf60a239126b55adcd/config/cfg_pretrain.yaml) | 학습률, 퍼즐 임베딩 학습률, weight decay 등의 ARC 기본값 |

전처리는 원본 builder를 subprocess로 실행한다. 모델 학습 시에는 URM이나 Hydra, WandB,
FlashAttention을 import하지 않으며 `torch`, `numpy`만 필요하다.

## 데이터 준비

저장소 루트에서 실행한다. 외부 checkout 경로는 필요에 따라 변경할 수 있다.

```bash
pip install torch numpy pydantic argdantic
git clone https://github.com/UbiquantAI/URM.git /tmp/URM
git -C /tmp/URM checkout c14e55f5f9227873617015cf60a239126b55adcd

python data/prep_arc_dataset.py \
  --urm-repo /tmp/URM \
  --output-dir data/arc1concept-aug-1000
```

기본 입력은 URM에 포함된 `kaggle/combined/arc-agi`이며 `training`, `evaluation`, `concept`를
사용하고 `evaluation`을 test set으로 지정한다. 이 commit에는 각 400, 400, 160과제가 있다.
`--input-file-prefix`, `--subsets`, `--test-set-name`, `--num-aug`, `--seed`로 바꿀 수 있다.

`--task-limit 2 --num-aug 2`를 추가하면 subset당 2과제, 원본+증강 2개만 만드는 실행 확인용
데이터가 된다. 이 제한은 점수 재현용 설정이 아니다. 완전한 데이터에서는 원본 JSON의 과제
순서를 보존하고, 제한할 때만 과제 이름순으로 선택한다.

출력에는 원본 `train/`, `test/`, `identifiers.json`, `test_puzzles.json` 외에
`provenance.json`을 저장한다. 기준 commit, 실제 commit, builder 해시, 원본 JSON 해시, seed와
전처리 옵션을 기록한다. 출력 폴더가 이미 채워져 있으면 파일을 혼합하지 않고 중단한다.
필요한 정답이 없으면 원본의 dummy 정답 대체를 허용하지 않고 중단한다.

### 분할의 의미

| 과제 subset | 과제 안의 `train` 시범 예제 | 과제 안의 `test` 질의 |
|---|---|---|
| training | 학습 | 학습 |
| concept | 학습 | 학습 |
| evaluation | **학습** | **평가만** |

따라서 이 인터페이스는 evaluation 과제의 공개 시범 예제를 학습에 포함한다. evaluation 질의의
정답은 역전파에 사용하지 않는다. 시범 입출력들을 하나의 긴 입력에 이어 붙이는 방식도 아니다.
각 입출력 쌍을 개별 예제로 학습하고, 같은 과제·같은 증강의 시범 예제와 질의가 같은 puzzle ID를
공유한다. 학습하지 않은 새 과제를 임의로 추가해 바로 처리하는 인터페이스는 아니다.

### 토큰, 증강, 식별자

- 30×30 캔버스를 펼친 900토큰, vocabulary 12: PAD=0, EOS=1, 색 0..9=토큰 2..11.
- 출력의 오른쪽과 아래쪽 경계에 가능한 범위에서 EOS를 넣는다. PAD 라벨만 -100으로 치환하며,
  EOS와 실제 색 토큰은 모두 감독한다. PAD 입력 칸도 모델 내부 상태와 통신에는 참여한다.
- 색 치환은 검정 0을 보존하고 나머지 9색을 섞는다. D4의 회전·반사 8가지와 결합하고 중복을 제거한다.
- `--num-aug 1000`은 원본 1개에 최대 1,000개 증강을 추가한다. 각 증강은 별도의 puzzle ID다.
- translation은 학습 split에만 적용한다. 각 증강 퍼즐의 예제 중 적어도 하나는 좌상단에 놓고,
  나머지는 입력과 출력에 같은 offset을 준다. 평가 예제는 좌상단에 둔다.
- puzzle ID 0은 배치 패딩용이다. 실제 ID는 1부터 시작하며 원본 `identifiers.json` 매핑을 그대로 쓴다.

## 학습과 재개

```bash
python lt/train_arc.py \
  --data data/arc1concept-aug-1000 \
  --out-dir runs/arc_v1_1 \
  --preset v1.1 --batch-size 2
```

처음에는 짧은 실행으로 학습과 메모리를 확인할 수 있다.

```bash
python lt/train_arc.py \
  --data data/arc1concept-aug-1000 \
  --out-dir runs/arc_v1_1_check \
  --max-steps 100 --eval-every 0 --no-eval-at-end
```

1 optimizer step은 한 세그먼트의 forward/backward/update다. `h,w`는 다음 스텝으로 이어지고
세그먼트 경계에서는 detach된다. 각 레인은 같은 퍼즐을 기본 16세그먼트 동안 유지한 뒤 교체한다.
URM처럼 그동안 새 candidate batch도 sampler에서 소비하되, 아직 끝나지 않은 레인에는 현재
입력과 라벨을 유지한다. 따라서 step 수나 sampler epoch 수를 완료한 퍼즐 수로 해석하면 안 된다.

Stablemax cross entropy를 예제의 유효 토큰 수로 나누고 배치 평균을 사용한다.
AdamATan2와 LT의 동역학 파라미터에 대한 weight decay 제외 규칙을 재사용한다.
puzzle embedding은 기존 sparse buffer와 전용 SignSGD를 사용한다. 각 ID의 gradient를 합친 후
sign을 취하므로 일반적인 dense Adam 임베딩과 다르다.

| 기본값 | 값 |
|---|---:|
| 모델 lr | 1e-4 |
| puzzle embedding lr | 1e-2 |
| 모델 / puzzle embedding weight decay | 0.1 / 0.1 |
| Adam β1 / β2 | 0.9 / 0.95 |
| warmup | 2,000 optimizer steps |
| warmup 후 lr | 상수 |
| EMA | 0.999 |
| batch | 2, 단일 프로세스·단일 장치 |
| 평가 / 저장 | 1,000 optimizer steps마다 |

원본과 같이 step 증가 전에 learning-rate schedule을 계산하므로 warmup이 켜져 있으면 첫
optimizer update의 lr은 0이다. `--lr-warmup-steps 0`이면 첫 스텝부터 갱신한다.
EMA는 학습 `Parameter`만 평균한다. puzzle embedding은 buffer이므로 평가에도 raw 테이블을 쓴다.
LT는 학습하는 Q/ACT head가 없어 불필요한 상수 Q 손실을 제외한다.

```bash
python lt/train_arc.py \
  --data data/arc1concept-aug-1000 \
  --out-dir runs/arc_v1_1 \
  --resume runs/arc_v1_1/latest.pt \
  --max-steps 20000
```

`--max-steps`는 재개 이후 추가 스텝 수가 아니라 **절대 종료 스텝**이다.
체크포인트에서 모델 구조, optimizer 설정, batch size, sampler 설정을 복원한다. 실행 장치,
출력 폴더, 종료 스텝, 평가 옵션은 CLI 값을 사용한다. hidden state·coupling·현재 예제·반복 카운터,
sampler RNG/cursor, optimizer, EMA, RNG도 복원하므로 퍼즐 처리 중간에서 이어갈 수 있다.
CPU의 같은 설정에서는 중간 저장/재개와 연속 실행의 비트 단위 일치를 검증했다.

`latest.pt`를 임시 파일에 저장한 뒤 원자적으로 교체한다. 이전 마일스톤의 별도 보관은 자동으로 하지
않는다. Ctrl-C는 현재 optimizer step을 완료한 뒤 저장하고 끝낸다. 긴 평가 중 요청한 경우 해당
평가가 끝난 뒤 종료한다. 강제 kill이나 장치 오류가 나면 마지막으로 저장된 checkpoint에서 재개한다.

데이터 fingerprint가 다르면 재개를 거부한다. fingerprint는 JSON 내용과 각 배열의 크기 및 앞뒤
64KiB를 확인하며, 전체 `.npy` 내용의 암호학적 동일성을 보장하는 검사는 아니다.
Sudoku checkpoint는 vocab, 위치, ID 테이블과 과제가 달라 `--resume`으로 받지 않는다.

### 메모리

900토큰의 dense 결합 기억 자체가 `batch × heads × 900²` 원소다. 기본 batch=2, 8헤드에서
FP32 coupling carry만 약 49.4MiB다. 실제 학습에는 8블록의 attention·주소·값·MLP 중간값과
역전파, 모델·optimizer·EMA 및 puzzle embedding 테이블이 추가된다.

960과제에 모두 1,000개 증강이 만들어지면 ID는 blank 포함 최대 960,961개다.
832차원 FP32 puzzle embedding 테이블 **하나만 약 2.98GiB**이며, 중복 증강 제거가 있으면 줄어든다.
배치를 줄여도 이 테이블 크기는 변하지 않는다. 기본 폭/반복/batch의 GPU 검증은 작은 ID 테이블로
수행했으며, 전체 증강 데이터의 최고 메모리 사용량이나 학습 처리량은 아직 측정하지 않았다.

CUDA에서는 기본 BF16 autocast를 사용하고 상태는 FP32를 유지한다. CPU에서는 AMP를 끈다.
`--no-amp`, `--compile`, `--hidden-size`, `--num-heads`, `--blocks-per-seg`, `--loops`,
`--puzzle-emb-dim`을 조정할 수 있다. 폭과 반복을 줄인 결과는 기본 v1.1 설정의 결과와 구분해야 한다.
다중 GPU와 gradient accumulation은 이 CLI에 구현하지 않았다.

## 평가와 외삽

```bash
python lt/train_arc.py \
  --data data/arc1concept-aug-1000 \
  --out-dir runs/arc_v1_1 \
  --resume runs/arc_v1_1/latest.pt \
  --eval-only --eval-segments 128
```

정답을 입력에 넣지 않고 각 query를 지정한 세그먼트 수만큼 실행한다. 출력 토큰에서 좌상단에
붙은 가장 큰 유효 색 직사각형을 찾는 원본 crop 규칙을 쓴다. 정답 격자 크기로 crop하지 않는다.
그다음 회전·반사와 색 치환을 되돌리고, 원래 task와 입력 격자로 예측을 모아 동일한 출력에 투표한다.
빈도순으로 서로 다른 상위 K개 중 정답이 있는지 검사한다.

`ARC/pass@K`는 **각 과제의 test query 성공률을 구한 뒤 과제별 평균**이다. 복수 query가 있는
과제를 전부 맞아야만 1로 세는 점수도, 모든 query를 한꺼번에 평균한 점수도 아니다.
별도로 출력하는 `token_accuracy`, `sequence_exact`는 증강 예제의 유효 토큰 기준 지표다.
크롭 후 원본 ARC 정답과 비교하는 pass@K와 같지 않다.

LT의 `q_halt_logits`는 항상 -5이므로 URM의 confidence 값은 모두 동일하다.
원본의 `(투표수, 평균 q, 최대 log q)` 순위는 여기서는 **빈도 투표와 입력 순서에 따른 동률 처리**로
줄어든다. 기본 top-2 시도와 점수는 `eval_step_N_segS_submission.json`, `eval_step_N_segS.json`에 저장한다.

빠른 진단에는 `--eval-max-tasks 8 --eval-max-augmentations 8`을 추가할 수 있다. 과제 이름순 부분집합과
각 과제의 앞쪽 증강만 평가하며 JSON에 평가 과제 수, 전체 과제 수, 증강 상한을 기록한다.
부분 평가 점수를 전체 ARC-AGI-1 점수로 보고하면 안 된다. 옵션을 생략하면 모든 과제와 증강을 사용한다.

## URM과의 차이

데이터 인터페이스를 가져온 것이며 URM 논문 점수 재현을 의미하지 않는다.

- 네트워크는 LT다. URM의 short convolution, Transformer 블록과 내부 반복 구성은 가져오지 않았다.
- URM은 puzzle embedding을 prefix token으로 넣지만 LT는 기존대로 각 셀 입력에 더한다. 따라서
  모델 토큰 수는 900 그대로다. 위치 회전·거리 감쇠·결합 기억도 LT의 기존 2차원 정의를 사용한다.
- 원본 ARC script는 8 GPU, global batch 768을 사용한다. 이 CLI는 단일 장치 batch 2가 기본이며
  배치 축소에 맞춰 lr를 자동 재조정하지 않는다. 성능에 적절한 최종 학습 설정은 별도 실험이 필요하다.
- 원본 평가 주기는 epoch 묶음 기준이다. 이 CLI의 `--eval-every`는 optimizer step 기준이고,
  sampler의 epoch 묶음은 `--epochs-per-iter`로 따로 지정한다.
- 원본 evaluator의 `aggregated_voting=True`는 이전 평가의 예측까지 누적한다. 여기서는 매 평가마다
  투표를 새로 시작해 해당 checkpoint/세그먼트 수의 결과만 평가한다. 원본의 샘플링 `maj@N` 지표는
  구현하지 않았으며 `pass@1/2/5/10`만 출력한다.
- LT의 학습 가능한 Q head/ACT가 없으므로 고정된 반복 수를 사용한다.

## 완료한 검증

```bash
python -m unittest lt.test_train_arc -v
```

4개 테스트는 실제 900토큰 형식에서 다음을 검증한다.

- 출력 크기 복원, 8가지 공간 역변환과 색 역치환, EOS/PAD 처리.
- ID 패딩과 라벨 ignore, sampler 상태 복원, 합성 held-out query 정답의 학습 split 비포함.
- 다중 query 과제의 pass@K 평균과 빈도 투표.
- optimizer/sparse embedding 실제 갱신, 3스텝의 미완료 carry 저장 뒤 6스텝까지 재개한 결과가
  연속 6스텝의 모든 raw model tensor 및 carry와 비트 단위로 일치, 저장 가중치의 3세그먼트 평가.

추가로 공식 builder를 통해 실제 ARC 각 subset 2과제·증강 2개 데이터를 만들고 확인했다.

- 고정 seed에서 원본 URM sampler와 연속 40개 train batch, 전체 3개 eval batch가 동일했다.
- 80개 격자/증강 사례에서 원본 인코딩과 역증강을 대조했다.
- CPU의 축소 모델로 4 optimizer step, 전체 평가 예제 6개와 제출 JSON 생성을 완료했다.
- RTX 4090에서 **기본 LT 폭 832·8헤드·8블록·batch 2·CUDA AMP**로 2 optimizer step을 실행했다.
  저장 파라미터가 모두 유한했고 sparse embedding이 실제 갱신됐다. 해당 checkpoint를 다시 불러
  2세그먼트 평가·투표·제출 JSON 생성도 확인했다.

전체 960과제×1,000증강 학습, 유효 ARC 정확도 달성, 장기 외삽 개선, 대규모 최고 메모리는 검증하지 않았다.
