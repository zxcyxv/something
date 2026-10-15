# v1.71, 세그먼트당 16회 반복: Kaggle 단일 셀

파일: [v171_bps16_cell.py](v171_bps16_cell.py). 이 파일 전체를 Kaggle 노트북 셀 하나에 붙여넣고 실행한다. 저장소 import는 없다.

## 무엇인가

2026-09-27의 v1.71 Kaggle 셀(아카이브 브랜치 `archive/pre-refactor-2026-10-15`, 커밋 `7f1e2b7`의 `lt/train.py`)에서
학습 규약 하나만 바꾼 것이다.

| 항목 | v1.71 Kaggle 런 (204k) | 이 셀 |
| --- | --- | --- |
| 세그먼트당 블록 반복 (`blocks_per_seg`) | 8 | **16** (모두 기울기 있음, 손실은 세그먼트 끝 1회) |
| 퍼즐당 세그먼트 (`loops`) | 16 | 16 |
| 퍼즐당 학습 블록 | 128 | 256 |
| 모델 | v1.71 (QR 없는 선형 W_C, 주소 이력 Z, β 초기값 +π/2, post MLP) | 같음 |
| 최적화·평가·milestone | 배치 128, AdamATan2 lr 1e−4(warmup 2,000), weight decay 1.0, EMA 0.999, 250에폭(1,953스텝)마다 held-out 2,048, 1만 스텝마다 512문제 128세그먼트 외삽 | 같음 |
| 정밀도 | `bfloat16` | `auto` (native BF16이면 BF16, T4·P100은 FP32) |
| 시작 | 44,919스텝 재개 | 처음부터 (`resume_from=None`) |
| 출력 | `/kaggle/working/lt_v171` | `/kaggle/working/lt_v171_bps16` |

모델 코드는 `blocks_per_seg`를 그대로 읽으므로(`for _ in range(self.config.blocks_per_seg)`) 모델 쪽 수정은 없다.
한 스텝 = 한 세그먼트 = 16블록이고, 평가도 세그먼트당 16블록으로 돌아간다(16세그먼트 = 256블록).

## Kaggle에서 실행

1. `sudoku_lt_1k.npz`를 Input으로 연결한다(기본 경로 `/kaggle/input/datasets/jrjinwoo/sudoku-lt-1k/sudoku_lt_1k.npz`, 다르면 `CFG['data_npz']`).
2. 파일 전체를 셀 하나에 붙여넣고 실행한다. 학습은 별도 프로세스에서 돈다(GPU가 둘이면 torchrun, 배치 128을 rank당 64로).
3. `max_hours=11.5`에 닿으면 완료된 스텝 경계에서 저장하고 멈춘다. 같은 세션에서 셀을 다시 실행하면 `out_dir`의 마지막 `step_*.pt`에서 이어 간다.
   세션을 바꿀 때는 이전 output을 Input으로 연결하고 `CFG['resume_from']`에 그 체크포인트(파일 또는 폴더)를 지정한다. `max_steps`는 절대 종료 스텝이다.
4. 일반 체크포인트는 2,000스텝마다 저장하고 최근 2개만 둔다. milestone 체크포인트(`milestones/`)는 따로 보관된다.

## 로컬 확인

```bash
python kaggle/v171_bps16_cell.py --selftest                 # CPU 자체검사
python kaggle/v171_bps16_cell.py --config run.json          # JSON이 내장 기본값을 덮어쓴다. 로컬 런은 "blocks_per_seg": 16도 적어야 한다
```

2026-10-15 RTX 4090 확인
- CPU 자체검사 7/7 통과.
- 셀 런처 경로(노트북과 같은 서브프로세스 실행, compile 켬): 60스텝 학습, 중단 시점 평가(2,048문제, 세그먼트당 16블록), 체크포인트 저장까지 정상. 파라미터 7,156,075(v1.71과 같음).
- 속도·메모리(eager, BF16, activation checkpoint): 348 ms/step(1스텝 = 16블록 × 배치 128), 최대 GPU 메모리 3.5 GB. Kaggle T4(FP32)에서는 이보다 느리다.

## 비교 대상

v1.71 8블록 런의 기록은 아카이브 브랜치의 `docs/v171.md`, `docs/v18.md` §1에 있다(100k 55.8% 외삽, 204k 후반 저하).
