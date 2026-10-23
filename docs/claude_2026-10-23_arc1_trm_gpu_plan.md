# ARC-AGI-1 TRM 비교 런: 4×H100 기준 예측과 GPU 선택 (2026-10-23)

내일(10-24) 4×H100과 4×RTX PRO 6000을 동시에 띄워 속도를 재고, **스텝당 비용이 낮은 쪽만 남긴다**.
H100 시간당 가격이 PRO 6000의 약 2배이므로, H100이 우리 모델에서 **2배 이상 빨라야** 손해가 없다.
이 문서는 그 판정 전에 4090 실측과 커널 프로파일로 예측한 값이다. 최종 판정은 내일의 벤치 실측으로 한다(§6).

## 0. 요약

| | 4×H100 SXM | 4×H100 PCIe | 4×PRO 6000 WS | 4×PRO 6000 Server |
|---|---:|---:|---:|---:|
| 예측 s/step (배치 768, loops 16 ACT, FP32) | **0.28** | 0.46 | 0.53 | 0.59 |
| 4090 대비 배수 | 3.6× | 2.2× | 1.9× | 1.7× |
| PRO 6000 WS 대비 속도비 | 1.89 | 0.87 | 1 | 0.90 |
| TRM 전체 예산 519,200스텝(TRM 데이터 100k에폭) | **40 h (1.7일)** | 66 h | 76 h (3.2일) | 85 h (3.5일) |
| 같은 예산의 비용 (PRO 시간당 가격 P, H100 = 2P) | 81 P | 133 P | **76 P** | 85 P |
| TRM 공식(4×H100, README) | 약 3일 = 0.5 s/step | | | |

- **SXM이면 거의 손익분기다.** SXM 대 PRO 6000 예측 속도비는 1.9~2.1로, 가격비 2와 구별되지 않는다. 그래서 벤치로 정해야 한다.
- **H100 PCIe면 PRO 6000이 확실히 싸다.** 속도비가 1.15다.
- **BF16 bands는 이득이 없다(§3).** 시간의 76%가 대역 흔적 갱신의 메모리 트래픽이고, 행렬곱 정밀도와 무관하다. 모델은 FP32 그대로 간다.
- **loops는 스텝당 시간을 바꾸지 않는다.** 한 최적화 스텝은 loops와 무관하게 세그먼트 1개(8블록)다. 그래서 **내일 런은 loops 16(ACT, TRM과 같은 깊은 감독)으로 한다(10-23 결정).** 4090에서 loops 16 + ACT + grad_accum 2가 레인당 5.17 ms로 loops 1(5.2 ms)과 같음을 확인했다.
- **배치는 TRM과 같은 글로벌 768이다.** GPU당 192레인을 한 번에 올리지 않고 `grad_accum 2`(96레인 × 2)로 나누는 것이 빠르다(§4). 손실을 글로벌 배치로 나눠 기울기를 누적하므로 768 배치와 같은 기울기다.

## 1. TRM 기준 (공식 레시피, 4×H100)

출처: SamsungSAILMontreal/TinyRecursiveModels README·`config/cfg_pretrain.yaml`·`config/arch/trm.yaml`. 재현 자료는 서브에이전트가 조사했다(§7).

- **명령:** `torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 ema=True`
  - README 표기는 "assuming 4 H-100 GPUs", "~3 days"다.
  - 논문은 L_cycles 6, README는 4다.
- **최적화:** 글로벌 배치 768, 100,000에폭, eval_interval 10,000에폭(평가 10번). lr 1e-4(워밍업 2,000 후 고정), puzzle_emb_lr 1e-2, wd 0.1(둘 다), AdamATan2 β 0.9/0.95, EMA 0.999.
- **모델:** hidden 512, 헤드 8, 확장 4, halt_max_steps 16, 탐색 0.1, puzzle_emb_len 16, bf16, stablemax CE.
- **데이터:** training + evaluation + concept, 증강 1000. **training과 concept 과제는 시험 쌍까지 학습에 들어간다.** evaluation 과제는 시범 쌍만 학습에 들어가고, 시험 쌍 419개가 시험셋이다.
  - 에폭당 약 3,988개 예시: training 1,718 + evaluation 1,363 + concept 907.
  - 100k에폭 = **518,071스텝**(ARC Prize 재현 런의 마지막 스텝과 같다).
  - TRM 속도로 환산하면 3일 / 518k스텝 ≈ 0.5 s/step(4×H100)이다.
- **보고 수치:** 논문 pass@2 44.6%. ARC Prize 검증 40%(semi-private). URM 논문의 재실행은 pass@1 40.0%다.

## 2. 우리 런 설정과 TRM과의 차이

설정은 `configs/arc1_trm_loops16.json`(내일 런, ACT: q_head를 앞 토큰에, q_halt > 0이면 정지, 탐색 0.1, 손실 + 0.5 BCE)이다. `configs/arc1_trm_loops1.json`은 깊은 감독 없는 대조용으로 남겨 둔다. bands 8블록/세그먼트, hidden 512, 헤드 8, 앞 토큰 과제 임베딩 1개, 증강 1000(ID 960 × 1001). 최적화 값은 TRM과 같다. 평가는 500에폭(2,596스텝)마다 16세그먼트로 한다.

| 항목 | TRM | 우리 | 비교에 주는 영향 |
|---|---|---|---|
| 학습 데이터 | 에폭당 3,988(training·concept 시험 쌍 포함) | `data/arc1_trm.npz`, 같은 분할 3,988 | 같음. 100k에폭 = 519,200스텝(TRM 518,071, 정수 나눗셈 차이 0.2%) |
| 세그먼트 | ACT 최대 16 | ACT 최대 16 (loops 16) | 같음 |
| 세그먼트당 연산 | 2층 × 30회(앞 20회 무기울기) | 8블록, 전부 역전파 | 대략 TRM 스텝 연산의 절반 |
| 평가 | 1000판 투표 pass@k, 과제 단위 | 원본 판 1개, 시험 쌍 단위 정확, `loops` 세그먼트 | 가장 가까운 TRM 지표는 `all/exact_accuracy`(판마다 따로 정답, 16스텝). TRM 기준 정체값은 약 0.265 |
| 퍼즐 임베딩 | 16토큰 | 1토큰 | — |

## 3. 커널 프로파일: 어디에 시간이 드나 (4090, loops 1, 레인 16, act_ckpt 끔)

| 설정 | 메모리 대역(원소별·리덕션·복사) | BF16 텐서 GEMM | FP32 GEMM | 기타 | s/step |
|---|---:|---:|---:|---:|---:|
| FP32 (현재, `matmul_precision highest`) | 76.4% | 18.2% | 2.7% | 2.7% | 0.0854 |
| TF32 (`high`) | 77.1% | 18.4% | 1.8% | 2.7% | 0.0846 |
| BF16 bands (`band_bf16 true`) | 77.3% | 19.9% | 0 | 2.8% | 0.0827 |

- 가장 큰 커널들은 대역 흔적 갱신(E_K, E_V의 복소 감쇠, cos/exp/sin이 융합된 Triton 원소별 커널)이다. 흔적 상태는 토큰당 FP32 32 KB다.
- bands 행렬곱 4개를 BF16으로 바꿔도 3%만 빨라진다. 대신 변환 커널(`_to_copy`)이 새로 생긴다. TF32는 1%다.
- 따라서 이 워크로드의 속도는 **HBM/GDDR 대역폭이 거의 정한다.** FP32 비텐서 연산(PRO 6000 120 TFLOPS 대 H100 SXM 약 67 TFLOPS)의 차이는 영향이 없다(2.7%).
- `band_bf16` 옵션(기본 꺼짐, `tests/test_bands.py`)은 남겨 두었다. 흔적 메모리를 절반으로 줄이지만, `grad_accum`으로 같은 배치를 만들 수 있으므로 쓸 이유가 없다.

## 4. 속도·메모리 예측

**4090 실측 (loops 1, 증강 1000, 피크는 `torch.cuda.max_memory_allocated`)**

| act_ckpt | 레인당 시간 | 레인당 메모리 | 상수항 |
|---|---:|---:|---:|
| 켬 (b16/32/48: 0.112/0.233/0.342 s) | 7.1 ms | 0.387 GiB | 3.8 GiB |
| 끔 (b8/16: 0.045/0.084 s) | 5.2 ms | 0.509 GiB | 3.8 GiB |

- 레인 16 이상이면 시간이 레인 수에 선형이다(GPU가 포화). 그래서 `grad_accum`이 속도를 바꾸지 않는다.
- GPU당 192레인: act_ckpt를 켜면 78.1 GiB다. H100 80 GB에서는 OOM 경계이고, PRO 6000 96 GB에는 들어간다. act_ckpt를 끄면 101.5 GiB라 둘 다 안 된다.
- **둘 다 최적은 `grad_accum 2`, act_ckpt 끔이다.** 96레인이고, 다른 마이크로배치의 carry를 더해 약 56 GiB다. 4090 환산 1.0 s/step이다.

**GPU 환산.** 시간 = 0.818 × (4090 대역폭 / 대역폭) + 0.182 × (4090 BF16 / BF16) × 1.0 s. 원소별 커널이 피크 대역폭에 같은 비율로 닿는다고 가정했다.

| GPU | 대역폭 | BF16 dense | 출처 |
|---|---:|---:|---|
| RTX 4090 | 1,008 GB/s | 165 TFLOPS (FP32 누적) | NVIDIA Ada 백서 |
| H100 SXM | 3,350 GB/s | 989 | [flopper.io](https://flopper.io/compare/nvidia-h100-sxm5-80gb-vs-nvidia-h100-pcie-80gb/spec-sheet), [spheron](https://www.spheron.network/blog/nvidia-h100-specs/) |
| H100 PCIe | 2,000 GB/s | 756 | 같음 |
| PRO 6000 Workstation | 1,792 GB/s | 약 500 (추정) | [PNY 목록](https://www.memoryexpress.com/Products/MX00134219) |
| PRO 6000 Server | 1,597 GB/s | 약 500 (추정) | NVIDIA 서버판 데이터시트(FP32 120 TFLOPS, FP4 4 PFLOPS 희소, PCIe 5.0, NVLink 없음) |

PRO 6000의 BF16 값은 FP4 4 PFLOPS(희소)에서 ÷8로 환산한 추정이다. BF16 비중이 18%라 이 오차는 결과를 거의 바꾸지 않는다.

**불확실성.**
- 원소별 커널의 대역폭 효율은 GPU마다 다르다. L2 크기가 4090 72 MB, H100 50 MB, PRO 6000 128 MB로 차이가 난다.
- `torch.compile`이 sm_90과 sm_120에서 다른 커널을 고를 수 있다.
- 그래서 예측 속도비 1.9~2.1과 가격비 2는 오차 범위 안에서 구별되지 않는다. **벤치 실측으로 정한다.**
- 통신은 무시할 수 있다. 스텝당 기울기 약 14 MB + 임베딩 기울기 약 1.5 MB라서, NVLink 없는 PRO 6000에도 불리하지 않다.

## 5. 학습곡선 비교에서 볼 것

참조 곡선은 `docs/evidence/trm_arc1/reference.json`이다. 공개된 유일한 ARC-AGI-1 TRM 곡선인 gaoxin492 재현 런(8×H200, 배치 4608, 200k에폭, pass@2 41.0%)을 W&B 화면에서 수치화했다. 근사치이고, x는 에폭이다.

- **x축은 에폭 또는 samples(본 예시 수)로 맞춘다.** 그 런은 배치가 6배라 step은 바로 비교할 수 없다. step 축은 공식 레시피(배치 768) 환산이다.
- **비교할 짝**
  - train lm_loss / exact ↔ `train/lm_loss`, `train/exact_accuracy`. 정의가 같다(끝난 퍼즐 기준, 손실은 글로벌 배치로 나눔).
  - 우리 eval exact(원본 판) ↔ `all/exact_accuracy`(판별, 16스텝). 우리 쪽은 1세그먼트 8블록이다. pass@2는 투표 지표라 직접 비교하지 않는다.
- **TRM 기준점(에폭)**

  | 지표 | 10k | 20k | 30k | 50k | 100k | 정체 |
  |---|---:|---:|---:|---:|---:|---:|
  | `all/exact_accuracy` | 0.004 | 0.054 | 0.141 | 0.226 | 0.264 | 약 0.265 |
  | `train/exact_accuracy` | — | — | — | — | — | 30k에폭 무렵 0.8 |

  TRM의 평균 ACT 스텝은 16에서 시작해 120k에폭 이후 약 1.1까지 내려간다. 즉 학습 후반의 TRM도 사실상 1~2세그먼트로 답한다. 우리 쪽 `train.jsonl`의 `steps`가 같은 양이다(`status`·곡선에서 비교).
- **예측(검증 전).** 같은 에폭에서 우리 eval exact가 TRM `all/exact_accuracy`보다 높을지 낮을지는 근거가 없다. 유일한 선례는 10-18 ARC-1 bands 런(배치 32, loops 16)인데, 4,831스텝(약 15만 샘플, 이 레시피의 약 200스텝 분량)에서 0/419였다. 너무 이른 시점이라 판단 근거가 되지 않는다.
- **중간 점검 기준:** 30k에폭(약 156k스텝, H100 SXM 예측 약 12시간, PRO 6000 약 23시간)에 TRM은 0.141이다. 이때 우리가 0.05 미만이면 구조 문제로 보고, 계속할지 다시 정한다.

## 6. 내일 절차 (두 인스턴스에서 같은 명령)

전제: 오늘 작업을 커밋·푸시해야 두 인스턴스에서 clone할 수 있다. 디스크는 30 GB 이상이어야 한다(체크포인트 1개 약 6 GB, 2개 보관 + 쓰는 중 1개).

```bash
git clone https://github.com/zxcyxv/something.git && cd something
bash scripts/arc1_trm.sh setup                 # torch 빌드 vs GPU 아키텍처(sm_90 / sm_120), NCCL, 디스크, 테스트
PRICE=<인스턴스 시간당 $> bash scripts/arc1_trm.sh bench
#   fp32에서 메모리상 들어가는 가장 빠른 grad_accum x act_ckpt를 40스텝씩 잼(체크포인트 안 씀)
#   마지막 줄 BENCH_JSON {...}을 복사
bash scripts/arc1_trm.sh decide bench/<label>/bench.json '<다른 쪽 BENCH_JSON의 {...}>'
#   또는 숫자로: bash scripts/arc1_trm.sh decide H100:<s/step>:<$/h> PRO6000:<s/step>:<$/h>
```

남길 쪽에서:

```bash
bash scripts/arc1_trm.sh start                 # runs/arc1_trm_l16, setsid nohup, 같은 명령으로 재개
bash scripts/arc1_trm.sh status                # 마지막 [LT]/[EVAL], s/step, ETA, GPU, 디스크
bash scripts/arc1_trm.sh curves                # runs/arc1_trm_l16/curves/{curves_*.png, summary.md}
bash scripts/arc1_trm.sh stop                  # 워커 PID에 SIGTERM -> 체크포인트 저장 후 종료
bash scripts/arc1_trm.sh archive               # docs/evidence/run_arc1_trm_l16/ (그 다음 커밋·푸시)
```

- 시간 제한은 `MAX_HOURS=30 bash scripts/arc1_trm.sh start`처럼 건다.
- 기본값이 loops 16(`configs/arc1_trm_loops16.json`, `runs/arc1_trm_l16`)이다. loops 1 대조는 `CONFIG=configs/arc1_trm_loops1.json RUN=arc1_trm_l1`로 띄운다.

## 7. 출처

- **TRM 레시피:** SamsungSAILMontreal/TinyRecursiveModels README, `config/cfg_pretrain.yaml`, `config/arch/trm.yaml`, `dataset/build_arc_dataset.py`
- **재현 곡선:** huggingface.co/gaoxin492/TinyRecursiveModels-ARC-AGI-1(W&B 화면, 같은 런이 GitHub issue #19에도 있음)
- **ARC Prize 재현:** huggingface.co/arcprize/trm_arc_prize_verification(8×H100, 20~30 h, 최종 518,071스텝)
- **다른 수치:** URM 논문(arXiv 2512.14693)의 TRM 재실행, arXiv 2512.11847, alphaXiv 재현(8×A100, pass@2 43.0%)

## 8. 학습 데이터 (TRM과 같게, 10-23 결정)

- **생성:** `data/arc1_trm.npz`는 `python data/prep_arc1.py --kaggle .../kaggle/combined/arc-agi --trm-split --out data/arc1_trm.npz`로 만들었다. 입력은 TRM 입력 형식인 `kaggle/combined/arc-agi_{training,evaluation,concept}_{challenges,solutions}.json`이다.
- **구성:** 학습 쌍 3,988개 = 시범 3,092 + training 시험 416 + concept 시험 480. 시험셋은 evaluation 시험 쌍 419개로, `arc1.npz`와 비트 단위로 같다.
- **기존 데이터와의 대조:** 같은 입력에서 `--trm-split` 없이 만들면 `arc1.npz`와 3,092쌍 중 한 쌍만 다르다(ConceptARC `Order5`의 한 출력 격자에서 두 칸의 색이 서로 바뀜). TRM이 쓰는 kaggle 사본의 값을 따랐다.
