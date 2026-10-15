# B-EMA, 세그먼트당 16블록 Kaggle 셀

`kaggle/b_ema_bps16_cell.py`를 노트북 셀 하나에 통째로 붙여 실행한다. `kaggle/lt_cell.py`(현재 `lt/`)에서 CFG만 바꿨다:
outer_rho 0.0, outer_eta 0.125, blocks_per_seg 16, act_ckpt False, out_dir /kaggle/working/lt_b_ema_bps16.
GPU가 여럿이면 torchrun으로 GPU당 한 프로세스(4×L4면 GPU당 배치 32). 메모리가 부족하면 act_ckpt를 True로(같은 수학, 느림).
검증: 문법 검사만. 4090에서 배치 128, 16블록은 20 GB를 넘었다.
