# bands, 16블록/세그먼트, 43,325스텝 체크포인트

런 `runs/bands_bps16`(설정 `configs/bands_bps16.json`)의 마지막 체크포인트 `step_43325.pt`에서 레인별 carry를 None으로 비운 것이다.
가중치(model = EMA, raw_model), EMA shadow, 옵티마이저 상태, RNG, cfg는 그대로다. 재개하면 모든 레인이 새 퍼즐로 시작한다(비트 단위 재개는 아님).

복원:

```bash
cat checkpoints/bands_bps16_step43325/step_43325_nocarry.pt.zst.part* | zstd -d -o step_43325.pt
sha256sum step_43325.pt     # step_43325_nocarry.pt.sha256 와 같아야 한다
python -m lt.analysis.extrapolate step_43325.pt
```

held-out 2048 (같은 런의 가까운 평가): 41,013 86.3% / 1,338, 42,966 85.7% / 1,281.
외삽(앞 512문제, carry 유지): 256블록(학습 깊이) 306 → 10,000블록 420, 오차 감소 55.3%.
자세한 내용은 `docs/claude_2026-10-18_journal.md` §5, `docs/evidence/run_bands_bps16/`.

이 체크포인트는 fast-weight FFN 런(`configs/sudoku_bps16_ffw43k.json`)의 `init_from`이기도 하다.
