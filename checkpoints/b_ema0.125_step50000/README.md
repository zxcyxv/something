# B-EMA (outer_eta 0.125) 50,000스텝 체크포인트

런 `runs/b_ema0.125_20261015`(설정 `configs/b_ema.json`)의 마일스톤 `step_50000.pt`에서 레인별 carry(321 MB)를 None으로 비운 것이다.
가중치(model = EMA, raw_model), EMA shadow, 옵티마이저 상태, RNG, cfg는 그대로다. 재개하면 모든 레인이 새 퍼즐로 시작한다(비트 단위 재개는 아님).

복원:

```bash
cat checkpoints/b_ema0.125_step50000/step_50000_nocarry.pt.zst.part* | zstd -d -o step_50000.pt
sha256sum step_50000.pt     # step_50000_nocarry.pt.sha256 와 같아야 한다
python -m lt.analysis.extrapolate step_50000.pt
```

held-out 2048: 48,825 83.3% / 1,107, 50,778 83.1% / 1,090. 외삽(앞 512문제) 50k: 259 → 287 (seg 118), 오차 감소 11.1%.
