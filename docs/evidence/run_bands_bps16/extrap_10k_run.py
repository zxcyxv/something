"""Extrapolate a checkpoint to SEGS segments (carry kept, EMA weights) with per-segment exact/accuracy and progress logs.
Usage: python run.py CKPT SEGS N OUT_JSON"""
import json, sys, time, torch
sys.path.insert(0, '/workspace/something')
from lt.analysis.common import load_model
from lt.data import load_data, load_task_ids, eval_batches
ck, segs, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
model, cfg = load_model(ck, dev, weights='ema', kernel='auto', amp=True)
_, _, x, y, path, _ = load_data(cfg.get('data_npz'))
_, ids = load_task_ids(path)
x, y = x[:n], y[:n]
ids = None if ids is None else ids[:n]
model.eval(); model.config.loops = segs + 1
bps = model.config.blocks_per_seg
exact, acc, churn = [0] * segs, [0.0] * segs, [0.0] * segs
t0 = time.monotonic()
with torch.no_grad():
    for batch in eval_batches(x, y, n, 0, 1, ids):
        b = {k: v.to(dev) for k, v in batch.items()}
        carry, prev = model.initial_carry(b), None
        for s in range(segs):
            carry, logits = model(carry, b)
            pred = logits.argmax(-1); ok = pred == b['labels']
            exact[s] += int(ok.all(-1).sum()); acc[s] += float(ok.double().mean(-1).sum())
            if prev is not None: churn[s] += float((pred != prev).double().mean(-1).sum())
            prev = pred
            if (s + 1) % 25 == 0 or s == 15:
                base = exact[15]
                best = max(exact[:s + 1])
                print(f'[{time.monotonic() - t0:6.0f}s] seg {s + 1} ({(s + 1) * bps} blocks): exact {exact[s]}/{n}, '
                      f'best {best} (seg {exact.index(best) + 1}), err.red vs seg16 {(best - base) / (n - base) * 100:.1f}% '
                      f'(current {(exact[s] - base) / (n - base) * 100:.1f}%), churn {churn[s] / n:.5f}', flush=True)
                json.dump(dict(ckpt=ck, n=n, segs_done=s + 1, blocks_per_seg=bps, exact=exact[:s + 1],
                               accuracy=[a / n for a in acc[:s + 1]], churn=[c / n for c in churn[:s + 1]]), open(out, 'w'))
