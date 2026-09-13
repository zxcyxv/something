"""Bounded CPU-only checkpoint pilot for an even-plasticity hypothesis.

Predeclared default: held-out indices 128:144, original versus converted even
plasticity, fresh starts, up to 1024 blocks. These puzzles were outside the
earlier 128-puzzle relation analyses. This is a no-retraining diagnostic, not
an architecture leaderboard or evidence about training speed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from even_plasticity import from_v11
from research_cpu import CPUBlocks, load_cpu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=128)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--out", default="runs/even_plasticity_pilot_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    start_time = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load("data/sudoku_lt_1k.npz", allow_pickle=False) as z:
        x = z["test_inputs"].reshape(-1, 81)[args.start:args.start+args.n].copy()
        y = z["test_labels"].reshape(-1, 81)[args.start:args.start+args.n].copy()
    batch = dict(inputs=torch.from_numpy(x+1).long(), labels=torch.from_numpy(y+1).long(),
                 puzzle_identifiers=torch.zeros(args.n, dtype=torch.int32))
    model, meta = load_cpu(dtype=torch.float32)
    blocks = CPUBlocks(model, batch)
    even = from_v11(model).eval()
    ei, el = even.inner, even.inner.layers[0]
    eab, ekc = ei.W_C(el), ei.kernel(el)
    h = model.inner.init_hidden.expand(args.n, 81, -1).clone()
    he, w, p = h.clone(), None, None
    report = dict(checkpoint_step=meta["step"], puzzle_indices=list(range(args.start, args.start+args.n)),
                  precision="FP32 CPU, two threads", retrained=False,
                  planned_blocks=args.blocks, max_seconds=args.max_seconds,
                  original_config="v1.1 unmodified", candidate="even signed spectrum; bootstrap first write; packed state",
                  records=[], status="running")
    predictions = {}
    first_solved = {name: np.full(args.n, -1, dtype=np.int32) for name in ("original", "even")}
    def save():
        temp = out/"summary.tmp"
        temp.write_text(json.dumps(report, indent=2)+"\n")
        temp.replace(out/"summary.json")
    save()
    for k in range(1, args.blocks+1):
        h, w = blocks.block(h, w)
        qe = ei.boundary(el, he)+blocks.inj
        he, p, _ = ei.step(el, qe, eab, ekc, p)
        if k % 8 == 0:
            metrics = {}
            for name, state, decoder in (("original", h, model.inner.w_cls), ("even", he, ei.w_cls)):
                pred = decoder(state).argmax(-1)
                correct = pred.eq(batch["labels"])
                solved = correct.all(-1).numpy()
                new = solved & (first_solved[name] < 0)
                first_solved[name][new] = k
                metrics[name] = dict(solved=int(solved.sum()), cell_accuracy=float(correct.float().mean()))
                if k % 64 == 0:
                    predictions[f"{name}_{k}"] = pred.numpy().astype(np.int8)
            if k % 64 == 0:
                row = dict(block=k, seconds=time.monotonic()-start_time, **metrics)
                report["records"].append(row)
                report["completed_blocks"] = k
                save()
                print(json.dumps(row), flush=True)
        if time.monotonic()-start_time >= args.max_seconds:
            report["status"] = "stopped_at_declared_time_limit"
            break
    else:
        report["status"] = "completed"
    report["completed_blocks"] = k
    report["elapsed_seconds"] = time.monotonic()-start_time
    report["first_solved_at_segment_boundaries"] = {name: a.tolist() for name, a in first_solved.items()}
    report["cuda_initialized"] = torch.cuda.is_initialized()
    final = {}
    for name, state, decoder in (("original", h, model.inner.w_cls), ("even", he, ei.w_cls)):
        pred = decoder(state).argmax(-1)
        correct = pred.eq(batch["labels"])
        final[name] = dict(solved=int(correct.all(-1).sum()), cell_accuracy=float(correct.float().mean()),
                           solved_indices=np.flatnonzero(correct.all(-1).numpy()).tolist())
        predictions[f"{name}_final"] = pred.numpy().astype(np.int8)
    report["final"] = final
    save()
    np.savez_compressed(out/"predictions.npz", **predictions)
    print(json.dumps(dict(status=report["status"], blocks=k, final=final,
                          elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
