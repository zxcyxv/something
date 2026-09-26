"""Evaluate LT or a public TRM on the same held-out Sudoku puzzles.

Keep every segment prediction, use fixed-length inference without resets, and
retry at batch 1024 only if the requested batch 2048 runs out of CUDA memory.
TRM requires its source repository and the checkpoint's all_config.yaml.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_model(args, batch_size):
    if args.model == "lt":
        from lt import train
        from lt.ckpt_npz import load_lt
        model, cfg, step = load_lt(str(args.checkpoint), mod=train,
                                  batch_size=batch_size, loops=args.segs + 1)
        return model, dict(config=cfg, checkpoint_step=step,
                           blocks_per_segment=cfg["blocks_per_seg"] * cfg.get("num_layers", 1),
                           precision="float32 state/weights, bfloat16 CUDA autocast",
                           weights="ema; large checkpoint tensors stored as fp16")

    import yaml
    sys.path.insert(0, str(args.trm_source.resolve()))
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    cfg = yaml.safe_load(args.trm_config.read_text())["arch"]
    cfg.update(batch_size=batch_size, seq_len=81, num_puzzle_identifiers=1,
               vocab_size=11, halt_max_steps=args.segs + 1)
    model = TinyRecursiveReasoningModel_ACTV1(cfg).cuda().eval()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", "").removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model, dict(config=cfg,
                       source_commit=subprocess.check_output(
                           ["git", "-C", str(args.trm_source), "rev-parse", "HEAD"], text=True).strip(),
                       blocks_per_segment=cfg["H_cycles"] * (cfg["L_cycles"] + 1) * cfg["L_layers"],
                       precision=cfg["forward_dtype"], weights="public checkpoint state_dict, loaded strictly")


@torch.inference_mode()
def evaluate(args, batch_size, oom_attempts):
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, info = load_model(args, batch_size)
    with np.load(args.data) as z:
        x = z["test_inputs"].reshape(-1, 81)[:args.n].astype(np.int32) + 1
        y = z["test_labels"].reshape(-1, 81)[:args.n].astype(np.int32) + 1
    assert len(x) == args.n and np.all((x == 1) | (x == y))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pred_store = np.lib.format.open_memmap(output / "predictions.npy", mode="w+",
                                         dtype=np.uint8, shape=(args.segs, args.n, 81))
    exact = np.zeros(args.segs, np.int64)
    correct = np.zeros(args.segs, np.int64)
    clue_errors = np.zeros(args.segs, np.int64)
    changed = np.zeros(args.segs, np.int64)
    metadata = dict(model=args.model, checkpoint=str(args.checkpoint),
                    checkpoint_sha256=sha256(args.checkpoint), data=str(args.data),
                    data_sha256=sha256(args.data), n=args.n, batch_size=batch_size,
                    requested_batch_size=args.batch_size, oom_attempts=oom_attempts.copy(),
                    segments=args.segs, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(), tf32=False, compile=False,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    input_encoding="blank=1, digits=2..10", evaluation="fixed horizon; no early stopping, no puzzle reset, no clue clamping",
                    **info)
    save_json(output / "metadata.json", metadata)
    start = time.monotonic()
    for offset in range(0, args.n, batch_size):
        stop = min(offset + batch_size, args.n)
        batch = dict(inputs=torch.from_numpy(x[offset:stop]).cuda(),
                     labels=torch.from_numpy(y[offset:stop]).cuda().long(),
                     puzzle_identifiers=torch.zeros(stop - offset, dtype=torch.int32, device="cuda"))
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        prev = None
        for seg in range(args.segs):
            carry, out = model(carry, batch)
            pred = out["logits"].argmax(-1).cpu().numpy().astype(np.uint8)
            pred_store[seg, offset:stop] = pred
            match = pred == y[offset:stop]
            exact[seg] += match.all(-1).sum()
            correct[seg] += match.sum()
            clue_errors[seg] += ((pred != x[offset:stop]) & (x[offset:stop] != 1)).sum()
            if prev is not None:
                changed[seg] += ((pred != prev) & (x[offset:stop] == 1)).sum()
            prev = pred
            if seg + 1 in (1, 2, 4, 8, 16) or (seg + 1) % 32 == 0 or seg + 1 == args.segs:
                progress = dict(completed_batches=offset // batch_size, current_batch=[offset, stop],
                                current_segment=seg + 1, exact_at_segment=int(exact[seg]),
                                evaluated_puzzles_at_segment=stop, elapsed_seconds=time.monotonic()-start,
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save_json(output / "progress.json", progress)
                pred_store.flush()
                print(json.dumps(progress), flush=True)
        np.savez_compressed(output / "partial_metrics.npz", exact=exact, correct_cells=correct,
                            clue_errors=clue_errors, blank_changes=changed, completed_puzzles=stop)
    rows = [dict(segment=i+1, exact=int(exact[i]), exact_pct=float(100*exact[i]/args.n),
                 cell_accuracy=float(correct[i]/(args.n*81)), clue_errors=int(clue_errors[i]),
                 blank_changes=int(changed[i])) for i in range(args.segs)]
    all_exact = (pred_store == y[None]).all(-1)
    result = dict(**metadata, elapsed_seconds=time.monotonic()-start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  rows=rows, best_exact=int(exact.max()), best_segment=int(exact.argmax()+1),
                  ever_solved=int(all_exact.any(0).sum()), final_exact=int(exact[-1]),
                  solved_at_16_lost_at_end=int((all_exact[15] & ~all_exact[-1]).sum()) if args.segs>=16 else None)
    save_json(output / "summary.json", result)
    pred_store.flush()
    print("FINISHED", args.model, result["final_exact"], "/", args.n, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["lt", "trm"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/sudoku_lt_1k.npz")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--segs", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--fallback-batch-size", type=int, default=1024)
    p.add_argument("--trm-source", type=Path)
    p.add_argument("--trm-config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    oom_attempts = []
    try:
        evaluate(args, args.batch_size, oom_attempts)
        return
    except torch.cuda.OutOfMemoryError:
        oom_attempts.append(args.batch_size)
        save_json(args.output / "oom.json", dict(failed_batch_sizes=oom_attempts))
        print(f"CUDA OOM at batch {args.batch_size}; retrying {args.fallback_batch_size}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    if args.fallback_batch_size >= args.batch_size:
        raise RuntimeError("Fallback must be smaller than requested batch")
    evaluate(args, args.fallback_batch_size, oom_attempts)


if __name__ == "__main__":
    main()
