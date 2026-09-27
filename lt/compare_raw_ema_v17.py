"""Compare stored v1.7 raw/EMA states using the original model and fixed inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt.ckpt_npz import load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def import_original(path):
    spec = importlib.util.spec_from_file_location("original_v17_comparison", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize(module, state):
    state = module.strip_prefix(state)
    result = {key.removeprefix("model."): value for key, value in state.items()}
    assert len(result) == len(state), "Prefix removal caused a collision"
    return result


def state_manifest(raw, ema, model):
    assert set(raw) == set(ema) == set(model.state_dict())
    params = set(dict(model.named_parameters()))
    rows = []
    for key in sorted(raw):
        a, b = raw[key], ema[key]
        assert a.shape == b.shape and a.dtype == b.dtype
        diff = a.double() - b.double()
        l2 = float(diff.norm())
        rows.append(dict(name=key, kind="parameter" if key in params else "persistent_buffer",
                         shape=list(a.shape), dtype=str(a.dtype), numel=a.numel(),
                         identical=torch.equal(a, b), l2_difference=l2,
                         relative_l2_to_raw=l2 / max(float(a.double().norm()), 1e-30),
                         max_abs_difference=float(diff.abs().max()),
                         raw_l2=float(a.double().norm()), ema_l2=float(b.double().norm())))
    for key in ("inner.puzzle_emb.weights", "inner.init_hidden"):
        assert torch.equal(raw[key], ema[key]), f"Fresh-state/common buffer mismatch: {key}"
    return rows


def make_batch(x, y):
    return dict(inputs=torch.from_numpy(x).cuda(), labels=torch.from_numpy(y).cuda().long(),
                puzzle_identifiers=torch.zeros(len(x), dtype=torch.int32, device="cuda"))


@torch.inference_mode()
def predict(model, batch, segs):
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    assert carry.halted.all() and carry.coupling is None and carry.trace is None
    predictions = []
    for seg in range(segs):
        if seg:
            assert not carry.halted.any(), "Unexpected early reset/halt"
        carry, out = model(carry, batch)
        assert torch.isfinite(out["logits"]).all(), "Nonfinite logits"
        predictions.append(out["logits"].argmax(-1).cpu().numpy().astype(np.uint8))
    assert torch.all(carry.steps == segs)
    return np.stack(predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/v17_recovery/checkpoints/v17_step_380000.npz")
    p.add_argument("--source", type=Path, default=ROOT / "runs/v17_recovery/source/2026-09-09/analysis/train_v17.py")
    p.add_argument("--data", type=Path, default=ROOT / "data/sudoku_lt_1k.npz")
    p.add_argument("--output", type=Path, default=ROOT / "runs/v17_recovery/raw_ema_380k")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--segments", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--timing-swaps", action="store_true", help="Also exchange the three stored timing tensors in both directions")
    p.add_argument("--reuse-baselines", action="store_true", help="Validate and reuse completed raw/EMA results")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = import_original(args.source)
    raw, meta = load(str(args.checkpoint), which="raw")
    ema, ema_meta = load(str(args.checkpoint), which="ema")
    assert raw and ema and meta == ema_meta
    states = dict(raw=normalize(original, raw), ema=normalize(original, ema))
    cfg = dict(meta["cfg"])
    cfg.update(batch_size=args.batch_size, seq_len=81, num_puzzle_identifiers=1,
               loops=args.segments, amp=True, compile=False)
    model = original.LT(cfg).eval()
    manifest = state_manifest(states["raw"], states["ema"], model)
    write_json(args.output / "state_manifest.json", manifest)
    swap_manifest = {}
    if args.timing_swaps:
        timing_keys = {"inner.layers.0." + suffix for suffix in ("mu_rho_raw", "mu_omega", "eta_raw")}
        assert timing_keys <= states["raw"].keys()
        for name, backbone, donor in (("raw_ema_timing", "raw", "ema"), ("ema_raw_timing", "ema", "raw")):
            states[name] = {key: states[donor if key in timing_keys else backbone][key]
                            for key in states[backbone]}
            changed = [key for key in states[name] if not torch.equal(states[name][key], states[backbone][key])]
            assert set(changed) == timing_keys
            assert all(torch.equal(states[name][key], states[backbone][key])
                       for key in states[name] if key not in timing_keys)
            swap_manifest[name] = dict(backbone=backbone, timing_donor=donor,
                                       replaced_keys=sorted(timing_keys), changed_keys=sorted(changed),
                                       all_other_persistent_tensors_identical_to_backbone=True)
        write_json(args.output / "timing_swap_manifest.json", swap_manifest)
    loading = {}
    for name, state in states.items():
        report = model.load_state_dict(state, strict=True)
        loading[name] = dict(missing_keys=report.missing_keys, unexpected_keys=report.unexpected_keys)
    model.cuda()
    with np.load(args.data, allow_pickle=False) as z:
        x = z["test_inputs"].reshape(-1, 81)[:args.n].astype(np.int32) + 1
        y = z["test_labels"].reshape(-1, 81)[:args.n].astype(np.int32) + 1
    assert len(x) == args.n and np.all((x == 1) | (x == y))
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                    checkpoint_step=meta["step"], source=str(args.source), source_sha256=digest(args.source),
                    data=str(args.data), data_sha256=digest(args.data), config=cfg,
                    n=args.n, segments=args.segments, batch_size=args.batch_size,
                    input_encoding="original test arrays + 1; blank=1, digits=2..10",
                    precision="float32 parameters/state; original internal CUDA bfloat16 autocast",
                    compile=False, tf32=False, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(), seed=0, strict_loading=loading,
                    identical_common_buffers=["inner.puzzle_emb.weights", "inner.init_hidden"],
                    protocol="fixed weights; fresh carry per batch; fixed segments; no early halt, reset, or clue clamp",
                    scope="380k saved checkpoint comparison only; does not establish the cause of the earlier collapse")
    cached = set()
    if args.reuse_baselines:
        old_metadata = json.loads((args.output / "metadata.json").read_text())
        for key in ("checkpoint_sha256", "source_sha256", "data_sha256", "config", "n", "segments", "batch_size", "precision", "compile", "tf32", "seed", "torch_version", "gpu"):
            assert old_metadata[key] == metadata[key], f"Cached protocol mismatch: {key}"
        for name in ("raw", "ema"):
            assert (args.output / f"{name}_metrics.json").exists()
            assert (args.output / f"{name}_predictions.npy").exists()
            cached.add(name)
    metadata["reused_baseline_predictions"] = sorted(cached)
    metadata["timing_swaps"] = swap_manifest
    prefix = "timing_" if args.timing_swaps else ""
    write_json(args.output / f"{prefix}metadata.json", metadata)
    tiny = {}
    tiny_batch = make_batch(x[:2], y[:2])
    for name, state in states.items():
        if name in cached:
            continue
        model.load_state_dict(state, strict=True)
        a = predict(model, tiny_batch, args.segments)
        b = predict(model, tiny_batch, args.segments)
        assert np.array_equal(a, b), "Fresh-state tiny repeat was not identical"
        tiny[name] = dict(n=2, segments=args.segments, repeated_predictions_identical=True,
                          final_exact=int((a[-1] == y[:2]).all(-1).sum()))
    write_json(args.output / f"{prefix}tiny_validation.json", tiny)
    print(json.dumps(dict(event="tiny_validation_passed", details=tiny)), flush=True)
    predictions, metrics = {}, {}
    for name, state in states.items():
        if name in cached:
            store = np.load(args.output / f"{name}_predictions.npy", mmap_mode="r")
            assert store.shape == (args.segments, args.n, 81) and store.dtype == np.uint8
            saved_metrics = json.loads((args.output / f"{name}_metrics.json").read_text())
            matches = store == y[None]
            assert int(matches[-1].sum()) == saved_metrics["correct_cells"]
            assert int(matches[-1].all(-1).sum()) == saved_metrics["exact"]
            assert all(int(matches[i].all(-1).sum()) == row["exact"]
                       and float(matches[i].mean()) == row["cell_accuracy"]
                       for i, row in enumerate(saved_metrics["per_segment"]))
            predictions[name], metrics[name] = store, saved_metrics
            print(json.dumps(dict(event="reused_validated_baseline", weights=name, exact=saved_metrics["exact"])), flush=True)
            continue
        model.load_state_dict(state, strict=True)
        variant_start = time.monotonic()
        store = np.lib.format.open_memmap(args.output / f"{name}_predictions.npy", mode="w+",
                                         dtype=np.uint8, shape=(args.segments, args.n, 81))
        for offset in range(0, args.n, args.batch_size):
            stop = min(offset + args.batch_size, args.n)
            store[:, offset:stop] = predict(model, make_batch(x[offset:stop], y[offset:stop]), args.segments)
            store.flush()
            progress = dict(weights=name, completed_puzzles=stop, n=args.n,
                            exact_so_far=int((store[-1, :stop] == y[:stop]).all(-1).sum()),
                            elapsed_seconds=time.monotonic() - started)
            write_json(args.output / f"{prefix}progress.json", progress)
            print(json.dumps(progress), flush=True)
        # Check all persistent tensors stayed fixed during inference.
        assert all(torch.equal(value.cpu(), state[key]) for key, value in model.state_dict().items())
        matches = store == y[None]
        metrics[name] = dict(exact=int(matches[-1].all(-1).sum()),
                             correct_cells=int(matches[-1].sum()), cell_accuracy=float(matches[-1].mean()),
                             elapsed_seconds=time.monotonic() - variant_start,
                             per_segment=[dict(segment=i + 1, exact=int(matches[i].all(-1).sum()),
                                               cell_accuracy=float(matches[i].mean())) for i in range(args.segments)])
        predictions[name] = store
        write_json(args.output / f"{name}_metrics.json", metrics[name])
    raw_ok = (predictions["raw"][-1] == y).all(-1)
    ema_ok = (predictions["ema"][-1] == y).all(-1)
    cross = dict(both_success=int((raw_ok & ema_ok).sum()), raw_only_success=int((raw_ok & ~ema_ok).sum()),
                 ema_only_success=int((~raw_ok & ema_ok).sum()), neither_success=int((~raw_ok & ~ema_ok).sum()))
    success = {name: (value[-1] == y).all(-1) for name, value in predictions.items()}
    pairwise = {}
    names = list(predictions)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = success[first], success[second]
            pairwise[f"{first}__vs__{second}"] = dict(both_success=int((a & b).sum()),
                first_only_success=int((a & ~b).sum()), second_only_success=int((~a & b).sum()),
                neither_success=int((~a & ~b).sum()))
    result = dict(**metadata, metrics=metrics, success_cross_table=cross, pairwise_success_cross_tables=pairwise,
                  identical_final_predictions=int((predictions["raw"][-1] == predictions["ema"][-1]).all(-1).sum()),
                  different_final_cells=int((predictions["raw"][-1] != predictions["ema"][-1]).sum()),
                  elapsed_seconds=time.monotonic() - started)
    write_json(args.output / f"{prefix}results.json", result)
    per_puzzle = {f"{name}_success": value for name, value in success.items()}
    per_puzzle.update({f"{name}_correct_cells": (value[-1] == y).sum(-1) for name, value in predictions.items()})
    np.savez_compressed(args.output / f"{prefix}per_puzzle.npz", **per_puzzle)
    print(json.dumps(dict(event="complete", metrics=metrics, success_cross_table=cross)), flush=True)


if __name__ == "__main__":
    main()
