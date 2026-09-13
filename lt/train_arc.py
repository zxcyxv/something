"""Train LinearTuring on the official URM ARC dataset interface, on one device.

Reference: UbiquantAI/URM, c14e55f5f9227873617015cf60a239126b55adcd.
See docs/arc_training.md for preprocessing, protocol and differences from URM.
Only torch and numpy are needed after dataset preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

try:
    from .train import (ACTLossHead, EMAHelper, IGNORE_LABEL_ID, LT, LTCarry,
                        LTConfig, PRESETS, _EMASwap, create_optimizers,
                        cosine_schedule_with_warmup_lr_lambda)
except ImportError:
    from train import (ACTLossHead, EMAHelper, IGNORE_LABEL_ID, LT, LTCarry,
                       LTConfig, PRESETS, _EMASwap, create_optimizers,
                       cosine_schedule_with_warmup_lr_lambda)


URM_COMMIT = "c14e55f5f9227873617015cf60a239126b55adcd"
ARRAY_FIELDS = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")


class ARCSplit:
    """Memory-mapped URM arrays; only sampled examples are materialized."""

    def __init__(self, root: Path, split: str):
        self.root, self.split = root, split
        self.metadata = json.loads((root / split / "dataset.json").read_text())
        m = self.metadata
        if (m["seq_len"], m["vocab_size"], m["pad_id"], m["ignore_label_id"],
                m["blank_identifier_id"]) != (900, 12, 0, 0, 0):
            raise ValueError("Expected URM ARC encoding: 900 tokens, vocab 12, PAD/ignore/blank ID 0")
        self.arrays = {}
        for name in m["sets"]:
            if Path(name).name != name:
                raise ValueError(f"Invalid set name: {name}")
            data = {field: np.load(root / split / f"{name}__{field}.npy", mmap_mode="r")
                    for field in ARRAY_FIELDS}
            self._validate(data)
            self.arrays[name] = data
        if not self.arrays:
            raise ValueError(f"Empty dataset split {split}")

    def _validate(self, data):
        inputs, labels = data["inputs"], data["labels"]
        pi, gi, ids = (data[k] for k in ("puzzle_indices", "group_indices", "puzzle_identifiers"))
        if inputs.ndim != 2 or inputs.shape != labels.shape or inputs.shape[1] != 900:
            raise ValueError("ARC inputs/labels must both have shape [examples, 900]")
        if pi.ndim != 1 or gi.ndim != 1 or ids.ndim != 1 or not len(ids):
            raise ValueError("Invalid or empty puzzle/group index arrays")
        if len(pi) != len(ids) + 1 or pi[0] != 0 or pi[-1] != len(inputs) or np.any(np.diff(pi) <= 0):
            raise ValueError("Puzzle indices must delimit nonempty contiguous example ranges")
        if len(gi) < 2 or gi[0] != 0 or gi[-1] != len(ids) or np.any(np.diff(gi) <= 0):
            raise ValueError("Group indices must delimit nonempty contiguous puzzle ranges")
        if ids.min() <= 0 or ids.max() >= self.metadata["num_puzzle_identifiers"]:
            raise ValueError("Puzzle identifier out of metadata range (0 is reserved for padding)")

    def batch(self, data, example_indices, puzzle_indices, size=None):
        inputs = np.array(data["inputs"][example_indices], dtype=np.int64, copy=True)
        labels = np.array(data["labels"][example_indices], dtype=np.int64, copy=True)
        identifiers = np.array(data["puzzle_identifiers"][puzzle_indices], dtype=np.int64, copy=True)
        if np.any((inputs < 0) | (inputs >= 12)) or np.any((labels < 0) | (labels >= 12)):
            raise ValueError("Invalid ARC token; input and stored label IDs must be 0..11")
        labels[labels == 0] = IGNORE_LABEL_ID
        if size is not None and len(identifiers) < size:
            pad = size - len(identifiers)
            inputs = np.pad(inputs, ((0, pad), (0, 0)))
            labels = np.pad(labels, ((0, pad), (0, 0)), constant_values=IGNORE_LABEL_ID)
            identifiers = np.pad(identifiers, (0, pad))
        return {"inputs": torch.from_numpy(inputs), "labels": torch.from_numpy(labels),
                "puzzle_identifiers": torch.from_numpy(identifiers)}


class ARCTrainStream:
    """URM group -> augmentation -> examples sampler, with resumable RNG/cursor.

    Like upstream, concatenate shuffled group epochs, select one augmented puzzle
    per group, sample examples without replacement, and drop partial final batches.
    Candidate batches are still consumed while LT carries an unfinished puzzle.
    """

    def __init__(self, dataset: ARCSplit, batch_size: int, seed: int, epochs_per_iter: int):
        self.dataset, self.batch_size = dataset, batch_size
        self.seed, self.epochs_per_iter = seed, epochs_per_iter
        self.names = list(dataset.arrays)
        self.iteration, self.set_index, self.position = 0, -1, 0
        self.order = np.empty(0, dtype=np.int64)
        self.rng = np.random.Generator(np.random.Philox(seed))
        capacity = max(sum(min(int(n), batch_size) for n in np.diff(data["puzzle_indices"]))
                       for data in dataset.arrays.values())
        if capacity * epochs_per_iter < batch_size:
            raise ValueError("Dataset/epochs-per-iter too small to produce one full batch")

    def _begin_set(self):
        self.iteration += 1
        self.set_index = (self.set_index + 1) % len(self.names)
        data = self.dataset.arrays[self.names[self.set_index]]
        self.rng = np.random.Generator(np.random.Philox(self.seed + self.iteration))
        self.order = np.concatenate([self.rng.permutation(len(data["group_indices"]) - 1)
                                     for _ in range(self.epochs_per_iter)])
        self.position = 0

    def __next__(self):
        for _ in range(len(self.names) + 2):
            if self.position >= len(self.order):
                self._begin_set()
            data = self.dataset.arrays[self.names[self.set_index]]
            pi, gi = data["puzzle_indices"], data["group_indices"]
            example_ids, puzzle_ids, count = [], [], 0
            while self.position < len(self.order) and count < self.batch_size:
                group = self.order[self.position]
                puzzle = int(self.rng.integers(gi[group], gi[group + 1]))
                self.position += 1
                length = int(pi[puzzle + 1] - pi[puzzle])
                take = min(length, self.batch_size - count)
                example_ids.extend(pi[puzzle] + self.rng.choice(length, take, replace=False))
                puzzle_ids.extend([puzzle] * take)
                count += take
            if count == self.batch_size:
                return self.dataset.batch(data, example_ids, puzzle_ids)
        raise ValueError("No full training batch; lower batch size or increase --epochs-per-iter")

    def state_dict(self):
        return dict(iteration=self.iteration, set_index=self.set_index, position=self.position,
                    order=self.order, rng_state=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        for key in ("iteration", "set_index", "position", "order"):
            setattr(self, key, state[key])
        self.rng.bit_generator.state = state["rng_state"]


def dihedral(grid, transform):
    if transform < 4:
        return np.rot90(grid, transform)
    if transform == 4:
        return np.fliplr(grid)
    if transform == 5:
        return np.flipud(grid)
    if transform == 6:
        return grid.T
    if transform == 7:
        return np.fliplr(np.rot90(grid))
    raise ValueError(f"Invalid dihedral transform {transform}")


def inverse_augmentation(name, grid):
    if "|||" not in name:
        return name, grid
    original, trans, permutation = name.split("|||")
    inverse = [0, 3, 2, 1, 4, 5, 6, 7][int(trans[1:])]
    grid = dihedral(grid, inverse)
    return original, np.argsort(list(permutation)).astype(np.uint8)[grid]


def crop_grid(tokens):
    """Upstream evaluator's largest all-color rectangle anchored at (0, 0).

    Do not crop using target dimensions: output size is part of the prediction.
    Evaluation inputs are never translation-augmented by the URM builder.
    """
    grid = np.asarray(tokens).reshape(30, 30)
    width, best_area, best = 30, 0, (0, 0)
    for height in range(1, 31):
        for col in range(width):
            if grid[height - 1, col] < 2 or grid[height - 1, col] > 11:
                width = col
                break
        if height * width > best_area:
            best_area, best = height * width, (height, width)
    return (grid[:best[0], :best[1]] - 2).astype(np.uint8)


def grid_key(grid):
    return tuple(grid.shape), np.ascontiguousarray(grid, dtype=np.uint8).tobytes()


def eval_batches(dataset, identifiers, tasks, batch_size, max_augmentations):
    for data in dataset.arrays.values():
        examples, puzzles = [], []
        counts = Counter()
        for puzzle, identifier in enumerate(data["puzzle_identifiers"]):
            name = identifiers[int(identifier)].split("|||")[0]
            if name not in tasks or (max_augmentations and counts[name] >= max_augmentations):
                continue
            counts[name] += 1
            start, end = data["puzzle_indices"][puzzle:puzzle + 2]
            for example in range(int(start), int(end)):
                examples.append(example)
                puzzles.append(puzzle)
                if len(examples) == batch_size:
                    yield dataset.batch(data, examples, puzzles, batch_size)
                    examples, puzzles = [], []
        if examples:
            yield dataset.batch(data, examples, puzzles, batch_size)


class ARCVotes:
    """One evaluation sweep, inverse augmentation, frequency-ranked unique grids.

    LT q logits are constant. URM's (count, mean q, max log q) ranking therefore
    reduces to count with stable ties. No cross-checkpoint vote accumulation.
    """

    def __init__(self, identifiers, tasks):
        self.identifiers, self.tasks = identifiers, tasks
        self.votes = defaultdict(Counter)
        self.grids = {}

    def update(self, batch, predictions):
        for identifier, inputs, pred in zip(batch["puzzle_identifiers"].cpu().numpy(),
                                           batch["inputs"].cpu().numpy(), predictions.cpu().numpy()):
            if identifier == 0:
                continue
            name = self.identifiers[int(identifier)]
            task, inp = inverse_augmentation(name, crop_grid(inputs))
            _, output = inverse_augmentation(name, crop_grid(pred))
            key = grid_key(output)
            self.votes[task, grid_key(inp)][key] += 1
            self.grids[key] = output

    def result(self):
        scores = {k: 0.0 for k in (1, 2, 5, 10)}
        submission, covered = {}, 0
        for name, task in self.tasks.items():
            correct = {k: 0 for k in scores}
            submission[name] = []
            for pair in task["test"]:
                votes = self.votes[name, grid_key(np.array(pair["input"], dtype=np.uint8))]
                ranked = [key for key, _ in votes.most_common()]
                if not ranked:
                    raise ValueError(f"Evaluation task {name} has a query without predictions")
                covered += 1
                label = grid_key(np.array(pair["output"], dtype=np.uint8))
                for k in scores:
                    correct[k] += label in ranked[:k]
                top2 = ranked[:2]
                top2 += [top2[0]] * (2 - len(top2))
                submission[name].append({f"attempt_{i + 1}": self.grids[key].tolist()
                                         for i, key in enumerate(top2)})
            for k in scores:
                scores[k] += correct[k] / len(task["test"])
        return ({f"ARC/pass@{k}": value / len(self.tasks) for k, value in scores.items()}
                | {"ARC/tasks": len(self.tasks), "ARC/queries": covered}), submission


@torch.inference_mode()
def evaluate(base, ema, dataset, identifiers, tasks, args, device):
    selected = {name: tasks[name] for name in sorted(tasks)[:args.eval_max_tasks or None]}
    if not selected:
        raise ValueError("No evaluation tasks")
    lt = base.model
    loops = lt.config.loops
    was_training = base.training
    lt.config.loops = args.eval_segments or loops
    votes = ARCVotes(identifiers, selected)
    totals = Counter()
    started = time.monotonic()
    with _EMASwap(base, ema):
        base.eval()
        try:
            for index, batch in enumerate(eval_batches(dataset, identifiers, selected, args.batch_size,
                                                        args.eval_max_augmentations)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.device(device):
                    carry = lt.initial_carry(batch)
                for _ in range(lt.config.loops):
                    carry, out = lt(carry, batch)
                pred = out["logits"].argmax(-1)
                mask = batch["labels"] != IGNORE_LABEL_ID
                valid = mask.any(-1)
                correct = (pred == batch["labels"]) & mask
                totals["examples"] += int(valid.sum())
                totals["accuracy"] += float((correct.sum(-1) / mask.sum(-1).clamp_min(1))[valid].sum())
                totals["exact"] += int(((correct.sum(-1) == mask.sum(-1)) & valid).sum())
                votes.update(batch, pred)
                if (index + 1) % 100 == 0:
                    print(f"[ARC eval] {index + 1} batches, {time.monotonic() - started:.1f}s", flush=True)
        finally:
            lt.config.loops = loops
            base.train(was_training)
    metrics, submission = votes.result()
    metrics.update(token_accuracy=totals["accuracy"] / max(totals["examples"], 1),
                   sequence_exact=totals["exact"] / max(totals["examples"], 1),
                   evaluated_augmented_examples=totals["examples"],
                   eval_segments=args.eval_segments or loops,
                   max_augmentations_per_task=args.eval_max_augmentations,
                   total_available_tasks=len(tasks), seconds=time.monotonic() - started)
    return metrics, submission


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    return value


def dataset_fingerprint(root, splits):
    """Hash metadata, identifiers, provenance and sampled array bytes.

    Full multi-gigabyte array hashes are deliberately omitted at training launch.
    File size plus head/tail blocks detect common accidental dataset replacements.
    """
    digest = hashlib.sha256()
    for name in ("identifiers.json", "test_puzzles.json", "provenance.json"):
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    for split in splits:
        digest.update(json.dumps(split.metadata, sort_keys=True).encode())
        for path in sorted((root / split.split).glob("*.npy")):
            digest.update(f"{split.split}/{path.name}:{path.stat().st_size}".encode())
            with path.open("rb") as handle:
                digest.update(handle.read(65536))
                handle.seek(max(0, path.stat().st_size - 65536))
                digest.update(handle.read())
    return digest.hexdigest()


def save_checkpoint(path, base, optimizers, ema, carry, stream, step, args, fingerprint):
    raw = tree_to(base.state_dict(), "cpu")
    ema_state = tree_to(ema.state_dict(), "cpu") if ema is not None else None
    weights = dict(raw)
    if ema_state:
        weights.update(ema_state)
    state = dict(format="lt_arc_v1", step=step, model_cfg=asdict(base.model.config),
                 run_cfg=vars(args), dataset_fingerprint=fingerprint, urm_commit=URM_COMMIT,
                 raw_model_state_dict=raw, model_state_dict=weights, ema_shadow=ema_state,
                 optimizer_states=tree_to([optimizer.state_dict() for optimizer in optimizers], "cpu"),
                 carry=tree_to({field.name: getattr(carry, field.name) for field in fields(LTCarry)}, "cpu")
                 if carry is not None else None,
                 sampler=stream.state_dict(), torch_rng=torch.get_rng_state(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state_all() if next(base.parameters()).device.type == "cuda" else None)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/arc_v1_1"))
    parser.add_argument("--preset", choices=PRESETS, default="v1.1")
    parser.add_argument("--hidden-size", type=int, default=832)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--puzzle-emb-dim", type=int)
    parser.add_argument("--loops", type=int, default=16)
    parser.add_argument("--blocks-per-seg", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--epochs-per-iter", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, help="Absolute optimizer step at which to stop, including on resume")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--puzzle-emb-weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-min-ratio", type=float, default=1.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--eval-every", type=int, default=1000, help="Optimizer steps; 0 disables periodic evaluation")
    parser.add_argument("--eval-segments", type=int, default=0, help="0 uses training loops; larger values test extrapolation")
    parser.add_argument("--eval-max-tasks", type=int, default=0, help="Diagnostic task subset; 0 evaluates all tasks")
    parser.add_argument("--eval-max-augmentations", type=int, default=0, help="Per-task augmentation cap, including identity; 0 uses all")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    for name in ("batch_size", "hidden_size", "num_heads", "loops", "blocks_per_seg", "num_layers",
                 "epochs", "epochs_per_iter", "log_every", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("eval_every", "eval_segments", "eval_max_tasks", "eval_max_augmentations", "save_every",
                 "lr_warmup_steps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume with an ARC checkpoint")
    if args.hidden_size % (2 * args.num_heads):
        parser.error("--hidden-size must be divisible by twice --num-heads")
    if not (0 <= args.lr_min_ratio <= 1 and 0 <= args.ema_rate < 1):
        parser.error("lr-min-ratio must be in [0,1] and ema-rate in [0,1)")
    if args.puzzle_emb_dim is not None and not 0 < args.puzzle_emb_dim <= args.hidden_size:
        parser.error("--puzzle-emb-dim must be in 1..hidden-size")
    if any(getattr(args, name) < 0 for name in ("lr", "puzzle_emb_lr", "weight_decay", "puzzle_emb_weight_decay")):
        parser.error("Learning rates and weight decays must be nonnegative")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error("Optimizer beta1 and beta2 must be in [0,1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.resume is None and (args.out_dir / "latest.pt").exists():
        raise ValueError("Output directory already contains latest.pt; choose a new --out-dir or use --resume")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This CLI supports one process/device; do not launch it with multi-process torchrun")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint is not None:
        if checkpoint.get("format") != "lt_arc_v1":
            raise ValueError("--resume requires an lt/train_arc.py checkpoint; Sudoku checkpoints are incompatible")
        # Preserve architecture, optimization and data-stream settings. Runtime
        # controls (device, output, stop step, evaluation) remain CLI-controlled.
        for key in ("batch_size", "epochs", "epochs_per_iter", "lr", "puzzle_emb_lr", "weight_decay",
                    "puzzle_emb_weight_decay", "beta1", "beta2", "lr_warmup_steps", "lr_min_ratio",
                    "ema", "ema_rate", "seed"):
            setattr(args, key, checkpoint["run_cfg"][key])
    root = args.data.resolve()
    train_data, test_data = ARCSplit(root, "train"), ARCSplit(root, "test")
    identifiers = json.loads((root / "identifiers.json").read_text())
    tasks = json.loads((root / "test_puzzles.json").read_text())
    if train_data.metadata["num_puzzle_identifiers"] != test_data.metadata["num_puzzle_identifiers"]:
        raise ValueError("Train/test puzzle identifier vocabularies differ")
    if len(identifiers) != train_data.metadata["num_puzzle_identifiers"]:
        raise ValueError("identifiers.json length disagrees with dataset metadata")
    for name, task in tasks.items():
        if not task.get("test") or any("output" not in pair for pair in task["test"]):
            raise ValueError(f"Missing reference test outputs for task {name}")
    fingerprint = dataset_fingerprint(root, (train_data, test_data))
    if checkpoint is not None and checkpoint["dataset_fingerprint"] != fingerprint:
        raise ValueError("Dataset fingerprint differs from checkpoint; ID embeddings cannot be safely reused")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = checkpoint["model_cfg"] if checkpoint is not None else dict(
        batch_size=args.batch_size, seq_len=900, grid=30, vocab_size=12,
        num_puzzle_identifiers=len(identifiers), puzzle_emb_ndim=args.puzzle_emb_dim or args.hidden_size,
        hidden_size=args.hidden_size, num_heads=args.num_heads, loops=args.loops,
        blocks_per_seg=args.blocks_per_seg, num_layers=args.num_layers, **PRESETS[args.preset])
    cfg = dict(cfg, amp=args.amp and device.type == "cuda", forward_dtype="float32")
    if not 0 < cfg["puzzle_emb_ndim"] <= cfg["hidden_size"]:
        raise ValueError("Puzzle embedding dimension must be in 1..hidden_size")
    # Sparse trainable embedding buffers must be created on-device: moving a
    # requires_grad buffer afterwards can turn it into a non-leaf tensor.
    with torch.device(device):
        base = ACTLossHead(LT(cfg), "stablemax_cross_entropy", q_weight=0)
    optimizers, base_lrs = create_optimizers(base, vars(args), world_size=1)
    ema = EMAHelper(args.ema_rate) if args.ema else None
    if ema is not None:
        ema.register(base)
    stream = ARCTrainStream(train_data, args.batch_size, args.seed, args.epochs_per_iter)
    carry, step = None, 0
    if checkpoint is not None:
        base.load_state_dict(checkpoint["raw_model_state_dict"], strict=True, assign=False)
        for optimizer, state in zip(optimizers, checkpoint["optimizer_states"]):
            optimizer.load_state_dict(tree_to(state, device))
        if ema is not None:
            ema.load_state_dict(tree_to(checkpoint["ema_shadow"], device))
        if checkpoint["carry"] is not None:
            carry = LTCarry(**tree_to(checkpoint["carry"], device))
        stream.load_state_dict(checkpoint["sampler"])
        step = checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"[ARC] resumed step {step}; model/optimizer/sampler/carry restored", flush=True)
    del checkpoint
    args.out_dir.mkdir(parents=True, exist_ok=True)
    planned_steps = max(1, int(args.epochs * train_data.metadata["total_groups"] *
                               train_data.metadata["mean_puzzle_examples"] / args.batch_size))
    stop_step = args.max_steps if args.max_steps is not None else planned_steps
    coupling_bytes = args.batch_size * cfg["num_heads"] * 900 * 900 * 4
    print(f"[ARC] device={device} tokens=900 IDs={len(identifiers)} batch={args.batch_size} "
          f"loops={cfg['loops']} blocks={cfg['blocks_per_seg']} stop_step={stop_step}\n"
          f"[ARC] coupling carry alone={coupling_bytes / 2**20:.1f} MiB; activations/gradients require extra memory", flush=True)
    config_path = args.out_dir / (f"resume_step_{step}_config.json" if args.resume else "config.json")
    config_path.write_text(json.dumps(dict(model_cfg=asdict(base.model.config), run_cfg=vars(args),
                                           dataset_fingerprint=fingerprint, urm_commit=URM_COMMIT), default=str, indent=2) + "\n")

    def run_eval():
        result, submission = evaluate(base, ema, test_data, identifiers, tasks, args, device)
        result["step"] = step
        result["weights"] = "ema_parameters_raw_puzzle_embeddings" if ema else "raw"
        tag = f"eval_step_{step}_seg{args.eval_segments or base.model.config.loops}"
        (args.out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out_dir / f"{tag}_submission.json").write_text(json.dumps(submission) + "\n")
        print("[ARC eval] " + json.dumps(result), flush=True)

    if args.eval_only:
        run_eval()
        return
    base.train()
    model = torch.compile(base) if args.compile else base
    last_eval, started = -1, time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("[ARC] stop requested; completing the current step before checkpointing", flush=True)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        while step < stop_step and not stop_requested:
            batch = {key: value.to(device) for key, value in next(stream).items()}
            if carry is None:
                with torch.device(device):
                    carry = base.initial_carry(batch)
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=set())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            (loss / args.batch_size).backward()
            for optimizer, lr in zip(optimizers, base_lrs):
                scheduled = cosine_schedule_with_warmup_lr_lambda(
                    # Upstream URM and lt/train.py schedule before incrementing
                    # step: the first optimizer update has lr=0 during warmup.
                    current_step=step, base_lr=lr, num_warmup_steps=args.lr_warmup_steps,
                    num_training_steps=planned_steps, min_ratio=args.lr_min_ratio)
                for group in optimizer.param_groups:
                    group["lr"] = scheduled
                optimizer.step()
                optimizer.zero_grad()
            step += 1
            if ema is not None:
                ema.update(base)
            if step == 1 or step % args.log_every == 0:
                report = dict(step=step, loss=float(loss.detach()) / args.batch_size,
                              halted=int(metrics["count"]), seconds=time.monotonic() - started)
                if report["halted"]:
                    report["exact"] = float(metrics["exact_accuracy"]) / report["halted"]
                print("[ARC train] " + json.dumps(report), flush=True)
                with (args.out_dir / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(report) + "\n")
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(args.out_dir / "latest.pt", base, optimizers, ema, carry, stream,
                                step, args, fingerprint)
            if args.eval_every and step % args.eval_every == 0 and not stop_requested:
                run_eval()
                last_eval = step
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    save_checkpoint(args.out_dir / "latest.pt", base, optimizers, ema, carry, stream, step, args, fingerprint)
    if args.eval_at_end and last_eval != step and not stop_requested:
        run_eval()
    print(f"[ARC] checkpoint -> {args.out_dir / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
