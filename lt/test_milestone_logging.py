"""CPU checks for the supplied Kaggle harness's interruptible milestone reports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import train


class ToyLT(nn.Module):
    def __init__(self, loops=2, fail=False):
        super().__init__()
        self.config = SimpleNamespace(loops=loops)
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.fail = fail
        self.seen = []

    def initial_carry(self, batch):
        self.seen.extend((batch["inputs"][:, 0] - 1).tolist())
        return 0

    def forward(self, carry, batch):
        if self.fail:
            raise RuntimeError("toy forward failure")
        segment = carry + 1
        pred = batch["labels"].clone().long()
        wrong = (batch["inputs"][:, 0] - 1) % 3 >= segment - 1
        if self.offset.item() > 0:
            wrong[:] = False
        pred[wrong, 0] = (pred[wrong, 0] + 1) % 11
        return segment, {"logits": torch.nn.functional.one_hot(pred, 11).float()}


class ToyHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ToyLT(**kwargs)


def data(n=7):
    inputs = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None, None], (n, 9, 9)).copy()
    return inputs, np.ones_like(inputs)


def config(**updates):
    result = dict(train.CFG, global_batch_size=4, milestone_extrap_n=3,
                  milestone_extrap_segs=4, address_projection="linear")
    result.update(updates)
    return result


class MilestoneLoggingTests(unittest.TestCase):
    def test_global_cap_partial_batch_and_configurable_training_segment(self):
        base = ToyHead(loops=2).eval()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "extrap_step_20.txt"
            result = train.extrapolate(base, *data(), config(), 0, 1, "cpu", 20, None, 4, path)
            text = path.read_text()
            saved = json.loads(path.with_suffix(".json").read_text())
            history = [json.loads(line) for line in (Path(directory) / "extrap_results.jsonl").read_text().splitlines()]
        self.assertEqual(base.model.seen, [0, 1, 2])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["exact"], [0, 1, 2, 3])
        self.assertEqual(result["train"]["segment"], 2)
        self.assertEqual(result["train"]["exact"], 1)
        self.assertEqual(result["best_exact"]["segment"], 4)
        self.assertEqual(result["weights"], "raw")
        self.assertEqual(result, saved)
        self.assertEqual(history, [saved])
        self.assertIn(text.rstrip(), output.getvalue())
        self.assertIn("train: seg2", text)
        self.assertNotIn("seg16", text)
        self.assertEqual(base.model.config.loops, 2)
        self.assertFalse(base.training)

    def test_short_run_uses_ema_and_restores_parameters_and_mode(self):
        base = ToyHead(loops=16)
        ema = train.EMAHelper()
        ema.register(base)
        ema.shadow["model.offset"].fill_(1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            result = train.extrapolate(base, *data(), config(milestone_extrap_n=1),
                                       0, 1, "cpu", 9, ema, 2, Path(directory) / "short.txt")
        self.assertEqual(result["weights"], "ema")
        self.assertEqual(result["exact"], [1, 1])
        self.assertIsNone(result["train"])
        self.assertIn("train: seg16 not evaluated", output.getvalue())
        self.assertEqual(base.model.offset.item(), 0)
        self.assertEqual(base.model.config.loops, 16)
        self.assertTrue(base.training)

    def test_distributed_global_cap_including_rank_with_no_samples(self):
        # Accumulate non-root rank contributions first, then deliver their sum
        # to root. This exercises the real sharding loop without a GPU/process group.
        totals = {}
        counts = {}
        stop_calls = {}
        current_rank = None

        def reduce(tensor, dst):
            self.assertEqual(dst, 0)
            call = counts.get(current_rank, 0)
            counts[current_rank] = call + 1
            totals[call] = totals.get(call, torch.zeros_like(tensor)) + tensor.clone()
            if current_rank == 0:
                tensor.copy_(totals[call])

        def stop_requested(device, deadline):
            stop_calls[current_rank] = stop_calls.get(current_rank, 0) + 1
            return False

        seen = []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                mock.patch.object(train.dist, "reduce", side_effect=reduce), \
                mock.patch.object(train, "stop_requested", side_effect=stop_requested):
            for current_rank in (3, 2, 1, 0):
                base = ToyHead()
                result = train.extrapolate(base, *data(), config(), current_rank, 4, "cpu",
                                           12, None, 4, Path(directory) / "distributed.txt")
                seen.extend(base.model.seen)
                if current_rank != 0:
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result["n"], 3)
                    self.assertEqual(result["exact"], [0, 1, 2, 3])
        self.assertEqual(sorted(seen), [0, 1, 2])
        self.assertEqual(counts, {3: 2, 2: 2, 1: 2, 0: 2})
        self.assertEqual(stop_calls, {3: 4, 2: 4, 1: 4, 0: 4})

    def test_forward_error_restores_mode_loops_and_ema(self):
        base = ToyHead(loops=5, fail=True)
        ema = train.EMAHelper()
        ema.register(base)
        ema.shadow["model.offset"].fill_(1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "toy forward failure"):
            train.extrapolate(base, *data(), config(), 0, 1, "cpu", 4, ema, 2,
                              Path(directory) / "failure.txt")
        self.assertEqual(base.model.config.loops, 5)
        self.assertEqual(base.model.offset.item(), 0)
        self.assertTrue(base.training)

    def test_interruption_preserves_per_segment_counts_and_best_cohort(self):
        base = ToyHead()
        # One whole batch, then only segment 1 of the second global batch.
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output, \
                mock.patch.object(train, "stop_requested", side_effect=[False]*5+[True]):
            path = Path(directory) / "partial.txt"
            result = train.extrapolate(base, *data(), config(milestone_extrap_n=6),
                                       0, 1, "cpu", 4, None, 4, path)
            saved = json.loads(path.with_suffix(".json").read_text())
        self.assertTrue(result["partial"])
        self.assertEqual(result["count"], [6, 4, 4, 4])
        self.assertEqual(result["best_comparison_n"], 6)
        self.assertEqual(result["best_exact"]["segment"], 1)
        self.assertEqual(result["train"]["n"], 4)
        self.assertEqual(result["final"]["n"], 4)
        self.assertIn("[same n=6]", output.getvalue())
        self.assertNotIn("# best-train:", output.getvalue())
        self.assertEqual(saved, result)
        self.assertEqual(base.model.config.loops, 2)

    def test_expired_deadline_returns_empty_partial_report(self):
        base = ToyHead(loops=16)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            result = train.extrapolate(base, *data(), config(), 0, 1, "cpu", 4, None, 2,
                                       Path(directory) / "expired.txt", deadline=float("-inf"))
        self.assertTrue(result["partial"])
        self.assertEqual(result["count"], [0, 0])
        self.assertIsNone(result["train"])
        self.assertIsNone(result["best_exact"])
        self.assertIsNone(result["final"])
        self.assertIn("not evaluated", output.getvalue())
        self.assertEqual(base.model.config.loops, 16)


if __name__ == "__main__":
    unittest.main()
