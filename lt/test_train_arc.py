"""Bounded ARC interface/optimizer/resume checks: python -m unittest lt.test_train_arc -v."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .train_arc import (ARCSplit, ARCTrainStream, ARCVotes, crop_grid, dihedral,
                        inverse_augmentation, main)


def encode(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    rows, cols = grid.shape
    result = np.zeros((30, 30), dtype=np.uint8)
    result[:rows, :cols] = grid + 2
    if rows < 30:
        result[rows, :cols] = 1
    if cols < 30:
        result[:rows, cols] = 1
    return result.reshape(900)


def make_dataset(root):
    """One held-out task's demonstrations share its ID; query answer is test-only."""
    demos = [(np.array([[0, 1], [2, 0]]), np.array([[1, 0], [0, 2]])),
             (np.array([[0, 3], [4, 0]]), np.array([[3, 0], [0, 4]]))]
    query = (np.array([[0, 5], [6, 0]]), np.array([[5, 0], [0, 6]]))
    names = ["<blank>", "task", "task|||t1|||0123456789"]
    (root / "identifiers.json").write_text(json.dumps(names))
    task = {"train": [{"input": x.tolist(), "output": y.tolist()} for x, y in demos],
            "test": [{"input": query[0].tolist(), "output": query[1].tolist()}]}
    (root / "test_puzzles.json").write_text(json.dumps({"task": task}))
    for split, examples in (("train", demos), ("test", [query])):
        directory = root / split
        directory.mkdir()
        metadata = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=12,
                        seq_len=900, num_puzzle_identifiers=3, total_groups=1,
                        mean_puzzle_examples=len(examples), sets=["all"])
        (directory / "dataset.json").write_text(json.dumps(metadata))
        inputs, labels = [], []
        for transform in (0, 1):
            for x, y in examples:
                inputs.append(encode(dihedral(x, transform)))
                labels.append(encode(dihedral(y, transform)))
        arrays = dict(inputs=inputs, labels=labels, puzzle_identifiers=[1, 2],
                      puzzle_indices=[0, len(examples), 2 * len(examples)], group_indices=[0, 2])
        for name, values in arrays.items():
            np.save(directory / f"all__{name}.npy", np.asarray(values, dtype=np.int32))


class ARCInterfaceTests(unittest.TestCase):
    def test_crop_size_and_inverse_augmentation(self):
        grid = np.array([[0, 1, 4], [7, 2, 8]], dtype=np.uint8)
        permutation = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)
        for transform in range(8):
            mapped = dihedral(permutation[grid], transform)
            name, recovered = inverse_augmentation(f"example|||t{transform}|||0987654321", crop_grid(encode(mapped)))
            self.assertEqual(name, "example")
            np.testing.assert_array_equal(recovered, grid)
        np.testing.assert_array_equal(crop_grid(encode(np.zeros((30, 30)))), np.zeros((30, 30)))
        self.assertEqual(crop_grid(np.zeros(900)).shape, (0, 0))
        # The evaluator recovers a rectangle from predictions, never target size.
        self.assertEqual(crop_grid(encode(np.zeros((3, 5)))).shape, (3, 5))

    def test_sampler_padding_ignore_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            dataset = ARCSplit(root, "train")
            stream = ARCTrainStream(dataset, batch_size=2, seed=11, epochs_per_iter=3)
            batch = next(stream)
            self.assertTrue(bool((batch["puzzle_identifiers"] > 0).all()))
            self.assertTrue(bool((batch["labels"] == -100).any()))
            self.assertFalse(bool((batch["labels"] == 0).any()))
            state = stream.state_dict()
            restored = ARCTrainStream(dataset, batch_size=2, seed=11, epochs_per_iter=3)
            restored.load_state_dict(state)
            for _ in range(5):
                left, right = next(stream), next(restored)
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            padded = dataset.batch(dataset.arrays["all"], [0], [0], size=2)
            self.assertEqual(int(padded["puzzle_identifiers"][1]), 0)
            self.assertTrue(bool((padded["labels"][1] == -100).all()))
            # Held-out query color never appears as a supervised train output.
            supervised = np.concatenate([crop_grid(x).reshape(-1) for x in dataset.arrays["all"]["labels"]])
            self.assertNotIn(5, supervised)
            self.assertNotIn(6, supervised)

    def test_voting_is_macro_average_over_tasks(self):
        tasks = {
            "a": {"test": [{"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}]},
            "b": {"test": [{"input": [[5]], "output": [[6]]}]},
        }
        votes = ARCVotes(["<blank>", "a", "b"], tasks)
        batch = {"puzzle_identifiers": torch.tensor([1, 1, 2, 0]),
                 "inputs": torch.tensor(np.stack([encode([[1]]), encode([[3]]), encode([[5]]), encode([[0]])]))}
        preds = torch.tensor(np.stack([encode([[2]]), encode([[9]]), encode([[6]]), encode([[9]])]))
        votes.update(batch, preds)
        metrics, submission = votes.result()
        self.assertEqual(metrics["ARC/pass@1"], 0.75)
        self.assertEqual(metrics["ARC/queries"], 3)
        self.assertEqual(submission["a"][0]["attempt_1"], [[2]])
        # A second, less frequent candidate affects pass@2 but not pass@1.
        votes.update({key: value[:2] for key, value in batch.items()}, preds[:2])
        alternate = torch.tensor(np.stack([encode([[2]]), encode([[4]])]))
        votes.update({key: value[:2] for key, value in batch.items()}, alternate)
        metrics, _ = votes.result()
        self.assertEqual(metrics["ARC/pass@1"], 0.75)
        self.assertEqual(metrics["ARC/pass@2"], 1.0)

    def test_training_resumes_inside_segment_schedule_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"
            dataset.mkdir()
            make_dataset(dataset)
            common = ["--data", str(dataset), "--hidden-size", "16", "--num-heads", "2",
                      "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2",
                      "--epochs-per-iter", "3", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(common + ["--out-dir", str(root / "full"), "--max-steps", "6"])
                main(common + ["--out-dir", str(root / "split"), "--max-steps", "3"])
                partial = torch.load(root / "split/latest.pt", weights_only=False)
                self.assertEqual(partial["carry"]["steps"].tolist(), [1, 1])
                self.assertFalse(bool(partial["carry"]["halted"].any()))
                self.assertGreater(float(partial["raw_model_state_dict"]["model.inner.puzzle_emb.weights"].abs().max()), 0)
                main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "6",
                      "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                      "--device", "cpu", "--cpu-threads", "2"])
                main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only",
                      "--resume", str(root / "split/latest.pt"), "--eval-segments", "3", "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertEqual(resumed["step"], 6)
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            report = json.loads((root / "split/eval_step_6_seg3.json").read_text())
            self.assertEqual(report["ARC/tasks"], 1)
            self.assertEqual(report["evaluated_augmented_examples"], 2)
            self.assertEqual(report["eval_segments"], 3)


if __name__ == "__main__":
    unittest.main()
