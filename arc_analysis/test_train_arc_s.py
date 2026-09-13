"""캔버스 S 래퍼 검사: python -m unittest arc_analysis.test_train_arc_s -v

  1. S=10 합성 데이터에서 train_arc 의 기존 검사(크롭·역증강·샘플러·투표·재개 비트 일치)를 그대로 통과하는가
  2. URM 공식 builder 로 만든 실제 S=10 데이터(과제 3개·증강 4)에서 학습 2 step → 저장 → 재개 → 평가·제출 JSON 이 도는가
  3. S=30 데이터에서는 crop_grid 가 train_arc 원본과 같은가 (S=30 은 원본과 동일해야 한다)
  4. frame 부호화: 4변 테두리·PAD 무시·크롭 복원, 평행이동 예제도 처리
  5. --model d4 --encoding frame 으로 학습→중간 저장→재개 비트 일치, 공식 builder 데이터에서 끝까지
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import train_arc                        # noqa: E402
import train_arc_s                      # noqa: E402
from train_arc import ARCTrainStream, ARCVotes, dihedral, inverse_augmentation   # noqa: E402

S = 10


def encode(grid, S=S):
    grid = np.asarray(grid, dtype=np.uint8)
    rows, cols = grid.shape
    result = np.zeros((S, S), dtype=np.uint8)
    result[:rows, :cols] = grid + 2
    if rows < S:
        result[rows, :cols] = 1
    if cols < S:
        result[:rows, cols] = 1
    return result.reshape(S * S)


def make_dataset(root, S=S):
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
                        seq_len=S * S, num_puzzle_identifiers=3, total_groups=1,
                        mean_puzzle_examples=len(examples), sets=["all"])
        (directory / "dataset.json").write_text(json.dumps(metadata))
        inputs, labels = [], []
        for transform in (0, 1):
            for x, y in examples:
                inputs.append(encode(dihedral(x, transform), S))
                labels.append(encode(dihedral(y, transform), S))
        arrays = dict(inputs=inputs, labels=labels, puzzle_identifiers=[1, 2],
                      puzzle_indices=[0, len(examples), 2 * len(examples)], group_indices=[0, 2])
        for name, values in arrays.items():
            np.save(directory / f"all__{name}.npy", np.asarray(values, dtype=np.int32))


class CanvasTests(unittest.TestCase):
    def setUp(self):
        train_arc_s.install(S)

    def test_crop_and_inverse_augmentation_on_small_canvas(self):
        crop = train_arc_s.crop_grid
        grid = np.array([[0, 1, 4], [7, 2, 8]], dtype=np.uint8)
        permutation = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)
        for transform in range(8):
            mapped = dihedral(permutation[grid], transform)
            name, recovered = inverse_augmentation(f"example|||t{transform}|||0987654321", crop(encode(mapped)))
            self.assertEqual(name, "example")
            np.testing.assert_array_equal(recovered, grid)
        np.testing.assert_array_equal(crop(encode(np.zeros((S, S)))), np.zeros((S, S)))
        self.assertEqual(crop(np.zeros(S * S)).shape, (0, 0))
        self.assertEqual(crop(encode(np.zeros((3, 5)))).shape, (3, 5))
        # S=30 에서는 원본 crop_grid 와 동일
        train_arc_s.install(30)
        for shape in ((1, 1), (3, 5), (30, 30), (12, 7)):
            tokens = encode(np.random.randint(0, 10, shape), 30)
            np.testing.assert_array_equal(train_arc_s.crop_grid(tokens), ORIGINAL_CROP(tokens))

    def test_frame_encoding(self):
        train_arc_s.install(S, "base", "frame")
        x = encode(np.array([[0, 1, 4], [7, 2, 8]]))
        y = train_arc_s.frame_encode(x[None])[0].reshape(S, S)
        self.assertEqual(int((y == 1).sum()), 3 + 2 + 1)        # 아래 행 3 + 오른쪽 열 2 + 모서리 1 (좌·상은 캔버스 밖)
        np.testing.assert_array_equal(train_arc_s.crop_grid(y.reshape(-1)), np.array([[0, 1, 4], [7, 2, 8]]))
        shifted = np.zeros((S, S), dtype=np.int64); shifted[2:4, 3:6] = np.array([[0, 1, 4], [7, 2, 8]]) + 2
        shifted[4, 3:6] = 1; shifted[2:4, 6] = 1                 # URM 평행이동 부호화
        z = train_arc_s.frame_encode(shifted.reshape(1, -1))[0].reshape(S, S)
        self.assertEqual(int((z == 1).sum()), 4 * 5 - 2 * 3)   # 4×5 상자 − 2×3 격자 = 14 (모서리 포함)
        self.assertTrue((z[1, 2:7] == 1).all() and (z[4, 2:7] == 1).all() and (z[1:5, 2] == 1).all() and (z[1:5, 6] == 1).all())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); make_dataset(root, S)
            dataset = train_arc_s.ARCSplitS(root, "train")
            batch = dataset.batch(dataset.arrays["all"], [0, 1], [0, 0], size=3)
            inp = batch["inputs"][0].numpy().reshape(S, S); lab = batch["labels"][0].numpy().reshape(S, S)
            self.assertEqual(int((inp == 1).sum()), 5)             # 2×2 격자, 좌상단: 3×3 상자 − 4
            self.assertEqual(int((lab == 1).sum()), 5)
            self.assertEqual(int((lab == -100).sum()), S * S - 9)  # 격자 4 + 테두리 5 만 감독
            self.assertTrue(bool((batch["labels"][2] == -100).all()))

    def test_split_rejects_wrong_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root, S)
            dataset = train_arc_s.ARCSplitS(root, "train")
            self.assertEqual(dataset.arrays["all"]["inputs"].shape[1], S * S)
            train_arc_s.install(12)
            with self.assertRaises(ValueError):
                train_arc_s.ARCSplitS(root, "train")

    def test_sampler_and_votes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root, S)
            dataset = train_arc_s.ARCSplitS(root, "train")
            stream = ARCTrainStream(dataset, batch_size=2, seed=11, epochs_per_iter=3)
            batch = next(stream)
            self.assertEqual(batch["inputs"].shape, (2, S * S))
            self.assertTrue(bool((batch["labels"] == -100).any()))
            supervised = np.concatenate([train_arc_s.crop_grid(x).reshape(-1) for x in dataset.arrays["all"]["labels"]])
            self.assertNotIn(5, supervised)
        tasks = {"a": {"test": [{"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}]},
                 "b": {"test": [{"input": [[5]], "output": [[6]]}]}}
        votes = ARCVotes(["<blank>", "a", "b"], tasks)
        batch = {"puzzle_identifiers": torch.tensor([1, 1, 2, 0]),
                 "inputs": torch.tensor(np.stack([encode([[1]]), encode([[3]]), encode([[5]]), encode([[0]])]))}
        preds = torch.tensor(np.stack([encode([[2]]), encode([[9]]), encode([[6]]), encode([[9]])]))
        votes.update(batch, preds)
        metrics, submission = votes.result()
        self.assertEqual(metrics["ARC/pass@1"], 0.75)
        self.assertEqual(submission["a"][0]["attempt_1"], [[2]])

    def test_resume_is_bitwise_on_synthetic_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"; dataset.mkdir()
            make_dataset(dataset, S)
            common = ["--data", str(dataset), "--hidden-size", "16", "--num-heads", "2",
                      "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2",
                      "--epochs-per-iter", "3", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_s.main(common + ["--out-dir", str(root / "full"), "--max-steps", "6"])
                train_arc_s.main(common + ["--out-dir", str(root / "split"), "--max-steps", "3"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "6",
                                  "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                  "--device", "cpu", "--cpu-threads", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only",
                                  "--resume", str(root / "split/latest.pt"), "--eval-segments", "3", "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertEqual(full["model_cfg"]["seq_len"], S * S)
            self.assertEqual(full["model_cfg"]["grid"], S)
            self.assertEqual(full["carry"]["current_hidden"].shape[1], S * S)
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            report = json.loads((root / "split/eval_step_6_seg3.json").read_text())
            self.assertEqual(report["ARC/tasks"], 1)
            self.assertEqual(report["eval_segments"], 3)

    def test_d4_resume_is_bitwise_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"; dataset.mkdir()
            make_dataset(dataset, S)
            common = ["--data", str(dataset), "--hidden-size", "32", "--num-heads", "8", "--model", "d4", "--encoding", "frame",
                      "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2",
                      "--epochs-per-iter", "3", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_s.main(common + ["--out-dir", str(root / "full"), "--max-steps", "6"])
                train_arc_s.main(common + ["--out-dir", str(root / "split"), "--max-steps", "3"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "6",
                                  "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                  "--device", "cpu", "--cpu-threads", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only",
                                  "--resume", str(root / "split/latest.pt"), "--eval-segments", "3", "--device", "cpu"])
                with self.assertRaises(ValueError):
                    train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only", "--model", "base",
                                      "--resume", str(root / "split/latest.pt"), "--device", "cpu"])
            self.assertEqual(json.loads((root / "split/arc_s.json").read_text()), {"canvas": S, "model": "d4", "encoding": "frame", "stdp": True, "transport": "adj", "sheaf_cond": "none"})
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertIn("model.inner.layers.0.wc_base", full["raw_model_state_dict"])
            self.assertGreater(float(full["raw_model_state_dict"]["model.inner.puzzle_emb.weights"].abs().max()), 0)
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            self.assertEqual(json.loads((root / "split/eval_step_6_seg3.json").read_text())["ARC/tasks"], 1)

    @unittest.skipUnless(Path("/tmp/URM/data/build_arc_dataset.py").exists(), "URM checkout not present")
    def test_end_to_end_on_official_builder_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "arc_s10"
            subprocess.run([sys.executable, str(HERE / "prep_arc_canvas.py"), "--urm-repo", "/tmp/URM", "--canvas", str(S),
                            "--num-aug", "4", "--task-limit", "2", "--output-dir", str(data)],
                           check=True, capture_output=True)
            prov = json.loads((data / "provenance.json").read_text())
            self.assertEqual(prov["canvas"], S)
            self.assertEqual(json.loads((data / "train/dataset.json").read_text())["seq_len"], S * S)
            for name, task in json.loads((data / "test_puzzles.json").read_text()).items():
                for pair in task["train"] + task["test"]:
                    for g in (pair["input"], pair["output"]):
                        self.assertLessEqual(max(len(g), len(g[0])), S)
            for model, heads in (("base", "4"), ("d4", "8")):
                common = ["--data", str(data), "--out-dir", str(root / model), "--hidden-size", "32", "--num-heads", heads,
                          "--model", model, "--encoding", "frame",
                          "--blocks-per-seg", "2", "--loops", "2", "--batch-size", "4", "--epochs-per-iter", "2",
                          "--lr-warmup-steps", "0", "--eval-every", "0", "--save-every", "0", "--device", "cpu", "--cpu-threads", "2"]
                with contextlib.redirect_stdout(io.StringIO()):
                    train_arc_s.main(common + ["--max-steps", "2", "--eval-segments", "3", "--eval-max-augmentations", "2"])
                report = json.loads((root / model / "eval_step_2_seg3.json").read_text())
                submission = json.loads((root / model / "eval_step_2_seg3_submission.json").read_text())
                self.assertEqual(report["ARC/tasks"], len(submission))
                self.assertEqual(report["eval_segments"], 3)
                for attempts in submission.values():
                    for attempt in attempts:
                        self.assertLessEqual(len(attempt["attempt_1"]), S)


ORIGINAL_CROP = train_arc.crop_grid     # install() 전에 잡아 둔 원본

if __name__ == "__main__":
    unittest.main()


class ColorModelTests(unittest.TestCase):
    """색 슬롯 LT: 색 순열 등변 + 학습→재개 비트 일치 (합성 S=10)."""

    def test_color_train_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"; dataset.mkdir()
            make_dataset(dataset, S)
            common = ["--data", str(dataset), "--hidden-size", "96", "--num-heads", "8", "--model", "color", "--encoding", "frame",
                      "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2",
                      "--epochs-per-iter", "3", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_s.main(common + ["--out-dir", str(root / "full"), "--max-steps", "6"])
                train_arc_s.main(common + ["--out-dir", str(root / "split"), "--max-steps", "3"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "6",
                                  "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                  "--device", "cpu", "--cpu-threads", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only",
                                  "--resume", str(root / "split/latest.pt"), "--eval-segments", "3", "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertEqual(full["model_cfg"]["slot_dim"], 8)
            self.assertIn("model.inner.layers.0.wcb_raw", full["raw_model_state_dict"])
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            self.assertEqual(json.loads((root / "split/eval_step_6_seg3.json").read_text())["ARC/tasks"], 1)

    def test_v3_train_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"; dataset.mkdir()
            make_dataset(dataset, S)
            common = ["--data", str(dataset), "--hidden-size", "1536", "--num-heads", "8", "--puzzle-emb-dim", "140",
                      "--model", "v3", "--encoding", "frame", "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2",
                      "--epochs-per-iter", "3", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_s.main(common + ["--out-dir", str(root / "full"), "--max-steps", "4"])
                train_arc_s.main(common + ["--out-dir", str(root / "split"), "--max-steps", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "4",
                                  "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                  "--device", "cpu", "--cpu-threads", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only",
                                  "--resume", str(root / "split/latest.pt"), "--eval-segments", "2", "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertTrue(full["model_cfg"]["sym_equiv"] and full["model_cfg"]["write_addr"])
            self.assertEqual(full["model_cfg"]["addr_p"], 192)
            self.assertIn("model.inner.layers.0.wc_raw_b", full["raw_model_state_dict"])
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            self.assertEqual(json.loads((root / "split/eval_step_4_seg2.json").read_text())["ARC/tasks"], 1)

    def test_color_permutation_equivariance(self):
        sys.path.insert(0, str(HERE))
        import lt_color
        from train import PRESETS
        torch.manual_seed(0)
        Sg = 6
        cfg = dict(batch_size=1, seq_len=Sg * Sg, grid=Sg, vocab_size=12, num_puzzle_identifiers=3, puzzle_emb_ndim=192,
                   hidden_size=192, num_heads=8, slot_dim=16, loops=2, blocks_per_seg=2, amp=False, forward_dtype="float32", **PRESETS["v1.1"])
        m = lt_color.ColorLT(cfg).eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "b_down" in n:
                    p.normal_(0, 0.05)
            m.inner.puzzle_emb.weights.normal_(0, 0.05)

        def run(tok, pid):
            b = {"inputs": tok, "puzzle_identifiers": torch.tensor([pid])}
            c = m.initial_carry(b)
            with torch.no_grad():
                for _ in range(2):
                    c, o = m(c, b)
            return o["logits"][0]
        x = torch.randint(0, 12, (1, Sg * Sg)); base = run(x, 1)
        perm = torch.randperm(9); tokmap = torch.arange(12); tokmap[3:] = perm + 3
        with torch.no_grad():
            e = m.inner.puzzle_emb.weights[1].view(12, -1).clone(); e2 = e.clone(); e2[3:] = e[3:][torch.argsort(perm)]
            m.inner.puzzle_emb.weights[2] = e2.reshape(-1)
        got = run(tokmap[x], 2)
        self.assertLess(float((got[:, perm + 3] - base[:, 3:]).norm() / base[:, 3:].norm()), 1e-4)
        self.assertLess(float((got[:, :3] - base[:, :3]).norm() / base[:, :3].norm()), 1e-4)


class SheafTests(unittest.TestCase):
    """v3 + laplacian 수송 + 채널 스케일: 학습→재개 비트 일치, ID 폭 자동 확장, 사이드카 기록."""

    def test_v3_sheaf_train_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"; dataset.mkdir()
            make_dataset(dataset, S)
            common = ["--data", str(dataset), "--hidden-size", "1536", "--num-heads", "8", "--model", "v3", "--encoding", "frame",
                      "--transport", "laplacian", "--sheaf-cond", "scale",
                      "--blocks-per-seg", "1", "--loops", "2", "--batch-size", "2", "--epochs-per-iter", "3",
                      "--lr-warmup-steps", "0", "--eval-every", "0", "--save-every", "0", "--no-eval-at-end",
                      "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_s.main(common + ["--out-dir", str(root / "full"), "--max-steps", "4"])
                train_arc_s.main(common + ["--out-dir", str(root / "split"), "--max-steps", "2"])
                train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--max-steps", "4",
                                  "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                  "--device", "cpu", "--cpu-threads", "2"])
                with self.assertRaises(ValueError):
                    train_arc_s.main(["--data", str(dataset), "--out-dir", str(root / "split"), "--eval-only", "--transport", "adj",
                                      "--resume", str(root / "split/latest.pt"), "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertEqual(full["model_cfg"]["puzzle_emb_ndim"], 140 + 128)
            self.assertEqual((full["model_cfg"]["transport"], full["model_cfg"]["sheaf_cond"]), ("laplacian", "scale"))
            self.assertEqual(full["raw_model_state_dict"]["model.inner.puzzle_emb.weights"].shape[1], 268)
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            side = json.loads((root / "split/arc_s.json").read_text())
            self.assertEqual((side["transport"], side["sheaf_cond"]), ("laplacian", "scale"))
