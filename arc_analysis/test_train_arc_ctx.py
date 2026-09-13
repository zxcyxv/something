"""In-context 프로토콜 검사: python -m unittest arc_analysis.test_train_arc_ctx -v

  1. D4CtxInner: 슬롯마다 내용을 같은 g 로 회전하면 로짓이 슬롯별로 따라온다 (fp32 오차)
  2. 데이터: 질의 출력이 입력에 새지 않음, 빈 슬롯 PAD, 시범 마스킹, inductive 가 evaluation 과제를 제외
  3. 공식 builder 데이터(S=10)에서 base·d4 학습 3 step → 저장 → 재개 → 6 step 이 연속 6 step 과 비트 일치, 평가·제출 JSON
"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import d4                                   # noqa: E402
import ctx_data                             # noqa: E402
import train_arc_s                          # noqa: E402
import train_arc_ctx                        # noqa: E402
from lt_ctx import D4CtxInner, CtxConfig    # noqa: E402
from train import LTCarry, PRESETS          # noqa: E402


def build_smoke(root):
    subprocess.run([sys.executable, str(HERE / "prep_arc_canvas.py"), "--urm-repo", "/tmp/URM", "--canvas", "10",
                    "--num-aug", "2", "--task-limit", "2", "--output-dir", str(root)], check=True, capture_output=True)


class CtxTests(unittest.TestCase):
    def test_d4_slotwise_equivariance(self):
        torch.manual_seed(0)
        S, K = 6, 2; n = 2 * (K + 1)
        cfg = CtxConfig.from_dict(dict(batch_size=1, seq_len=n * S * S, grid=S, vocab_size=12, num_puzzle_identifiers=3,
                                       puzzle_emb_ndim=64, hidden_size=64, num_heads=8, loops=2, blocks_per_seg=2, amp=False,
                                       forward_dtype="float32", n_slots=n, **PRESETS["v1.1"]))
        m = D4CtxInner(cfg).eval()
        with torch.no_grad():
            for name, p in m.named_parameters():
                if "down" in name or "psi_slot" in name:
                    p.normal_(0, 0.1)
            m.puzzle_emb.weights.normal_(0, 0.05)
        x = torch.randint(0, 12, (1, n * S * S))

        def run(tokens, pid):
            carry = LTCarry(current_hidden=m.init_hidden.expand(1, n * S * S, -1).clone())
            with torch.no_grad():
                for _ in range(2):
                    carry, logits = m(carry, {"inputs": tokens, "puzzle_identifiers": torch.tensor([pid])})
            return logits[0]

        base = run(x, 1)
        for g in range(1, 8):
            with torch.no_grad():                    # 회전한 과제 = ID 행의 슬라이스 순열
                m.puzzle_emb.weights[2] = d4.act_slices(m.puzzle_emb.weights[1], g, m.dh)
            xs = x.view(1, n, S * S)
            xg = torch.stack([d4.act_field(xs[:, s], S, g) for s in range(n)], 1).reshape(1, -1)
            got = run(xg, 2).view(n, S * S, -1)
            exp = torch.stack([d4.act_field(base.view(n, S * S, -1)[s].unsqueeze(0), S, g)[0] for s in range(n)])
            self.assertLess(float((got - exp).norm() / exp.norm()), 1e-4, f"g={g}")

    def test_data_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arc"; build_smoke(root)
            train_arc_s.install(10, "base", "frame")
            tr = ctx_data.CtxSplit(root, "train", 2, inductive=True)
            tr_all = ctx_data.CtxSplit(root, "train", 2, inductive=False)
            evals = set(json.loads((root / "test_puzzles.json").read_text()))
            self.assertEqual(len(tr_all.allowed_groups["all"]) - len(tr.allowed_groups["all"]), len(evals))
            st = ctx_data.CtxStream(tr, 3, 0, 4, mask_prob=1.0)
            b = next(st)
            qi, qo = tr.query_slots()
            self.assertTrue(bool((b["inputs"][:, qo] == 0).all()))                 # 질의 출력 슬롯 입력은 PAD
            self.assertTrue(bool((b["labels"][:, qo] != -100).any(-1).all()))       # 질의 출력은 감독
            self.assertTrue(bool((b["labels"][:, qi] == -100).all()))
            for i in range(3):                                                    # 마스킹: 시범 출력 슬롯 하나가 PAD + 감독
                masked = [k for k in range(2) if (b["labels"][i, (2 * k + 1) * 100:(2 * k + 2) * 100] != -100).any()]
                for k in masked:
                    self.assertTrue(bool((b["inputs"][i, (2 * k + 1) * 100:(2 * k + 2) * 100] == 0).all()))
                    self.assertTrue(bool((b["inputs"][i, (2 * k) * 100:(2 * k + 1) * 100] > 0).any()))
            te = ctx_data.CtxSplit(root, "test", 2, inductive=True)
            ids = json.loads((root / "identifiers.json").read_text()); tasks = json.loads((root / "test_puzzles.json").read_text())
            n = 0
            for eb in ctx_data.eval_batches(te, ids, tasks, 4, 0):
                n += int((eb["puzzle_identifiers"] > 0).sum())
                x = eb["inputs"][0]
                self.assertTrue(bool((x[:100] > 0).any()) and bool((x[qi] > 0).any()))   # 시범 1 과 질의 입력 존재
                self.assertTrue(bool((x[qo] == 0).all()))
            self.assertEqual(n, sum(len(t["test"]) for t in tasks.values()) * 3)          # 원본 + 증강 2

    @unittest.skipUnless(Path("/tmp/URM/data/build_arc_dataset.py").exists(), "URM checkout not present")
    def test_no_puzzle_id_train_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "arc"; build_smoke(data)
            common = ["--data", str(data), "--model", "d4", "--k-demos", "2", "--no-puzzle-id",
                      "--hidden-size", "32", "--num-heads", "8", "--blocks-per-seg", "1", "--loops", "2",
                      "--batch-size", "2", "--epochs-per-iter", "4", "--lr-warmup-steps", "0", "--eval-every", "0",
                      "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                train_arc_ctx.main(common + ["--out-dir", str(root / "full"), "--max-steps", "6"])
                train_arc_ctx.main(common + ["--out-dir", str(root / "split"), "--max-steps", "3"])
                train_arc_ctx.main(["--data", str(data), "--out-dir", str(root / "split"), "--max-steps", "6",
                                    "--resume", str(root / "split/latest.pt"), "--eval-every", "0", "--no-eval-at-end",
                                    "--device", "cpu", "--cpu-threads", "2"])
                train_arc_ctx.main(["--data", str(data), "--out-dir", str(root / "split"), "--eval-only",
                                    "--resume", str(root / "split/latest.pt"), "--eval-segments", "2", "--device", "cpu"])
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "split/latest.pt", weights_only=False)
            self.assertFalse(full["model_cfg"]["use_puzzle_id"])
            self.assertFalse(any("puzzle_emb" in k for k in full["raw_model_state_dict"]))
            self.assertEqual(len(full["optimizer_states"]), 1)
            sigma = full["raw_model_state_dict"]["model.inner.psi_slot"]
            self.assertGreater(float(sigma.std()), 1.0)                     # [−π, π] 균등 → std ≈ 1.81
            for key, value in full["raw_model_state_dict"].items():
                torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
            for key in ("current_hidden", "coupling", "steps", "halted"):
                torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
            self.assertFalse(json.loads((root / "split/eval_step_6_seg2.json").read_text())["puzzle_id"])

    @unittest.skipUnless(Path("/tmp/URM/data/build_arc_dataset.py").exists(), "URM checkout not present")
    def test_train_resume_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "arc"; build_smoke(data)
            for model, heads in (("base", "4"), ("d4", "8")):
                common = ["--data", str(data), "--model", model, "--k-demos", "2", "--mask-prob", "0.5",
                          "--hidden-size", "32", "--num-heads", heads, "--blocks-per-seg", "1", "--loops", "2",
                          "--batch-size", "2", "--epochs-per-iter", "4", "--lr-warmup-steps", "0", "--eval-every", "0",
                          "--save-every", "0", "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"]
                with contextlib.redirect_stdout(io.StringIO()):
                    train_arc_ctx.main(common + ["--out-dir", str(root / f"{model}_full"), "--max-steps", "6"])
                    train_arc_ctx.main(common + ["--out-dir", str(root / f"{model}_split"), "--max-steps", "3"])
                    train_arc_ctx.main(["--data", str(data), "--out-dir", str(root / f"{model}_split"), "--max-steps", "6",
                                        "--resume", str(root / f"{model}_split/latest.pt"), "--eval-every", "0",
                                        "--no-eval-at-end", "--device", "cpu", "--cpu-threads", "2"])
                    train_arc_ctx.main(["--data", str(data), "--out-dir", str(root / f"{model}_split"), "--eval-only",
                                        "--resume", str(root / f"{model}_split/latest.pt"), "--eval-segments", "3", "--device", "cpu"])
                    with self.assertRaises(ValueError):
                        train_arc_ctx.main(["--data", str(data), "--out-dir", str(root / f"{model}_split"), "--eval-only", "--k-demos", "3",
                                            "--resume", str(root / f"{model}_split/latest.pt"), "--device", "cpu"])
                full = torch.load(root / f"{model}_full/latest.pt", weights_only=False)
                resumed = torch.load(root / f"{model}_split/latest.pt", weights_only=False)
                self.assertEqual(full["model_cfg"]["seq_len"], 600); self.assertEqual(full["model_cfg"]["n_slots"], 6)
                self.assertGreater(float(full["raw_model_state_dict"]["model.inner.puzzle_emb.weights"].abs().max()), 0)
                for key, value in full["raw_model_state_dict"].items():
                    torch.testing.assert_close(value, resumed["raw_model_state_dict"][key], rtol=0, atol=0)
                for key in ("current_hidden", "coupling", "steps", "halted"):
                    torch.testing.assert_close(full["carry"][key], resumed["carry"][key], rtol=0, atol=0)
                report = json.loads((root / f"{model}_split/eval_step_6_seg3.json").read_text())
                self.assertEqual(report["protocol"], "in-context"); self.assertFalse(report["inductive"])
                self.assertEqual(report["eval_segments"], 3); self.assertIn("exact_hit_count", report)
                sub = json.loads((root / f"{model}_split/eval_step_6_seg3_submission.json").read_text())
                self.assertEqual(len(sub), report["ARC/tasks"])


if __name__ == "__main__":
    unittest.main()
