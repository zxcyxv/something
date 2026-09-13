"""Run the official URM ARC-AGI-1 builder and record its provenance.

No data conversion is reimplemented here. Install numpy, pydantic and argdantic
in the Python environment used by this command; a URM checkout is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


URM_URL = "https://github.com/UbiquantAI/URM"
URM_COMMIT = "c14e55f5f9227873617015cf60a239126b55adcd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urm-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-file-prefix", type=Path,
                        help="Defaults to URM/kaggle/combined/arc-agi")
    parser.add_argument("--subsets", nargs="+", default=["training", "evaluation", "concept"])
    parser.add_argument("--test-set-name", default="evaluation")
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-limit", type=int, default=0,
                        help="Smoke runs only: keep first N tasks per subset; 0 keeps all.")
    args = parser.parse_args()
    repo, output = args.urm_repo.resolve(), args.output_dir.resolve()
    if not (repo / "data/build_arc_dataset.py").is_file():
        parser.error("--urm-repo must contain data/build_arc_dataset.py")
    if args.num_aug < 0 or args.task_limit < 0:
        parser.error("--num-aug and --task-limit must be nonnegative")
    if args.test_set_name not in args.subsets:
        parser.error("--test-set-name must occur in --subsets")
    if len(set(args.subsets)) != len(args.subsets):
        parser.error("--subsets must not contain duplicates")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be absent or empty (avoid mixing dataset versions)")
    prefix = (args.input_file_prefix or repo / "kaggle/combined/arc-agi").resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != URM_COMMIT:
        print(f"[ARC prep] checkout {commit}; validated reference is {URM_COMMIT}", flush=True)
    sources, all_task_ids = {}, set()
    # Prepare bounded fixtures if requested, and reject the upstream builder's
    # dummy-label fallback: those would make evaluation silently meaningless.
    with tempfile.TemporaryDirectory(prefix="lt-arc-input-") as temporary:
        selected_prefix = Path(temporary) / "arc-agi"
        for subset in args.subsets:
            challenge_file = Path(f"{prefix}_{subset}-challenges.json")
            solution_file = Path(f"{prefix}_{subset}-solutions.json")
            puzzles = json.loads(challenge_file.read_text())
            solutions = json.loads(solution_file.read_text()) if solution_file.exists() else {}
            for path in (challenge_file, solution_file):
                if path.exists():
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            names = list(puzzles)
            if args.task_limit:
                names = sorted(names)[:args.task_limit]
            selected, labels = {}, {}
            for name in names:
                if name in all_task_ids:
                    parser.error(f"task ID {name} occurs in multiple subsets")
                all_task_ids.add(name)
                task = puzzles[name]
                if not task.get("train") or not task.get("test"):
                    parser.error(f"task {name} must have both train and test examples")
                for pair in task["train"]:
                    if "output" not in pair:
                        parser.error(f"task {name} has a demonstration without output")
                answer = solutions.get(name, [pair.get("output") for pair in task["test"]])
                if len(answer) != len(task["test"]) or any(grid is None for grid in answer):
                    parser.error(f"missing test solutions for {name}; dummy labels are not permitted")
                selected[name], labels[name] = task, answer
            if not selected:
                parser.error(f"subset {subset} is empty")
            Path(f"{selected_prefix}_{subset}-challenges.json").write_text(json.dumps(selected))
            Path(f"{selected_prefix}_{subset}-solutions.json").write_text(json.dumps(labels))
        command = [sys.executable, "-m", "data.build_arc_dataset",
                   "--input-file-prefix", str(selected_prefix), "--output-dir", str(output),
                   "--subsets", *args.subsets, "--test-set-name", args.test_set_name,
                   "--num-aug", str(args.num_aug), "--seed", str(args.seed)]
        # Fix Python's hash iteration order because the upstream builder uses sets.
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed))
        subprocess.run(command, cwd=repo, env=env, check=True)
    provenance = {
        "repository": URM_URL, "commit": commit, "validated_commit": URM_COMMIT,
        "builder_sha256": hashlib.sha256((repo / "data/build_arc_dataset.py").read_bytes()).hexdigest(),
        "source_sha256": sources, "subsets": args.subsets, "test_set_name": args.test_set_name,
        "num_aug": args.num_aug, "seed": args.seed, "task_limit": args.task_limit,
        "python_hash_seed": args.seed,
        "protocol": "evaluation demonstrations are training data; evaluation test outputs are held out",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[ARC prep] {len(all_task_ids)} tasks -> {output}", flush=True)


if __name__ == "__main__":
    main()
