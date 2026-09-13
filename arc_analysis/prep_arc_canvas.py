"""캔버스 S×S 로 축소한 ARC 데이터를 URM 공식 builder 로 만든다.

`data/prep_arc_dataset.py` 와 같은 절차(과제 검증 → 임시 JSON → builder subprocess → provenance)에
  · 과제 필터: 모든 입출력 격자의 최대 변 ≤ S 인 과제만
  · builder 의 모듈 상수 ARCMaxGridSize 를 S 로 바꿔 실행 (파일은 수정하지 않음; seq_len = S² 가 metadata 에 기록된다)
을 더한 것이다. 부호화·증강·ID 규칙은 원본 그대로다.

사용:
  python arc_analysis/prep_arc_canvas.py --urm-repo /tmp/URM --canvas 10 --output-dir data/arc_s10_aug250 --num-aug 250
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

RUNNER = """
import sys
import data.build_arc_dataset as b
b.ARCMaxGridSize = int(sys.argv[1])
sys.argv = [sys.argv[0]] + sys.argv[2:]
b.cli()
"""


def task_max_side(task, answer):
    grids = [g for pair in task["train"] for g in (pair["input"], pair["output"])]
    grids += [pair["input"] for pair in task["test"]] + list(answer)
    return max(max(len(g), len(g[0])) for g in grids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urm-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canvas", type=int, required=True, help="캔버스 변 S (원본 30). 최대 변 ≤ S 인 과제만 남긴다")
    parser.add_argument("--input-file-prefix", type=Path, help="기본 URM/kaggle/combined/arc-agi")
    parser.add_argument("--subsets", nargs="+", default=["training", "evaluation", "concept"])
    parser.add_argument("--test-set-name", default="evaluation")
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-limit", type=int, default=0, help="실행 확인용: subset 당 앞 N 과제(이름순)만")
    args = parser.parse_args()
    repo, output = args.urm_repo.resolve(), args.output_dir.resolve()
    if not (repo / "data/build_arc_dataset.py").is_file():
        parser.error("--urm-repo must contain data/build_arc_dataset.py")
    if not 1 <= args.canvas <= 30:
        parser.error("--canvas must be in 1..30")
    if args.num_aug < 0 or args.task_limit < 0:
        parser.error("--num-aug and --task-limit must be nonnegative")
    if args.test_set_name not in args.subsets or len(set(args.subsets)) != len(args.subsets):
        parser.error("--test-set-name must occur once in --subsets, which must not repeat")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be absent or empty")
    prefix = (args.input_file_prefix or repo / "kaggle/combined/arc-agi").resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != URM_COMMIT:
        print(f"[ARC prep] checkout {commit}; validated reference is {URM_COMMIT}", flush=True)
    sources, all_task_ids, kept = {}, set(), {}
    with tempfile.TemporaryDirectory(prefix="lt-arc-canvas-") as temporary:
        selected_prefix = Path(temporary) / "arc-agi"
        for subset in args.subsets:
            challenge_file = Path(f"{prefix}_{subset}-challenges.json")
            solution_file = Path(f"{prefix}_{subset}-solutions.json")
            puzzles = json.loads(challenge_file.read_text())
            solutions = json.loads(solution_file.read_text()) if solution_file.exists() else {}
            for path in (challenge_file, solution_file):
                if path.exists():
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            selected, labels, dropped = {}, {}, 0
            for name in list(puzzles):                     # 원본 JSON 순서 보존
                if name in all_task_ids:
                    parser.error(f"task ID {name} occurs in multiple subsets")
                all_task_ids.add(name)
                task = puzzles[name]
                if not task.get("train") or not task.get("test") or any("output" not in p for p in task["train"]):
                    parser.error(f"task {name} lacks demonstrations or test queries")
                answer = solutions.get(name, [pair.get("output") for pair in task["test"]])
                if len(answer) != len(task["test"]) or any(grid is None for grid in answer):
                    parser.error(f"missing test solutions for {name}; dummy labels are not permitted")
                if task_max_side(task, answer) > args.canvas:
                    dropped += 1
                    continue
                selected[name], labels[name] = task, answer
            fit = len(selected)
            if args.task_limit:
                names = sorted(selected)[:args.task_limit]
                selected = {n: selected[n] for n in names}; labels = {n: labels[n] for n in names}
            if not selected:
                parser.error(f"subset {subset} is empty after the canvas filter")
            kept[subset] = {"fit_canvas": fit, "kept": len(selected), "dropped_by_canvas": dropped, "total": len(puzzles)}
            print(f"[ARC prep] {subset}: {fit}/{len(puzzles)} tasks fit canvas {args.canvas}, {len(selected)} kept", flush=True)
            Path(f"{selected_prefix}_{subset}-challenges.json").write_text(json.dumps(selected))
            Path(f"{selected_prefix}_{subset}-solutions.json").write_text(json.dumps(labels))
        command = [sys.executable, "-c", RUNNER, str(args.canvas),
                   "--input-file-prefix", str(selected_prefix), "--output-dir", str(output),
                   "--subsets", *args.subsets, "--test-set-name", args.test_set_name,
                   "--num-aug", str(args.num_aug), "--seed", str(args.seed)]
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONPATH=str(repo))
        subprocess.run(command, cwd=repo, env=env, check=True)
    meta = json.loads((output / "train" / "dataset.json").read_text())
    if meta["seq_len"] != args.canvas ** 2:
        raise RuntimeError(f"builder wrote seq_len={meta['seq_len']}, expected {args.canvas ** 2}")
    provenance = {
        "repository": URM_URL, "commit": commit, "validated_commit": URM_COMMIT,
        "builder_sha256": hashlib.sha256((repo / "data/build_arc_dataset.py").read_bytes()).hexdigest(),
        "builder_patch": {"ARCMaxGridSize": args.canvas}, "canvas": args.canvas,
        "source_sha256": sources, "subsets": args.subsets, "test_set_name": args.test_set_name,
        "num_aug": args.num_aug, "seed": args.seed, "task_limit": args.task_limit, "python_hash_seed": args.seed,
        "task_filter": f"all grids of a task have max(height, width) <= {args.canvas}", "tasks": kept,
        "protocol": "evaluation demonstrations are training data; evaluation test outputs are held out",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[ARC prep] canvas {args.canvas}: {sum(k['kept'] for k in kept.values())} tasks -> {output}", flush=True)


if __name__ == "__main__":
    main()
