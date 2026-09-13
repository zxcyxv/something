"""S=10 사다리: 기존 LT vs D4LT 의 증강 수 곡선. seed 0 하나. 동시 2런.

  base × 증강 {1000, 250, 60, 8}
  d4   × 증강 {1000, 125, 31, 8}      (125·31 은 base 의 1000·250 의 1/8)
둘 다 --encoding frame (부호화 통제), 20k step × 배치 128, 중간 평가 2k step 마다 증강 32개 상한.
끝나면 전체 증강 seg16 평가와 seg128 외삽(증강 128 상한)을 따로 돌린다.
결과: runs/arc_s10/<model>_aug<N>/  (stdout.log, train.jsonl, eval_step_*.json)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/arc_s10"
STEPS = int(os.environ.get("LADDER_STEPS", 20000))
CONCURRENCY = int(os.environ.get("LADDER_CONCURRENCY", 2))
GRID = [("base", 8), ("d4", 8), ("base", 60), ("d4", 31), ("base", 250), ("d4", 125), ("base", 1000), ("d4", 1000)]
COMMON = ["--encoding", "frame", "--batch-size", "128", "--seed", "0", "--eval-every", "2000", "--eval-max-augmentations", "32",
          "--save-every", "1000", "--log-every", "100", "--lr-warmup-steps", "2000"]


def data_dir(aug):
    return ROOT / f"data/arc_s10_aug{aug}"


def wait_for_data():
    needed = {data_dir(aug) for _, aug in GRID}
    while True:
        missing = [d for d in needed if not (d / "provenance.json").exists()]
        if not missing:
            return
        print(f"[ladder] waiting for {[m.name for m in missing]}", flush=True)
        time.sleep(30)


def stages(model, aug):
    out = RUNS / f"{model}_aug{aug}"
    py = [sys.executable, str(ROOT / "arc_analysis/train_arc_s.py"), "--data", str(data_dir(aug)), "--out-dir", str(out)]
    train = py + ["--model", model, "--max-steps", str(STEPS)] + COMMON
    if (out / "latest.pt").exists():
        train = py + ["--resume", str(out / "latest.pt"), "--max-steps", str(STEPS)] + COMMON[4:]
    resume = ["--resume", str(out / "latest.pt"), "--batch-size", "128"]
    final16 = py + resume + ["--eval-only", "--eval-segments", "16"]
    final128 = py + resume + ["--eval-only", "--eval-segments", "128", "--eval-max-augmentations", "128"]
    return out, [("train", train), ("eval16", final16), ("eval128", final128)]


def main():
    wait_for_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    queue = list(GRID)
    active = {}
    started = time.monotonic()
    while queue or active:
        while queue and len(active) < CONCURRENCY:
            model, aug = queue.pop(0)
            out, plan = stages(model, aug)
            if (out / f"eval_step_{STEPS}_seg128.json").exists():
                print(f"[ladder] skip {out.name}: done", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "stdout.log", "a")
            active[out.name] = dict(plan=plan, log=log, proc=None, out=out)
            print(f"[ladder] start {out.name}", flush=True)
        for name, st in list(active.items()):
            if st["proc"] is None or st["proc"].poll() is not None:
                if st["proc"] is not None:
                    stage_name = st["stage"]
                    rc = st["proc"].returncode
                    print(f"[ladder] {name} {stage_name} exit {rc} at {(time.monotonic() - started) / 3600:.2f} h", flush=True)
                    if rc != 0:
                        st["log"].close(); del active[name]; continue
                if not st["plan"]:
                    st["log"].close(); del active[name]
                    print(f"[ladder] done {name}", flush=True)
                    continue
                stage_name, cmd = st["plan"].pop(0)
                st["stage"] = stage_name
                st["log"].write(f"\n===== {stage_name}: {' '.join(cmd)}\n"); st["log"].flush()
                st["proc"] = subprocess.Popen(cmd, stdout=st["log"], stderr=subprocess.STDOUT, cwd=ROOT)
        time.sleep(20)
    summary = {}
    for model, aug in GRID:
        out = RUNS / f"{model}_aug{aug}"
        for seg in (16, 128):
            p = out / f"eval_step_{STEPS}_seg{seg}.json"
            if p.exists():
                summary[f"{model}_aug{aug}_seg{seg}"] = json.loads(p.read_text())
    (RUNS / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[ladder] all done in {(time.monotonic() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
