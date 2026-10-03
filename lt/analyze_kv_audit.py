"""Matched training-window comparisons for the implementation-audit controls."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "baseline_ng0": "Original STDP",
    "kv_historical_key_trace_ng0": "Historical rotated K trace",
    "kv_read_gain_quarter_ng0": "Original STDP, read gain 0.25",
}


def read_rows(path):
    rows = []
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently written final line may be incomplete.
            if row.get("_count_raw", 0) > 0:
                rows.append(row)
    return rows


def window(rows, step):
    selected = [row for row in rows if step-256 < row["step"] <= step]
    return dict(step=step, samples=len(selected),
                loss=sum(row["lm_loss"] for row in selected)/len(selected),
                accuracy=sum(row["accuracy"] for row in selected)/len(selected),
                exact=sum(row["exact_accuracy"] for row in selected)/len(selected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/kv_collapse_20261003")
    args = ap.parse_args()
    root = Path(args.root)
    result = dict(window="Terminal training segments only, steps (endpoint-256, endpoint]. No EMA/test metrics.", runs={})
    csv_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    available = {run: read_rows(root/run/"train.jsonl") for run in RUNS if (root/run/"train.jsonl").exists()}
    for run, rows in available.items():
        if not rows:
            continue
        end = rows[-1]["step"]
        points = sorted(set([x for x in (512, 1008, 1504, 2000, 2496, 2992, 4000, 5008, 6000) if x <= end]+[end]))
        values = [window(rows, step) for step in points]
        rolling = [window(rows, r["step"]) for r in rows if r["step"] >= 256]
        best = max(rolling, key=lambda r: r["accuracy"]) if rolling else values[-1]
        result["runs"][run] = dict(last_terminal_step=end, checkpoints=values, best_window=best,
                                    latest_window=values[-1], latest_minus_best_accuracy=values[-1]["accuracy"]-best["accuracy"])
        for row in values:
            csv_rows.append(dict(run=run, **row))
        x = [r["step"] for r in rolling]
        axes[0].plot(x, [r["loss"] for r in rolling], label=RUNS[run])
        axes[1].plot(x, [100*r["accuracy"] for r in rolling], label=RUNS[run])
        baseline = window(available["baseline_ng0"], end)
        result["runs"][run]["baseline_same_endpoint"] = baseline
        print(run, end, "loss", round(values[-1]["loss"], 6), "accuracy", round(100*values[-1]["accuracy"], 3),
              "baseline_accuracy", round(100*baseline["accuracy"], 3), flush=True)
    for ax, y in zip(axes, ("Training loss (256-step window)", "Training cell accuracy (%)")):
        ax.set_xlabel("Optimizer step"); ax.set_ylabel(y); ax.grid(alpha=.25); ax.legend(fontsize=8)
    out = root/"audit"
    out.mkdir(exist_ok=True)
    (out/"training_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out/"training_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("run", "step", "samples", "loss", "accuracy", "exact"))
        writer.writeheader(); writer.writerows(csv_rows)
    fig.savefig(out/"training_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
