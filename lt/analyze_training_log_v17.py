"""Read the recovered v1.7 log; quantify evaluation dips without running a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def moments(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(n=int(len(a)), mean=float(a.mean()), sd=float(a.std()),
                minimum=float(a.min()), maximum=float(a.max()))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path(
        "runs/v17_recovery/source/2026-09-09/train_v17.log"))
    parser.add_argument("--out", type=Path, default=Path("runs/v17_recovery/analysis"))
    args = parser.parse_args()
    raw = args.log.read_text()
    train, evaluation, starts = [], [], []
    attempt = -1
    train_re = re.compile(
        r"\[LT\] step (\d+)  lm_loss ([0-9.eE+-]+)  acc ([0-9.]+)  "
        r"exact ([0-9.]+) \(~(\d+)/(\d+)\) \[halt step (\d+)\]")
    eval_re = re.compile(r"\[EVAL\] step (\d+)  acc ([0-9.]+)  exact (\d+)/(\d+)")
    for line_no, line in enumerate(raw.splitlines(), 1):
        if line.startswith("[LT] num_processes="):
            attempt += 1
        if "시작 step=" in line or "체크포인트 재개:" in line or "EMA shadow 복원" in line:
            starts.append(dict(line=line_no, attempt=attempt, text=line))
        if match := train_re.fullmatch(line):
            step, loss, acc, exact, count, denom, halt = match.groups()
            train.append(dict(line=line_no, attempt=attempt, step=int(step), loss=float(loss),
                              acc=float(acc), exact=float(exact), exact_rounded_count=int(count),
                              batch_size=int(denom), halt_step=int(halt),
                              step_mod_16=int(step) % 16))
        elif match := eval_re.fullmatch(line):
            step, acc, count, denom = match.groups()
            evaluation.append(dict(line=line_no, attempt=attempt, step=int(step), acc=float(acc),
                                   exact_count=int(count), n=int(denom), exact=int(count)/int(denom)))

    # Minimum parsing audit: no numerical training/evaluation rows silently disappear.
    assert len(train) == sum(line.startswith("[LT] step ") for line in raw.splitlines())
    assert len(evaluation) == sum(line.startswith("[EVAL]") for line in raw.splitlines())
    assert len(evaluation) == 194 and len(starts) == 4
    assert all(0 <= r["exact_count"] <= r["n"] and 0 <= r["acc"] <= 1 for r in evaluation)
    assert all(np.isfinite([r["loss"], r["acc"], r["exact"]]).all() for r in train)

    # Preserve every raw row in CSV. In step-indexed summaries keep the later attempt
    # where the restart reran an optimizer step (106250..107000).
    latest = {r["step"]: r for r in train}
    unique_train = sorted(latest.values(), key=lambda r: r["step"])
    selected = [r for r in evaluation if r["step"] >= 105000]
    count_stats = moments([r["exact_count"] for r in selected])
    threshold = count_stats["mean"] - 1.5 * count_stats["sd"]
    low_indices = [i for i, r in enumerate(selected) if r["exact_count"] < threshold]
    groups = []
    for i in low_indices:
        if not groups or i != groups[-1][-1] + 1:
            groups.append([i])
        else:
            groups[-1].append(i)

    phase_means = {}
    for phase in sorted({r["step_mod_16"] for r in unique_train}):
        phase_means[phase] = moments([r["loss"] for r in unique_train
                                     if r["step"] >= 105000 and r["step_mod_16"] == phase])
    for r in unique_train:
        r["phase_centered_loss"] = r["loss"] - phase_means[r["step_mod_16"]]["mean"]

    def train_window(left, right):
        rows = [r for r in unique_train if left <= r["step"] < right]
        return dict(left_inclusive=left, right_exclusive=right,
                    loss=moments([r["loss"] for r in rows]),
                    phase_centered_loss=moments([r["phase_centered_loss"] for r in rows]),
                    halted_acc=moments([r["acc"] for r in rows]),
                    halted_exact=moments([r["exact"] for r in rows]))

    episodes = []
    for group in groups:
        first, last = group[0], group[-1]
        trough = min((selected[i] for i in group), key=lambda r: r["exact_count"])
        preceding = selected[first-1] if first else None
        following = selected[last+1] if last+1 < len(selected) else None
        recovery = next((r for r in selected[last+1:] if preceding is not None
                         and r["exact_count"] >= preceding["exact_count"]), None)
        episodes.append(dict(first_low=selected[first], last_low=selected[last], trough=trough,
                             preceding=preceding, following=following,
                             first_return_to_preceding_exact=recovery,
                             preceding_4000_train=train_window(trough["step"]-4000,trough["step"]),
                             preceding_1000_train=train_window(trough["step"]-1000,trough["step"]),
                             following_4000_train=train_window(trough["step"],trough["step"]+4000)))

    summary = dict(
        source=str(args.log), source_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
        raw_line_count=len(raw.splitlines()), train_rows=len(train), eval_rows=len(evaluation),
        rerun_train_steps=len(train)-len(unique_train), restart_events=starts,
        metric_semantics=dict(
            evaluation="EMA weights, held-out 2048, fresh carry, final seg16; stablemax cell loss training",
            training_loss="Current segment's raw-model lm_loss, logged every 250 optimizer steps",
            training_acc_exact="Latest halted seg16 raw-model minibatch128 metrics; halt_step is explicit",
            phase_control="Loss centered by post105k mean at global step modulo16; descriptive, not causal"),
        detection=dict(
            purpose="Reproduce the existing README's z < -1.5 statement, not a failure criterion",
            selection="All evaluation rows with step >=105000, through last logged evaluation378882",
            population_sd_ddof=0, count_statistics=count_stats,
            coefficient_of_variation=count_stats["sd"]/count_stats["mean"],
            threshold=threshold, low_points=[selected[i] for i in low_indices],
            low_point_count=len(low_indices), consecutive_episode_count=len(groups)),
        episodes=episodes,
        best_eval=max(evaluation,key=lambda r:r["exact_count"]),
        last_eval=evaluation[-1], last_train=train[-1],
        local_328k_evals=[r for r in evaluation if 310000<=r["step"]<=340000],
        train_loss_by_step_mod16=phase_means,
        post105k_train=train_window(105000,unique_train[-1]["step"]+1),
        local_328k_train_windows={
            "before_320k_324k":train_window(320000,324000),
            "approach_324k_328k":train_window(324000,328000),
            "recovery_328k_332k":train_window(328000,332000)},
        limitations=[
            "No parameter, gradient, hidden-state or trace norms were logged here; no norm-divergence conclusion.",
            "Training is raw weights on current augmented minibatches; eval is EMA on fixed held-out puzzles.",
            "No raw-weight evaluation: EMA cancellation versus raw-model test instability is unresolved.",
            "250-step loss sampling cannot exclude transient excursions between logged steps.",
            "No causal intervention or new training/evaluation was performed.",
            "Four-decimal logged metrics have rounding error; exact eval counts are integers."])
    args.out.mkdir(parents=True, exist_ok=True)
    # phase_centered_loss was attached to selected row objects; normalize fieldnames for raw CSV.
    for r in train:
        r["phase_centered_loss"] = r["loss"]-phase_means[r["step_mod_16"]]["mean"]
    write_csv(args.out/"train.csv",train)
    write_csv(args.out/"eval.csv",evaluation)
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts=np.array([r["step"] for r in unique_train]); es=np.array([r["step"] for r in evaluation])
    def rolling(field):
        values=np.array([r[field] for r in unique_train])
        return np.array([values[max(0,i-15):i+1].mean() for i in range(len(values))])
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex="col")
    for col,(left,right) in enumerate([(0,382000),(310000,340000)]):
        for row,(field,label) in enumerate([("exact","Puzzle exact (%)"),("acc","Cell accuracy (%)")]):
            ax=axes[row,col]
            ax.plot(ts/1000,100*rolling(field),color="#8b8b8b",lw=1,label="Train raw, last 16 logged batches")
            ax.plot(es/1000,[100*r[field] for r in evaluation],color="#1b639c",marker=".",ms=3,lw=1,label="Eval EMA, 2048 puzzles")
            if row==0:
                ax.scatter([selected[i]["step"]/1000 for i in low_indices],
                           [100*selected[i]["exact"] for i in low_indices],color="#c84436",s=22,zorder=4,label="README z < -1.5 points")
            ax.set_ylabel(label); ax.set_ylim(0,101)
        ax=axes[2,col]
        ax.scatter(ts/1000,[r["loss"] for r in unique_train],s=3,alpha=.2,color="#888888",label="Logged current-segment loss")
        ax.plot(ts/1000,rolling("loss"),color="#204f7a",lw=1,label="16-point mean (4000 steps)")
        ax.set_ylabel("Train lm_loss");ax.set_xlabel("Optimizer step (thousands)")
        for row in range(3):
            ax=axes[row,col]; ax.set_xlim(left/1000,right/1000);ax.grid(alpha=.2)
            for ep in episodes:
                start=(ep["preceding"]["step"]+ep["first_low"]["step"])/2
                end=(ep["last_low"]["step"]+ep["following"]["step"])/2
                ax.axvspan(start/1000,end/1000,color="#c84436",alpha=.07)
            if col==0:ax.axvline(106,color="#6b4b94",ls="--",lw=.8)
    axes[0,0].legend(fontsize=8,loc="lower right")
    axes[2,0].legend(fontsize=8)
    axes[0,0].set_title("Recovered v1.7 training log; restart at106k")
    axes[0,1].set_title("328104-step evaluation dip and recovery")
    fig.suptitle("Recovered v1.7 training and evaluation metrics",fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out/"training_curves.png",dpi=170)
    fig.savefig(args.out/"training_curves.pdf")
    plt.close(fig)
    print(json.dumps(dict(eval_rows=len(evaluation),train_rows=len(train),
                          low_points=len(low_indices),episodes=len(groups),
                          threshold=threshold,output=str(args.out)),ensure_ascii=False))


if __name__ == "__main__":
    main()
