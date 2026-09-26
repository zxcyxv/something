"""Publication/export plots for the phase/time research, from saved results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def save(fig, root, name):
    fig.savefig(root/f"{name}.png", dpi=180)
    fig.savefig(root/f"{name}.pdf")
    plt.close(fig)


def main():
    root = Path("runs/phase_timing_v11")
    rank = read(root/"projective_time/summary.json")
    fit = read(root/"time_analysis/fit_summary.json")
    audit = read(root/"fp64_audit/summary.json")
    ablation = read(root/"write_ablation/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for puzzle, color in zip((58, 209, 230), ("#0072B2", "#D55E00", "#009E73")):
        cell = {58: 26, 209: 19, 230: 28}[puzzle]
        rows = [r for r in rank["cases"] if r["puzzle"] == puzzle and
                r["cell"] == cell and r["file"].startswith("late_")]
        axes[0].plot([r["head_1based"] for r in rows],
                     [r["sigma6_over_sigma1_quantiles"]["median"] for r in rows],
                     "o-", label=f"Late puzzle {puzzle}", color=color)
    reference = rank["positive_control"]["sigma6_over_sigma1_quantiles"]["median"]
    axes[0].axhline(reference, color="black", ls="--", label="Known temporal trace + arbitrary gauge")
    axes[0].axhline(.001, color="gray", ls=":", label="Minimum rejection threshold")
    axes[0].set(yscale="log", ylim=(3e-9, 1), xlabel="Head",
                ylabel="Median smallest / largest singular value",
                title="Necessary temporal-frequency identity fails")
    axes[0].legend(fontsize=8, loc="center left")
    ranks = fit["protocol"]["ranks"]
    for label, title, color in (("train", "Training trajectories", "gray"),
                                ("holdout_early", "Unseen puzzles, early", "#0072B2"),
                                ("holdout_late", "Late correction windows", "#D55E00")):
        values = np.array([[r["scores"][label]["relative_residual_norm"] for r in h["fits"]]
                           for h in fit["heads"]])
        axes[1].plot(ranks, np.median(values, axis=0), "o-", label=title, color=color)
        axes[1].fill_between(ranks, values.min(0), values.max(0), alpha=.12, color=color)
    axes[1].axhline(.1, color="black", ls=":", label="Predeclared 10% approximation target")
    axes[1].set(xscale="log", xticks=ranks, xticklabels=ranks,
                xlabel="Number of free real innovation components",
                ylabel="Unexplained increment norm / target increment norm",
                title="Best fitted damped rotations still leave large error", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    save(fig, root, "temporal_correspondence")

    labels = {
        "original": "Original",
        "cell_exchange_even": "Cell-exchange symmetric write",
        "intrinsic_phase_even": "Intrinsic-phase even write",
        "intrinsic_phase_even_energy_matched": "Phase-even, paired magnitude retained",
        "intrinsic_phase_mirrored": "Intrinsic-phase reversed write",
    }
    colors = {"original": "black", "cell_exchange_even": "#009E73",
              "intrinsic_phase_even": "#0072B2", "intrinsic_phase_even_energy_matched": "#E69F00",
              "intrinsic_phase_mirrored": "#CC79A7"}
    arrays = {}
    with np.load("runs/late_puzzle_probe_v11/baseline.npz") as z:
        arrays["original"] = z["predictions"]
        ids, gold = z["indices"], z["Y"]+1
    for name in labels:
        if name != "original":
            with np.load(root/"write_ablation"/f"{name}.npz") as z:
                arrays[name] = z["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, predictions in arrays.items():
        complete = (predictions == gold[None]).all(-1).sum(-1)
        axes[0].plot(np.arange(len(complete))[::8]/8, complete[::8],
                     label=labels[name], color=colors[name], lw=1.4, alpha=.85)
    axes[0].set(xlabel="Segment (8 blocks)", ylabel="Currently complete puzzles / 12",
                title="Fresh-state interventions; twelve preselected puzzles", ylim=(-.2, 12.2))
    axes[0].legend(fontsize=7.5, loc="lower right")
    rows = {"original": ablation["original"], **ablation["modes"]}
    order = list(labels)
    x = np.arange(len(order))
    axes[1].bar(x-.17, [rows[k]["final_complete_count"] for k in order], width=.34, label="At block8192")
    axes[1].bar(x+.17, [rows[k]["final_256_stable_count"] for k in order], width=.34, label="Correct throughout last256 blocks")
    axes[1].set(xticks=x, xticklabels=["Original", "Cell-even", "Phase-even", "Phase-even\nenergy matched", "Phase\nreversed"],
                ylabel="Complete puzzles / 12", ylim=(0, 12), title="Endpoint and sustained completion")
    axes[1].legend(fontsize=8)
    save(fig, root, "write_ablation")

    eta = np.array(audit["outer_memory"]["eta"])
    nu = np.linspace(0, np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for h in range(8):
        response = eta[h]/np.abs(1-(1-eta[h])*np.exp(-1j*nu))
        axes[0].plot(nu/np.pi, response, label=f"Head {h+1}")
    axes[0].set(xlabel="Temporal frequency / pi (radians per block)",
                ylabel="Steady-state amplitude gain", yscale="log",
                title="Actual time filter: W's learned EMA")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].bar(np.arange(1, 9), audit["outer_memory"]["half_life_blocks"])
    axes[1].set(xlabel="Head", ylabel="Memory half-life (blocks)",
                title="Explicit W retention time")
    save(fig, root, "memory_time_filter")
    print("saved phase/time research figures", flush=True)


if __name__ == "__main__":
    main()
