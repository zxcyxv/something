"""Plot fixed-clock, forward/reverse coupling responses from the order probe."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/order_pair_v11")
REPORT = json.loads((ROOT / "summary.json").read_text())


def window(start, head, edge):
    points = []
    for case in REPORT["cases"]:
        if case["snapshot_after_block"] != start:
            continue
        delta = case["lag_blocks"]
        row = case["heads"][head]["W_after_tail"][edge]
        # Keep the same physical clock in both directed couplings:
        # delta = time(cell48) - time(cell66).
        positive, negative = ((row["source_first"], row["target_first"])
                              if edge == "forward_edge" else
                              (row["target_first"], row["source_first"]))
        points.append((delta, positive))
        if delta:
            points.append((-delta, negative))
    return np.asarray(sorted(points))


def panel(ax, start, head):
    for edge, color, label in (("forward_edge", "#1864ab", "cell 66 -> cell 48"),
                               ("reverse_edge", "#c2255c", "cell 48 -> cell 66")):
        data = window(start, head, edge)
        ax.plot(data[:, 0], data[:, 1], "o-", color=color, ms=3.5, lw=1.5, label=label)
    ax.axhline(0, color="#555", lw=.6)
    ax.axvline(0, color="#777", lw=.6, ls=":")
    ax.set_xscale("symlog", linthresh=2)
    ticks = [-16, -4, -1, 0, 1, 4, 16]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.grid(alpha=.15)
    ax.set_title(f"After block {start} | head {head + 1}", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for ax, start in zip(axes, (152, 192)):
    panel(ax, start, 2)
    ax.set_xlabel("Lag: time(cell 48) - time(cell 66), blocks")
axes[0].set_ylabel("Paired memory contribution / epsilon^2")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Order response: 8 blocks after the second pulse\n"
             "Single-pulse effects subtracted; positive lag = cell 66 first", fontsize=11)
for extension in ("png", "pdf"):
    fig.savefig(ROOT / f"order_pair_head3.{extension}", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(4, 4, figsize=(14, 10), layout="constrained")
for rowbase, start in ((0, 152), (2, 192)):
    for head in range(8):
        panel(axes[rowbase + head // 4, head % 4], start, head)
fig.suptitle("All heads: paired memory response, 8 blocks after the second pulse\n"
             "Blue: 66 -> 48; red: 48 -> 66. Same lag convention for both edges.", fontsize=12)
fig.supxlabel("Lag: time(cell 48) - time(cell 66), blocks (symmetric log axis)")
fig.supylabel("Paired memory contribution / epsilon^2")
for extension in ("png", "pdf"):
    fig.savefig(ROOT / f"order_pair_all_heads.{extension}", dpi=170)
plt.close(fig)
print("saved order-response plots", ROOT)
