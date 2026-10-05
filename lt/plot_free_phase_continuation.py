"""Replot the archived October 5 continuation without a live run/checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'docs/research/2026-10-05/continuation_audit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (args.archive / 'train_terminal_metrics.csv').open() as stream:
        train = list(csv.DictReader(stream))
    evaluations = json.loads((args.archive / 'eval_history.json').read_text())['rows']
    raw = []
    for path in sorted(args.archive.glob('raw_vs_ema_step*/results.json')):
        result = json.loads(path.read_text())
        raw.append(dict(step=result['step'], **result['results']['raw']))
    summary = json.loads((args.archive / 'summary.json').read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'free-phase-continuation-20261005'})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True, layout='constrained')
    window = 32
    train_steps = np.array([int(row['step']) for row in train])
    for axis, metric, label in zip(axes, ('accuracy', 'exact_accuracy'),
                                    ('Cell accuracy (%)', 'Solved puzzles (%)')):
        values = np.array([float(row[metric]) for row in train]) * 100
        mean = np.convolve(values, np.ones(window) / window, mode='valid')
        axis.plot(train_steps[window-1:], mean, color='#8b949e', linewidth=1.3,
                  label='Online raw train: 32 terminal-batch mean')
        axis.plot([r['step'] for r in evaluations], [100*r[metric] for r in evaluations],
                  color='#1261a0', marker='o', markersize=3, linewidth=1.8,
                  label='Held-out EMA: 2,048 puzzles, 16 segments')
        axis.scatter([r['step'] for r in raw], [100*r[metric] for r in raw],
                     color='#d66800', marker='D', s=45, zorder=5,
                     label='Held-out raw: two measured checkpoints')
        axis.axvline(41013, color='#1261a0', alpha=.25, linestyle='--')
        axis.set_ylabel(label)
        axis.set_ylim(0, 100)
        axis.grid(alpha=.2)
    axes[0].legend(loc='lower right', fontsize=8)
    axes[1].set_xlabel('Optimizer step (8 recurrent blocks per step)')
    axes[1].annotate('Best EMA: 668/2048 (32.62%)', xy=(41013, 32.6171875),
                     xytext=(28500, 48), arrowprops={'arrowstyle': '->', 'color': '#1261a0'},
                     color='#1261a0', fontsize=9)
    fig.suptitle('Free-phase Fourier R=4 continuation — 2026-10-05\n'
                 f"Snapshot step {summary['cutoff_step']:,}; train/eval protocols differ", fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'learning_curves.png', dpi=160)
    svg = args.output / 'learning_curves.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
