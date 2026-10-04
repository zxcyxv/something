"""Compare all-segment and same-step terminal training metrics from scalar logs.

Reads live run directories or archived train.jsonl.gz files. The shared horizon
ends on a complete 16-segment episode; an unfinished live-log line is excluded.
No checkpoints or GPU are needed. Accuracy is never paired with another step's
loss, and runs with loops != 16 are excluded from matched comparisons.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.is_dir():
        path = next((path / name for name in ('train.jsonl', 'train.jsonl.gz')
                     if (path / name).exists()), path / 'train.jsonl')
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    if data and not data.endswith(b'\n'):
        data = data.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in data else b''
    rows = [json.loads(line) for line in data.splitlines() if line]
    steps = [row['step'] for row in rows]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f'Non-increasing optimizer steps in {path}')
    return rows


def window(rows, low, high):
    selected = [row for row in rows if low < row['step'] <= high]
    terminal = [row for row in selected if row.get('_count_raw', 0) > 0]
    result = dict(start_exclusive=low, end_inclusive=high,
                  steps=len(selected), terminal_batches=len(terminal))
    if selected:
        result['all_segment_loss'] = statistics.mean(row['lm_loss'] for row in selected)
    if terminal:
        weight = sum(row['_count_raw'] for row in terminal)
        for field, source in (('terminal_loss', 'lm_loss'), ('cell_accuracy', 'accuracy'),
                              ('exact_accuracy', 'exact_accuracy')):
            result[field] = sum(row[source] * row['_count_raw'] for row in terminal) / weight
    return result


def plot(runs, horizon, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ('All-segment loss', 'Terminal-segment loss', 'Terminal cell accuracy (%)')
    keys = ('all_segment_loss', 'terminal_loss', 'cell_accuracy')
    for label, rows in runs.items():
        windows = [window(rows, low, min(low+128, horizon))
                   for low in range(0, horizon, 128)]
        for axis, key in zip(axes, keys):
            values = [(w['end_inclusive'], w[key]) for w in windows if key in w]
            axis.plot([v[0] for v in values],
                      [v[1] * (100 if key == 'cell_accuracy' else 1) for v in values],
                      label=label, linewidth=1.7)
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel('Optimizer step')
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Aligned loops=16 training through step {horizon}; 128-step means')
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=DIRECTORY')
    ap.add_argument('--out', required=True, help='JSON destination; PNG uses the same stem')
    ap.add_argument('--window', type=int, default=512)
    args = ap.parse_args()
    runs, metadata = {}, {}
    for entry in args.run:
        label, source = entry.split('=', 1)
        directory = Path(source)
        cfg = json.loads((directory / 'config.json').read_text())
        if cfg.get('loops') != 16:
            raise ValueError(f'{label}: only loops=16 belongs in the matched comparison')
        rows = read_rows(directory)
        if not rows:
            raise ValueError(f'Empty scalar log: {directory}')
        runs[label] = rows
        metadata[label] = dict(source=str(directory), final_step=rows[-1]['step'],
                               model_id=cfg.get('model_id'), num_layers=cfg.get('num_layers'),
                               blocks_per_segment=cfg.get('blocks_per_seg'),
                               last_terminal=next((r for r in reversed(rows)
                                                   if r.get('_count_raw', 0) > 0), None))
        durations = [b['elapsed']-a['elapsed'] for a, b in zip(rows, rows[1:])
                     if b['step'] > 256 and a.get('elapsed') is not None and b.get('elapsed') is not None]
        if durations:
            metadata[label]['median_step_seconds_after_256'] = statistics.median(durations)
    horizon = min(rows[-1]['step'] for rows in runs.values()) // 16 * 16
    low = max(0, horizon-args.window)
    result = dict(
        aggregation='All optimizer-step losses separately from count>0 terminal loss/accuracy; matched identical step bounds',
        comparison_start_exclusive=low, comparison_end_inclusive=horizon,
        comparison={label: window(rows, low, horizon) for label, rows in runs.items()},
        runs=metadata,
        windows={label: [window(rows, start, min(start+args.window, horizon))
                        for start in range(0, horizon, args.window)] for label, rows in runs.items()})
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    plot(runs, horizon, dest.with_suffix('.png'))
    print(json.dumps(result['comparison'], indent=2))


if __name__ == '__main__':
    main()
