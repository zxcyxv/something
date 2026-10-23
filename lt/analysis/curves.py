"""Learning curves of lt.train runs against reference curves (TRM ARC-AGI-1 and others): one figure, a merged CSV and a
markdown table at the reference's x values.

    python -m lt.analysis.curves runs/arc1_trm_l1 --ref docs/evidence/trm_arc1/reference.json --out runs/compare
    python -m lt.analysis.curves runs/a runs/b --ref ref.json --ref wandb_export.csv:TRM-wandb --x epoch --out cmp

Our run (OUT of lt.train): config.json, train.jsonl (one row per optimizer step), eval.jsonl (one row per evaluation;
runs from before eval.jsonl: the [EVAL] lines of stdout*.log). Several directories of one run (resumed elsewhere) can be
joined with '+': runs/x+runs/x_resumed.

Metrics, ours -> the TRM/HRM W&B names they correspond to (same definitions in pretrain.py / ACTLossHead):
    train lm_loss          train/lm_loss          per-puzzle mean stablemax CE (pad ignored), summed over lanes / global batch
    train exact_accuracy   train/exact_accuracy   finished (halted) puzzles of the step with every cell right
    train accuracy         train/accuracy         cell accuracy of finished puzzles
    eval exact / n         all/exact_accuracy     test pairs right; ours: identity version only, all `loops` segments
                           ARC/pass@1, pass@2     TRM: 1000-version voting per test input, scored per task (not ours)
A reference file is JSON: {"name", "source", "global_batch_size", "examples_per_epoch", "curves": {METRIC: [[x, y], ...]},
"x_unit": {METRIC: "step" | "epoch"}, "notes"} with METRIC among the names above; or a W&B CSV export (a "Step" column
and one column per metric; 'file.csv:label' names it). Values in percent (> 1.5 for an accuracy) are divided by 100.

X axes: step (optimizer steps), epoch (= step * global batch / training pairs; TRM's epoch is the same pass over the
3,092 demonstration pairs), samples (lane-segments = step * global batch), hours (summed step time, evaluation excluded).
"""
import argparse
import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

TRAIN_KEYS = {'train/lm_loss': 'lm_loss', 'train/exact_accuracy': 'exact_accuracy', 'train/accuracy': 'accuracy',
              'train/steps': 'steps'}
EVAL_KEYS = {'all/exact_accuracy': 'exact', 'all/accuracy': 'accuracy'}
PANELS = (('train/lm_loss', 'train lm_loss (stablemax CE)', True),
          ('train/exact_accuracy', 'train exact (finished puzzles)', False),
          ('eval/exact', 'eval exact (test pairs / TRM pass@k)', False),
          ('cell', 'cell accuracy (train / eval)', False))
EVAL_RE = re.compile(r'^\[EVAL\] step (\d+) acc ([\d.]+) exact (\d+)/(\d+)')


def _jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def _train_pairs(cfg):
    """Training pairs per epoch (the denominator of lt.train's epoch)."""
    path = cfg.get('data_npz')
    meta = Path(path).with_suffix('.json') if path else None
    if meta and meta.exists():
        m = json.loads(meta.read_text())
        if 'train_pairs' in m:
            return int(m['train_pairs'])
    if path and Path(path).exists():
        with np.load(path, allow_pickle=False) as z:
            for key in ('arc_train_in', 'train_inputs', 'inputs'):
                if key in z.files:
                    return int(z[key].shape[0])
    return 3092


def load_run(spec):
    """'dir' or 'dir1+dir2+...' (one run continued in several directories) -> dict of arrays."""
    dirs = [Path(d) for d in spec.split('+')]
    cfg = json.loads((dirs[0] / 'config.json').read_text())
    rows, evals = {}, {}
    for d in dirs:
        for name in ('train.jsonl', 'train.jsonl.gz'):
            if (d / name).exists():
                for r in _jsonl(d / name):
                    rows[r['step']] = r                            # a resume rewrites steps after the checkpoint
        if (d / 'eval.jsonl').exists():
            for r in _jsonl(d / 'eval.jsonl'):
                if not r.get('tag'):
                    evals[r['step']] = dict(step=r['step'], accuracy=r['accuracy'], exact=r['exact'] / max(r['n'], 1),
                                            n=r['n'], partial=r.get('partial', False))
        else:
            for log in sorted(d.glob('stdout*.log')):
                for line in log.read_text(errors='replace').splitlines():
                    m = EVAL_RE.match(line)
                    if m:
                        s, acc, ex, n = int(m[1]), float(m[2]), int(m[3]), int(m[4])
                        evals[s] = dict(step=s, accuracy=acc, exact=ex / max(n, 1), n=n,
                                        partial='[partial]' in line)
    if not rows:
        raise FileNotFoundError(f'{spec}: no train.jsonl')
    steps = np.array(sorted(rows))
    get = lambda k: np.array([float(rows[s].get(k, np.nan)) for s in steps])
    gbs = int(cfg['global_batch_size'])
    run = dict(name=Path(dirs[-1]).name, cfg=cfg, gbs=gbs, pairs=_train_pairs(cfg), loops=cfg.get('loops'),
               step=steps.astype(float), seconds=get('seconds'), lm_loss=get('lm_loss'), accuracy=get('accuracy'),
               exact_accuracy=get('exact_accuracy'), count=get('count'), steps=get('steps'), mem_gb=get('mem_gb'))
    ev = [evals[s] for s in sorted(evals) if not evals[s]['partial']]
    run['eval_step'] = np.array([e['step'] for e in ev], float)
    run['eval_exact'] = np.array([e['exact'] for e in ev])
    run['eval_acc'] = np.array([e['accuracy'] for e in ev])
    run['eval_n'] = ev[-1]['n'] if ev else 0
    return run


def to_x(step, kind, gbs, pairs, run=None):
    step = np.asarray(step, float)
    if kind == 'step':
        return step
    if kind == 'epoch':
        return step * gbs / pairs
    if kind == 'samples':
        return step * gbs
    if kind == 'hours':
        if run is None:
            return np.full_like(step, np.nan)                          # reference curves have no wall clock
        cum = np.nancumsum(run['seconds']) / 3600
        return np.interp(step, run['step'], cum)
    raise ValueError(kind)


def smooth(y, window):
    """Trailing mean over `window` points, ignoring NaN."""
    if window <= 1 or len(y) < 2:
        return y
    ok = np.isfinite(y)
    c = np.cumsum(np.where(ok, y, 0.0))
    n = np.cumsum(ok)
    lo = np.maximum(np.arange(len(y)) - window, -1)
    s = c - np.where(lo >= 0, c[np.maximum(lo, 0)], 0)
    k = n - np.where(lo >= 0, n[np.maximum(lo, 0)], 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(k > 0, s / np.maximum(k, 1), np.nan)


def run_series(run, window):
    """Panel key -> list of (label suffix, step array, y array, style)."""
    fin = run['count'] > 0                                         # steps where some puzzle finished
    out = {'train/lm_loss': [('', run['step'], smooth(run['lm_loss'], window), '-')],
           'train/exact_accuracy': [('', run['step'][fin], smooth(run['exact_accuracy'][fin], window), '-')],
           'cell': [(' train', run['step'][fin], smooth(run['accuracy'][fin], window), '-')]}
    if len(run['eval_step']):
        out['eval/exact'] = [(f" eval exact/{run['eval_n']} (id. version)", run['eval_step'], run['eval_exact'], 'o-')]
        out['cell'].append((' eval', run['eval_step'], run['eval_acc'], 'o--'))
    return out


def _pct(metric, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    if 'loss' not in metric and 'steps' not in metric and len(pts) and np.nanmax(pts[:, 1]) > 1.5:
        pts = pts.copy()
        pts[:, 1] /= 100
    return pts


def load_ref(spec):
    """JSON reference or W&B CSV export ('path[:label]')."""
    path, _, label = spec.partition(':') if not Path(spec).exists() else (spec, '', '')
    path = Path(path)
    if path.suffix == '.csv':
        with path.open() as f:
            rows = list(csv.DictReader(f))
        step_col = next(c for c in rows[0] if c.lower() in ('step', '_step', 'train/step', 'global_step'))
        curves = {}
        for col in rows[0]:
            base = col.split(' - ')[-1].strip()                  # W&B: "run - metric", "run - metric__MIN"
            if col == step_col or base.endswith(('__MIN', '__MAX')):
                continue
            pts = [(float(r[step_col]), float(r[col])) for r in rows if r[col] not in ('', 'NaN', None)]
            if pts:
                curves[base] = pts
        ref = dict(name=label or path.stem, source=str(path), curves=curves, x_unit={})
    else:
        ref = json.loads(path.read_text())
        if label:
            ref['name'] = label
    ref['curves'] = {k: _pct(k, v) for k, v in ref['curves'].items() if len(v)}
    ref.setdefault('x_unit', {})
    ref.setdefault('global_batch_size', 768)
    ref.setdefault('examples_per_epoch', 3092)
    return ref


def ref_series(ref):
    """Panel key -> list of (label suffix, step array, y array, style, approx)."""
    out = {}
    for metric, pts in ref['curves'].items():
        unit = ref['x_unit'].get(metric, ref.get('default_x_unit', 'step'))
        x = pts[:, 0] if unit == 'step' else pts[:, 0] * ref['examples_per_epoch'] / ref['global_batch_size']
        if metric in ('train/lm_loss', 'train/exact_accuracy'):
            key, suffix = metric, ''
        elif metric == 'train/accuracy':
            key, suffix = 'cell', ' train'
        elif metric == 'all/accuracy':
            key, suffix = 'cell', ' eval (all versions)'
        elif metric == 'all/exact_accuracy':
            key, suffix = 'eval/exact', ' all/exact_accuracy (every version)'
        elif metric.startswith('ARC/pass@') or metric.startswith('eval/pass@'):
            key, suffix = 'eval/exact', ' ' + metric.split('/')[-1] + ' (1000-version vote, per task)'
        else:
            continue
        out.setdefault(key, []).append((suffix, x, pts[:, 1], 'x--' if len(x) < 40 else '--'))
    return out


def nearest(xs, ys, x, tol):
    if not len(xs):
        return math.nan
    i = int(np.argmin(np.abs(xs - x)))
    return float(ys[i]) if abs(xs[i] - x) <= tol else math.nan


def table(runs, refs, window):
    """Markdown: at every x (in steps) where a reference has a point, the reference value and each run's value."""
    lines = []
    for ref in refs:
        for key, series in ref_series(ref).items():
            for suffix, rx, ry, _ in series:
                lines.append(f"\n**{ref['name']}{suffix} — {key}** (x in optimizer steps of global batch "
                             f"{ref['global_batch_size']}; epoch = step × {ref['global_batch_size']} / {ref['examples_per_epoch']})\n")
                head = '| step | epoch | ref | ' + ' | '.join(r['name'] for r in runs) + ' |'
                lines += [head, '|' + '---:|' * (3 + len(runs))]
                for x, y in zip(rx, ry):
                    vals = []
                    for r in runs:
                        s = run_series(r, window).get(key, [])
                        if not s:
                            vals.append('—')
                            continue
                        _, sx, sy, _ = s[0]
                        tol = 0.05 * x + 2 * (r['cfg'].get('eval_interval', 1) * r['pairs'] / r['gbs']
                                              if key == 'eval/exact' else window)
                        v = nearest(sx, sy, x, tol)
                        vals.append('—' if math.isnan(v) else f'{v:.4f}')
                    ep = x * ref['global_batch_size'] / ref['examples_per_epoch']
                    lines.append(f'| {x:,.0f} | {ep:,.0f} | {y:.4f} | ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def run_summary(run):
    last = int(run['step'][-1])
    hours = float(np.nansum(run['seconds'])) / 3600
    fin = run['count'] > 0
    tail = slice(max(0, len(run['step']) - 200), None)
    s = (f"- **{run['name']}**: loops {run['loops']}, global batch {run['gbs']}, {last:,} steps "
         f"(epoch {last * run['gbs'] / run['pairs']:,.0f}), {hours:.2f} h of steps "
         f"({hours / max(last, 1) * 3600:.3f} s/step), peak memory {np.nanmax(run['mem_gb']) if np.isfinite(run['mem_gb']).any() else float('nan'):.1f} GB/GPU; "
         f"last 200 steps: lm_loss {np.nanmean(run['lm_loss'][tail]):.4f}, "
         f"train exact {np.nanmean(run['exact_accuracy'][tail][fin[tail]]) if fin[tail].any() else float('nan'):.4f}")
    if len(run['eval_step']):
        b = int(np.argmax(run['eval_exact']))
        s += (f"; eval exact last {run['eval_exact'][-1]:.4f} (step {int(run['eval_step'][-1]):,}), "
              f"best {run['eval_exact'][b]:.4f} (step {int(run['eval_step'][b]):,}) of {run['eval_n']} test pairs")
    return s


def plot(runs, refs, xkind, window, out_png, xlog=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, (key, title, logy) in zip(axes.flat, PANELS):
        for i, r in enumerate(runs):
            for suffix, sx, sy, style in run_series(r, window).get(key, []):
                ax.plot(to_x(sx, xkind, r['gbs'], r['pairs'], r), sy, style, color=colors[i % len(colors)],
                        lw=1.4, ms=3, label=f"{r['name']} (loops {r['loops']}){suffix}")
        for j, ref in enumerate(refs):
            c = colors[(len(runs) + j) % len(colors)]
            for k, (suffix, sx, sy, style) in enumerate(ref_series(ref).get(key, [])):
                x = to_x(sx, xkind, ref['global_batch_size'], ref['examples_per_epoch'])
                if np.isfinite(x).any():
                    ax.plot(x, sy, style, color=c, alpha=1 - 0.25 * k, lw=1.4, ms=6, label=f"{ref['name']}{suffix}")
        ax.set_title(title)
        ax.set_xlabel({'step': 'optimizer step', 'epoch': 'epoch (passes over the training pairs)',
                       'samples': 'lane-segments (step x global batch)', 'hours': 'hours of training steps'}[xkind])
        if logy:
            ax.set_yscale('log')
        if xlog:
            ax.set_xscale('log')
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle(f'ARC-AGI-1 learning curves (x = {xkind}; train curves: trailing mean over {window} steps)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_csv(runs, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'kind', 'step', 'epoch', 'hours', 'lm_loss', 'accuracy', 'exact_accuracy', 'count'])
        for r in runs:
            hrs = np.nancumsum(r['seconds']) / 3600
            for i, s in enumerate(r['step']):
                w.writerow([r['name'], 'train', int(s), s * r['gbs'] / r['pairs'], hrs[i], r['lm_loss'][i],
                            r['accuracy'][i], r['exact_accuracy'][i], r['count'][i]])
            for s, a, e in zip(r['eval_step'], r['eval_acc'], r['eval_exact']):
                w.writerow([r['name'], 'eval', int(s), s * r['gbs'] / r['pairs'],
                            float(np.interp(s, r['step'], hrs)), '', a, e, r['eval_n']])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help="lt.train output directories ('a+b' joins one run's directories)")
    ap.add_argument('--ref', action='append', default=[], help='reference curves: JSON, or W&B CSV export[:label]')
    ap.add_argument('--out', required=True, help='output directory (curves_<x>.png, curves.csv, summary.md)')
    ap.add_argument('--x', nargs='+', default=['step', 'epoch', 'samples', 'hours'], choices=['step', 'epoch', 'samples', 'hours'])
    ap.add_argument('--window', type=int, default=0, help='trailing mean of the train curves in steps (0: 1%% of the run)')
    ap.add_argument('--xlog', action='store_true')
    a = ap.parse_args(argv)
    runs = [load_run(s) for s in a.runs]
    refs = [load_ref(s) for s in a.ref]
    window = a.window or max(1, int(0.01 * max(len(r['step']) for r in runs)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for xk in a.x:
        plot(runs, refs, xk, window, out / f'curves_{xk}.png', a.xlog)
    write_csv(runs, out / 'curves.csv')
    md = ['# Learning curves', '', *[run_summary(r) for r in runs], '']
    for ref in refs:
        md.append(f"- reference **{ref['name']}**: {ref.get('source', '')}" + (f" — {ref['notes']}" if ref.get('notes') else ''))
    md += ['', f'Train values are trailing means over {window} steps; "—" = no point of the run near that x.',
           table(runs, refs, window), '']
    (out / 'summary.md').write_text('\n'.join(md))
    print('\n'.join(md[:len(runs) + 2]))
    print(f"wrote {', '.join(str(out / f'curves_{x}.png') for x in a.x)}, {out / 'curves.csv'}, {out / 'summary.md'}")


if __name__ == '__main__':
    main()
