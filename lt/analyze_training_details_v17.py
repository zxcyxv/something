"""Summarize observed MLP/Phi adjoints and trace statistics of the v1.7 replay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v17_recovery/train_probe_380k_details'


def read_rows(folder):
    return [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]


def stats(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max()))


def main():
    rows = read_rows(OUT)
    prior = read_rows(ROOT / 'runs/v17_recovery/train_probe_380k_lr1e4')
    same = {key: all(a[key] == b[key] for a, b in zip(rows, prior))
            for key in ['current_exact', 'current_cell_accuracy', 'metrics',
                        'dense_gradient_norm', 'dense_update_norm', 'updates']}
    assert len(rows) == len(prior) == 64 and all(same.values())
    blocks = [dict(step=r['step'], segment=r['segment'], **b)
              for r in rows for b in r['blocks']]
    pairs = {
        'mlp_residual': ('mlp_input_gradient', 'pre_phi_gradient'),
        'phi': ('pre_phi_gradient', 'post_phi_gradient'),
        'mlp_residual_and_phi': ('mlp_input_gradient', 'post_phi_gradient'),
    }
    ratios = {key: np.array([b[n]['norm'] / b[d]['norm'] for b in blocks])
              for key, (n, d) in pairs.items()}
    measurements = ['q_norm_mean', 'mlp_input_norm_mean', 'mlp_input_norm_max',
                    'mlp_output_delta_norm_mean', 'mlp_delta_over_input_mean',
                    'mlp_residual_over_input_mean', 'pre_phi_norm_mean',
                    'post_phi_norm_mean', 'mlp_input_above_sqrt_d_fraction',
                    'trace_norm_min', 'trace_norm_mean',
                    'trace_over_address_mean', 'trace_over_address_max']
    summary = dict(
        replay_identical=same, steps=len(rows), blocks=len(blocks),
        definitions={
            'adjoint_ratios': 'Global L2 norms over batch/cell/features in the actual loss direction; not spectral norms.',
            'norm_mean': 'Per-cell feature L2 norm, averaged over 128 puzzles and 81 cells within each block.',
            'trace_norm_min': 'Minimum complex trace norm over puzzle/cell/head within each block; before normalization.',
            'head_gradient_energy': 'Sum of squared raw-coordinate gradients over 64 steps; not functional sensitivity.',
        },
        measurements={k: stats([b[k] for b in blocks]) for k in measurements},
        adjoint_ratios={k: stats(v) | dict(count_above_one=int((v > 1).sum()))
                        for k, v in ratios.items()},
        trace_over_address_mean_per_head=np.mean(
            [b['trace_over_address_mean_per_head'] for b in blocks], axis=0).tolist(),
        gradient_squared_share_per_head={},
    )
    for key in rows[0]['gradients']:
        if '.mu_rho_raw' in key or '.mu_omega' in key:
            energy = np.array([r['gradients'][key]['per_head_norm'] for r in rows]) ** 2
            summary['gradient_squared_share_per_head'][key] = (energy.sum(0) / energy.sum()).tolist()
    (OUT / 'detail_analysis.json').write_text(json.dumps(summary, indent=2) + '\n')

    x = np.array([r['step'] for r in rows]) - 380000
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for key, label in [('mlp_input_norm_mean', 'y: MLP input'),
                       ('pre_phi_norm_mean', 'r: before Phi'),
                       ('post_phi_norm_mean', 'h: after Phi')]:
        a = np.array([b[key] for b in blocks]).reshape(64, 8)
        line, = axes[0].plot(x, a.mean(1), label=label)
        axes[0].fill_between(x, a.min(1), a.max(1), alpha=.12, color=line.get_color())
    axes[0].axhline(np.sqrt(832), color='gray', linestyle=':', label='sqrt(d)')
    axes[0].set_ylabel('Mean cell L2 norm')
    axes[0].legend(ncol=2)
    for key, label in [('mlp_residual', 'MLP + residual'), ('phi', 'Phi'),
                       ('mlp_residual_and_phi', 'MLP + residual + Phi')]:
        a = ratios[key].reshape(64, 8)
        line, = axes[1].plot(x, a.mean(1), label=label)
        axes[1].fill_between(x, a.min(1), a.max(1), alpha=.12, color=line.get_color())
    axes[1].axhline(1, color='gray', linestyle=':')
    axes[1].set_ylabel('Actual loss adjoint norm ratio')
    axes[1].legend()
    for key, label in [('trace_norm_min', 'Minimum over cells/heads'),
                       ('trace_norm_mean', 'Mean over cells/heads')]:
        a = np.array([b[key] for b in blocks]).reshape(64, 8)
        axes[2].plot(x, a.min(1) if key.endswith('min') else a.mean(1), label=label)
    axes[2].set_ylabel('Trace norm before normalization')
    axes[2].set_xlabel('Training update after raw checkpoint 380000')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
        for boundary in [16.5, 32.5, 48.5]:
            ax.axvline(boundary, color='gray', linewidth=.6)
    fig.suptitle('v1.7 actual training: 64 updates, 8 blocks/update\nShading: range over blocks, not a confidence interval')
    fig.savefig(OUT / 'mlp_trace_diagnostics.png', dpi=160)
    fig.savefig(OUT / 'mlp_trace_diagnostics.pdf')
    plt.close(fig)
    print(json.dumps(summary['adjoint_ratios'], indent=2))


if __name__ == '__main__':
    main()
