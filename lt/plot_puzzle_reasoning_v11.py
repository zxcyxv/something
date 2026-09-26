"""Plot puzzle states, all-head message contributions, and digit trajectories."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_puzzle_overview_v11 import draw_board


def main():
    root = Path('runs/puzzle_overview_v11')
    for puzzle, offset, cell in [(58, 0, 26), (209, -3, 19), (230, 0, 28)]:
        d = json.loads((root / f'puzzle_{puzzle}_reasoning.json').read_text())
        m = json.loads((root / f'puzzle_{puzzle}/metadata.json').read_text())
        e = next(e for e in d['events'] if e['offset'] == offset and e['cell'] == cell)
        frames = {b['offset']: b['prediction'] for b in d['blocks']}
        fig = plt.figure(figsize=(18, 11), facecolor='#f7f9fc')
        g = fig.add_gridspec(2, 4, height_ratios=[1, 1.7], hspace=.38, wspace=.33)
        for col, (f, title) in enumerate([(-16, 'Starting board: T-16'),
                                        (offset-1, f'Before focused block: T{offset-1:+d}'),
                                        (offset, f'After focused block: T{offset:+d}')]):
            draw_board(fig.add_subplot(g[0, col]), frames[f], m['givens'], m['gold'], title, cell)
        ax = fig.add_subplot(g[0, 3])
        votes = np.array([s['read'] for s in e['sources']]).reshape(9, 9)
        lim = float(np.abs(votes).max())
        im = ax.imshow(votes, cmap='RdBu_r', vmin=-lim, vmax=lim)
        for j, s in enumerate(e['sources']):
            r, c = divmod(j, 9)
            color = 'white' if abs(s['read']) > lim*.65 else '#243249'
            ax.text(c, r-.14, str(s['before']), ha='center', va='center', fontsize=10,
                    fontweight='bold' if m['givens'][j] else 'normal', color=color)
            ax.text(c, r+.24, f"{s['read']:+.1f}", ha='center', va='center', fontsize=7, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for k in [2.5, 5.5]:
            ax.axhline(k, color='#73839c', lw=1)
            ax.axvline(k, color='#73839c', lw=1)
        ax.set_title(f"All-head messages to {e['rc']}\nDigit {e['new']} minus {e['old']}; each tile = one source", fontsize=10, loc='left')
        fig.colorbar(im, ax=ax, shrink=.8, fraction=.045)

        ax = fig.add_subplot(g[1, :3])
        traj = np.array([c['trajectory'] for c in d['cells']])
        gold = np.array([c['gold'] for c in d['cells']])
        ax.imshow(traj != gold[:, None], cmap=matplotlib.colors.ListedColormap(['#e5f2f7', '#ffe0e5']),
                  interpolation='nearest', vmin=0, vmax=1, aspect='auto')
        for r in range(len(gold)):
            for c in range(17):
                ax.text(c, r, str(traj[r, c]), ha='center', va='center', fontsize=9,
                        fontweight='bold' if c and traj[r, c] != traj[r, c-1] else 'normal',
                        color='#b42339' if traj[r, c] != gold[r] else '#243249')
        ax.set_yticks(np.arange(len(gold)), [f"{c['rc']}  (gold {c['gold']})" for c in d['cells']], fontsize=9)
        ax.set_xticks(np.arange(17), [str(x) for x in range(-16, 1)], fontsize=9)
        ax.set_xticks(np.arange(-.5, 17, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(gold), 1), minor=True)
        ax.grid(which='minor', color='white', lw=.7)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.axvline(offset+16-.45, color='#7c3aed', lw=1)
        ax.axvline(offset+16+.45, color='#7c3aed', lw=1)
        ax.set_xlabel('Blocks relative to full-puzzle completion T')
        ax.set_title('Every cell that is wrong at least once in this window | pink: wrong; blue: correct; bold: changed', loc='left', fontsize=11)

        sub = g[1, 3].subgridspec(2, 1, height_ratios=[1, 1.2], hspace=.8)
        ax = fig.add_subplot(sub[0])
        groups = e['source_groups']
        peer = sum(v['read'] for k, v in groups.items() if k.startswith('peer_'))
        nonpeer = sum(v['read'] for k, v in groups.items() if k.startswith('nonpeer_'))
        values = [e['current']['prepared_q'], peer, nonpeer, groups['self']['read'], e['current']['bias']]
        labels = ['Prepared q', 'All peers', 'All nonpeers', 'Self read', 'Bias']
        ax.barh(labels, values, color=['#d45565' if v > 0 else '#4b83b6' for v in values])
        ax.invert_yaxis()
        ax.axvline(0, color='#65758b', lw=.6)
        ax.set_title(f"Current {e['new']}-{e['old']} margin = {e['margin']:+.3f}", loc='left', fontsize=10)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(values):
            ax.text(0, i, f' {v:+.3f}', va='center', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        ax = fig.add_subplot(sub[1])
        vals = list(e['temporal_decomposition'].values())
        labels = ['Prepared q change', 'Value change', 'a_psi change', 'W change', 'Normalization']
        ax.barh(labels, vals, color=['#d45565' if v > 0 else '#4b83b6' for v in vals])
        ax.invert_yaxis()
        ax.axvline(0, color='#65758b', lw=.6)
        ax.set_title(f"Margin change: {e['previous_margin']:+.3f} -> {e['margin']:+.3f}", loc='left', fontsize=10)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax.text(0, i, f' {v:+.3f}', va='center', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f"Puzzle {puzzle} | from digit exchanges to the {e['old']} -> {e['new']} transition at {e['rc']} (T{offset:+d})",
                     x=.08, ha='left', fontsize=17, fontweight='bold')
        fig.text(.08, .025, 'Message tiles sum all 8 heads and use the actual output normalization. Positive favors the new digit over the old digit. '
                 'These are normal-trajectory contributions, not intervention effects.', fontsize=9, color='#475569')
        path = root / f'puzzle_{puzzle}_reasoning.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(path, flush=True)


if __name__ == '__main__':
    main()
