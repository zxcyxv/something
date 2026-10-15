import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = sys.argv[1]; rows = json.load(open(f'{R}/progress.json')); FIG = f'{R}/fig'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
s = np.array([r['step'] for r in rows]); G = lambda b, k: np.array([r['blocks'][str(b)][k] for r in rows])
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2))
ax = axs[0, 0]; cm = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
for c, b in zip(cm, (2, 8, 16, 32, 64)):
    ax.plot(s, G(b, 'acc') * 100, color=c, lw=1.6, label=f'block {b}')
ax.set_title('exact rows on held-out (%)', fontsize=9); ax.legend(fontsize=8, frameon=False); ax.set_xlabel('training step')
ax = axs[0, 1]
for b, c in ((8, 'tab:blue'), (32, 'tab:red')):
    ax.plot(s, G(b, 'cos'), color=c, lw=1.6, label=f'Walsh cos to truth, block {b}')
    ax.plot(s, G(b, 'table'), color=c, lw=1.0, ls='--', label=f'truth table correct, block {b}')
ax.set_ylim(0.5, 1.01); ax.set_title('does the 512-entry table match Rule 110?', fontsize=9); ax.legend(fontsize=8, frameon=False); ax.set_xlabel('training step')
ax = axs[1, 0]; D = np.array([r['blocks']['8']['deg'] for r in rows]); cols = plt.cm.tab10(np.arange(10)); bottom = np.zeros(len(s))
for j in range(1, 8):
    ax.fill_between(s, bottom, bottom + D[:, j], color=cols[j], label=f'degree {j}', lw=0); bottom += D[:, j]
ax.set_ylim(0, 1); ax.set_title('Walsh energy by interaction degree, block 8', fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=4, loc='lower right'); ax.set_xlabel('training step')
ax = axs[1, 1]
ax.plot(s, G(8, 'r4'), color='tab:purple', lw=1.6, label='R² of the 9-cell cone, block 8')
ax.plot(s, G(32, 'r4'), color='tab:purple', lw=1.0, ls='--', label='same, block 32')
ax.plot(s, G(8, 'r3'), color='tab:gray', lw=1.2, label='R² of the 7-cell window, block 8')
ax.set_ylim(0, 1.01); ax.set_xlabel('training step'); ax.legend(fontsize=8, frameon=False, loc='lower left')
ax2 = ax.twinx(); ax2.plot(s, [r['pnorm'] for r in rows], color='tab:orange', lw=1.4); ax2.set_ylabel('weight norm', color='tab:orange')
ax2.spines['right'].set_visible(True); ax.set_title('locality and weight norm', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/f9_progress.png'); plt.close(fig)
