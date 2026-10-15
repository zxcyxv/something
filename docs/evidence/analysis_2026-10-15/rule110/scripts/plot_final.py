import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = sys.argv[1]; d = np.load(f'{R}/final.npz'); FIG = f'{R}/fig'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
BL = list(d['BL']); NB = list(d['NB']); PB = list(d['PB']); KB = list(d['KB'])
deg = np.array([bin(s).count('1') for s in range(512)])
st = d['spec_truth']; sup = np.abs(st) > 1e-6

# F1 extrapolation over training (run 1 evals)
log = json.load(open(f'{R}/log.json')); fig, ax = plt.subplots(figsize=(7, 3.6))
cm = plt.cm.viridis(np.linspace(0.15, 0.9, len(log)))
for c, r in zip(cm, log):
    ax.plot(range(1, 129), np.array(r['exact']) * 100, color=c, lw=1.6, label=f"{r['step']:,}")
ax.axvspan(1, 8, color='0.92'); ax.text(1.3, 5, 'trained\nhorizon', fontsize=8, color='0.35')
ax.set_xscale('log', base=2); ax.set_xlabel('block'); ax.set_ylabel('exact rows (%)'); ax.legend(title='step', fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/f1_extrap.png'); plt.close(fig)

# F2 light cone R^2 heatmap + table accuracy
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
im = axs[0].imshow(d['R2'].T, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
axs[0].set_xticks(range(len(BL))); axs[0].set_xticklabels(BL, fontsize=7); axs[0].set_yticks(range(7))
axs[0].set_xlabel('block'); axs[0].set_ylabel('window radius r'); axs[0].axhline(3.5, color='c', lw=1, ls='--')
axs[0].set_title('variance of the output logit explained by the radius-r window', fontsize=9); fig.colorbar(im, ax=axs[0], fraction=0.04)
axs[1].plot(BL, d['R2'][:, 4] * 100, 'o-', label='R² of the 9-cell light cone', color='tab:purple')
axs[1].plot(BL, d['table_acc'] * 100, 's-', label='512-entry truth table correct', color='tab:green')
axs[1].plot(BL, d['row_acc'] * 100, '^-', label='exact rows', color='tab:gray')
axs[1].set_xscale('log', base=2); axs[1].axvspan(1, 8, color='0.92'); axs[1].set_xlabel('block'); axs[1].set_ylabel('%'); axs[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/f2_lightcone.png'); plt.close(fig)

# F3 Walsh spectra
fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
order = np.lexsort((np.arange(512), deg))
for ax, b in zip(axs[:2], (8, 128)):
    sm = d['spec_model'][BL.index(b)]; sm = sm / np.linalg.norm(sm); stn = st / np.linalg.norm(st)
    ax.scatter(stn, sm, s=8, c=deg, cmap='tab10', vmin=0, vmax=9)
    lim = max(abs(stn).max(), abs(sm).max()) * 1.05; ax.plot([-lim, lim], [-lim, lim], color='0.7', lw=0.8)
    cos = sm @ st / np.linalg.norm(sm) / np.linalg.norm(st)
    ax.set_title(f'block {b}: cos = {cos:.3f}', fontsize=9); ax.set_xlabel('true Walsh coefficient (unit norm)'); ax.set_ylabel('model coefficient (unit norm)')
E = np.array([[(d['spec_model'][k][deg == j] ** 2).sum() for j in range(9)] for k in range(len(BL))]); E /= E.sum(1, keepdims=True)
Et = np.array([(st[deg == j] ** 2).sum() for j in range(9)]); Et /= Et.sum()
bottom = np.zeros(len(BL)); cols = plt.cm.tab10(np.arange(9))
for j in range(1, 8):
    axs[2].bar(range(len(BL)), E[:, j], bottom=bottom, color=cols[j], label=f'degree {j}', width=0.85); bottom += E[:, j]
axs[2].set_xticks(range(len(BL))); axs[2].set_xticklabels(BL, fontsize=7, rotation=90); axs[2].set_xlabel('block')
axs[2].set_title('energy by interaction degree (truth: ' + ' '.join(f'{Et[j]:.2f}' for j in range(1, 7)) + ')', fontsize=8)
axs[2].legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(f'{FIG}/f3_walsh.png'); plt.close(fig)

# F4 truth table images: rows = offsets -4..-1 (4 bits), cols = offsets 0..+4 (5 bits)
def grid(v):
    return np.array([[v[l | (r << 4)] for r in range(32)] for l in range(16)])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
axs[0].imshow(grid(d['f']), cmap='Greys', vmin=0, vmax=1); axs[0].set_title('truth: centre after 4 steps', fontsize=9)
for ax, b in zip(axs[1:], (8, 128)):
    g = d['g'][BL.index(b)]; lim = np.abs(g).max()
    ax.imshow(grid(g), cmap='RdBu_r', vmin=-lim, vmax=lim)
    wrong = (g > 0) != (d['f'] > 0.5); yy, xx = np.where(grid(wrong.astype(float)) > 0)
    ax.scatter(xx, yy, marker='x', color='k', s=14, lw=0.9)
    ax.set_title(f'model mean logit, block {b} ({wrong.sum()} wrong)', fontsize=9)
for ax in axs:
    ax.set_xlabel('cells 0..+4 (5 bits)'); ax.set_ylabel('cells -4..-1'); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f'{FIG}/f4_table.png'); plt.close(fig)

# F5 probes
sel = [1, 2, 8, 128]; fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
for ax, b in zip(axs, sel):
    im = ax.imshow(d['PR'][PB.index(b)] * 100, cmap='viridis', vmin=50, vmax=100, origin='lower')
    for t in range(5):
        for o in range(9):
            ax.text(o, t, f"{d['PR'][PB.index(b)][t, o]*100:.0f}", ha='center', va='center', fontsize=6, color='w' if d['PR'][PB.index(b)][t, o] < .8 else 'k')
    ax.set_xticks(range(9)); ax.set_xticklabels(range(-4, 5)); ax.set_yticks(range(5)); ax.set_xlabel('cell offset'); ax.set_ylabel('CA step t')
    ax.set_title(f'block {b}', fontsize=9)
fig.colorbar(im, ax=axs, fraction=0.015, label='linear probe accuracy (%)'); fig.savefig(f'{FIG}/f5_probe.png', bbox_inches='tight'); plt.close(fig)

# F6 FFN neurons (block 2)
k = NB.index(2); fs = d['f_spec'][k]                            # [512 S, 256]
e = fs ** 2; e[0] = 0; tot = e.sum(0) + 1e-12
inf = np.array([[e[[s for s in range(512) if s >> p & 1], n].sum() / tot[n] for p in range(9)] for n in range(256)])
dg = np.array([[e[deg == j, n].sum() / tot[n] for j in range(1, 8)] for n in range(256)])
com = (inf * np.arange(9)).sum(1) / inf.sum(1); o = np.argsort(com)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 0.85]))
im = axs[0].imshow(inf[o].T, aspect='auto', cmap='YlGnBu', origin='lower'); axs[0].set_yticks(range(9)); axs[0].set_yticklabels(range(-4, 5))
axs[0].set_ylabel('cell offset'); axs[0].set_xlabel('FFN unit (sorted by where it reads)'); axs[0].set_title('share of each unit\'s Walsh energy involving the cell', fontsize=9)
fig.colorbar(im, ax=axs[0], fraction=0.03)
im = axs[1].imshow(dg[o].T, aspect='auto', cmap='OrRd', origin='lower', extent=(0, 256, 0.5, 7.5)); axs[1].set_ylabel('interaction degree')
axs[1].set_xlabel('FFN unit (same order)'); axs[1].set_title('energy by degree per unit', fontsize=9); fig.colorbar(im, ax=axs[1], fraction=0.03)
fig.tight_layout(); fig.savefig(f'{FIG}/f6_neurons.png'); plt.close(fig)
np.save(f'{R}/neuron_inf.npy', inf)
# example units: the 8 with the largest std
top = np.argsort(-d['f_std'][k])[:8]; fig, axs = plt.subplots(2, 4, figsize=(12, 4.6))
for ax, n in zip(axs.flat, top):
    v = d['FM'][k][:, n]; ax.imshow(grid(v), cmap='RdBu_r', vmin=-abs(v).max(), vmax=abs(v).max()); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'unit {n}: offsets ' + ','.join(str(p - 4) for p in np.argsort(-inf[n])[:3]), fontsize=8)
fig.suptitle('mean activation over the 512 windows (same layout as the truth table)', fontsize=9); fig.tight_layout(); fig.savefig(f'{FIG}/f7_units.png'); plt.close(fig)

# F8 read kernel
fig, ax = plt.subplots(figsize=(6.5, 3.2)); cmk = plt.cm.plasma(np.linspace(0, 0.85, len(KB)))
for c, b, kr in zip(cmk, KB, d['KER']):
    ax.plot(range(-8, 9), kr, 'o-', color=c, ms=3, label=f'block {b}')
ax.axhline(0, color='0.7', lw=0.8); ax.set_xlabel('key cell offset j - i'); ax.set_ylabel('mean q~_i · k~_j'); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/f8_kernel.png'); plt.close(fig)
print('dependence on offset -4 (mean unit share)', inf[:, 0].mean(), 'units with >5%', (inf[:, 0] > .05).sum())
print('unit mean degree', (dg * np.arange(1, 8)).sum(1).mean())
