"""Exact lag trace S = sum_j w_j a_{b-j} c_{b-j}^H (w_j = eta (1-eta)^j), error of Im S (R=1 timing term).
per-neuron : (sum_j w_j a_j)(sum_l w_l c_l)^H                   (current trace)
co-rotated : same after removing the known common rotation
recent K   : exact sum over j < K + (1-eta)^K * per-neuron product of the normalized tail
"""
import numpy as np
rng = np.random.default_rng(0)
P, B, eta = 16, 300, 0.2
w = eta * (1 - eta) ** np.arange(B)

def rev(x): return x[::-1]
def exact(a, c): return np.einsum('j,ja,jb->ab', w, rev(a), rev(c).conj())
def per_neuron(a, c, rot=None):
    ar, cr = rev(a), rev(c)
    if rot is not None:
        cum = np.exp(1j * np.concatenate(([0.], np.cumsum(rev(rot)[:-1]))))[:, None]
        ar, cr = ar * cum, cr * cum
    return np.outer(w @ ar, (w @ cr).conj())
def recent(a, c, K):
    ar, cr = rev(a), rev(c)
    head = np.einsum('j,ja,jb->ab', w[:K], ar[:K], cr[:K].conj())
    wt = w[K:] / w[K:].sum()
    tail = np.outer(wt @ ar[K:], (wt @ cr[K:]).conj()) * w[K:].sum()
    return head + tail
def err(s, ref): return np.linalg.norm(s.imag - ref.imag) / np.linalg.norm(ref.imag)

def scenario(name, phiV, phiK, rot=None):
    a, c = np.exp(1j * phiV), np.exp(1j * phiK)
    ref = exact(a, c)
    cols = [err(per_neuron(a, c), ref), err(per_neuron(a, c, rot), ref) if rot is not None else float('nan')]
    cols += [err(recent(a, c, K), ref) for K in (1, 2, 4, 8)]
    print(f"{name:<40} per-neuron {cols[0]:.3f}  co-rot {cols[1]:.3f}  recent K=1 {cols[2]:.3f}  K=2 {cols[3]:.3f}  K=4 {cols[4]:.3f}  K=8 {cols[5]:.3f}")

base = rng.uniform(-np.pi, np.pi, (2, P))
t = np.arange(B)[:, None]
scenario('stationary', base[0] + 0 * t, base[1] + 0 * t)
scenario('common drift 0.3 rad/block', base[0] + .3 * t, base[1] + .3 * t, rot=np.full(B, .3))
x, y = rng.uniform(-np.pi, np.pi, (2, 2, P))
alt = (np.arange(B) % 2)[:, None]
scenario('period-2 alternation', np.where(alt, x[0], y[0]), np.where(alt, x[1], y[1]))
z3 = rng.uniform(-np.pi, np.pi, (3, 2, P)); m3 = np.arange(B) % 3
scenario('period-3 cycle', z3[m3, 0], z3[m3, 1])
for s in (0.05, 0.15, 0.3):
    walk = np.cumsum(rng.normal(0, s, (B, 2, P)), 0)
    scenario(f'independent random walk {s}', base[0] + walk[:, 0], base[1] + walk[:, 1])
common = rng.normal(0, .15, B)
walk = np.cumsum(rng.normal(0, .05, (B, 2, P)), 0)
cc = np.cumsum(common)[:, None]
scenario('common walk 0.15 + own walk 0.05', base[0] + walk[:, 0] + cc, base[1] + walk[:, 1] + cc,
         rot=np.concatenate((common[1:], [0.])))

print('\n-- polyphase: one per-neuron trace per block-index class (b mod F), each normalized by its class weight --')
def polyphase(a, c, F, rot=None):
    ar, cr = rev(a), rev(c)
    if rot is not None:
        cum = np.exp(1j * np.concatenate(([0.], np.cumsum(rev(rot)[:-1]))))[:, None]
        ar, cr = ar * cum, cr * cum
    out = 0
    for p in range(F):
        m = (np.arange(B) % F) == p
        W = w[m].sum()
        out = out + np.outer(w[m] @ ar[m], (w[m] @ cr[m]).conj()) / W
    return out

def scen2(name, phiV, phiK, rot=None):
    a, c = np.exp(1j * phiV), np.exp(1j * phiK)
    ref = exact(a, c)
    cols = [err(polyphase(a, c, F), ref) for F in (1, 2, 3, 4, 6)]
    cr = [err(polyphase(a, c, F, rot), ref) for F in (1, 2, 6)] if rot is not None else [float('nan')] * 3
    print(f"{name:<40} F=1 {cols[0]:.3f}  F=2 {cols[1]:.3f}  F=3 {cols[2]:.3f}  F=4 {cols[3]:.3f}  F=6 {cols[4]:.3f}"
          f"   +co-rot F=1 {cr[0]:.3f} F=2 {cr[1]:.3f} F=6 {cr[2]:.3f}")

scen2('common drift 0.3 rad/block', base[0] + .3 * t, base[1] + .3 * t, rot=np.full(B, .3))
scen2('period-2 alternation', np.where(alt, x[0], y[0]), np.where(alt, x[1], y[1]))
scen2('period-3 cycle', z3[m3, 0], z3[m3, 1])
z4 = rng.uniform(-np.pi, np.pi, (4, 2, P)); m4 = np.arange(B) % 4
scen2('period-4 cycle', z4[m4, 0], z4[m4, 1])
scen2('period-2 + common drift 0.3', np.where(alt, x[0], y[0]) + .3 * t, np.where(alt, x[1], y[1]) + .3 * t, rot=np.full(B, .3))
noisy = np.where(alt, x[0], y[0]) + np.cumsum(rng.normal(0, .1, (B, P)), 0)
noisk = np.where(alt, x[1], y[1]) + np.cumsum(rng.normal(0, .1, (B, P)), 0)
scen2('period-2 + own walk 0.1', noisy, noisk)
for s in (0.15, 0.3):
    walk = np.cumsum(rng.normal(0, s, (B, 2, P)), 0)
    scen2(f'independent random walk {s}', base[0] + walk[:, 0], base[1] + walk[:, 1])
