"""Numerical checks for the pair-STDP reading of the pairangle write (theory review 2026-10-12)."""
import math
import numpy as np

tau = 2.0                                     # tau_phi = omega * tau (phase units)
L = lambda s: np.tanh(s / tau) / np.cosh(s / tau)   # odd tanh*sech window, s = t_post - t_pre (phase units)

# 1. periodization: all spike pairings of two phase-locked trains -> sine series
r = np.arange(1, 9)
b = tau ** 2 * r / np.cosh(math.pi * r * tau / 2)
d = np.linspace(-math.pi, math.pi, 2001)
K = np.arange(-60, 61)
WT = L(d[:, None] + 2 * math.pi * K[None]).sum(1)                 # sum over cycle offsets k
series = (b[None] * np.sin(d[:, None] * r[None])).sum(1)
c = (WT @ series) / (series @ series)
print('1. periodized sum = c * sum_r b_r sin(r d): c = %.6f (1/pi = %.6f), max rel err %.2e'
      % (c, 1 / math.pi, np.abs(WT - c * series).max() / np.abs(WT).max()))
scale = 2.8566329144042717                    # model: coefficients scaled so that peak = 1
coef = scale * b[:2]
print('   model coefficients (R=2):', np.round(coef, 6), ' b2/b1 = %.4f' % (b[1] / b[0]))
W2 = lambda x: coef[0] * np.sin(x) + coef[1] * np.sin(2 * x)
WTn = WT / np.abs(WT).max()
for R in (1, 2, 3):
    s = (b[None, :R] * np.sin(d[:, None] * r[None, :R])).sum(1)
    print('   truncation R=%d: max |normalized WT - normalized series| = %.4f' % (R, np.abs(WTn - s / np.abs(s).max()).max()))
print('   share of the window value carried by cross-cycle pairings (|k|>=1) at d=0.4: %.3f'
      % (1 - L(0.4) / L(0.4 + 2 * math.pi * K).sum()))

# 2. explicit spike trains over N cycles -> per-cycle weight change equals the periodized window
def pair_sum(phi_pre, phi_post, N=400, rho_pre=1.0, rho_post=1.0):
    tp = phi_pre + 2 * math.pi * np.arange(N)
    tq = phi_post + 2 * math.pi * np.arange(N)
    return rho_pre * rho_post * L(tq[:, None] - tp[None]).sum() / N
for (pk, pv) in ((0.6, 0.3), (0.7, 1.1)):
    print('2. pre %.1f, post %.1f: per-cycle pair sum %.6f vs c*W_T %.6f'
          % (pk, pv, pair_sum(pk, pv), c * (b * np.sin((pv - pk) * r)).sum()))

# 3. the per-token 2x2 block is (scalar STDP weight) x (phase carrier); the read copies V's phase
rng = np.random.default_rng(0)
zv, zk, zq = (rng.normal(size=2) @ np.array([1, 1j]) for _ in range(3))
g = 1 + 1.3 * W2(np.angle(zv) - np.angle(zk))
uv, uk, uq = (np.array([z.real, z.imag]) for z in (zv, zk, zq))
block = np.outer(uv, uk) * g
ev, ek = uv / abs(zv), uk / abs(zk)
print('3. block - |v||k| g e_v e_k^T:', np.abs(block - abs(zv) * abs(zk) * g * np.outer(ev, ek)).max())
out = block @ uq
print('   read output phase - phi_V (mod pi):', (np.angle(out[0] + 1j * out[1]) - np.angle(zv) + math.pi / 2) % math.pi - math.pi / 2,
      '; amplitude / (|v| g Re(conj zk zq)):', np.hypot(*out) / abs(abs(zv) * g * (np.conj(zk) * zq).real))
cpart = 0.5 * zv * np.conj(zk) * g; dpart = 0.5 * zv * zk * g
print('   complex-linear + antilinear parts reproduce the read:', abs((cpart * zq + dpart * np.conj(zq)) - (out[0] + 1j * out[1])))

# 4. the user's example with the trained alphas
alpha100k = np.array([1.292, 1.314, 1.211, 1.307, 1.273, 1.306, 1.901, 1.423])
for name, delta in (('block b   (K 0.6, V 0.3)', 0.3 - 0.6), ('block b+1 (K 0.7, V 1.1)', 1.1 - 0.7)):
    w = W2(delta)
    print('4. %s: delta %+.1f, W %+.4f, gain 1+aW over heads [%.3f, %.3f], head 6 %.3f'
          % (name, delta, w, 1 + alpha100k.min() * w if w > 0 else 1 + alpha100k.max() * w,
             1 + alpha100k.max() * w if w > 0 else 1 + alpha100k.min() * w, 1 + alpha100k[6] * w))
for name, (pk, pv) in (('swapped roles: K 0.3->1.1, V 0.6->0.7', ((0.3, 1.1), (0.6, 0.7))),):
    print('   %s: delta %+.1f -> %+.1f' % (name, pv[0] - pk[0], pv[1] - pk[1]))
neg = d[1 + alpha100k.min() * W2(d) < 0]
print('   gain < 0 needs delta in [%.2f, %.2f] (alpha %.2f) / [%.2f, %.2f] (alpha %.2f)'
      % (neg.min(), neg.max(), alpha100k.min(), d[1 + alpha100k.max() * W2(d) < 0].min(), d[1 + alpha100k.max() * W2(d) < 0].max(), alpha100k.max()))

# 5. if a block is one cycle (T = 2 pi) and phases move between blocks, the true pairings
T = 2 * math.pi
pre = np.array([0.6, T + 0.7]); post = np.array([0.3, T + 1.1])
pairs = post[:, None] - pre[None]
print('5. lags (post - pre), rows post b, b+1; cols pre b, b+1:\n', np.round(pairs, 3))
print('   L of each pairing:\n', np.round(L(pairs), 4))
print('   same-block (stationary assumption) per-block windows: %.4f, %.4f; true cross-block pairs add %.4f'
      % (c * (b * np.sin(-0.3 * r)).sum(), c * (b * np.sin(0.4 * r)).sum(), L(pairs[1, 0]) + L(pairs[0, 1])))

# 6. carriers: the two observations store different rank-1 directions
e = lambda p: np.array([math.cos(p), math.sin(p)])
Cb, Cb1 = np.outer(e(0.3), e(0.6)), np.outer(e(1.1), e(0.7))
print('6. carrier overlap <C_b, C_b+1>_F = %.4f (= cos(0.8) cos(0.1) = %.4f)'
      % ((Cb * Cb1).sum(), math.cos(0.8) * math.cos(0.1)))
for lam in (0.0, 0.5, 0.9):
    Gt = lam * W2(-0.3) * Cb + W2(0.4) * Cb1            # timing component only, leaky accumulation
    print('   timing part with leak %.1f: projection on C_b+1 %+.4f, on C_b %+.4f' % (lam, (Gt * Cb1).sum(), (Gt * Cb).sum()))
