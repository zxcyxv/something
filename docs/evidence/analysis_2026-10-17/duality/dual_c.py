import numpy as np
lam, w, c = 0.5, 0.34, 0.14
rng = np.random.default_rng(1); N, P = 50, 30
K = lambda d: (1-lam)*lam**abs(d)*np.exp(1j*w*d)          # fixed complex window (Hermitian)
def direct(v, k, pv, pk):                                  # pair weight with per-event latency
    n = len(v); return sum((1-lam)*lam**abs(s-t)*np.sin(w*(s-t)+pv[s]-pk[t])*v[s]*k[t] for s in range(n) for t in range(n))
def change(v, k, pv, pk):                                  # change domain on latency-modulated flows
    n = len(v); fv, fk = v*np.exp(1j*pv), k*np.exp(-1j*pk)
    dv, dk = np.diff(fv, prepend=0), np.diff(fk, prepend=0)
    # brute force: sum_{s,t} K(s-t) fv_s fk_t = sum_{a,b} dv_a dk_b * sum_{s>=a, t>=b} K(s-t)
    tot = 0
    for a in range(n):
        for b in range(n):
            if dv[a] == 0 or dk[b] == 0: continue
            tot += dv[a]*dk[b]*sum(K(s-t) for s in range(a, n) for t in range(b, n))
    return tot.imag
for kind in ('random walk + latency', 'constant state (fixed point) + latency'):
    if kind.startswith('random'):
        v, k = rng.normal(size=N).cumsum(), rng.normal(size=N).cumsum()
        pv, pk = c*np.sin(rng.normal(size=N).cumsum()), c*np.sin(rng.normal(size=N).cumsum())
    else:
        v, k, pv, pk = np.ones(N), np.ones(N), np.full(N, 0.1), np.full(N, -0.05)
    pad = lambda x: np.r_[x, np.zeros(P)]
    print(kind, 'direct %.12f  change %.12f' % (direct(*map(pad, (v, k, pv, pk))), change(*map(pad, (v, k, pv, pk)))))
