import numpy as np
lam, w = 0.5, 0.34
L = lambda d: np.sign(d)*(1-lam)*lam**abs(d)*np.sin(abs(d)*w)   # odd window, latency c=0
D = np.arange(1, 200)
print('sum L(D>=1)=%.4f  L(1)=%.4f L(8)=%.4f  m1=%.4f' % (L(D).sum(), L(1), L(8), 2*(D*L(D)).sum()))
G = lambda g: sum(min(d, g)*L(d) for d in D)
print('G:', [round(G(g), 3) for g in (1,2,3,4,6,8)])
# duality on random trajectories (finite, states start from 0 before n=0)
rng = np.random.default_rng(0); N = 60
for _ in range(3):
    v, k = rng.normal(size=N).cumsum(), rng.normal(size=N).cumsum()
    lhs = sum(L(s-t)*v[s]*k[t] for s in range(N) for t in range(N))
    dv, dk = np.diff(v, prepend=0), np.diff(k, prepend=0)
    Gf = lambda g: np.sign(g)*G(abs(g))
    rhs = sum(Gf(s-t)*dv[s]*dk[t] for s in range(N) for t in range(N))
    print('dual lhs %.6f rhs %.6f' % (lhs, rhs))
# model A reads only the per-block write Im S_b (pairs whose later member is block b), not the time integral
for g in (1, 3, 8):
    k = (np.arange(N) >= 20).astype(float); v = (np.arange(N) >= 20+g).astype(float)
    S = [sum(L(b-t)*v[b]*k[t] for t in range(b)) + sum(L(s-b)*v[s]*k[b] for s in range(b)) for b in range(N)]
    nz = [(b, round(x, 3)) for b, x in enumerate(S) if abs(x) > 1e-3]
    print(f'g={g}: per-block Im S_b', nz[:8], ' sum=%.3f' % sum(S))
