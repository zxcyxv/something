import numpy as np
exec(open(__file__.replace('dual2.py', 'dual.py')).read().split('# duality')[0])
Gf = lambda g: np.sign(g)*G(abs(g))
rng = np.random.default_rng(0); N = 60
def both(v, k):
    n = len(v)
    lhs = sum(L(s-t)*v[s]*k[t] for s in range(n) for t in range(n))
    dv, dk = np.diff(v, prepend=0), np.diff(k, prepend=0)
    return lhs, sum(Gf(s-t)*dv[s]*dk[t] for s in range(n) for t in range(n))
for tail in ('ends at 0', 'held constant for 40 blocks'):
    for _ in range(2):
        v, k = rng.normal(size=N).cumsum(), rng.normal(size=N).cumsum()
        if tail == 'ends at 0': v, k = np.r_[v, 0, np.zeros(40)], np.r_[k, 0, np.zeros(40)]
        else: v, k = np.r_[v, np.full(40, v[-1])], np.r_[k, np.full(40, k[-1])]
        print(tail, 'lhs %.5f rhs %.5f' % both(v, k))
