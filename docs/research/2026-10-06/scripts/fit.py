import numpy as np
from scipy.optimize import minimize
x=np.linspace(-np.pi,np.pi,4001); xp=x[x>0]
def eps_win(d,e=0.5):
    r=np.sqrt(d*d+e*e); y=d/r*np.exp(-r); return y/np.abs(y).max()
cands={
 'tanh(x/0.5)*sech(x) sharp':lambda d:np.tanh(d/0.5)/np.cosh(d),




}
def fit(f,R):
    y=f(xp); dy=np.gradient(f(x),x)[x>0]
    best=None
    for seed in range(30):
        rng=np.random.default_rng(seed); w0=np.sort(rng.uniform(0.3,7,R))
        def obj(w):
            A=np.sin(np.outer(xp,w)); c,*_=np.linalg.lstsq(A,y,rcond=None)
            dA=np.cos(np.outer(xp,w))*w
            return np.mean((A@c-y)**2)+0.05*np.mean((dA@c-dy)**2)
        r=minimize(obj,w0,method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-8,fatol=1e-14))
        if best is None or r.fun<best.fun: best=r
    w=best.x; A=np.sin(np.outer(xp,w)); c,*_=np.linalg.lstsq(A,y,rcond=None)
    dA=np.cos(np.outer(xp,w))*w
    pk=np.abs(y).max()
    return np.sqrt(np.mean((A@c-y)**2))/pk, np.linalg.norm(dA@c-dy)/np.linalg.norm(dy)
for n,f in cands.items():
    y=f(x); d=np.gradient(y,x); d2=np.gradient(d,x)
    i=np.argmax(y)
    print(f'{n:28s} peak={y.max():.3f}@{x[i]:.2f} slope0={d[2000]:.2f} max|L\"|={np.abs(d2[5:-5]).max():.2f}')
    for R in (2,3,4):
        e,de=fit(f,R); print(f'    R={R}: rel RMSE={e*100:6.3f}%  deriv rel L2={de*100:6.2f}%')
