# Equal real-rank comparison: pure sine modes (rank 2 each) vs odd damped modes
# {sinh(l d)cos(w d), cosh(l d)sin(w d)} (rank 4 each), fit on d in [-pi,pi].
import numpy as np
from scipy.optimize import minimize
x=np.linspace(-np.pi,np.pi,4001); m=x>0; xp=x[m]
targets={'tanh*sech':lambda d:np.tanh(d)/np.cosh(d),
         'eps=0.5 current':lambda d:(lambda r:d/r*np.exp(-r))(np.sqrt(d*d+.25))}
def basis_sine(p): return [np.sin(w*xp) for w in p]
def basis_damp(p):
    out=[]
    for l,w in p.reshape(-1,2):
        out+= [np.sinh(l*xp)*np.cos(w*xp), np.cosh(l*xp)*np.sin(w*xp)]
    return out
def fit(f,basis,npar,lo,hi,seeds=40):
    y=f(xp); y=y/np.abs(y).max(); dy=np.gradient(f(x),x)[m]/np.abs(f(xp)).max()
    best=(1e9,None)
    for s in range(seeds):
        p0=np.random.default_rng(s).uniform(lo,hi,npar)
        def obj(p):
            B=np.stack(basis(p),1); c,*_=np.linalg.lstsq(B,y,rcond=None)
            h=1e-4; dB=(np.stack(basis(p),1)) # placeholder
            return np.mean((B@c-y)**2)
        r=minimize(obj,p0,method='Nelder-Mead',options=dict(maxiter=6000,xatol=1e-9,fatol=1e-16))
        if r.fun<best[0]: best=(r.fun,r.x)
    p=best[1]; B=np.stack(basis(p),1); c,*_=np.linalg.lstsq(B,y,rcond=None)
    fit_y=B@c; d_fit=np.gradient(np.concatenate([-fit_y[::-1],[0],fit_y]),x)[m]
    return np.sqrt(np.mean((fit_y-y)**2)), np.linalg.norm(d_fit-dy)/np.linalg.norm(dy), np.round(p,3), np.abs(c).max()
for n,f in targets.items():
    for rank in (4,8):
        e1=fit(f,basis_sine,rank//2,0.3,7)
        e2=fit(f,basis_damp,rank//2,0.0,3.5) if rank>=4 else None
        print(f'{n:16s} rank{rank}: sine RMSE={e1[0]*100:.3f}% dL2={e1[1]*100:.2f}% | damped RMSE={e2[0]*100:.3f}% dL2={e2[1]*100:.2f}% params={e2[2]} max|c|={e2[3]:.1f}')
