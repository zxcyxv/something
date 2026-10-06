# Sign-guaranteed parametrization: L(d) = s * sin(w0 d) * |sum_k a_k e^{i k w0 d}|^2  (>=0 on (0,pi) if w0<=1)
# expands to N+1 sine modes at frequencies w0*{1..N+1}/... ; fit to tanh*sech, slope normalised.
import numpy as np
from scipy.optimize import minimize
x=np.linspace(-np.pi,np.pi,4001); m=x>0; xp=x[m]
y=np.tanh(xp)/np.cosh(xp); dy=(1/np.cosh(xp))**2*(1-2*np.sinh(xp)**2)/1  # d/dx tanh*sech = sech^3 - tanh^2 sech
dy=1/np.cosh(xp)**3-np.tanh(xp)**2/np.cosh(xp)
def L(p,N,w0,d):
    a=p[:N+1]+1j*np.r_[0,p[N+1:]]
    q=sum(a[k]*np.exp(1j*k*w0*d) for k in range(N+1))
    return np.sin(w0*d)*np.abs(q)**2
for w0 in (0.5,0.75,1.0):
  for N in (1,2,3):
    best=None
    for s in range(40):
        p0=np.random.default_rng(s).normal(size=2*N+1)
        r=minimize(lambda p:np.mean((L(p,N,w0,xp)-y)**2),p0,method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-10,fatol=1e-16))
        if best is None or r.fun<best.fun: best=r
    f=L(best.x,N,w0,xp); h=1e-6; df=(L(best.x,N,w0,xp+h)-L(best.x,N,w0,xp-h))/2/h
    print(f'w0={w0} N={N} modes={N+1} rank={2*(N+1)}: RMSE={np.sqrt(np.mean((f-y)**2))/0.5*100:.3f}% dL2={np.linalg.norm(df-dy)/np.linalg.norm(dy)*100:.2f}% min_on_(0,pi)={f.min():.2e}')
