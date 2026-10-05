"""Test a predeclared mechanism on identical recorded K/V and phase states.

Prediction: finite phase perturbations that cross a discontinuity produce a
jump absent from the local derivative. Continuous windows should have smaller
Taylor residuals. This tests the operator, not a causal explanation of training.
"""
import json

import torch

from .research_free_phase_windows import DEST, sine_window, sine_derivative
from .biexponential_phase_window import window as biexp, derivative as dbiexp


def main():
    torch.set_num_threads(2)
    data=torch.load(DEST/'activation_samples.pt',map_location='cpu',weights_only=False)
    fits=json.loads((DEST/'shape_study.json').read_text())['rows']
    fit=next(r for r in fits if r['family']=='optimized_frequencies' and r['epsilon']==.5 and r['modes']==4)
    omega=torch.tensor(fit['frequencies'],dtype=torch.float64)
    coeff=torch.tensor(fit['coefficients'],dtype=torch.float64)
    families={
        'signed_exp':(lambda x:x.sign()*(-x.abs()).exp(),lambda x:torch.where(x==0,0.,-(-x.abs()).exp())),
        'fourier4_e05':(lambda x:sine_window(x,omega,coeff),lambda x:sine_derivative(x,omega,coeff)),
        'biexponential':(biexp,dbiexp)}
    result=dict(prediction=__doc__,source_checkpoint=data['checkpoint'],source_step=data['step'],
                protocol='Same recorded raw states and per-neuron phase directions for every window; K,V,Q held fixed; FP64; two blocks, one example/head each.',rows=[])
    for sample in data['samples']:
        q,k,v,pk,pv=[sample[n][:1,:1].double() for n in ('q','k','v','pk','pv')]
        torch.manual_seed(85+sample['block'])
        uk,uv=torch.randn_like(pk),torch.randn_like(pv)
        delta=pv[..., :, None]-pk[..., None, :]
        direction=uv[..., :, None]-uk[..., None, :]
        activity=v[..., :, None]*k[..., None, :]
        for name,(f,df) in families.items():
            original=f(delta)
            slope=df(delta)*direction
            for eps in (1e-3,1e-4,1e-5):
                moved=f(delta+eps*direction)
                actual=moved-original
                predicted=eps*slope
                residual=actual-predicted
                dg=(activity*actual).mean(-3)
                predg=(activity*predicted).mean(-3)
                read=q@dg.transpose(-1,-2)
                predread=q@predg.transpose(-1,-2)
                result['rows'].append(dict(block=sample['block'],window=name,phase_noise_std=eps,
                    ordering_flip_fraction=float((delta*(delta+eps*direction)<0).double().mean()),
                    window_change_rms=float(actual.square().mean().sqrt()),
                    window_taylor_residual_rms=float(residual.square().mean().sqrt()),
                    window_change_max=float(actual.abs().max()),
                    read_change_norm=float(read.norm()),
                    read_taylor_relative_residual=float((read-predread).norm()/read.norm().clamp_min(1e-30))))
    (DEST/'window_linearization.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:
        if r['phase_noise_std']==1e-4:print(r)


if __name__=='__main__':main()
