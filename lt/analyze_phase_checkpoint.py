"""Inspect learned phase offsets; no inference-time temporal aliasing claim."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch


def stats(x):
    x=np.asarray(x).ravel()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),
                std=float(x.std()),quantiles=dict(zip(['p01','p05','p25','p50','p75','p95','p99'],
                map(float,np.quantile(x,[.01,.05,.25,.5,.75,.95,.99])))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint'); ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),step=ck['step'],
                units='radians',scope='parameter phase offsets; signed activity phase flips not included',
                temporal_aliasing='not measurable: no sampled oscillator exists in this implementation',states={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,state in enumerate(['raw_model_state_dict','model_state_dict']):
        sd=ck[state]; layers={}
        for name,value in sd.items():
            if not name.endswith('theta_k_raw'): continue
            pk=(math.pi/2*value.double().tanh()).numpy()
            pv=(math.pi/2*sd[name.replace('theta_k_raw','theta_v_raw')].double().tanh()).numpy()
            # Only channels within the same head connect through the fast matrix.
            delta=pv[:,:,None]-pk[:,None,:]
            ad=np.abs(delta)
            entry=dict(K=stats(pk),V=stats(pv),delta=stats(delta),
                       fraction_abs_delta_over_pi_2=float((ad>math.pi/2).mean()),
                       fraction_abs_delta_over_0_9pi=float((ad>.9*math.pi).mean()),
                       fraction_phase_near_bound=float((np.concatenate([np.abs(pk).ravel(),np.abs(pv).ravel()]) > .9*math.pi/2).mean()),
                       fraction_abs_sine_under_0_1=float((np.abs(np.sin(delta))<.1).mean()),
                       positive_delta_fraction=float((delta>0).mean()),
                       per_head=[dict(K=stats(k),V=stats(v),delta=stats(d),
                           fraction_abs_delta_over_pi_2=float((np.abs(d)>math.pi/2).mean()))
                           for k,v,d in zip(pk,pv,delta)])
            layers[name.rsplit('.',1)[0]]=entry
            for col,(x,label,bounds) in enumerate([(pk,'K phase',(-math.pi/2,math.pi/2)),
                                                      (pv,'V phase',(-math.pi/2,math.pi/2)),
                                                      (delta,'V-K phase difference',(-math.pi,math.pi))]):
                axes[row,col].hist(x.ravel(),bins=60,range=bounds,density=True,alpha=.7)
                axes[row,col].set_title(state+': '+label)
                axes[row,col].set_xlabel('radians')
                axes[row,col].axvline(0,color='black',lw=.7)
                if col==2:
                    for b in [-math.pi/2,math.pi/2]:axes[row,col].axvline(b,color='red',ls='--')
        report['states'][state]=layers
    fig.suptitle('Phase offsets and within-head pair differences; step '+str(ck['step']))
    fig.tight_layout(); fig.savefig(out/'phase_distribution.png',dpi=160);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
