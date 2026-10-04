"""Independent confirmation of one exploratory fixed-activity STDP window.

The discovery sweep chose lp=.1, lm=.9, ap=am=1 for shared background.
This file freezes that choice and tests fresh banks; it never trains or retunes.
The same frozen Hopfield read is used for all three writing rules.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .simulate_associative_stdp import bipolar, hopfield_recall


def verify(seed,patterns,condition,trials=1024,dim=32):
    rng=np.random.default_rng(seed)
    bank=bipolar(rng,(trials,patterns,dim))
    target=bank[np.arange(trials),rng.integers(patterns,size=trials)]
    observed=rng.random((trials,dim))>.5
    observed[:,0]=False;observed[:,1]=True
    cue=np.where(observed,target,0.)
    h=rng.normal(size=bank.shape)
    background=.6*bipolar(rng,(trials,1,dim))
    streams=[]
    for _ in range(16):
        h=.6*h+.4*bank
        streams.append(h.copy()+background if condition=='shared_background' else h.copy())
    settings={'hebb':(.1,.1),'balanced_stdp':(.1,.1),'selected_stdp':(.1,.9)}
    results={};per_sample={}
    reference=np.einsum('bpi,bpj->bij',bank,bank)/dim
    reference[:,np.arange(dim),np.arange(dim)]=0
    for rule,(lp,lm) in settings.items():
        ek=np.zeros_like(bank);ev=ek.copy();m=np.zeros((trials,dim,dim))
        for x in streams:
            g=np.einsum('bpi,bpj->bij',x,x) if rule=='hebb' else (
                np.einsum('bpi,bpj->bij',x,ek)-np.einsum('bpi,bpj->bij',ev,x))
            m+=g/(dim*len(streams))
            ek=lp*ek+(1-lp)*x;ev=lm*ev+(1-lm)*x
        m[:,np.arange(dim),np.arange(dim)]=0
        pred=hopfield_recall(m,cue,observed)[-1]
        acc=((pred==target)*(~observed)).sum(-1)/(~observed).sum(-1)
        per_sample[rule]=acc
        cosine=(m*reference).sum((1,2))/(np.linalg.norm(m,axis=(1,2)).clip(1e-15)
            *np.linalg.norm(reference,axis=(1,2)).clip(1e-15))
        results[rule]=dict(hidden_accuracy=float(acc.mean()),exact=float((pred==target).all(-1).mean()),
                           clean_memory_cosine=float(cosine.mean()))
    delta=per_sample['selected_stdp']-per_sample['hebb']
    mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(trials))
    return dict(seed=seed,patterns=patterns,condition=condition,trials=trials,
                rules=results,paired_accuracy_difference=mean,paired_se=se,
                paired_approx_95_interval=[mean-1.96*se,mean+1.96*se])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--trials',type=int,default=1024)
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise SystemExit('Use a new output directory.')
    source=Path(__file__).read_bytes();(out/'source_snapshot.py').write_bytes(source)
    protocol=dict(selection='Window selected on discovery fixed_results.json; held-out evaluation never retunes',
        window=dict(lambda_plus=.1,lambda_minus=.9,amplitude_plus=1,amplitude_minus=1),
        axis='Internal settling, all patterns simultaneous',seeds=[71001,71002,71003],trials=args.trials,
        conditions=['settling','shared_background'],source_sha256=hashlib.sha256(source).hexdigest(),
        interpretation='A conditional advantage tests a temporal filtering mechanism; it cannot establish universal STDP superiority or selective weakening in the learned programmer.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for seed in protocol['seeds']:
        for p in (4,8):
            for condition in protocol['conditions']:
                result=verify(seed+p*100,p,condition,args.trials)
                rows.append(result)
                print(json.dumps({k:result[k] for k in ('seed','patterns','condition','paired_accuracy_difference','paired_se')}),flush=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
