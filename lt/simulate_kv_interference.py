"""CPU toy measurements of associative/STDP memory interference, without training.

Targets are explicitly chosen associations, not Sudoku labels or Hopfield attractors.
All vectors have unit norm. The experiments isolate read interference before feedback.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def outer(v, k):
    return v[..., :, None] * k[..., None, :]


def summary(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))


def metrics(signal, history):
    total = signal + history
    sn = np.linalg.norm(signal, axis=-1)
    tn = np.linalg.norm(total, axis=-1)
    hn = np.linalg.norm(history, axis=-1)
    projection = (history * signal).sum(-1) / sn**2
    cosine = (total * signal).sum(-1) / np.maximum(tn * sn, 1e-30)
    return dict(history_to_signal_rms=float(np.sqrt(np.mean(hn**2 / sn**2))),
                relative_read_error=summary(hn / sn), cosine_to_target=summary(cosine),
                history_signed_projection=summary(projection),
                negative_history_projection_fraction=float((projection < 0).mean()))


def stdp_step(k, v, ek, ev, lam):
    g = outer(v, ek) - outer(ev, k)
    return g, lam * ek + (1-lam) * k, lam * ev + (1-lam) * v


def verify():
    rng = np.random.default_rng(1)
    k, v = rng.normal(size=7), rng.normal(size=7)
    z = np.zeros(7)
    g1, ek, ev = stdp_step(k, z, z, z, .1)
    g2, _, _ = stdp_step(z, v, ek, ev, .1)
    timed_error = float(np.max(np.abs(g1 + g2 - .9 * outer(v, k))))
    kc, vc = ek + 1j*z, ev + 1j*v
    complex_error = float(np.max(np.abs(g2 - np.imag(outer(vc, np.conj(kc))))))
    assert timed_error < 1e-12 and complex_error < 1e-12
    return dict(timed_pair_STDP_error=timed_error, complex_product_error=complex_error)


def load_sweep(rng, trials):
    rows = []
    for d in (16, 32, 64, 104):
        for c in (0., .5, .9):
            q = unit(rng.normal(size=(trials, d)))
            target = unit(rng.normal(size=(trials, d)))
            memory = outer(target, q)
            history_read = np.zeros_like(target)
            endpoints = sorted(set([0, 1, d//8, d//2, d, 2*d, 4*d]))
            for n in range(4*d + 1):
                if n:
                    noise = rng.normal(size=(trials, d))
                    noise -= (noise*q).sum(-1, keepdims=True)*q
                    # c is the common component; c=0 uses independent isotropic keys.
                    key = unit(rng.normal(size=(trials, d))) if c == 0 else c*q + np.sqrt(1-c*c)*unit(noise)
                    value = unit(rng.normal(size=(trials, d)))
                    memory += outer(value, key)
                    history_read += value*(key*q).sum(-1, keepdims=True)
                if n in endpoints:
                    sv = np.linalg.svd(memory[:min(8,trials)], compute_uv=False)
                    numeric_rank = (sv > sv[:, :1]*1e-10).sum(-1)
                    effective_rank = (sv**2).sum(-1)**2 / (sv**4).sum(-1)
                    direct_error = np.max(np.abs(np.einsum('bij,bj->bi', memory, q)-target-history_read))
                    assert direct_error < 1e-10
                    # Each association can be written by a strict-past STDP key pulse
                    # then value pulse, with zeroed eligibility between episodes.
                    # G=.9*v*k^T, so read with q/.9 gives exactly this measurement.
                    rows.append(dict(d=d, past_associations=n, key_common_component=c,
                        independent_key_expected_noise_rms=float(np.sqrt(n/d)) if c==0 else None,
                        numeric_rank_mean=float(numeric_rank.mean()),
                        participation_rank_mean=float(effective_rank.mean()),
                        reconstruction_error=float(direct_error), **metrics(target,history_read)))
    return rows


def coherent_examples(rng, trials, d=32, n=32):
    q = unit(rng.normal(size=(trials,d)))
    v = unit(rng.normal(size=(trials,d)))
    orth = rng.normal(size=(trials,d)); orth -= (orth*v).sum(-1,keepdims=True)*v
    orth = unit(orth)
    cases = {}
    for name, history in [('helpful_same_association',n*v),
                           ('conflicting_same_key',-n*v),
                           ('orthogonal_value_same_key',n*orth)]:
        cases[name] = dict(d=d,past_associations=n,memory_rank_upper_bound=2,
                          **metrics(v,history))
    return cases


def cyclic_stdp(rng, trials):
    rows = []
    d = 16
    kcycle = unit(rng.normal(size=(2,trials,d)))
    vcycle = unit(rng.normal(size=(2,trials,d)))
    q = unit(rng.normal(size=(trials,d)))
    for lam in (.1,.9):
        for prefix in (0,16,64,256):
            ek=np.zeros((trials,d));ev=ek.copy();memory=np.zeros((trials,d,d))
            for _ in range(prefix):
                k=unit(rng.normal(size=(trials,d)));v=unit(rng.normal(size=(trials,d)))
                g,ek,ev=stdp_step(k,v,ek,ev,lam);memory+=g
            reads=[];writes=[];operators=[];errors=[]
            # 512 blocks wash out trace differences even at lambda=.9.
            for r in range(512):
                g,ek,ev=stdp_step(kcycle[r%2],vcycle[r%2],ek,ev,lam)
                memory+=g
                if r>=480:
                    signal=np.einsum('bij,bj->bi',g,q)
                    past=np.einsum('bij,bj->bi',memory-g,q)
                    reads.append(signal+past);writes.append(signal);operators.append(memory.copy())
                    errors.append(np.max(np.abs(signal+past-np.einsum('bij,bj->bi',memory,q))))
            reads=np.stack(reads);writes=np.stack(writes);operators=np.stack(operators)
            mean_m=operators.mean(0)
            cycle_mean_read=np.einsum('bij,bj->bi',mean_m,q)
            write_energy=np.mean(np.sum(writes**2,axis=-1))
            rows.append(dict(d=d,lambda_value=lam,prefix_blocks=prefix,
                tail_G_read_rms=float(np.sqrt(write_energy)),
                tail_M_read_rms=float(np.sqrt(np.mean(np.sum(reads**2,axis=-1)))),
                cycle_mean_M_read_to_G_rms=float(np.sqrt(np.mean(np.sum(cycle_mean_read**2,axis=-1))/write_energy)),
                M_alternating_read_to_G_rms=float(np.sqrt(np.mean(np.sum((reads-reads.mean(0))**2,axis=-1))/write_energy)),
                reconstruction_error=float(max(errors)),
                note='Same prescribed final cycle and fixed query; no target declares the persistent mean useful or harmful.'))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--trials',type=int,default=256);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261004)
    result=dict(seed=20261004,trials=args.trials,precision='float64',checks=verify(),
        scope='Prescribed unit-norm activities, independent trials, no optimizer, no neural feedback, no Sudoku. Not a Hopfield capacity measurement.',
        association_protocol='Current target v0 at q=k0; historical unit keys/values accumulated as outer products. Exact equivalent to isolated lambda=.1 STDP key-then-value episodes with trace reset and q/.9. G-only retrieves only the latest association by design.',
        load_sweep=load_sweep(rng,args.trials),coherent_examples=coherent_examples(rng,args.trials),
        continuous_STDP_cycle=cyclic_stdp(rng,args.trials))
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    flat=[]
    for row in result['load_sweep']:
        item={k:v for k,v in row.items() if not isinstance(v,dict)}
        for k,v in row.items():
            if isinstance(v,dict):
                item[k+'_mean']=v['mean'];item[k+'_se']=v['se']
        flat.append(item)
    with (args.out/'load_sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for d in (16,32,64,104):
        rows=[r for r in result['load_sweep'] if r['d']==d and r['key_common_component']==0]
        axes[0].plot([r['past_associations']/d for r in rows],[r['history_to_signal_rms'] for r in rows],'o-',label=f'd={d}')
    x=np.linspace(0,4,100);axes[0].plot(x,np.sqrt(x),'k--',label='sqrt(history / d)')
    axes[0].set(xlabel='Past associations / channel dimension',ylabel='Interference / target RMS',title='Independent keys and values');axes[0].legend()
    for c in (0.,.5,.9):
        rows=[r for r in result['load_sweep'] if r['d']==104 and r['key_common_component']==c]
        axes[1].plot([r['past_associations'] for r in rows],[r['cosine_to_target']['mean'] for r in rows],'o-',label=f'key common component={c}')
    axes[1].set(xlabel='Past associations',ylabel='Mean cosine to current target',title='Read interference at d=104');axes[1].legend()
    fig.tight_layout();fig.savefig(args.out/'interference.png',dpi=160);plt.close(fig)
    print(json.dumps({'out':str(args.out),'checks':result['checks'],'rows':len(flat)},indent=2))


if __name__=='__main__':
    main()
