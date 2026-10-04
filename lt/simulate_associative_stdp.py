"""Paired associative-memory experiments along the INTERNAL recurrence axis.

Pattern lanes are simultaneous, never a causal token stream. The main evaluation
freezes memory after storage. Hebbian, project pair-STDP and general pair-STDP
share their reader and recurrence. No teacher error enters a plasticity update.
Run: python -m lt.simulate_associative_stdp --out runs/associative_stdp
"""
import argparse
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


RULES = ('hebb', 'balanced_stdp', 'general_stdp')


def outer(v, k):
    return v[..., :, None] * k[..., None, :]


def numpy_step(k, v, ek, ev, rule, lp=.1, lm=.1, ap=1., am=1.):
    """Strict-past, normalized exponential traces, same convention as project."""
    plus = outer(v, ek)
    minus = outer(ev, k)
    write = outer(v, k) if rule == 'hebb' else ap*plus-am*minus
    return write, lp*ek+(1-lp)*k, lm*ev+(1-lm)*v


def bipolar(rng, shape):
    return (2*rng.integers(0, 2, shape)-1).astype(np.float64)


def hopfield_recall(memory, cue, observed, iterations=12):
    """Synchronous recall; preserve supplied known bits, hold ties unchanged.

    Report cycles explicitly: synchronous Hopfield is not guaranteed to descend
    energy, especially for asymmetric matrices. Classical memory has zero diag.
    """
    x = cue.copy()
    trajectory = [x.copy()]
    for _ in range(iterations):
        field = np.einsum('bij,bj->bi', memory, x)
        proposed = np.where(field > 1e-12, 1., np.where(field < -1e-12, -1., x))
        x = np.where(observed, cue, proposed)
        trajectory.append(x.copy())
    return np.stack(trajectory)


def score_numpy(trajectory, target, observed):
    prediction = trajectory[-1]
    hidden = ~observed
    hits = prediction == target
    sample_acc = (hits*hidden).sum(-1)/hidden.sum(-1).clip(1)
    return dict(hidden_accuracy=float(sample_acc.mean()),
                unresolved_hidden_fraction=float(((prediction==0)*hidden).sum()/hidden.sum()),
                hidden_accuracy_se=float(sample_acc.std(ddof=1)/np.sqrt(len(sample_acc))),
                exact=float(hits.all(-1).mean()),
                fixed_point_fraction=float((trajectory[-1] == trajectory[-2]).all(-1).mean()),
                two_cycle_fraction=float(((trajectory[-1] == trajectory[-3]).all(-1)
                    & ~(trajectory[-1] == trajectory[-2]).all(-1)).mean()),
                hidden_accuracy_by_recurrence=[float(((x == target)*hidden).sum()/hidden.sum())
                                               for x in trajectory])


def mechanism_checks():
    """Analytic controls are NOT evidence of task-level STDP superiority."""
    rng = np.random.default_rng(123)
    k, v = rng.normal(size=(2, 12))
    z = np.zeros_like(k)
    _, ek, ev = numpy_step(k, z, z, z, 'balanced_stdp', lp=.7, lm=.7)
    g, _, _ = numpy_step(z, v, ek, ev, 'balanced_stdp', lp=.7, lm=.7)
    delay_error = float(np.max(np.abs(g-.3*outer(v, k))))
    _, ek, ev = numpy_step(z, v, z, z, 'balanced_stdp', lp=.7, lm=.7)
    reverse, _, _ = numpy_step(k, z, ek, ev, 'balanced_stdp', lp=.7, lm=.7)
    reverse_error = float(np.max(np.abs(reverse+.3*outer(v, k))))
    ek=ev=z.copy()
    static_max=0.
    for _ in range(20):
        g, ek, ev = numpy_step(k, v, ek, ev, 'balanced_stdp')
        static_max=max(static_max,float(np.max(np.abs(g))))
    xs=rng.normal(size=(20,12)); ek=ev=z.copy(); m=np.zeros((12,12))
    for x in xs:
        g,ek,ev=numpy_step(x,x,ek,ev,'balanced_stdp');m+=g
    skew_error=float(np.max(np.abs(m+m.T)))
    retained=np.eye(12)+m
    frozen_before=retained.copy()
    # Zero update preserves an existing memory; G=0 does not imply erasure.
    retained += numpy_step(k,v,k,v,'balanced_stdp')[0]
    retention_error=float(np.max(np.abs(retained-frozen_before)))
    assert max(delay_error,reverse_error,static_max,skew_error,retention_error)<1e-12
    return dict(forward_pair_error=delay_error,reverse_pair_error=reverse_error,
                constant_activity_write_absmax=static_max,identity_population_skew_error=skew_error,
                zero_update_retention_error=retention_error,
                interpretation='Pulse order only checks signs. Balanced zero-lag cancellation and skew symmetry are algebraic limitations, not a domain verdict.')


def fixed_activity_experiments(trials=128, dim=32):
    """Hopfield anchor -> time-lag writing on identical activity trajectories.

    Full patterns clamp STORAGE activity only. Recall sees partial cues. General
    windows are a predeclared sweep, not selected on test recall performance.
    """
    rows=[]
    windows=[('hebb',.1,.1,1.,1.),('balanced_stdp',.1,.1,1.,1.),
             ('balanced_stdp',.9,.9,1.,1.)]
    windows += [('general_stdp',lp,lm,ap,2-ap)
                for lp,lm in ((.1,.1),(.9,.9),(.1,.9),(.9,.1))
                for ap in (.5,1.,1.5)]
    for p in (2,4,8,16):
        rng=np.random.default_rng(100+p)
        patterns=bipolar(rng,(trials,p,dim))
        idx=rng.integers(0,p,trials);target=patterns[np.arange(trials),idx]
        observed=rng.random((trials,dim))>.5
        observed[:,0]=False;observed[:,1]=True
        cue=np.where(observed,target,0.)
        canonical=np.einsum('bpi,bpj->bij',patterns,patterns)/dim
        canonical[:,np.arange(dim),np.arange(dim)]=0
        result=score_numpy(hopfield_recall(canonical,cue,observed),target,observed)
        rows.append(dict(patterns=p,condition='canonical_hopfield',rule='hebb',**result))
        for condition in ('static','settling','noisy_settling','shared_background'):
            local=np.random.default_rng(1000+p)
            h=local.normal(size=patterns.shape)
            background=.6*bipolar(local,(trials,1,dim))
            stream=[]
            for r in range(16):
                if condition=='static':h=patterns.copy()
                else:
                    h=.6*h+.4*patterns
                    if condition=='noisy_settling':h+=.25*local.normal(size=h.shape)
                stream.append(h.copy()+background if condition=='shared_background' else h.copy())
            for rule,lp,lm,ap,am in windows:
                ek=np.zeros_like(patterns);ev=ek.copy();m=np.zeros_like(canonical)
                plus_norm=minus_norm=0.
                for x in stream:
                    # Avoid BLAS dot-product thread fanout for a scalar reduction.
                    plus_norm+=float(np.sqrt(np.square(np.einsum('bpi,bpj->bij',x,ek)).sum()))
                    minus_norm+=float(np.sqrt(np.square(np.einsum('bpi,bpj->bij',ev,x)).sum()))
                    # One channel population: K=V=x. All pattern lanes simultaneous.
                    if rule=='hebb':write=np.einsum('bpi,bpj->bij',x,x)
                    else:write=ap*np.einsum('bpi,bpj->bij',x,ek)-am*np.einsum('bpi,bpj->bij',ev,x)
                    m+=write/(dim*len(stream))
                    ek=lp*ek+(1-lp)*x;ev=lm*ev+(1-lm)*x
                sym=(m+m.transpose(0,2,1))/2;skew=(m-m.transpose(0,2,1))/2
                denom=np.linalg.norm(m,axis=(1,2)).clip(1e-15)
                symmetric_fraction=float((np.linalg.norm(sym,axis=(1,2))/denom).mean())
                m[:,np.arange(dim),np.arange(dim)]=0
                dot=(m*canonical).sum((1,2))
                reference_norm=np.linalg.norm(canonical,axis=(1,2)).clip(1e-15)
                alignment=dot/(np.linalg.norm(m,axis=(1,2)).clip(1e-15)*reference_norm)
                # Components orthogonal to the desired clean outer-product memory.
                residual=m-(dot/reference_norm**2)[:,None,None]*canonical
                result=score_numpy(hopfield_recall(m,cue,observed),target,observed)
                rows.append(dict(patterns=p,condition=condition,rule=rule,lambda_plus=lp,
                    lambda_minus=lm,amplitude_plus=ap,amplitude_minus=am,
                    matrix_rms=float(np.sqrt(np.mean(m*m))),symmetric_norm_fraction=symmetric_fraction,
                    clean_memory_cosine=float(alignment.mean()),orthogonal_memory_rms=float(np.sqrt(np.mean(residual**2))),
                    skew_rms=float(np.sqrt(np.mean(skew*skew))),plus_norm=plus_norm,minus_norm=minus_norm,
                    **result))
    return rows


def rms_norm(x):
    dtype=x.dtype
    x=x.float()
    return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)).to(dtype)


class AssociativeMemory(nn.Module):
    """One-head Fast Weight reader with learned channels and internal recurrence.

    Pattern lanes are summed within each recurrence. Two post-residual RMSNorms,
    unchanged q/k/v/out/FFN across rules; no QK normalization, no token mask.
    General pair-STDP starts equal to the project rule, then learns window params.
    """
    def __init__(self,rule,dim=32,hidden=32,store_steps=6,recall_steps=4,write_gain=1.):
        super().__init__()
        self.rule=rule;self.dim=dim;self.hidden=hidden
        self.store_steps=store_steps;self.recall_steps=recall_steps;self.write_gain=write_gain
        self.input=nn.Linear(2*dim,hidden,bias=False)
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.k=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,hidden,bias=False)
        self.out=nn.Linear(hidden,hidden,bias=False)
        self.gate_up=nn.Linear(hidden,4*hidden,bias=False)
        self.down=nn.Linear(2*hidden,hidden,bias=False)
        self.decoder=nn.Linear(hidden,dim,bias=False)
        if rule=='general_stdp':
            self.balance_logit=nn.Parameter(torch.tensor(0.))
            self.trace_logits=nn.Parameter(torch.full((2,),math.log(.1/.9)))

    def window(self):
        if self.rule=='general_stdp':
            ap=2*self.balance_logit.sigmoid();am=2-ap
            lp,lm=self.trace_logits.sigmoid().unbind()
            return lp,lm,ap,am
        return .1,.1,1.,1.

    def write(self,k,v,memory,ek,ev):
        lp,lm,ap,am=self.window()
        plus=torch.einsum('bpi,bpj->bij',v,ek)
        minus=torch.einsum('bpi,bpj->bij',ev,k)
        g=torch.einsum('bpi,bpj->bij',v,k) if self.rule=='hebb' else ap*plus-am*minus
        memory=memory+self.write_gain*g/(k.shape[1]*self.store_steps)
        return memory,lp*ek+(1-lp)*k,lm*ev+(1-lm)*v

    def block(self,h,drive,memory,ek=None,ev=None,plastic=False):
        h=h+drive
        q=self.q(h)
        if plastic:
            k,v=self.k(h),self.v(h)
            memory,ek,ev=self.write(k,v,memory,ek,ev)
        read=torch.einsum('bij,bpj->bpi',memory,q)
        h=rms_norm(h+self.out(read))
        gate,up=self.gate_up(h).chunk(2,dim=-1)
        h=rms_norm(h+self.down(.5*gate*up))
        return h,memory,ek,ev

    def store(self,patterns):
        b,p,_=patterns.shape
        drive=self.input(torch.cat((patterns,torch.ones_like(patterns)),dim=-1))
        h=drive.new_zeros(b,p,self.hidden)
        m=drive.new_zeros(b,self.hidden,self.hidden)
        ek=ev=torch.zeros_like(h)
        for _ in range(self.store_steps):h,m,ek,ev=self.block(h,drive,m,ek,ev,True)
        return m

    @torch.no_grad()
    def storage_diagnostics(self,patterns):
        """Separate signal size from cancellation and latent matrix structure.

        With linear K/V projections and shared activities, M = Wv A Wk.T.
        Balanced shared traces force A to be skew, even though M need not be.
        This analysis does not modify training or normalize the written memory.
        """
        b,p,_=patterns.shape
        drive=self.input(torch.cat((patterns,torch.ones_like(patterns)),dim=-1))
        h=torch.zeros_like(drive);ek=ev=torch.zeros_like(h)
        ep=em=torch.zeros_like(h)
        m=drive.new_zeros(b,self.hidden,self.hidden);latent=torch.zeros_like(m)
        lp,lm,ap,am=self.window();steps=[]
        for r in range(self.store_steps):
            z=h+drive;k=self.k(z);v=self.v(z)
            plus=torch.einsum('bpi,bpj->bij',v,ek)
            minus=torch.einsum('bpi,bpj->bij',ev,k)
            g=torch.einsum('bpi,bpj->bij',v,k) if self.rule=='hebb' else ap*plus-am*minus
            a=torch.einsum('bpi,bpj->bij',z,z) if self.rule=='hebb' else (
                ap*torch.einsum('bpi,bpj->bij',z,ep)-am*torch.einsum('bpi,bpj->bij',em,z))
            latent+=self.write_gain*a/(p*self.store_steps)
            norm=lambda x:float(x.square().mean().sqrt())
            steps.append(dict(recurrence=r+1,positive_term_rms=norm(ap*plus),
                negative_term_rms=norm(am*minus),write_rms=norm(g),
                write_to_terms_ratio=norm(g)/(norm(ap*plus)+norm(am*minus)+1e-12)))
            ep=lp*ep+(1-lp)*z;em=lm*em+(1-lm)*z
            h,m,ek,ev=self.block(h,drive,m,ek,ev,True)
        mapped=self.v.weight[None]@latent@self.k.weight.T[None]
        sym=(latent+latent.transpose(-1,-2))/2
        skew=(latent-latent.transpose(-1,-2))/2
        return dict(recurrences=steps,latent_symmetric_rms=norm(sym),latent_skew_rms=norm(skew),
                    latent_mapping_max_error=float((mapped-m).abs().max()))

    def recall(self,memory,cue,observed,steps=None,plastic=False):
        drive=self.input(torch.cat((cue,observed.to(cue.dtype)),dim=-1))[:,None]
        h=torch.zeros_like(drive);ek=ev=torch.zeros_like(h)
        predictions=[]
        # Recall traces begin empty: no arbitrary correspondence to storage lanes.
        for _ in range(self.recall_steps if steps is None else steps):
            h,memory,ek,ev=self.block(h,drive,memory,ek,ev,plastic)
            predictions.append(self.decoder(h[:,0]))
        return torch.stack(predictions),memory

    def forward(self,patterns,cue,observed):
        memory=self.store(patterns)
        predictions,_=self.recall(memory,cue,observed)
        return predictions,memory


def episodes(generator,batch,patterns,dim,missing=.5,flip=0.):
    bank=2*torch.randint(2,(batch,patterns,dim),generator=generator).float()-1
    idx=torch.randint(patterns,(batch,),generator=generator)
    target=bank[torch.arange(batch),idx]
    observed=torch.rand(batch,dim,generator=generator)>=missing
    observed[:,0]=False;observed[:,1]=True
    flips=torch.rand(batch,dim,generator=generator)<flip
    cue=torch.where(observed,torch.where(flips,-target,target),0.)
    return bank,cue,observed,target


def reconstruction_loss(prediction,target,observed):
    error=(prediction-target).square()
    hidden=~observed
    return (error*hidden).sum()/hidden.sum()+.1*(error*observed).sum()/observed.sum()


def score_torch(predictions,target,observed):
    last=predictions[-1];hits=(last>=0)==(target>0);hidden=~observed
    per_sample=(hits*hidden).sum(-1)/hidden.sum(-1)
    return dict(hidden_accuracy=float(per_sample.mean()),
                hidden_accuracy_se=float(per_sample.std(unbiased=True)/math.sqrt(len(target))),
                all_accuracy=float(hits.float().mean()),exact=float(hits.all(-1).float().mean()),
                hidden_mse=float(((last-target).square()*hidden).sum()/hidden.sum()),
                hidden_accuracy_by_recurrence=[float((((x>=0)==(target>0))*hidden).sum()/hidden.sum())
                                               for x in predictions])


@torch.no_grad()
def evaluate(model,batch,mode='frozen',steps=None,memory_scale=1.):
    bank,cue,observed,target=batch
    memory=model.store(bank)
    before=memory.clone()
    if mode=='zero':memory=torch.zeros_like(memory)
    elif mode=='shuffled':memory=memory.roll(1,0)
    memory=memory*memory_scale
    predictions,after=model.recall(memory,cue,observed,steps=steps,plastic=mode=='online')
    if mode=='frozen':torch.testing.assert_close(memory,after,rtol=0,atol=0)
    result=score_torch(predictions,target,observed)
    result.update(memory_rms=float(before.square().mean().sqrt()),
                  read_memory_change_rms=float((after-memory).square().mean().sqrt()))
    return result


def learned_experiments(out,seeds,steps=1500,dim=32,hidden=32,batch_size=16):
    evaluations=[];histories=[];diagnostics=[]
    started=time.monotonic()
    for seed in seeds:
        torch.manual_seed(seed)
        base=AssociativeMemory('hebb',dim,hidden)
        models={}
        for rule in RULES:
            model=AssociativeMemory(rule,dim,hidden)
            model.load_state_dict(base.state_dict(),strict=False)
            models[rule]=model
        opts={r:torch.optim.Adam(m.parameters(),lr=1e-3) for r,m in models.items()}
        best={r:(float('inf'),None) for r in RULES}
        g=torch.Generator().manual_seed(2000+seed)
        validation=episodes(torch.Generator().manual_seed(12000+seed),256,4,dim)
        for step in range(1,steps+1):
            p=(2,4,8)[int(torch.randint(3,(),generator=g))]
            data=episodes(g,batch_size,p,dim,missing=(.25,.5,.75)[step%3])
            bank,cue,observed,target=data
            for rule,model in models.items():
                opts[rule].zero_grad(set_to_none=True)
                predictions,memory=model(bank,cue,observed)
                loss=reconstruction_loss(predictions[-1],target,observed)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {rule} seed={seed} step={step}')
                loss.backward()
                if step%100==0 or step==1:
                    qkv_grad={n:float(getattr(model,n).weight.grad.norm()) for n in ('q','k','v')}
                    lp,lm,ap,am=model.window()
                    histories.append(dict(seed=seed,rule=rule,step=step,loss=float(loss.detach()),
                        memory_rms=float(memory.detach().square().mean().sqrt()),gradients=qkv_grad,
                        lambda_plus=float(torch.as_tensor(lp).detach()),lambda_minus=float(torch.as_tensor(lm).detach()),
                        amplitude_plus=float(torch.as_tensor(ap).detach()),amplitude_minus=float(torch.as_tensor(am).detach())))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opts[rule].step()
            if step%250==0 or step==steps:
                status={}
                for rule,model in models.items():
                    val=evaluate(model,validation)
                    if val['hidden_mse']<best[rule][0]:best[rule]=(val['hidden_mse'],copy.deepcopy(model.state_dict()))
                    status[rule]=round(val['hidden_accuracy'],4)
                print(json.dumps(dict(seed=seed,step=step,validation_hidden_accuracy=status,
                    elapsed=round(time.monotonic()-started,1))),flush=True)
        for rule,model in models.items():
            model.load_state_dict(best[rule][1])
            torch.save(dict(state_dict=model.state_dict(),seed=seed,rule=rule,dim=dim,hidden=hidden,
                            validation_hidden_mse=best[rule][0]),out/f'{rule}_seed{seed}.pt')
            lp,lm,ap,am=model.window()
            window=dict(lambda_plus=float(torch.as_tensor(lp).detach()),lambda_minus=float(torch.as_tensor(lm).detach()),
                        amplitude_plus=float(torch.as_tensor(ap).detach()),amplitude_minus=float(torch.as_tensor(am).detach()))
            bank=episodes(torch.Generator().manual_seed(44000+seed),64,4,dim)[0]
            diagnostics.append(dict(seed=seed,rule=rule,**window,**model.storage_diagnostics(bank)))
            for p in (2,4,8,16):
                for missing in (.25,.5,.75):
                    test=episodes(torch.Generator().manual_seed(22000+seed*100+p*10+int(missing*4)),512,p,dim,missing)
                    modes=('frozen','zero','shuffled')
                    if p==4 and missing==.5:modes+=('online',)
                    for mode in modes:
                        for r in ((4,8,16) if mode=='frozen' and p==4 and missing==.5 else (4,)):
                            evaluations.append(dict(seed=seed,rule=rule,patterns=p,missing=missing,
                                mode=mode,recall_steps=r,**window,**evaluate(model,test,mode,r)))
                    if p==4 and missing==.5:
                        for gain in (.25,4.,16.):
                            evaluations.append(dict(seed=seed,rule=rule,patterns=p,missing=missing,
                                mode='read_gain_control',read_gain=gain,recall_steps=4,**window,
                                **evaluate(model,test,'frozen',4,memory_scale=gain)))
            # Same held-out episodes for all models, with corrupted observed bits.
            noisy=episodes(torch.Generator().manual_seed(33000+seed),512,4,dim,.5,.1)
            evaluations.append(dict(seed=seed,rule=rule,patterns=4,missing=.5,flip=.1,
                mode='frozen',recall_steps=4,**window,**evaluate(model,noisy)))
        (out/'training.json').write_text(json.dumps(histories,indent=2))
        (out/'learned_results.json').write_text(json.dumps(evaluations,indent=2))
        (out/'storage_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    return histories,evaluations


def export(out,fixed,history,learned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for rule in RULES:
        rs=[r for r in fixed if r['rule']==rule and r['condition']=='settling'
            and r['lambda_plus']==.1 and r['lambda_minus']==.1
            and r['amplitude_plus']==(1.5 if rule=='general_stdp' else 1.)]
        axes[0].plot([r['patterns'] for r in rs],[r['hidden_accuracy'] for r in rs],'-o',label=rule)
        hs=[r for r in history if r['rule']==rule]
        ts=sorted({r['step'] for r in hs})
        axes[1].plot(ts,[np.mean([r['loss'] for r in hs if r['step']==t]) for t in ts],label=rule)
        ls=[r for r in learned if r['rule']==rule and r['mode']=='frozen'
            and r['missing']==.5 and r['recall_steps']==4 and not r.get('flip')]
        ps=sorted({r['patterns'] for r in ls})
        axes[2].plot(ps,[np.mean([r['hidden_accuracy'] for r in ls if r['patterns']==p]) for p in ps],'-o',label=rule)
    axes[0].set(title='Fixed activity: same reader, identity K/V',xlabel='Stored patterns',ylabel='Hidden-bit recall accuracy')
    axes[1].set(title='Learned reader: paired training batches',xlabel='Optimizer steps',ylabel='Training reconstruction loss')
    axes[2].set(title='Learned reader: frozen-memory test',xlabel='Stored patterns',ylabel='Hidden-bit recall accuracy')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    flat=[dict(stage=stage,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))})
          for stage,rows in (('fixed_activity',fixed),('learned',learned)) for r in rows]
    fields=sorted({k for r in flat for k in r})
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    selected=[]
    for rule in RULES:
        for mode in ('frozen','zero','shuffled','online'):
            rs=[r for r in learned if r['rule']==rule and r['patterns']==4 and r['missing']==.5
                and r['mode']==mode and r['recall_steps']==4 and not r.get('flip')]
            if rs:
                selected.append(dict(rule=rule,mode=mode,seeds=len(rs),
                    hidden_accuracy_mean=float(np.mean([r['hidden_accuracy'] for r in rs])),
                    hidden_accuracy_seed_std=float(np.std([r['hidden_accuracy'] for r in rs],ddof=1)) if len(rs)>1 else None,
                    hidden_mse_mean=float(np.mean([r['hidden_mse'] for r in rs])),
                    exact_mean=float(np.mean([r['exact'] for r in rs]))))
    (out/'summary.json').write_text(json.dumps(selected,indent=2))
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in
        ('rule','mode','hidden_accuracy_mean','hidden_mse_mean','exact_mean'))+'</tr>' for r in selected)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Associative STDP</title>'
        '<h1>재귀축 연상기억: 헤비안과 STDP</h1>'
        '<p>주 평가: 새로운 패턴들을 저장한 뒤 M을 고정하고, 가려진 단서를 복원한다. '
        '시간축은 내부 반복이며 패턴들은 병렬이다. 느린 가중치의 역전파와 빠른 기억의 '
        '비지도 갱신을 구분한다. 모든 신규 학습 모델에 두 번의 residual RMSNorm을 적용한다.</p>'
        '<img src="comparison.png" style="max-width:100%"><h2>패턴 4개, 50% 가림</h2>'
        '<table border="1"><tr><th>Rule</th><th>Memory</th><th>Hidden accuracy</th>'
        '<th>Hidden MSE</th><th>Exact</th></tr>'+table+'</table>'
        '<p>zero/shuffled는 기억 활용 대조군이다. online은 보조 평가이며 회상 흔적을 0에서 시작한다. '
        '일반 STDP가 약화 진폭을 0에 가깝게 줄이는 경우, 강화·약화의 우위로 해석하지 않는다. '
        '고정 활동 sweep의 가장 좋은 설정은 탐색 결과이며 검증된 우위가 아니다. '
        '홉필드 참조의 sign 회상과 학습 모델의 회상은 서로 다른 모델이므로 직접 순위를 매기지 않는다.</p>'
        '<p>균형 STDP의 K=V 갱신은 반대칭이다. 이것은 구현 검사의 기준이지, STDP 전체가 '
        '연상기억에 부적합하다는 결론이 아니다. 시험 정답은 갱신이나 단서에 들어가지 않는다.</p>')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--fixed-trials',type=int,default=128)
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise SystemExit('Use a new output directory; overwriting experiments is disabled.')
    source=Path(__file__).read_bytes();(out/'source_snapshot.py').write_bytes(source)
    protocol=dict(args=vars(args),source_sha256=hashlib.sha256(source).hexdigest(),
        torch=torch.__version__,numpy=np.__version__,axis='internal recurrence; parallel pattern lanes',
        dim=32,hidden=32,store_steps=6,recall_steps=4,batch_size=16,
        primary='Frozen-memory recall on unseen random bipolar pattern banks',
        write_scale='write_gain/(number_of_patterns*store_steps), write_gain=1 for all rules',
        optimizer='Adam lr=1e-3, weight_decay=0, clip_grad_norm=1; identical batches per rule',
        normalization='Two affine-free post-residual RMSNorms, FP32 eps=1e-5; QK raw',
        storage='All complete patterns are shown simultaneously on every storage recurrence; no query target is supplied at recall',
        trace='Strict past normalized EMA; no trace reset during storage; zero initial memory/traces',
        general_window='alpha_plus=2sigmoid(b), alpha_minus=2-alpha_plus; lambda_plus/minus=sigmoid(theta); initialized alpha=1, lambda=.1',
        slow_learning='Supervised reconstruction trains the programmer; fast updates never receive target errors',
        checkpoint_selection='Lowest held-out validation missing-bit MSE, validation evaluated every 250 steps',
        controls=['Canonical Hopfield','constant/correlated-in-time/noisy settling activities/shared background',
                  'zero/shuffled memory','fixed vs online recall','recall extrapolation 4/8/16','noisy cues'],
        limitations=['Continuous signed pair rule, not a spiking STDP reproduction',
                     'No positional RoPE and scalar traces in this toy: shared-latent skew factorization does not directly apply to the full Sudoku model with positional rotations and channel-dependent traces',
                     'General STDP has four effective-window scalars represented by three learned parameters; controls have no extra parameters',
                     'Fixed learning rate and write gain: a negative result cannot establish impossibility or domain mismatch',
                     'Frozen Hopfield reference clamps observed cue bits; learned reader receives a mask without output clamping',
                     'General windows are normalized-EMA windows: changing lambda also changes impulse amplitude'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    checks=mechanism_checks();(out/'mechanism_checks.json').write_text(json.dumps(checks,indent=2))
    fixed=fixed_activity_experiments(args.fixed_trials)
    (out/'fixed_results.json').write_text(json.dumps(fixed,indent=2))
    history,learned=learned_experiments(out,args.seeds,args.steps)
    export(out,fixed,history,learned)
    (out/'completed.json').write_text(json.dumps(dict(steps=args.steps,seeds=args.seeds)))
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':main()
