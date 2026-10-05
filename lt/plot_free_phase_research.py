"""Standalone research figures from recorded numerical measurements."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .summarize_free_phase_research import record_directory


DEST = Path(__file__).resolve().parents[1] / 'docs/research/2026-10-05/free_phase_window_research'


def save_figure(fig, name):
    for extension in ('png', 'svg'):
        path = DEST/f'{name}.{extension}'
        fig.savefig(path, dpi=180)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main():
    rows = json.loads((DEST/'shape_study.json').read_text())['rows']
    x = np.linspace(-math.pi,math.pi,8001)
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    a,b,c,d = axes.flat
    a.plot(x,np.sign(x)*np.exp(-abs(x)),color='#111827',lw=1.6,label='Original signed exponential')
    # Leave a visible discontinuity in the black line.
    a.lines[-1].set_ydata(np.where(abs(x)<.002,np.nan,np.sign(x)*np.exp(-abs(x))))
    colors = ['#d97706','#0891b2','#9333ea']
    for (epsilon,modes),color in zip([(.2,8),(.35,8),(.5,4)],colors):
        row = next(r for r in rows if r['family']=='optimized_frequencies' and r['epsilon']==epsilon and r['modes']==modes)
        omega,coef=np.array(row['frequencies']),np.array(row['coefficients'])
        y = (np.sin(x[:,None]*omega)*coef).sum(-1)
        dy = (np.cos(x[:,None]*omega)*coef*omega).sum(-1)
        label=f'epsilon={epsilon}, R={modes}'
        a.plot(x,y,color=color,label=label)
        b.plot(x,dy,color=color,label=label)
    b.plot(x,-np.exp(-abs(x)),color='#111827',ls='--',label='Original: derivative away from 0')
    a.set(title='Bounded-delay STDP window candidates',xlabel='Current delay difference',ylabel='L(delta)',xlim=(-math.pi,math.pi))
    b.set(title='Smoothing changes the central gradient',xlabel='Current delay difference',ylabel="L\u2032(delta)",xlim=(-1.5,1.5))
    a.legend(fontsize=8); b.legend(fontsize=8)
    strict = json.loads((DEST/'gpu_kernel_manual_highest.json').read_text())['rows']
    mixed = json.loads((DEST/'gpu_kernel_bf16.json').read_text())['rows']
    lookup = {r['name']:r for r in mixed+strict}
    names=['fixed_exp','exact_dynamic_exp','manual_4','manual_8','bf16_4','bf16_8']
    labels=['Fixed exp','Free exp','R4 FP32','R8 FP32','R4 mixed','R8 mixed']
    values=[lookup[n]['forward_backward_median_ms'] for n in names]
    bars=c.barh(labels,values,color=['#475569','#111827','#0891b2','#0891b2','#d97706','#d97706'])
    c.bar_label(bars,fmt='%.2f',padding=3)
    c.invert_yaxis(); c.set(title='RTX 4090: write + read + backward',xlabel='Milliseconds (phase generation and model excluded)',xlim=(0,max(values)*1.2))
    for epsilon,color in zip([.08,.2,.35,.5],['#b91c1c','#d97706','#0891b2','#9333ea']):
        candidates=sorted((r for r in rows if r['family']=='optimized_frequencies' and r['epsilon']==epsilon),key=lambda r:r['modes'])
        d.plot([r['modes'] for r in candidates],[r['derivative_relative_l2']*100 for r in candidates],marker='o',label=f'epsilon={epsilon}',color=color)
    d.set(title='Sharp transitions need more components',xlabel='Sine components R (separation rank <= 2R)',ylabel='Derivative relative L2 error (%)',xticks=[1,2,4,8],yscale='log')
    d.legend(fontsize=8)
    for ax in (a,b,d):
        ax.axhline(0,color='#cbd5e1',lw=.6); ax.grid(alpha=.15)
    fig.suptitle('Free-order phase windows: shape, gradient and actual compute cost',fontsize=15)
    save_figure(fig, 'research_overview')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    runs={
        'Fixed exponential':('kv_phase_exp_current_fresh_20261005','#111827'),
        'Fixed smooth R4':('research_free_phase_fixed_fourier4_e05_20261005','#2563eb'),
        'Free smooth R4':('research_free_phase_free_fourier4_e05_20261005','#059669'),
        'Free exponential':('research_free_phase_free_exact_exp_20261005','#dc2626')}
    for label,(directory,color) in runs.items():
        path=record_directory(directory)/'train.jsonl'
        if not path.exists():continue
        records=[json.loads(line) for line in path.read_text().splitlines()]
        records=[r for r in records if r['step']<=3008 and r.get('_count_raw',0)>0]
        for ax,metric in zip(axs,('accuracy','exact_accuracy')):
            y=np.array([r[metric]*100 for r in records])
            if len(y)<16:continue
            ax.plot([r['step'] for r in records][15:],np.convolve(y,np.ones(16)/16,mode='valid'),label=label,color=color)
    for ax,title in zip(axs,('Training cell accuracy','Training exact puzzle accuracy')):
        ax.set(title=title,xlabel='Optimizer step',ylabel='Percent',xlim=(0,3008))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('One seed; 16-terminal-batch rolling mean; no clue/blank split')
    save_figure(fig, 'training_controls')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    row=next(r for r in rows if r['family']=='optimized_frequencies' and r['epsilon']==.5 and r['modes']==4)
    w,c=np.array(row['frequencies']),np.array(row['coefficients'])
    slow,fast=1.,.1
    peak=np.log(slow/fast)/(1/fast-1/slow)
    scale=1/(np.exp(-peak/slow)-np.exp(-peak/fast))
    for ax,bound in zip(axs,(math.pi,.65)):
        x=np.linspace(-bound,bound,8001)
        old=np.sign(x)*np.exp(-abs(x));old[abs(x)<bound/4000]=np.nan
        ax.plot(x,old,label='Original signed exponential',color='#111827')
        ax.plot(x,(np.sin(x[:,None]*w)*c).sum(-1),label='Trained smooth R4 candidate',color='#059669')
        ax.plot(x,scale*np.sign(x)*(np.exp(-abs(x))-np.exp(-abs(x)/fast)),label='C1 exponential difference (preflight failed)',color='#d97706')
        ax.axhline(0,lw=.5,color='#64748b');ax.grid(alpha=.2)
        ax.set(xlabel='Current delay difference',ylabel='L(delta)',title='Full bounded range' if bound>1 else 'Near the sign reversal')
        ax.legend(fontsize=8)
    save_figure(fig, 'final_window_candidates')


if __name__=='__main__':
    main()
