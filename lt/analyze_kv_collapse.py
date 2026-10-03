"""Summarize completed or live KV collapse experiments without touching jobs."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    if not path.exists():
        return []
    result=[]
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live writer may have only written part of its final line.
    return result


def summarize(path):
    data=rows(path / "train.jsonl")
    diag=rows(path / "diagnostics.jsonl")
    windows=[]
    for start in range(0,len(data),256):
        group=data[start:start+256]
        terminal=[x for x in group if x["segment"]==16]
        windows.append(dict(start=group[0]["step"],end=group[-1]["step"],
                            loss=float(np.mean([x["lm_loss"] for x in group])),
                            terminal_loss=float(np.mean([x["lm_loss"] for x in terminal])) if terminal else None,
                            terminal_accuracy=float(np.mean([x["accuracy"] for x in terminal])) if terminal else None))
    result=dict(run=path.name,last_step=data[-1]["step"] if data else 0,windows=windows)
    return result,data,diag


def parse_reference(path):
    if not path.exists():
        return [], []
    text=path.read_text()
    train=[]
    for m in re.finditer(r'\[LT\] step (\d+)\s+lm_loss ([\d.]+)\s+acc ([\d.]+)\s+exact ([\d.]+).*?\[halt step (\d+)\]',text):
        step,loss,accuracy,exact,halt=m.groups()
        if int(step)<=6000:
            train.append(dict(step=int(step),halt_step=int(halt),loss=float(loss),
                              accuracy=float(accuracy),exact_accuracy=float(exact)))
    evaluation=[]
    for m in re.finditer(r'\[EVAL\] step (\d+).*?acc ([\d.]+)\s+exact (\d+)/(\d+)',text):
        step,accuracy,exact,count=m.groups()
        if int(step)<=6000:
            evaluation.append(dict(step=int(step),accuracy=float(accuracy),
                                   exact=int(exact),count=int(count)))
    return train,evaluation


def parse_r1b8(path):
    if not path.exists():
        return [], []
    text=path.read_text()
    training=[]
    for m in re.finditer(r'\[LT\] step (\d+)\s+lm_loss ([\d.]+)\s+acc ([\d.]+)\s+exact ([\d.]+)',text):
        step,loss,accuracy,exact=m.groups()
        if int(step)<=6000:
            training.append(dict(step=int(step),loss=float(loss),accuracy=float(accuracy),exact=float(exact)))
    _,evaluation=parse_reference(path)
    # The historical log prints two evaluations at 1953; retain both in JSON.
    return training,evaluation


def compare_r1b8(root,runs):
    reference,evaluation=parse_r1b8(root/'reference'/'R1B8_bilin_ok.log')
    if not reference:
        return
    v17,v17_eval=parse_reference(root/'reference'/'train_v17.log')
    fig,grid=plt.subplots(2,2,figsize=(13,8))
    axes=[grid[0,0],grid[1,0],grid[0,1],grid[1,1]]
    axes[0].plot([x['step'] for x in reference],[x['accuracy'] for x in reference],'s--',label='R1B8 bilinear historical')
    axes[1].plot([x['step'] for x in evaluation],[x['accuracy'] for x in evaluation],'s--',label='R1B8 bilinear historical EMA')
    axes[2].plot([x['step'] for x in reference],[x['exact'] for x in reference],'s--',label='R1B8 bilinear historical')
    axes[3].plot([x['step'] for x in evaluation],[x['exact']/x['count'] for x in evaluation],'s--',label='R1B8 bilinear historical EMA')
    axes[0].plot([x['halt_step'] for x in v17],[x['accuracy'] for x in v17],'o--',label='v1.7 historical')
    axes[1].plot([x['step'] for x in v17_eval],[x['accuracy'] for x in v17_eval],'o--',label='v1.7 historical EMA')
    axes[2].plot([x['halt_step'] for x in v17],[x['exact_accuracy'] for x in v17],'o--',label='v1.7 historical')
    axes[3].plot([x['step'] for x in v17_eval],[x['exact']/x['count'] for x in v17_eval],'o--',label='v1.7 historical EMA')
    paired=[]
    for path,data in runs:
        cfg=json.loads((path/'config.json').read_text())
        if cfg.get('nograd_fixed',0) or cfg.get('nograd_every',0):
            continue
        lookup={x['step']:x for x in data}
        current=[]
        for ref in reference:
            if ref['step'] in lookup:
                actual=lookup[ref['step']]
                record=dict(run=path.name,step=ref['step'],r1b8_accuracy=ref['accuracy'],
                            kv_accuracy=actual['accuracy'],difference_pp=100*(actual['accuracy']-ref['accuracy']),
                            r1b8_exact_accuracy=ref['exact'],kv_exact_accuracy=actual['exact_accuracy'])
                current.append(record);paired.append(record)
        axes[0].plot([x['step'] for x in current],[x['kv_accuracy'] for x in current],'o-',label=path.name)
        axes[2].plot([x['step'] for x in current],[x['kv_exact_accuracy'] for x in current],'o-',label=path.name)
        _,current_eval=parse_reference(path/'stdout.log')
        axes[1].plot([x['step'] for x in current_eval],[x['accuracy'] for x in current_eval],'o-',label=path.name)
        axes[3].plot([x['step'] for x in current_eval],[x['exact']/x['count'] for x in current_eval],'o-',label=path.name)
    for ax,label in zip(axes,['Training cell accuracy','EMA test cell accuracy','Training exact-match rate','EMA test exact-match rate']):
        ax.set_ylabel(label);ax.set_xlabel('Optimizer step');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('8 gradient blocks x 16 segments; R1B8 uses a historical data pipeline')
    fig.tight_layout();fig.savefig(root/'r1b8_comparison.png',dpi=160);plt.close(fig)
    (root/'r1b8_comparison.json').write_text(json.dumps(dict(paired_training=paired,
        historical_training=reference,historical_evaluation=evaluation,
        caveat='Same recurrent block count. Historical R1B8 has a different trainer, initialization and data pipeline; paired batch identity is not established.'),indent=2))


def compare_v17(root, runs):
    reference,reference_eval=parse_reference(root/'reference'/'train_v17.log')
    if not reference:
        return
    fig,axes=plt.subplots(2,1,figsize=(10,8))
    axes[0].plot([x['halt_step'] for x in reference],[x['accuracy'] for x in reference],
                 'o--',label='v1.7 historical training log')
    axes[1].plot([x['step'] for x in reference_eval],[x['accuracy'] for x in reference_eval],
                 'o--',label='v1.7 historical EMA test')
    comparisons=[]
    for path,data in runs:
        cfg=json.loads((path/'config.json').read_text())
        if cfg.get('nograd_fixed',0) or cfg.get('nograd_every',0):
            continue
        lookup={x['step']:x for x in data}
        paired=[]
        for ref in reference:
            if ref['halt_step'] in lookup and ref['step'] in lookup:
                accuracy=lookup[ref['halt_step']]['accuracy']
                paired.append(dict(run=path.name,reported_step=ref['step'],
                    accuracy_step=ref['halt_step'],v17_accuracy=ref['accuracy'],
                    kv_accuracy=accuracy,difference_pp=100*(accuracy-ref['accuracy']),
                    v17_loss=ref['loss'],kv_loss=lookup[ref['step']]['lm_loss']))
        comparisons.extend(paired)
        axes[0].plot([x['accuracy_step'] for x in paired],[x['kv_accuracy'] for x in paired],
                     'o-',label=path.name)
        _,evaluation=parse_reference(path/'stdout.log')
        axes[1].plot([x['step'] for x in evaluation],[x['accuracy'] for x in evaluation],
                     'o-',label=path.name)
    for ax,label in zip(axes,['Training cell accuracy: matched completed segment','EMA test cell accuracy: fresh 16-segment evaluation']):
        ax.set_ylabel(label);ax.set_xlabel('Optimizer step');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(root/'v17_comparison.png',dpi=160);plt.close(fig)
    (root/'v17_comparison.json').write_text(json.dumps(comparisons,indent=2))
    if comparisons:
        with (root/'v17_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)


def compare_read_pair(root,runs,names,labels,title,filename):
    """Compare matched read-path controls at the same recurrent depth."""
    selected={path.name:(path,data) for path,data in runs}
    if not all(name in selected for name in names):
        return
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    result={}
    for name,label in zip(names,labels):
        path,data=selected[name]
        completed=[r for r in data if r['segment']==16]
        # Both conditions see the same terminal batches at the same steps.
        windows=[]
        for end in range(15,len(completed),16):
            group=completed[end-15:end+1]
            windows.append(dict(step=group[-1]['step'],start=group[0]['step'],
                **{key:float(np.mean([x[key] for x in group]))
                   for key in ['accuracy','exact_accuracy','lm_loss']}))
        _,evaluation=parse_reference(path/'stdout.log')
        axes[0,0].plot([x['step'] for x in windows],[100*x['accuracy'] for x in windows],label=label)
        axes[0,1].plot([x['step'] for x in windows],[100*x['exact_accuracy'] for x in windows],label=label)
        axes[1,0].plot([x['step'] for x in evaluation],[100*x['accuracy'] for x in evaluation],'o-',label=label)
        axes[1,1].plot([x['step'] for x in evaluation],[x['exact'] for x in evaluation],'o-',label=label)
        result[name]=dict(last_step=data[-1]['step'] if data else 0,
            terminal_windows_256=windows,evaluation=evaluation,
            training_endpoints=[x for x in completed if x['step'] in (2000,4000,6000)])
    for ax,label in zip(axes.flat,['Training cell accuracy (%)','Training exact match (%)',
                                  'EMA test cell accuracy (%)','EMA test exact matches / 2048']):
        ax.set_xlabel('Optimizer step');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(title+'; 8 blocks x 16 segments\nTraining curves average 16 completed batches (256 optimizer steps)')
    fig.tight_layout();fig.savefig(root/f'{filename}.png',dpi=160);plt.close(fig)
    (root/f'{filename}.json').write_text(json.dumps(result,indent=2))


def compare_training_stability(root,runs):
    """Follow the user's primary criterion: sustained reversal of training."""
    selected={path.name:(path,data) for path,data in runs}
    labels={
        'baseline_ng0':'KV-STDP: accumulated M only',
        'kv_interpolated_read_ng0':'Real current KV + accumulated complex-STDP M',
        'kv_current_only_ng0':'Real current KV only: no traces or accumulation',
        'kv_complex_current_only_ng0':'STDP difference G only: traces, no accumulation',
        'kv_current_plus_stdp_ng0':'Current KV B + STDP difference G: no accumulation',
        'v17_no_address_norm':'v1.7: no address normalization, mixed read',
    }
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    result={}
    for name,label in labels.items():
        if name not in selected:
            continue
        _,data=selected[name]
        terminal=[r for r in data if r['segment']==16]
        if not terminal:
            continue
        # Every point averages the same completed batches in each condition.
        ends=list(range(256,terminal[-1]['step']+1,256))
        if terminal[-1]['step']>=256 and terminal[-1]['step'] not in ends:
            ends.append(terminal[-1]['step'])
        windows=[]
        for end in ends:
            group=[r for r in terminal if end-256<r['step']<=end]
            windows.append(dict(end=end,first_terminal_step=group[0]['step'],
                count=len(group),**{key:float(np.mean([r[key] for r in group]))
                    for key in ['lm_loss','accuracy','exact_accuracy']}))
        if not windows:
            continue
        axes[0].plot([r['end'] for r in windows],[r['lm_loss'] for r in windows],label=label)
        axes[1].plot([r['end'] for r in windows],[100*r['accuracy'] for r in windows],label=label)
        best_accuracy=max(windows,key=lambda r:r['accuracy'])
        best_loss=min(windows,key=lambda r:r['lm_loss'])
        result[name]=dict(last_step=data[-1]['step'],windows=windows,
            highest_window_accuracy=best_accuracy,
            lowest_window_loss=best_loss,
            final_window=windows[-1],
            accuracy_change_from_best_pp=100*(windows[-1]['accuracy']-best_accuracy['accuracy']),
            loss_change_from_best=windows[-1]['lm_loss']-best_loss['lm_loss'])
    for ax,label in zip(axes,['Training loss','Training cell accuracy (%)']):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Optimizer step')
    fig.suptitle('Training stability: 8 blocks x 16 segments, no no-grad prefix\n'
                 '256-step trailing means of completed batches; final partial interval also included')
    fig.tight_layout();fig.savefig(root/'training_stability.png',dpi=160);plt.close(fig)
    (root/'training_stability.json').write_text(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    args=ap.parse_args()
    fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
    results=[]
    runs=[]
    for path in sorted(args.root.iterdir()):
        if not path.is_dir() or not (path/"train.jsonl").exists():
            continue
        result,data,diag=summarize(path)
        results.append(result)
        runs.append((path,data))
        windows=result["windows"]
        axes[0].plot([x["end"] for x in windows],[x["loss"] for x in windows],label=path.name)
        axes[1].plot([x["end"] for x in windows],[x["terminal_accuracy"] for x in windows],label=path.name)
        # The v1.7 state is token x token, not the KV channel x channel matrix.
        cfg=json.loads((path/'config.json').read_text())
        selected=[x for x in diag if x["segment"]==1] if (
            cfg.get('memory_type')=='kv_stdp' and
            cfg.get('research_variant') not in ('current_only','complex_current_only','current_plus_stdp')) else []
        axes[2].plot([x["step"] for x in selected],[x["memory_spectrum"]["sigma_max"] for x in selected],label=path.name)
    for ax,label in zip(axes,["Training loss (256-step mean)","Training accuracy (segment 16)","Memory maximum singular value (segment 1)"]):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
    axes[-1].set_xlabel("Optimizer step")
    fig.tight_layout();fig.savefig(args.root/"comparison.png",dpi=160);plt.close(fig)
    (args.root/"summary.json").write_text(json.dumps(results,indent=2))
    compare_v17(args.root,runs)
    compare_r1b8(args.root,runs)
    compare_read_pair(args.root,runs,
        ['v17_no_address_norm','v17_no_norm_memory_read'],
        ['Mixed read: (1-lambda) A + lambda W','W-only read'],
        'v1.7 without address normalization','read_ablation_comparison')
    compare_read_pair(args.root,runs,
        ['baseline_ng0','kv_interpolated_read_ng0'],
        ['KV-STDP: accumulated M read','KV-STDP: current KV + accumulated M read'],
        'KV-STDP read interpolation','kv_interpolation_comparison')
    compare_read_pair(args.root,runs,
        ['kv_interpolated_read_ng0','kv_current_only_ng0'],
        ['Real current KV + accumulated complex-STDP M','Real current KV only'],
        'KV read control: removing recurrent memory','current_only_comparison')
    compare_read_pair(args.root,runs,
        ['kv_current_only_ng0','kv_complex_current_only_ng0'],
        ['Current KV B: no traces','STDP difference G: original traces'],
        'Current operator: B vs G, without matrix accumulation','real_vs_complex_current_comparison')
    compare_read_pair(args.root,runs,
        ['kv_current_only_ng0','kv_complex_current_only_ng0','kv_current_plus_stdp_ng0'],
        ['Current KV B only','STDP difference G only','Current KV B + STDP difference G'],
        'Current operator controls: B, G, B + G, without matrix accumulation',
        'current_plus_stdp_comparison')
    compare_training_stability(args.root,runs)
    for result in results:
        print(result["run"],"step",result["last_step"],"recent",result["windows"][-1] if result["windows"] else None)


if __name__=="__main__":
    main()
