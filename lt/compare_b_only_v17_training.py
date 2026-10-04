"""Compare raw training logs at matched cadence/phase; exclude evaluation metrics."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=Path('runs/kv_b_only_fresh_20261004'))
    ap.add_argument('--reference',type=Path,default=Path('docs/research/2026-10-03/reference/train_v17.log'))
    ap.add_argument('--out',type=Path,default=Path('runs/b_only_v17_training_comparison_20261004'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    raw=(args.run/'train.jsonl').read_text()
    if not raw.endswith('\n'):raw=raw[:raw.rfind('\n')+1]
    current=[json.loads(line) for line in raw.splitlines()]
    by={r['step']:r for r in current};last=current[-1]['step']
    pattern=re.compile(r'\[LT\] step (\d+)\s+lm_loss ([0-9.eE+-]+)\s+acc ([0-9.]+)\s+exact ([0-9.]+).*?\[halt step (\d+)\]')
    historical={}
    for line in args.reference.read_text().splitlines():
        m=pattern.search(line)
        if m:
            s,l,a,e,h=m.groups();s=int(s)
            if s<=last:historical[s]=dict(step=s,loss=float(l),accuracy=float(a),exact=float(e),halt=int(h))
    pairs=[]
    for s,o in sorted(historical.items()):
        if s in by and o['halt'] in by:
            assert by[o['halt']]['segment']==16 and by[o['halt']]['_count_raw']==128
            pairs.append(dict(step=s,halt_step=o['halt'],loss_segment=by[s]['segment'],
                v17_loss=o['loss'],b_only_loss=by[s]['lm_loss'],
                v17_accuracy=o['accuracy'],b_only_accuracy=by[o['halt']]['accuracy'],
                v17_exact=o['exact'],b_only_exact=by[o['halt']]['exact_accuracy']))
    write_csv(args.out/'matched_training.csv',pairs)
    windows=[]
    for lo in range(0,last,2000):
        hi=lo+2000
        rs=[r for r in pairs if lo<r['step']<=hi]
        allseg=[r for r in current if lo<r['step']<=hi]
        terminal=[r for r in allseg if r['segment']==16]
        if not rs or not terminal:continue
        item=dict(start_exclusive=lo,end_inclusive=min(hi,last),complete=hi<=last,matched_samples=len(rs))
        for key in ('v17_loss','b_only_loss','v17_accuracy','b_only_accuracy','v17_exact','b_only_exact'):
            item[key]=float(np.mean([r[key] for r in rs]))
        item.update(b_only_all_segment_loss=float(np.mean([r['lm_loss'] for r in allseg])),
                    b_only_terminal_loss=float(np.mean([r['lm_loss'] for r in terminal])),
                    b_only_terminal_accuracy=float(np.mean([r['accuracy'] for r in terminal])),
                    b_only_terminal_exact=float(np.mean([r['exact_accuracy'] for r in terminal])))
        windows.append(item)
    write_csv(args.out/'windows.csv',windows)
    noise={}
    complete_end=last//2000*2000
    selected=[r for r in pairs if 2000<r['step']<=complete_end]
    for model in ('v17','b_only'):
        noise[model]={}
        for metric in ('loss','accuracy','exact'):
            key=model+'_'+metric;x=np.array([r[key] for r in selected])
            residual=[]
            for lo in range(2000,complete_end,2000):
                v=np.array([r[key] for r in selected if lo<r['step']<=lo+2000])
                residual.extend(v-v.mean())
            noise[model][metric]=dict(mean=float(x.mean()),std=float(x.std()),
                adjacent_change_rms=float(np.sqrt(np.mean(np.diff(x)**2))),
                within_2000_step_window_residual_rms=float(np.sqrt(np.mean(np.array(residual)**2))))
    phase=[]
    for lo in (2000,14000):
        if lo+2000>last:continue
        for segment in range(1,17):
            rs=[r for r in current if lo<r['step']<=lo+2000 and r['segment']==segment]
            phase.append(dict(start_exclusive=lo,end_inclusive=lo+2000,segment=segment,n=len(rs),
                              mean_loss=float(np.mean([r['lm_loss'] for r in rs]))))
    write_csv(args.out/'b_only_loss_by_segment.csv',phase)
    result=dict(latest_step=last,matched_rows=len(pairs),reference=str(args.reference),run=str(args.run),
        protocol='Raw training only. Loss matched at logged optimizer step; accuracy/exact matched at halt step. Historical log has 250-step cadence, and loss can be from a different segment than accuracy. Complete 2000-step windows used for variability statistics.',
        limitations='Historical sparse loss samples are not full-stream averages. Metrics alone cannot identify a harness bug. Different model and initialization, same nominal training recipe; do not infer identical parameter updates.',
        windows=windows,variability_steps=[2001,complete_end],variability=noise,b_only_loss_by_segment=phase)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for model,label,color in [('v17','v1.7','#2563eb'),('b_only','B-only','#ea580c')]:
        for ax,metric in zip(axes,('loss','accuracy','exact')):
            scale=1 if metric=='loss' else 100
            ax.plot([r['step'] for r in pairs],[r[model+'_'+metric]*scale for r in pairs],color=color,alpha=.25)
            ws=[w for w in windows if w['complete']]
            ax.plot([(w['start_exclusive']+w['end_inclusive'])/2 for w in ws],
                    [w[model+'_'+metric]*scale for w in ws],color=color,marker='o',label=label)
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Loss (matched logged phase)');axes[1].set_ylabel('Terminal cell accuracy (%)')
    axes[2].set_ylabel('Terminal exact (%)');axes[2].set_xlabel('Optimizer step')
    axes[0].legend();fig.suptitle('Training only: raw matched samples and 2000-step means')
    fig.tight_layout();fig.savefig(args.out/'matched_training.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(latest_step=last,matched_rows=len(pairs),variability=noise),indent=2))


if __name__=='__main__':
    main()
