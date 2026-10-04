"""Run imported URM using the same data, loss and optimizer harness."""
import argparse
import json
from pathlib import Path
import shutil
import time
from . import train as t
from .urm_full_bptt import install


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/urm_swiglu_2layer_8iter_loops16.json')
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=20000)
    ap.add_argument('--ffn',choices=('convswiglu','swiglu'),default='swiglu')
    args=ap.parse_args()
    cfg=dict(t.CFG);cfg.update(json.loads(Path(args.config).read_text()))
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg.update(out_dir=str(out),max_steps=args.steps,run_selftests=False,research_arch="urm_full_bptt",urm_ffn=args.ffn)
    if cfg['memory_type']!='address' or cfg['loops'] not in (1,16) or cfg['num_layers']!=2 or cfg['blocks_per_seg']!=8:
        raise ValueError('Baseline requires address harness, loops=1 or 16, layers=2, iterations=8.')
    if any(cfg.get(k,0) for k in ('nograd_fixed','nograd_every','late_sup_prob')):
        raise ValueError('No no-grad iterations or extra unsupervised segments allowed.')
    protocol=dict(upstream_commit='c14e55f5f9227873617015cf60a239126b55adcd',
        layers=2,iterations=8,outer_calls=cfg['loops'],layer_applications_per_segment=16,ffn=args.ffn,act_enabled=False,
        no_grad_recurrence=False,internal_truncated_bptt=False,
        detach_between_segments=True,optimizer_update_per_segment=True,
        sample_retention_segments=cfg['loops'],
        supervision='one final loss per segment; retain samples until fixed loops exhausted',
        preserved=['URM softmax attention','1D RoPE',('SwiGLU' if args.ffn=='swiglu' else 'ConvSwiGLU'),'two post-residual RMSNorms',
                   'puzzle prefix token','input reinjection','URM initialization'],
        adapted=['shared Sudoku data/loss/AdamATan2/EMA/logging harness',
                 'fixed iteration count; ACT disabled; unused q_head retained',
                 'PyTorch SDPA fallback if flash-attn is unavailable',
                 'activation checkpointing recomputes the within-segment graph; no internal truncation'],
        comparison_limit=('Same loops=16 sample retention and per-segment updates as B/phase models; URM uses two layers instead of one.' if cfg['loops']==16 else 'Legacy loops=1 run; different sample retention from B/phase models.'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'launch_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for src in ['lt/train.py','lt/urm_full_bptt.py','lt/research_urm_full_bptt.py']:
        shutil.copyfile(src,out/(Path(src).stem+'_snapshot.py'))
    shutil.copytree('lt/urm_vendor',out/'urm_vendor_snapshot',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    original=t.train_batch
    def train_batch(*a,**kw):
        result=original(*a,**kw)
        if result is not None:
            record=dict(result,step=a[2].step,elapsed=time.monotonic()-start)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return result
    t.train_batch=train_batch
    install(args.ffn)
    print(f'[URM] layers=2 iterations/segment=8 loops={cfg['loops']}; FFN={args.ffn}; ACT=False; full within-segment BPTT; boundary detach; update/segment',flush=True)
    start=time.monotonic();t.main(cfg)
    (out/'finished.json').write_text(json.dumps(dict(elapsed=time.monotonic()-start))+'\n')


if __name__=='__main__':main()
