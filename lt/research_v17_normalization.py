"""Run controlled address-normalization and read-path ablations of v1.7.

The historical trainer snapshot is imported unchanged. `_unit` is replaced by
the identity for both instantaneous and trace addresses. Value cosine agreement,
QR address projection, Phi, optimizer and the original data stream are retained.
The optional memory-only read replaces exactly `a_eff = (1-lam)*a + lam*w`
with `a_eff = w`, preserving memory writes and all parameter initialization.
"""
import argparse
from dataclasses import asdict
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def trainer_source(path,read_mode='mixed'):
    source=Path(path).read_text()
    if read_mode=='memory':
        before='a_eff = (1 - lam) * a + lam * w'
        if source.count(before)!=1:
            raise ValueError('Expected exactly one historical interpolation equation.')
        source=source.replace(before,'a_eff = w')
    elif read_mode!='mixed':
        raise ValueError(f'Unknown read mode: {read_mode}')
    return source


def load_trainer(path,read_mode='mixed'):
    spec=importlib.util.spec_from_file_location("historical_v17",path)
    trainer=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=trainer
    exec(compile(trainer_source(path,read_mode),str(path),'exec'),trainer.__dict__)
    return trainer


def raw_addresses(self,x,y):
    return x,y


def cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [cpu_tree(v) for v in value]
    return value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='runs/kv_collapse_20261003/reference/train_v17.py')
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=6000)
    ap.add_argument('--keep-normalization',action='store_true',help='Unmodified control')
    ap.add_argument('--read-mode',choices=['mixed','memory'],default='mixed',
                    help='Read the original interpolation or only the updated W with current V')
    args=ap.parse_args()
    source=Path(args.source).resolve()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('step_*.pt')) or (out/'train.jsonl').exists():
        raise RuntimeError('Use an empty output directory; historical trainer does not restore carry exactly.')
    t=load_trainer(source,args.read_mode)
    variant='v17_original' if args.keep_normalization else 'v17_without_address_norm'
    if args.read_mode=='memory':
        variant+='_memory_read'
    if not args.keep_normalization:
        t.LT_Inner._unit=raw_addresses
    cfg=dict(t.CFG,data_npz=str(Path('data/sudoku_lt_1k.npz').resolve()),
        out_dir=str(out),max_steps=args.steps,max_hours=float('inf'),num_processes=1,
        expect_processes=None,resume_from=None,scan_kaggle_input=False,require_resume=False,
        log_every=16,save_every_steps=500,keep_last=20,milestone_every=0)
    cfg.update(research_variant=variant,research_read_mode=args.read_mode,
               research_address_norm=args.keep_normalization,nograd_fixed=0,nograd_every=0)
    t.CFG=cfg
    # Suppress only progress-bar rendering; retain every original training log.
    try:
        import tqdm.auto
        tqdm.auto.tqdm=lambda *a,**kw: None
    except ImportError:
        pass  # The historical trainer already handles absent tqdm.
    effective_source=trainer_source(source,args.read_mode)
    (out/'trainer_snapshot.py').write_text(effective_source)
    (out/'model_change.diff').write_text(''.join(difflib.unified_diff(
        source.read_text().splitlines(keepends=True),effective_source.splitlines(keepends=True),
        fromfile='historical_train_v17.py',tofile='effective_train_v17.py')))
    (out/'research_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    # Infinity is only an operational deadline sentinel, not a model setting.
    (out/'config.json').write_text(json.dumps(dict(cfg,max_hours=None),indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(variant=variant,
        original_trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        trainer_sha256=hashlib.sha256(effective_source.encode()).hexdigest(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        changed_equation='identity _unit(x,y) for instantaneous/trace QK addresses' if not args.keep_normalization else 'none',
        read_equation='a_eff = w' if args.read_mode=='memory' else 'a_eff = (1 - lam) * a + lam * w',
        preserved='value cosine agree, QR, Phi, trace, W update, token sum, initialization, optimizer and data stream',
        checkpoint_note='Original checkpoint plus separate full carry and RNG sidecar for diagnosis.'),indent=2))
    original_batch=t.train_batch
    original_save=t.save_checkpoint
    original_optimizers=t.create_optimizers
    current={}
    started=time.monotonic()

    def check_finite(optimizer,args,kwargs):
        grads=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if grads and not bool(torch.stack([g.isfinite().all() for g in grads]).all()):
            raise FloatingPointError('Nonfinite gradient before optimizer update')

    def optimizers(*args,**kwargs):
        opts,lrs=original_optimizers(*args,**kwargs)
        for opt in opts:
            opt.register_step_pre_hook(check_finite)
        return opts,lrs

    def batch(model,base,state,*args,**kwargs):
        current['state']=state
        before=time.monotonic()
        result=original_batch(model,base,state,*args,**kwargs)
        if result is not None:
            record=dict(step=state.step,segment=int(state.carry.steps.max()),
                seconds=time.monotonic()-before,elapsed=time.monotonic()-started,
                **{k:float(v) for k,v in result.items()})
            with (out/'train.jsonl').open('a') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n')
            if state.step%256<16:
                with torch.no_grad():
                    h=state.carry.current_hidden.float()
                    m=state.carry.coupling.float()
                    sv=torch.linalg.svdvals(m[:8])
                    d=dict(step=state.step,segment=record['segment'],
                        current_hidden=dict(rms=float(h.square().mean().sqrt()),absmax=float(h.abs().max())),
                        coupling=dict(rms=float(m.square().mean().sqrt()),absmax=float(m.abs().max())),
                        memory_spectrum=dict(sigma_max=float(sv[...,0].max()),sigma_mean=float(sv[...,0].mean())))
                    with (out/'diagnostics.jsonl').open('a') as f:
                        f.write(json.dumps(d,allow_nan=False)+'\n')
        return result

    def save(*args,**kwargs):
        path=original_save(*args,**kwargs)
        state=current.get('state')
        if state is not None:
            extra=dict(step=state.step,carry=cpu_tree(asdict(state.carry)),
                       numpy_rng_state=np.random.get_state(),
                       torch_rng_state=torch.random.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state_all())
            tmp=out/f'.carry_{state.step}.tmp'
            torch.save(extra,tmp);tmp.replace(out/f'carry_{state.step}.pt')
        return path

    t.train_batch=batch;t.save_checkpoint=save;t.create_optimizers=optimizers
    t.main()
    actual=current['state'].step
    status='completed' if actual==args.steps else 'stopped'
    (out/f'{status}.json').write_text(json.dumps(dict(requested_steps=args.steps,
        actual_step=actual,elapsed=time.monotonic()-started)))


if __name__=='__main__':
    main()
