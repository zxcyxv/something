"""Train 새코드3 on the existing Sudoku-Extreme 1k / fixed augmentation pool.

python -m lt.train_new3 --config configs/new3_sudoku.json
python -m lt.train_new3 --resume runs/new3_sudoku/checkpoints --eval-only --eval-segs 128

Single GPU (RTX 5090) or CPU. One update is one segment of eight blocks by
default; loops=1 resets h/S for every new batch. With loops>1 they persist
across updates and detach at segment boundaries.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .new3 import ACTLossHead, LTv5, LTv5Carry, LTv5Config
from .train import (AdamATan2, EMAHelper, _EMASwap, SudokuTrainDataset,
                    cosine_schedule_with_warmup_lr_lambda)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'new3-sudoku-axial-window-delta-v1'
DEFAULT_CFG = dict(
    **dataclasses.asdict(LTv5Config(batch_size=128, seq_len=81, grid=9,
                                   vocab_size=11, num_puzzle_identifiers=1,
                                   ckpt_blocks=True)),
    data_npz=str(ROOT/'data/sudoku_lt_1k.npz'), num_aug=1000, test_size=2048,
    global_batch_size=128, epochs=50000, eval_interval=250, eval_batch_size=128,
    lr=1e-4, lr_min_ratio=1.0, lr_warmup_steps=2000, weight_decay=1.0,
    beta1=0.9, beta2=0.95, ema_rate=0.999, seed=0, q_weight=0.5,
    focal_gamma=0.0, loss_type='stablemax_cross_entropy', router_balance_gamma=0.001,
    out_dir=str(ROOT/'runs/new3_sudoku/checkpoints'), max_steps=160000,
    max_hours=720.0, log_every=50, gradient_log_every=1,
    save_every_steps=2000, keep_last=2, milestone_every=10000,
    milestone_extrap_segs=128, milestone_extrap_n=512,
    dataloader_workers=1, cpu_threads=2,
)
_STOP = False
_STEP_RE = re.compile(r'step_(\d+)\.pt$')


def validate_config(cfg):
    cfg.setdefault('expert_schedule','routed')
    unknown = set(cfg)-set(DEFAULT_CFG)-{'data_fingerprint'}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    if cfg['global_batch_size'] < 2 or cfg['eval_batch_size'] < 2:
        raise ValueError('Train and evaluation batch sizes must be at least 2.')
    cfg['batch_size'] = cfg['global_batch_size']
    LTv5Config.from_dict(cfg)
    if cfg['epochs'] <= 0 or cfg['eval_interval'] <= 0 or cfg['epochs'] % cfg['eval_interval']:
        raise ValueError('epochs must be a positive multiple of eval_interval (in epochs).')
    if cfg['test_size'] < 2 or cfg['num_aug'] < 0 or cfg['max_steps'] < 0:
        raise ValueError('Invalid test_size, num_aug, or max_steps.')
    if cfg['milestone_extrap_n'] < 2 or cfg['milestone_extrap_segs'] < cfg['loops']:
        raise ValueError('Milestone evaluation needs >=2 puzzles and at least train loops segments.')
    if cfg['dataloader_workers'] not in (0,1):
        raise ValueError('Use 0 or 1 loader worker to preserve the established data order.')
    if not 0 <= cfg['ema_rate'] < 1 or not 0 <= cfg['lr_min_ratio'] <= 1:
        raise ValueError('Invalid EMA decay or LR minimum ratio.')
    if not 0 <= cfg['beta1'] < 1 or not 0 <= cfg['beta2'] < 1:
        raise ValueError('Optimizer betas must lie in [0,1).')
    if cfg['lr'] <= 0 or cfg['lr_warmup_steps'] < 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid learning rate, warmup, or weight decay.')
    if cfg['router_balance_gamma'] < 0 or cfg['moe_aux_weight'] < 0:
        raise ValueError('MoE balance coefficients must be nonnegative.')
    for key in ('save_every_steps','milestone_every','gradient_log_every','keep_last'):
        if cfg[key] < 0:
            raise ValueError(f'{key} must be nonnegative; zero disables it.')
    if cfg['log_every'] < 1 or cfg['cpu_threads'] < 1 or cfg['max_hours'] <= 0:
        raise ValueError('Invalid logging interval, CPU threads, or time budget.')


def load_data(cfg):
    path = Path(cfg['data_npz']).expanduser().resolve()
    arrays = []
    with np.load(path, allow_pickle=False) as data:
        for split in ('train','test'):
            x,y = data[f'{split}_inputs'],data[f'{split}_labels']
            if split == 'test':
                x,y = x[:cfg['test_size']],y[:cfg['test_size']]
            if x.ndim != 3 or x.shape[1:] != (9,9) or y.shape != x.shape or len(x) < 2:
                raise ValueError(f'Invalid {split} board shapes.')
            if not ((x>=0)&(x<=9)).all() or not ((y>=1)&(y<=9)).all():
                raise ValueError(f'Invalid {split} token values.')
            if not ((x==0)|(x==y)).all():
                raise ValueError(f'{split} clues disagree with solutions.')
            target = np.arange(1,10)
            boxes = y.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
            if not (np.all(np.sort(y,axis=1)==target[None,:,None]) and
                    np.all(np.sort(y,axis=2)==target) and np.all(np.sort(boxes,axis=2)==target)):
                raise ValueError(f'Invalid {split} Sudoku solutions.')
            arrays.extend([np.ascontiguousarray(x),np.ascontiguousarray(y)])
    digest=hashlib.sha256()
    for a in arrays:
        digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    cfg['data_npz'],cfg['data_fingerprint']=str(path),digest.hexdigest()
    return arrays


@dataclasses.dataclass
class TrainState:
    step: int = 0
    iter_id: int = 0
    batch_in_iter: int = 0
    carry: LTv5Carry | None = None
    in_step: bool = False


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(cpu_tree(v) for v in value)
    return value


def device_tree(value,device):
    if isinstance(value,torch.Tensor): return value.to(device)
    if isinstance(value,dict): return {k:device_tree(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return type(value)(device_tree(v,device) for v in value)
    return value


def make_training_objects(cfg,device):
    model=ACTLossHead(LTv5(cfg),loss_type=cfg['loss_type'],q_weight=cfg['q_weight'],
                      focal_gamma=cfg['focal_gamma'],moe_aux_weight=cfg['moe_aux_weight']).to(device)
    named=list(model.named_parameters())
    def no_decay(name,p): return p.ndim<=1 or name.endswith('alpha_raw')
    optimizer=AdamATan2([
        {'params':[p for n,p in named if no_decay(n,p)],'weight_decay':0.0},
        {'params':[p for n,p in named if not no_decay(n,p)],'weight_decay':cfg['weight_decay']}],
        lr=0.0,betas=(cfg['beta1'],cfg['beta2']))
    ema=EMAHelper(cfg['ema_rate']);ema.register(model)
    return model,optimizer,ema


def latest_checkpoint(path):
    if path is None: return None
    path=Path(path)
    if path.is_file(): return path
    choices=[(int(match[1]),p) for p in path.rglob('step_*.pt')
             if (match:=_STEP_RE.fullmatch(p.name))]
    return max(choices,key=lambda item:item[0])[1] if choices else None


def save_checkpoint(directory,state,model,optimizer,ema,cfg,keep=None):
    if state.in_step:
        raise RuntimeError('Cannot save a partially completed update.')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=cpu_tree(model.state_dict())
    with _EMASwap(model,ema): averaged=cpu_tree(model.state_dict())
    device=next(model.parameters()).device
    carry=None if state.carry is None else {f.name:getattr(state.carry,f.name)
                                           for f in dataclasses.fields(state.carry)}
    checkpoint=dict(model_id=MODEL_ID,schema_version=1,cfg=dict(cfg),
                    step=state.step,iter_id=state.iter_id,batch_in_iter=state.batch_in_iter,
                    raw_model_state_dict=raw,model_state_dict=averaged,
                    optimizer_state=cpu_tree(optimizer.state_dict()),ema_shadow=cpu_tree(ema.shadow),
                    carry=cpu_tree(carry),rng=dict(torch=torch.get_rng_state(),
                    numpy=np.random.get_state(),python=random.getstate(),
                    cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
    path=directory/f'step_{state.step}.pt';temporary=path.with_suffix('.pt.tmp')
    torch.save(checkpoint,temporary);os.replace(temporary,path)
    keep=cfg['keep_last'] if keep is None else keep
    if keep:
        files=sorted((int(m[1]),p) for p in directory.glob('step_*.pt')
                     if (m:=_STEP_RE.fullmatch(p.name)))
        for _,old in files[:-keep]: old.unlink()
    return path


_RUNTIME_KEYS={'out_dir','max_steps','max_hours','log_every','gradient_log_every',
               'save_every_steps','keep_last','milestone_every','milestone_extrap_n',
               'milestone_extrap_segs','eval_batch_size','dataloader_workers','cpu_threads',
               'amp','ckpt_blocks','data_npz'}


def load_checkpoint(path,state_model,optimizer,ema,cfg,device,training=True):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('model_id')!=MODEL_ID or ck.get('schema_version')!=1:
        raise ValueError('This is not a compatible new3 Sudoku training checkpoint.')
    saved_cfg=dict(ck['cfg'])
    saved_cfg.setdefault('expert_schedule','routed')  # checkpoints predating the schedule option
    if cfg.get('expert_schedule','routed')!=saved_cfg['expert_schedule']:
        raise ValueError('Checkpoint expert_schedule differs from the requested model.')
    if training:
        for k in set(cfg)-_RUNTIME_KEYS:
            if cfg[k]!=saved_cfg.get(k):
                raise ValueError(f'Resume changes {k}: {saved_cfg.get(k)!r} -> {cfg[k]!r}')
    state_model.load_state_dict(ck['raw_model_state_dict'],strict=True)
    if training: optimizer.load_state_dict(ck['optimizer_state'])
    ema.shadow=device_tree(ck['ema_shadow'],device)
    carry=None if ck['carry'] is None else LTv5Carry(**device_tree(ck['carry'],device))
    state=TrainState(ck['step'],ck['iter_id'],ck['batch_in_iter'],carry)
    if training:
        torch.set_rng_state(ck['rng']['torch']);np.random.set_state(ck['rng']['numpy'])
        random.setstate(ck['rng']['python'])
        if device.type=='cuda' and ck['rng']['cuda'] is not None:
            torch.cuda.set_rng_state(ck['rng']['cuda'],device)
    return state


def make_loader(x,y,cfg,state):
    dataset=SudokuTrainDataset(x,y,seed=cfg['seed'],num_aug=cfg['num_aug'],
        global_batch_size=cfg['global_batch_size'],rank=0,world_size=1,
        epochs_per_iter=cfg['eval_interval'],start_iter=state.iter_id,
        total_iters=cfg['epochs']//cfg['eval_interval'],skip_batches=state.batch_in_iter)
    kwargs=dict(batch_size=None,num_workers=cfg['dataloader_workers'],pin_memory=cfg['amp'],
                generator=torch.Generator().manual_seed(cfg['seed']))
    # A dedicated generator keeps loader construction independent of model RNG.
    if cfg['dataloader_workers']:
        kwargs.update(prefetch_factor=2,persistent_workers=False,multiprocessing_context='spawn')
    return DataLoader(dataset,**kwargs)


def append_json(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def train_step(model,optimizer,ema,state,batch,cfg,planned_steps):
    state.in_step=True;model.train()
    device=next(model.parameters()).device
    batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
    if state.carry is None: state.carry=model.initial_carry(batch)
    optimizer.zero_grad(set_to_none=True)
    carry,loss,metrics,_,_=model(return_keys=set(),carry=state.carry,batch=batch)
    (loss/cfg['global_batch_size']).backward()
    named=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
    with torch.no_grad():
        grad_norms=torch.stack([p.grad.float().norm() for _,p in named])
        grad_l2=grad_norms.norm()
        if not bool(torch.isfinite(loss).all() & torch.isfinite(grad_l2)):
            raise FloatingPointError(f'Nonfinite loss/gradient at update {state.step+1}.')
        record_grad=cfg['gradient_log_every'] and (state.step+1)%cfg['gradient_log_every']==0
        before=[p.detach().clone() for _,p in named] if record_grad else None
    lr=cosine_schedule_with_warmup_lr_lambda(state.step,base_lr=cfg['lr'],
        num_warmup_steps=cfg['lr_warmup_steps'],num_training_steps=planned_steps,
        min_ratio=cfg['lr_min_ratio'])
    for group in optimizer.param_groups: group['lr']=lr
    optimizer.step()
    # Count only the original forward, never checkpoint backward recomputations.
    for layer in model.model.inner.layers:
        router=layer.moe.router;counts=router.pop_counts()
        router.apply_balance(cfg['router_balance_gamma'],cfg['num_active_experts'],counts)
    ema.update(model)
    state.carry=carry;state.step+=1;state.in_step=False
    n=metrics['count'].item()
    record=dict(step=state.step,lr=lr,blocks=cfg['blocks_per_seg'],
        expert_schedule=model.model.config.expert_schedule,
        lm_loss=metrics['lm_loss'].item()/cfg['global_batch_size'],
        moe_aux_loss=metrics.get('moe_aux_loss',loss.new_zeros(())).item(),
        evaluated_count=int(n),exact_count=int(metrics['exact_accuracy'].item()),
        exact_accuracy=metrics['exact_accuracy'].item()/n if n else None,
        accuracy=metrics['accuracy'].item()/n if n else None,grad_l2=grad_l2.item())
    if record_grad:
        with torch.no_grad():
            deltas=torch.stack([(p-old).norm() for (_,p),old in zip(named,before)])
            diagnostic=dict(step=state.step,names=[n for n,_ in named],
                grad_l2=grad_norms.cpu().tolist(),update_l2=deltas.cpu().tolist(),
                hidden_rms=carry.hidden.float().square().mean().sqrt().item(),
                memory_rms=carry.coupling.float().square().mean().sqrt().item()
                           if carry.coupling is not None else None)
        append_json(Path(cfg['out_dir'])/'gradients.jsonl',diagnostic)
    return record


def advance_cursor(state,steps_per_iter):
    state.batch_in_iter+=1
    if state.batch_in_iter==steps_per_iter:
        state.iter_id+=1;state.batch_in_iter=0;return True
    return False


def evaluation_batches(x,y,size,device):
    start=0
    while start<len(x):
        end=min(start+size,len(x))
        if len(x)-end==1: end+=1  # never emit a batch of one
        if end-start<2: raise ValueError('Evaluation requires at least two puzzles.')
        yield dict(inputs=torch.tensor(x[start:end].reshape(-1,81).astype('int64')+1,device=device),
                   labels=torch.tensor(y[start:end].reshape(-1,81).astype('int64')+1,device=device),
                   puzzle_identifiers=torch.zeros(end-start,dtype=torch.int32,device=device))
        start=end


def valid_sudoku(pred):
    grids=pred.reshape(-1,9,9);digits=torch.arange(2,11,device=pred.device)
    rows=(grids.sort(2).values==digits).all((1,2))
    cols=(grids.sort(1).values==digits[None,:,None]).all((1,2))
    boxes=grids.reshape(-1,3,3,3,3).permute(0,1,3,2,4).reshape(-1,9,9)
    return rows & cols & (boxes.sort(2).values==digits).all((1,2))


@torch.no_grad()
def evaluate(model,ema,x,y,cfg,step,segs,n=None,deadline=float('inf')):
    if segs<1: raise ValueError('Evaluation segment count must be positive.')
    x,y=x[:n],y[:n]
    if len(x)<2: raise ValueError('Evaluation requires at least two puzzles.')
    device=next(model.parameters()).device
    old_training,old_loops=model.training,model.model.config.loops
    totals=torch.zeros(segs,7,dtype=torch.float64,device=device)
    interrupted=False
    with _EMASwap(model,ema):
        model.eval();model.model.config.loops=segs+1  # prevents reset during extrapolation
        try:
            for batch in evaluation_batches(x,y,cfg['eval_batch_size'],device):
                carry=model.initial_carry(batch);previous=None
                for i in range(segs):
                    if _STOP or time.monotonic()>=deadline:
                        interrupted=True;break
                    carry,outputs=model.model(carry,batch)
                    pred=outputs['logits'].argmax(-1);ok=pred==batch['labels']
                    wrong_hint=(batch['inputs']>1)&(pred!=batch['inputs'])
                    totals[i,0]+=len(pred);totals[i,1]+=ok.sum()
                    totals[i,2]+=ok.all(-1).sum();totals[i,3]+=wrong_hint.any(-1).sum()
                    totals[i,4]+=(~valid_sudoku(pred)).sum()
                    if previous is not None: totals[i,5]+=(pred!=previous).sum()
                    totals[i,6]+=wrong_hint.sum();previous=pred
                if interrupted: break
        finally:
            model.model.config.loops=old_loops;model.train(old_training)
    rows=[]
    for i,(count,cells,exact,hint_bad,invalid,changed,hint_cells) in enumerate(totals.cpu().tolist()):
        if count:
            rows.append(dict(segment=i+1,n=int(count),exact=int(exact),exact_rate=exact/count,
                cell_accuracy=cells/(count*81),hint_wrong_puzzles=int(hint_bad),
                hint_wrong_cells=int(hint_cells),invalid_sudoku=int(invalid),churn=changed/(count*81)))
    result=dict(step=step,weights='ema',blocks=cfg['blocks_per_seg'],train_segments=old_loops,
                expert_schedule=model.model.config.expert_schedule,
                requested_segments=segs,partial=interrupted,rows=rows)
    if not interrupted and segs>=old_loops:
        first,last=rows[old_loops-1],rows[-1]
        result['error_reduction']=(last['exact']-first['exact'])/(first['n']-first['exact']) \
            if first['exact']<first['n'] else None
    return result


def write_evaluation(directory,result):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'eval_step_{result["step"]}_seg{result["requested_segments"]}.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)
    if result['rows']:
        r=result['rows'][-1]
        print(f'[EVAL] step {result["step"]} seg{r["segment"]} exact {r["exact"]}/{r["n"]} '
              f'({100*r["exact_rate"]:.2f}%) partial={result["partial"]}',flush=True)
    return path


def run(cfg,resume=None,device_name='auto',eval_only=False,eval_segs=None,eval_n=None):
    global _STOP
    _STOP=False;validate_config(cfg)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('This trainer targets a single GPU; run python, not multi-rank torchrun.')
    torch.set_num_threads(cfg['cpu_threads'])
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)
    if device.type=='cuda':
        if device.index is None:
            device=torch.device('cuda',torch.cuda.current_device())
        torch.cuda.set_device(device)
        if cfg['amp'] and not torch.cuda.is_bf16_supported():
            raise ValueError('Native BF16 is required for amp; use --no-amp for FP32.')
    else: cfg['amp']=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    tr_x,tr_y,te_x,te_y=load_data(cfg)
    planned_steps=int(cfg['epochs']*len(tr_x)/cfg['global_batch_size'])
    steps_per_iter=cfg['eval_interval']*len(tr_x)//cfg['global_batch_size']
    if steps_per_iter<1: raise ValueError('Not enough examples to produce a full training batch.')
    stop_at=min(cfg['max_steps'],steps_per_iter*(cfg['epochs']//cfg['eval_interval']))
    model,optimizer,ema=make_training_objects(cfg,device)
    path=latest_checkpoint(cfg['out_dir'] if resume in (None,'auto') else resume)
    if resume not in (None,'auto') and path is None:
        raise FileNotFoundError(f'No checkpoint found at {resume}')
    if eval_only and path is None: raise FileNotFoundError('Evaluation requires a checkpoint.')
    state=load_checkpoint(path,model,optimizer,ema,cfg,device,training=not eval_only) if path else TrainState()
    if state.step!=state.iter_id*steps_per_iter+state.batch_in_iter or state.batch_in_iter>=steps_per_iter:
        raise ValueError('Checkpoint data cursor does not match its optimizer step.')
    if eval_only:
        result=evaluate(model,ema,te_x,te_y,cfg,state.step,eval_segs or cfg['milestone_extrap_segs'],eval_n)
        return write_evaluation(Path(cfg['out_dir'])/'evaluations',result)
    out=Path(cfg['out_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print(f'[NEW3] {MODEL_ID} device={device} amp={cfg["amp"]} '
          f'params={sum(p.numel() for p in model.parameters()):,} batch={cfg["global_batch_size"]} '
          f'blocks={cfg["blocks_per_seg"]} loops={cfg["loops"]} '
          f'expert_schedule={cfg["expert_schedule"]} num_aug={cfg["num_aug"]}',flush=True)
    print(f'[NEW3] {"resume "+str(path) if path else "fresh initialization"}; '
          f'updates={state.step}..{stop_at}; eval every {steps_per_iter} updates',flush=True)
    def request_stop(*_):
        global _STOP
        _STOP=True
    old_handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    start=time.monotonic();deadline=start+cfg['max_hours']*3600
    last_halt=None;last_eval=-1
    loader=make_loader(tr_x,tr_y,cfg,state)
    try:
        if state.step<stop_at:
            for iteration,batch in loader:
                if _STOP or time.monotonic()>=deadline: break
                if int(iteration)!=state.iter_id: raise RuntimeError('Data iterator cursor mismatch.')
                record=train_step(model,optimizer,ema,state,batch,cfg,planned_steps)
                boundary=advance_cursor(state,steps_per_iter)
                append_json(out/'training.jsonl',record)
                if record['evaluated_count']: last_halt=record
                if state.step%cfg['log_every']==0:
                    status='' if last_halt is None else (f' exact={last_halt["exact_accuracy"]:.4f}'
                                                        f' [halt step {last_halt["step"]}]')
                    print(f'[NEW3] step {state.step} lm_loss={record["lm_loss"]:.5f} '
                          f'grad={record["grad_l2"]:.4f} lr={record["lr"]:.3g}{status} '
                          f'elapsed_s={time.monotonic()-start:.1f}',flush=True)
                if _STOP or time.monotonic()>=deadline: break
                regular=cfg['save_every_steps'] and state.step%cfg['save_every_steps']==0
                milestone=cfg['milestone_every'] and state.step%cfg['milestone_every']==0
                if boundary or regular:
                    save_checkpoint(out,state,model,optimizer,ema,cfg)
                if boundary:
                    result=evaluate(model,ema,te_x,te_y,cfg,state.step,cfg['loops'],deadline=deadline)
                    write_evaluation(out/'evaluations',result);last_eval=state.step
                if milestone:
                    save_checkpoint(out/'milestones',state,model,optimizer,ema,cfg,keep=0)
                    result=evaluate(model,ema,te_x,te_y,cfg,state.step,cfg['milestone_extrap_segs'],
                                    cfg['milestone_extrap_n'],deadline)
                    write_evaluation(out/'milestones',result)
                if state.step>=stop_at or _STOP or time.monotonic()>=deadline: break
        path=save_checkpoint(out,state,model,optimizer,ema,cfg)
        if not _STOP and time.monotonic()<deadline and last_eval!=state.step:
            write_evaluation(out/'evaluations',evaluate(model,ema,te_x,te_y,cfg,state.step,
                                                       cfg['loops'],deadline=deadline))
        print(f'[NEW3] saved step={state.step}: {path}',flush=True)
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
    return state.step


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,help='Flat JSON configuration; see configs/new3_sudoku.json')
    parser.add_argument('--out-dir',type=str)
    parser.add_argument('--resume',type=str,default='auto',help='Checkpoint file or directory; default resumes output directory')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--device',default='auto',help='auto, cpu, or cuda[:index]')
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--eval-segs',type=int)
    parser.add_argument('--eval-n',type=int)
    parser.add_argument('--write-default-config',type=Path)
    args=parser.parse_args()
    cfg=dict(DEFAULT_CFG)
    # An explicit resume path also supplies the exact saved model/training configuration.
    if args.resume!='auto':
        path=latest_checkpoint(args.resume)
        if path is None: raise FileNotFoundError(args.resume)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        cfg.update(saved['cfg'])
    if args.config:
        supplied=json.loads(args.config.read_text())
        unknown=set(supplied)-set(DEFAULT_CFG)
        if unknown: raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        cfg.update(supplied)
        if 'data_npz' in supplied and not Path(cfg['data_npz']).is_absolute():
            cfg['data_npz']=str(ROOT/cfg['data_npz'])
        if 'out_dir' in supplied and not Path(cfg['out_dir']).is_absolute():
            cfg['out_dir']=str(ROOT/cfg['out_dir'])
    for arg,key in ((args.out_dir,'out_dir'),(args.max_steps,'max_steps'),(args.amp,'amp')):
        if arg is not None: cfg[key]=arg
    if args.write_default_config:
        validate_config(cfg);args.write_default_config.parent.mkdir(parents=True,exist_ok=True)
        args.write_default_config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');return
    run(cfg,args.resume,args.device,args.eval_only,args.eval_segs,args.eval_n)


if __name__=='__main__': main()
