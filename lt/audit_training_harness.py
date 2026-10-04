"""Compare the historical v1.7 and current harness using identical model/state.

No production run is modified. Full checkpoint checks operate on in-memory copies.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import train as t
from .kv_stability import install


def load_old(path):
    spec=importlib.util.spec_from_file_location('historical_v17_audit',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);module._TIME_SYNC=False
    return module


def delta(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.shape==b.shape
        if a.dtype==torch.bool:return float((a!=b).any())
        return float((a.detach().double()-b.detach().double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
        return max([delta(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max([delta(x,y) for x,y in zip(a,b)] or [0.])
    if a is None or isinstance(a,(str,bool)):return float(a!=b)
    return abs(float(a)-float(b))


def paired_updates(old,cfg,base,checkpoint=None,steps=32):
    device=next(base.parameters()).device
    a=t.ACTLossHead(copy.deepcopy(base.model),q_weight=cfg['q_weight'])
    b=old.ACTLossHead(copy.deepcopy(base.model),loss_type='stablemax_cross_entropy',q_weight=cfg['q_weight'])
    oa,la=t.create_optimizers(a,cfg,1);ob,lb=old.create_optimizers(b,cfg,1)
    ea=t.EMAHelper(cfg['ema_rate']);ea.register(a)
    eb=old.EMAHelper(cfg['ema_rate']);eb.register(b)
    if checkpoint:
        sa=t.load_training_checkpoint(checkpoint,a,oa,ea,cfg,0,1,device)
        sb=t.load_training_checkpoint(checkpoint,b,ob,eb,cfg,0,1,device)
    else:
        sa=t.TrainState(step=1992);sb=old.TrainState(step=1992)
    mismatch=[n for n,p in a.named_parameters() if t._is_no_decay(n,p,cfg.get('qkv_no_decay',False))!=old._is_no_decay(n,p)]
    assert not mismatch,mismatch
    gradients=[{},{}]
    for index,optimizers in enumerate((oa,ob)):
        for oi,opt in enumerate(optimizers):
            original=opt.step
            def step(*args,_original=original,_index=index,_oi=oi,_opt=opt,**kwargs):
                gradients[_index][_oi]=[None if p.grad is None else p.grad.detach().clone()
                    for group in _opt.param_groups for p in group['params']]
                return _original(*args,**kwargs)
            opt.step=step
    tr_x,tr_y,te_x,te_y,_,_=t.load_data(cfg)
    stream=t.SudokuTrainDataset(tr_x,tr_y,seed=cfg['seed'],num_aug=cfg['num_aug'],
        global_batch_size=cfg['global_batch_size'],rank=0,world_size=1,epochs_per_iter=cfg['eval_interval'],
        start_iter=sa.iter_id,total_iters=sa.iter_id+1,skip_batches=sa.batch_in_iter)
    it=iter(stream);records=[]
    for _ in range(steps):
        _,batch=next(it)
        ma=t.train_batch(a,a,sa,batch,cfg,oa,la,390625,0,1,device)
        mb=old.train_batch(b,b,sb,batch,cfg,ob,lb,390625,0,1,device)
        common={k:ma[k] for k in mb}
        record=dict(step=sa.step,metrics_max_abs=delta(common,mb),
            gradient_max_abs=delta(gradients[0],gradients[1]),
            model_state_max_abs=delta(a.state_dict(),b.state_dict()),
            carry_max_abs=delta(t._carry_dict(sa.carry),t._carry_dict(sb.carry)),
            optimizer_max_abs=delta([o.state_dict() for o in oa],[o.state_dict() for o in ob]))
        ea.update(a);eb.update(b)
        record['EMA_max_abs']=delta(ea.shadow,eb.shadow)
        records.append(record)
    return dict(no_decay_mismatches=mismatch,steps=steps,
        maxima={k:max(r[k] for r in records) for k in records[0] if k!='step'},records=records),a,sa,oa,ea


def data_parity(old,cfg):
    tr_x,tr_y,te_x,te_y,_,_=t.load_data(cfg)
    old_data=old.load_data(cfg)
    arrays=[np.array_equal(a,b) for a,b in zip((tr_x,tr_y,te_x,te_y),old_data)]
    assert all(arrays)
    counts={}
    for start in (0,1,3):
        kw=dict(seed=cfg['seed'],num_aug=cfg['num_aug'],global_batch_size=128,rank=0,world_size=1,
                epochs_per_iter=cfg['eval_interval'],start_iter=start,total_iters=start+1)
        streams=[iter(m.SudokuTrainDataset(tr_x,tr_y,**kw)) for m in (t,old)]
        digest=hashlib.sha256()
        for _ in range(64):
            x,y=[next(s) for s in streams]
            assert x[0]==y[0]
            for k in x[1]:
                assert torch.equal(x[1][k],y[1][k]),(start,k)
                digest.update(x[1][k].numpy().tobytes())
        counts[str(start)]=dict(batches=64,sha256=digest.hexdigest())
    return dict(dataset_arrays_identical=arrays,streams=counts)


def side_effects(old,cfg,base,state,opts,ema,out):
    device=next(base.parameters()).device
    before=copy.deepcopy(base.state_dict());carry=copy.deepcopy(t._carry_dict(state.carry))
    optim=copy.deepcopy([o.state_dict() for o in opts]);shadow=copy.deepcopy(ema.shadow)
    rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    _,_,tx,ty,_,_=t.load_data(cfg);tx,ty=tx[:128],ty[:128]
    result={}
    for label in ('current_eval','historical_eval','current_save'):
        if label=='current_eval':
            measured=t.evaluate(base,tx,ty,cfg,0,1,device,state.step,ema)
        elif label=='historical_eval':
            measured=old.evaluate(base,base,tx,ty,cfg,0,1,device,state.step,ema)
        else:
            measured=t.save_training_checkpoint(str(out/'checkpoint_side_effect'),state,base,opts,ema,cfg,0,1,device)
        row=dict(model_state_max_abs=delta(before,base.state_dict()),
            carry_max_abs=delta(carry,t._carry_dict(state.carry)),optimizer_max_abs=delta(optim,[o.state_dict() for o in opts]),
            EMA_max_abs=delta(shadow,ema.shadow),cpu_rng_equal=torch.equal(rng,torch.random.get_rng_state()),
            cuda_rng_equal=True if cuda_rng is None else torch.equal(cuda_rng,torch.cuda.get_rng_state(device)),
            training_mode=base.training)
        if isinstance(measured,dict):row['metrics']={k:float(v) for k,v in measured.items()}
        result[label]=row
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,default=Path('docs/research/2026-10-03/reference/train_v17.py'))
    ap.add_argument('--run',type=Path,default=Path('runs/kv_b_only_fresh_20261004'))
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--out',type=Path,default=Path('runs/harness_audit_20261004'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);install('current_only');old=load_old(args.reference)
    cfg=json.loads((args.run/'config.json').read_text())
    result=dict(reference=str(args.reference),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256(Path(t.__file__).read_bytes()).hexdigest(),
        scope='Same B-only model through old/new loss wrappers, optimizers and train_batch; architecture held fixed. One rank.',
        data=data_parity(old,cfg))
    small=dict(cfg,hidden_size=16,num_heads=2,puzzle_emb_ndim=16,global_batch_size=4,
               loops=4,blocks_per_seg=2,amp=False,compile=False,activation_checkpoint=False)
    torch.manual_seed(42)
    base=t.ACTLossHead(t.LT(dict(small,batch_size=4,seq_len=81,num_puzzle_identifiers=1)))
    result['CPU_FP32_pair'],_,_,_,_=paired_updates(old,small,base,steps=32)
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CPU paired:',result['CPU_FP32_pair']['maxima'],flush=True)
    if args.checkpoint:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.device(device):
            base=t.ACTLossHead(t.LT(dict(cfg,batch_size=128,seq_len=81,num_puzzle_identifiers=1)))
        result['checkpoint_pair'],a,sa,opts,ema=paired_updates(old,cfg,base,str(args.checkpoint),steps=16)
        result['side_effects']=side_effects(old,cfg,a,sa,opts,ema,args.out)
        print('Checkpoint paired:',result['checkpoint_pair']['maxima'],flush=True)
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',args.out/'results.json',flush=True)


if __name__=='__main__':main()
