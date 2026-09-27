"""Continue a converted v1.71 checkpoint with fixed-set raw/EMA diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt import train


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_batch(x,y):
    return dict(inputs=torch.from_numpy(x).cuda(),labels=torch.from_numpy(y).cuda().long(),
                puzzle_identifiers=torch.zeros(len(x),dtype=torch.int32,device='cuda'))


def scalarize(value):
    if isinstance(value,dict): return {k:scalarize(v) for k,v in value.items()}
    if isinstance(value,(np.generic,torch.Tensor)): return value.item()
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'runs/v1_71_resume_380k')
    p.add_argument('--steps',type=int,default=1024,help='Additional optimizer updates')
    p.add_argument('--eval-every',type=int,default=64)
    p.add_argument('--eval-n',type=int,default=128)
    p.add_argument('--final-eval-n',type=int,default=2048)
    p.add_argument('--save-every',type=int,default=128)
    args=p.parse_args()
    if min(args.steps,args.eval_every,args.eval_n,args.final_eval_n,args.save_every)<=0:
        p.error('Step counts, evaluation sizes, and intervals must be positive')
    if (args.out/'steps.jsonl').exists():
        p.error('Output already has a training log; select a new directory')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=dict(ck['cfg'])
    if cfg.get('address_projection')!='linear' or not cfg.get('use_trace') or cfg.get('block_order')!='post':
        raise ValueError('Expected an explicitly converted v1.71 checkpoint')
    cfg.update(batch_size=128,global_batch_size=128,seq_len=81,num_puzzle_identifiers=1,
               loops=16,blocks_per_seg=8,lr=1e-4,puzzle_emb_lr=1e-4,lr_min_ratio=1.,
               lr_warmup_steps=0,grad_accum_steps=1,amp=True,compile=False,
               data_npz=str(ROOT/'data/sudoku_lt_1k.npz'),out_dir=str(args.out))
    with torch.device('cuda'):
        base=train.ACTLossHead(train.LT(cfg),'stablemax_cross_entropy',q_weight=cfg['q_weight'])
    opts,rates=train.create_optimizers(base,cfg,world_size=1)
    # This also rejects an unconverted QR checkpoint before modifying the model.
    loaded=train.load_checkpoint(str(args.checkpoint),base,opts,torch.device('cuda'))
    state=train.TrainState(step=int(loaded['step']),iter_id=int(loaded.get('iter_id',0)),
                           batch_in_iter=int(loaded.get('batch_in_iter',0)))
    if loaded.get('carry') is not None:
        state.carry=loaded['carry']
    ema=train.EMAHelper(mu=cfg['ema_rate'])
    ema.register(base)
    if loaded.get('ema_shadow'):
        ema.load_state_dict({k:v.cuda() for k,v in loaded['ema_shadow'].items()})
    start_step=state.step
    data_path=ROOT/'data/sudoku_lt_1k.npz'
    with np.load(data_path,allow_pickle=False) as data:
        tx=data['test_inputs'].reshape(-1,81).astype(np.int32)+1
        ty=data['test_labels'].reshape(-1,81).astype(np.int32)+1
        ds=train.SudokuTrainDataset(data['train_inputs'].reshape(-1,9,9),
                data['train_labels'].reshape(-1,9,9),seed=cfg['seed'],num_aug=cfg['num_aug'],
                global_batch_size=128,rank=0,world_size=1,epochs_per_iter=250,
                start_iter=state.iter_id,total_iters=state.iter_id+2+(state.batch_in_iter+args.steps)//1953,
                skip_batches=state.batch_in_iter)
    assert max(args.eval_n,args.final_eval_n)<=len(tx)
    batches=iter(ds)
    current={}
    active_training=True
    def observe_forward(module, inputs, output):
        if active_training:
            carry,out=output
            correct=out['logits'].detach().argmax(-1)==carry.current_data['labels']
            current.update(segment=int(carry.steps[0]),train_exact=int(correct.all(-1).sum()),
                           train_cell_accuracy=float(correct.float().mean()))
    hook=base.model.register_forward_hook(observe_forward)
    old_step=opts[1].step
    def observed_step():
        grads=[p.grad.detach().float() for p in base.parameters() if p.grad is not None]
        current['dense_gradient_norm']=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
        assert all(bool(torch.isfinite(g).all()) for g in grads),'Nonfinite gradient'
        result=old_step()
        for layer in base.model.inner.layers:
            assert torch.isfinite(layer.wc).all(),'Nonfinite direct projection'
        return result
    opts[1].step=observed_step
    metadata=dict(architecture='v1.71',checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),
                  source_sha256=digest(ROOT/'lt/train.py'),data_sha256=digest(data_path),config=cfg,
                  start_step=start_step,additional_steps=args.steps,eval_n=args.eval_n,
                  final_eval_n=args.final_eval_n,eval_every=args.eval_every,
                  conversion=ck.get('conversion'),initial_optimizer_restored=bool(ck.get('optimizer_states')),
                  precision='float32 weights; original bfloat16 autocast',gradient_clipping=False,
                  checkpoint_note='Periodic checkpoints include optimizer, EMA, carry and data cursor',
                  evaluation='Fixed first N held-out puzzles, fresh 16 segments, raw and EMA separately')
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    del ck,loaded
    started=time.monotonic()
    eval_log=(args.out/'eval.jsonl').open('w',buffering=1)
    def evaluate(n,kind):
        nonlocal active_training
        active_training=False
        base.eval()
        exact=cells=0
        loss_total=0.
        predictions=[]
        with train._EMASwap(base,None if kind=='raw' else ema),torch.no_grad():
            for offset in range(0,n,128):
                batch=make_batch(tx[offset:offset+128][:n-offset],ty[offset:offset+128][:n-offset])
                with torch.device('cuda'):
                    carry=base.model.initial_carry(batch)
                for _ in range(16): carry,out=base.model(carry,batch)
                correct=out['logits'].argmax(-1)==batch['labels']
                exact+=int(correct.all(-1).sum());cells+=int(correct.sum())
                loss_total+=float(train.stablemax_cross_entropy(out['logits'],batch['labels']).sum())
                predictions.append(out['logits'].argmax(-1).cpu().numpy().astype(np.uint8))
        row=dict(step=state.step,weights=kind,n=n,exact=exact,cell_accuracy=cells/(81*n),
                 lm_loss=loss_total/(81*n),elapsed_seconds=time.monotonic()-started)
        np.save(args.out/f'{kind}_pred_{state.step}_{n}.npy',np.concatenate(predictions))
        eval_log.write(json.dumps(row)+'\n')
        print(json.dumps(dict(event='eval',**row)),flush=True)
        base.train();active_training=True
        return row
    def save():
        staging=args.out/'.staging'
        path=train.save_checkpoint(str(staging),state.step,base,opts,ema,state.iter_id,state.batch_in_iter,cfg,1)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        payload['carry']=state.carry
        tmp=Path(path).with_suffix('.carry.tmp')
        destination=args.out/Path(path).name
        torch.save(payload,tmp);tmp.replace(destination)
        Path(path).unlink()
        checkpoints=sorted(args.out.glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
        for previous in checkpoints[:-2]: previous.unlink()
        return str(destination)
    evaluate(args.eval_n,'raw');evaluate(args.eval_n,'ema')
    with (args.out/'steps.jsonl').open('w',buffering=1) as log:
        for local in range(1,args.steps+1):
            current.clear()
            iteration,batch=next(batches)
            if iteration!=state.iter_id:
                state.iter_id=iteration;state.batch_in_iter=0
            metrics=train.train_batch(base,base,state,batch,cfg,opts,rates,
                       total_steps=start_step+args.steps,rank=0,world_size=1,device=torch.device('cuda'))
            ema.update(base);state.batch_in_iter+=1
            current.update(step=state.step,metrics=scalarize(metrics),elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(current)+'\n')
            if local%16==0:
                print(json.dumps(dict(event='train',**current)),flush=True)
            if local%args.save_every==0: save()
            if local%args.eval_every==0:
                evaluate(args.eval_n,'raw');evaluate(args.eval_n,'ema')
    path=save()
    final=[evaluate(args.final_eval_n,k) for k in ['raw','ema']]
    hook.remove();eval_log.close()
    summary=dict(start_step=start_step,final_step=state.step,checkpoint=path,final_evaluation=final,
                 elapsed_seconds=time.monotonic()-started)
    (args.out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
