"""Measure frozen v1.7 dynamics, using the actual normalization ablation code."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .research_v17_normalization import load_trainer,raw_addresses


def rms(x):
    return float(x.float().square().mean().sqrt())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out',required=True)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--segments',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(2)
    path=Path(args.checkpoint)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    t=load_trainer(path.parent/'trainer_snapshot.py')
    if not ck['cfg'].get('research_address_norm',
                         ck['cfg'].get('research_variant')=='v17_original'):
        t.LT_Inner._unit=raw_addresses
    cfg=dict(ck['cfg'],batch_size=args.batch,seq_len=81,num_puzzle_identifiers=1)
    with np.load(cfg['data_npz']) as data:
        x,y=data['test_inputs'][:args.batch],data['test_labels'][:args.batch]
    batch=dict(inputs=torch.tensor(x.reshape(-1,81).astype(np.int32)+1,device='cuda'),
               labels=torch.tensor(y.reshape(-1,81).astype(np.int32)+1,device='cuda'),
               puzzle_identifiers=torch.zeros(args.batch,dtype=torch.int32,device='cuda'))
    with torch.device('cuda'):
        model=t.LT(cfg)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['raw_model_state_dict'].items()})
    model.eval();inner=model.inner
    original_attn,original_step,original_boundary,original_phi=inner.attn_xy,inner.step,inner.boundary,inner.phi
    records=[];current={};hidden_history=[inner.init_hidden.expand(args.batch,81,-1)]
    predictions=[];kernels=[]

    def attention(xy,kc):
        a=original_attn(xy,kc)
        kernels.append(dict(address_norm=float((xy[0].float().square()+xy[1].float().square()).sum(-1).sqrt().mean()),
                            kernel_rms=rms(a),kernel_max=float(a.abs().max())))
        return a

    def step(layer,h,*args,**kwargs):
        current.clear();kernels.clear()
        value,w,z=original_step(layer,h,*args,**kwargs)
        current.update(input_rms=rms(h),read_delta_rms=rms(value-h),coupling_rms=rms(w),trace_rms=rms(z),
                       read_kernel=dict(kernels[0]),write_kernel=dict(kernels[1]))
        return value,w,z

    def boundary(layer,h,*args,**kwargs):
        value=original_boundary(layer,h,*args,**kwargs)
        current.update(pre_ffn_rms=rms(h),ffn_delta_rms=rms(value-h),post_ffn_rms=rms(value))
        return value

    def phi(h):
        value=original_phi(h)
        pred=inner.w_cls(value).argmax(-1)
        current.update(block=len(records)+1,hidden_rms=rms(value),hidden_change_rms=rms(value-hidden_history[-1]),
                       accuracy=float((pred==batch['labels']).float().mean()))
        if len(hidden_history)>1:
            current['hidden_two_step_change_rms']=rms(value-hidden_history[-2])
        if predictions:
            current['prediction_flip_rate']=float((pred!=predictions[-1]).float().mean())
        if len(predictions)>1:
            current['prediction_two_step_flip_rate']=float((pred!=predictions[-2]).float().mean())
        hidden_history.append(value);del hidden_history[:-2]
        predictions.append(pred);del predictions[:-2]
        records.append(dict(current))
        return value

    inner.attn_xy,inner.step,inner.boundary,inner.phi=attention,step,boundary,phi
    segments=[]
    with torch.no_grad(),torch.device('cuda'):
        carry=model.initial_carry(batch)
        for segment in range(1,args.segments+1):
            carry,output=model(carry,batch)
            logits=output['logits'];pred=logits.argmax(-1)
            segments.append(dict(segment=segment,accuracy=float((pred==batch['labels']).float().mean()),
                                 loss=float(t.stablemax_cross_entropy(logits,batch['labels']).mean())))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(checkpoint=str(path.resolve()),step=ck['step'],data='test',
                                  segments=segments,blocks=records),indent=2))
    print(segments[-1],flush=True)


if __name__=='__main__':
    main()
