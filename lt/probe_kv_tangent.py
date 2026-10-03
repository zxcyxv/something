"""Measure coupled hidden/memory/trace tangent growth at frozen checkpoints."""
import argparse
import json
import math
from pathlib import Path

import torch

from . import train as t
from .kv_stability import install


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out",required=True)
    ap.add_argument("--blocks",type=int,default=64)
    ap.add_argument("--batch",type=int,default=2)
    args=ap.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    install(ck['cfg'].get('research_variant','original'))
    cfg=dict(ck['cfg'],batch_size=args.batch,seq_len=81,num_puzzle_identifiers=1,
             activation_checkpoint=False,compile=False)
    with torch.device('cuda'):
        model=t.LT(cfg)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['raw_model_state_dict'].items()})
    model.eval()
    saved=ck['rank_states'][0]['carry']
    batch={k:v[:args.batch].cuda() for k,v in saved['current_data'].items()}
    inner=model.inner
    inj=inner.injection(batch).detach()
    layer=inner.layers[0]
    state=tuple(saved[name][:args.batch].cuda() for name in ('current_hidden','coupling','key_trace','value_trace'))
    tangent=tuple(torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)*.173+i).reshape_as(x) for i,x in enumerate(state))
    def norm(vec):
        return torch.sqrt(sum(x.float().square().sum() for x in vec))
    size=norm(tangent)
    tangent=tuple(x/size for x in tangent)
    def fn(h,m,ek,ev):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            return inner.block(layer,h,inj,m,ek,ev,None)
    log_growth=0.
    records=[]
    for step in range(1,args.blocks+1):
        state,tangent=torch.func.jvp(fn,state,tangent)
        state=tuple(x.detach() for x in state)
        tangent=tuple(x.detach() for x in tangent)
        size=float(norm(tangent))
        log_growth+=math.log(size)
        records.append(dict(block=step,gain=size,log_growth=log_growth,mean_log_gain=log_growth/step))
        tangent=tuple(x/size for x in tangent)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(step=ck['step'],initial_state='saved training carry',records=records),indent=2))
    print(ck['step'],records[-1],flush=True)


if __name__=='__main__':
    main()
