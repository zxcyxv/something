"""Read-only CPU audit of whether trained local phases actually change order."""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .experiment_free_phase_windows import model_class
from .research_free_phase_windows import DEST


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    opt=ap.parse_args()
    torch.set_num_threads(2)
    protocol=json.loads((opt.run/'protocol.json').read_text())
    path=t.find_latest_checkpoint(str(opt.run))
    ck=torch.load(path,map_location='cpu',weights_only=False)
    cfg=dict(ck['cfg'],amp=False,amp_dtype='float32',compile=False,batch_size=2,seq_len=81,num_puzzle_identifiers=1)
    t.KVSTDPInner=model_class(protocol['window'],protocol['phase_dynamic'],protocol['modes'],protocol['epsilon'],protocol['generator'])
    model=t.ACTLossHead(t.LT(cfg),q_weight=cfg['q_weight'])
    model.load_state_dict(ck['raw_model_state_dict'],strict=True)
    model.eval()
    carrydata=ck['rank_states'][0]['carry']
    full_batch=carrydata['current_hidden'].shape[0]
    def take(value):
        if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==full_batch:
            return value[:2]
        if isinstance(value,dict):return {k:take(v) for k,v in value.items()}
        return value
    batch=take(carrydata['current_data'])
    carry=model.model.initial_carry(batch)
    inner=model.model.inner
    original=inner.memory_step
    records=[]
    capture_enabled=False
    def capture(layer,q,k,v,*args,**kwargs):
        if not capture_enabled:
            return original(layer,q,k,v,*args,**kwargs)
        kr=inner.apply_rope(k.float(),layer)
        pk,pv=inner.phases(layer,kr,v.float())
        basek,basev=[inner.phase_limit*x.tanh() for x in (layer.theta_k_raw,layer.theta_v_raw)]
        delta=pv[..., :, None]-pk[..., None, :]
        delta0=(basev[..., :, None]-basek[..., None, :])[None,:,None]
        both=(delta.amin(-3)<0)&(delta.amax(-3)>0)
        correction=torch.cat((pk-basek[None,:,None],pv-basev[None,:,None]),-1)
        records.append(dict(block=len(records),phase_correction_rms=float(correction.square().mean().sqrt()),
                            phase_correction_max=float(correction.abs().max()),
                            order_reversed_vs_current_base=float(((delta*delta0)<0).float().mean()),
                            pair_has_both_orders_over_tokens=float(both.float().mean()),
                            near_zero_002=float((delta.abs()<.02).float().mean())))
        return original(layer,q,k,v,*args,**kwargs)
    inner.memory_step=capture
    for segment in range(16):
        capture_enabled=segment==15
        carry,_=model.model(carry,batch)
    report=dict(checkpoint=str(path),step=ck['step'],weights='raw',inference='CPU FP32, two checkpoint training puzzles rolled out from fresh carry for 16 segments; record the final 8 blocks; not an accuracy evaluation',
                phase_gain_norms={n:float(p.norm()) for n,p in inner.named_parameters() if 'phase_local_gain' in n},blocks=records)
    out=DEST/(opt.run.name+'_phase_audit.json')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
