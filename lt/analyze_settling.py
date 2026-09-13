"""Compare temporary correct outputs with a settled correct joint state.

Capture the original batch-128 trajectory under a 1 GiB PyTorch allocator cap,
then use batch 1 for matched state-swap continuations. No other GPU processes
are changed. Compare each small-batch native control with its original trace.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import train
from ckpt_npz import load_data, load_lt
from probe_phase_feedback import Blocks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="runs/relation_transport_v11")
    ap.add_argument("--out",default="runs/settling_v11")
    ap.add_argument("--steps",type=int,default=256)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cap=1024**3
    torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
    model,_,_=load_lt("checkpoints/v1.1_step160000.npz",mod=train,batch_size=128,loops=129,amp=False)
    _,_,batch=load_data(n=128);b=Blocks(model,batch)
    puzzle,cell=1,48
    labels=batch["labels"][puzzle].cpu().numpy()
    original=np.load(Path(args.root)/"normal.npz")["pred"][:,puzzle]
    h=b.inner.init_hidden.expand(128,81,-1).clone();w=None
    snapshots={};records=[];previous_h=None;previous_w=None
    for k in range(1,257):
        hn,wn=b.block(h,w)
        logits=b.inner.w_cls(hn)[puzzle]
        pred=logits.argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(pred,original[k])
        gold=logits.gather(-1,batch["labels"][puzzle,:,None]).squeeze(-1)
        others=logits.clone().scatter_(-1,batch["labels"][puzzle,:,None],-torch.inf).amax(-1)
        margin=gold-others
        dh=float((hn[puzzle]-h[puzzle]).norm()/h[puzzle].norm())
        dw=None if w is None else float((wn[puzzle]-w[puzzle]).norm()/w[puzzle].norm())
        records.append(dict(block=k,wrong=int((pred!=labels).sum()),target_digit=int(pred[cell])-1,
                            target_margin=float(margin[cell]),min_margin=float(margin.min()),
                            hidden_relative_step=dh,memory_relative_step=dw))
        h,w=hn,wn
        if k in (152,156,160,192,256):
            snapshots[k]=(h[puzzle:puzzle+1].cpu(),w[puzzle:puzzle+1].cpu())
        if k in (152,153,154,155,156,157,158,159,160,168,192,256):print("TRACE",records[-1],flush=True)
    capture_peak=torch.cuda.max_memory_allocated()
    single_inj=b.inj[puzzle:puzzle+1].clone()
    del h,w,hn,wn,logits,gold,others,margin,batch
    b.inj=single_inj
    torch.cuda.empty_cache()
    report={"puzzle":puzzle,"cell":cell,"precision":"FP32; autocast/TF32 off",
            "capture_batch":128,"continuation_batch":1,"allocator_cap_bytes":cap,
            "capture_peak_allocated_bytes":capture_peak,"records":records,"swaps":{}}
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    conditions=[]
    for local,rest,memory in itertools.product((152,192),repeat=3):
        conditions.append((f"local{local}_rest{rest}_memory{memory}",local,rest,memory))
    for hidden,memory in itertools.product((156,192),repeat=2):
        name=f"hidden{hidden}_memory{memory}"
        conditions.append((name,hidden,hidden,memory))
    traces={}
    for name,local,rest,memory in conditions:
        h=snapshots[rest][0].cuda().clone()
        h[:,cell]=snapshots[local][0][:,cell].cuda()
        w=snapshots[memory][1].cuda().clone()
        predictions=[b.inner.w_cls(h)[0].argmax(-1).cpu().numpy()]
        for step in range(1,args.steps+1):
            h,w=b.block(h,w)
            predictions.append(b.inner.w_cls(h)[0].argmax(-1).cpu().numpy())
        p=np.stack(predictions)
        errors=(p!=labels).sum(-1);target_bad=p[:,cell]!=labels[cell]
        wrong_steps=np.flatnonzero(errors>0)
        target_wrong_steps=np.flatnonzero(target_bad)
        result={"initial_error_count":int(errors[0]),"initial_target_digit":int(p[0,cell])-1,
                "target_wrong_steps":target_wrong_steps.tolist(),
                "max_error_count":int(errors.max()),"final_error_count":int(errors[-1]),
                "last_puzzle_wrong_step":int(wrong_steps[-1]) if len(wrong_steps) else None,
                "first_12_error_counts":errors[:13].tolist(),"first_12_target_digits":(p[:13,cell]-1).tolist()}
        if local==rest==memory:
            reference=original[local:local+len(p)]
            result["small_batch_vs_original_cell_disagreements"]=int((p!=reference).sum())
        report["swaps"][name]=result;traces[name]=p
        (out/"summary.json").write_text(json.dumps(report,indent=2))
        print("SWAP",name,result,flush=True)
    np.savez_compressed(out/"swaps.npz",**traces)
    torch.save(snapshots,out/"snapshots.pt")
    print("saved",out,flush=True)


if __name__=="__main__":main()
