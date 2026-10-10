import json, sys, re, collections
import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.urm_full_bptt import install
run='/workspace/something/runs/urm_swiglu_1layer_loops16_20261010'
cfg=dict(t.CFG); cfg.update(json.load(open(f'{run}/config.json')))
cfg['activation_checkpoint'] = sys.argv[1] != 'nockpt'
install('swiglu', 1)
dev=torch.device('cuda'); t._resolve_precision(cfg, dev)
tr_x,tr_y,*_=t.load_data(cfg); batch=next(t.eval_batches(tr_x[:128],tr_y[:128],128,0,1))
with torch.device(dev):
    base=t.ACTLossHead(t.LT(dict(cfg,batch_size=128,seq_len=81,num_puzzle_identifiers=1)),q_weight=cfg['q_weight'])
base.train(); opts,lrs=t.create_optimizers(base,cfg,1)
import torch._inductor.config as ic; ic.triton.persistent_reductions=False
comp=torch.compile(base,dynamic=False); st=t.TrainState()
step=lambda: t.train_batch(comp,base,st,batch,cfg,opts,lrs,390625,0,1,dev)
for _ in range(6): step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()
ev=[e for e in prof.events() if e.device_type==DeviceType.CUDA]
tot=sum(e.device_time for e in ev)/5/1000
c=collections.Counter()
for e in ev:
    n=e.name; k='GEMM' if ('gemm' in n or 'cutlass' in n) else 'attention (flash/sdpa)' if re.search('flash|fmha|attention|sdpa',n) else 'elementwise/other'
    c[k]+=e.device_time
print(f'URM ckpt={cfg["activation_checkpoint"]}: kernel time/step {tot:.1f} ms, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
for k,v in c.most_common(): print(f'  {v/5/1000:6.2f} ms  {k}')
