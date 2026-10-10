"""Profile compiled training steps of alpha=1 (+SwiGLU) and report where GPU time goes."""
import json, sys, time, statistics
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.experiment_free_phase_windows import model_class

precision = sys.argv[1] if len(sys.argv) > 1 else 'float32'
run = Path('/workspace/something/runs/pairangle_unrotated_dca1_swiglu_tau2_r2_qkl2_sum_20261010')
ck = torch.load(run / 'step_3475.pt', map_location='cpu', weights_only=False)
protocol = json.loads((run / 'protocol.json').read_text())
cfg = dict(ck['cfg']); cfg['activation_checkpoint'] = (sys.argv[2] != 'nockpt') if len(sys.argv) > 2 else True
dev = torch.device('cuda')
torch.set_float32_matmul_precision(protocol['precision'])
t._resolve_precision(cfg, dev)
tr_x, tr_y, *_ = t.load_data(cfg)
batch = next(t.eval_batches(tr_x[:128], tr_y[:128], 128, 0, 1))
t.KVSTDPInner = model_class('pairangle', True, 2, 0.35, 'dense', precision, protocol.get('window_scale_factor', 1.0),
    False, False, True, True, 2.0, 0.5, 'none', False, 1.0, 0.0, 'unrotated', True, 1.0, 'swiglu', sys.argv[3] if len(sys.argv) > 3 else 'torch')
with torch.device(dev):
    base = t.ACTLossHead(t.LT(dict(cfg, batch_size=128, seq_len=81, num_puzzle_identifiers=1)), q_weight=cfg['q_weight'])
base.load_state_dict(ck['raw_model_state_dict']); base.train()
opts, lrs = t.create_optimizers(base, cfg, 1)
import torch._inductor.config as ic
ic.triton.persistent_reductions = False
import torch._dynamo
torch._dynamo.config.verbose = False
compiled = torch.compile(base, dynamic=False)
state = t.TrainState()
step = lambda: t.train_batch(compiled, base, state, batch, cfg, opts, lrs, 390625, 0, 1, dev)
for _ in range(6): step()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5): step()
    torch.cuda.synchronize()
import re, collections
from torch.autograd import DeviceType
ev=[e for e in prof.events() if e.device_type==DeviceType.CUDA]
tot=sum(e.device_time for e in ev)/5/1000
span=(max(e.time_range.end for e in ev)-min(e.time_range.start for e in ev))/5/1000
print(f'kernels/step {len(ev)//5}  kernel time/step {tot:.1f} ms  GPU span/step {span:.1f} ms  (idle {span-tot:.1f} ms)')
def cat(n):
    if re.search(r'sgemm|gemm.*f32|_s1688gemm_fp32|cublas.*fp32', n) and 'bf16' not in n: return 'GEMM fp32'
    if 'gemm' in n or 'cutlass' in n: return 'GEMM bf16'
    if re.search(r'_fwd_kernel|_bwd_dg|_bwd_v|_bwd_k|_bwd_q', n): return 'FUSED pairangle kernels'
    if re.search(r'atan2|cos|sin', n): return 'phase elementwise (atan2/sin/cos fused)'
    if 'silu' in n: return 'SwiGLU elementwise'
    if re.search(r'norm|rsqrt|mean|pow', n): return 'norm / reduction'
    if 'triton' in n: return 'other triton fused'
    if 'elementwise' in n or 'vectorized' in n or 'copy' in n.lower() or 'fill' in n.lower(): return 'eager elementwise/copy'
    return 'other'
c=collections.Counter(); k=collections.Counter()
for e in ev: c[cat(e.name)]+=e.device_time; k[cat(e.name)]+=1
for name,v in c.most_common(): print(f'{v/5/1000:7.2f} ms  {v/5/1000/tot*100:5.1f}%  kernels {k[name]//5:4d}  {name}')
others=collections.Counter()
for e in ev:
    if cat(e.name) not in ('GEMM bf16','FUSED pairangle kernels'): others[e.name[:150]]+=e.device_time
print('-- top other:')
for n,v in others.most_common(14): print(f'{v/5/1000:6.2f} ms  {n}')
print(f'peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, activation_checkpoint={cfg["activation_checkpoint"]}')
fk=collections.Counter(); fn=collections.Counter()
for e in ev:
    if cat(e.name)=='FUSED pairangle kernels': fk[e.name[:40]]+=e.device_time; fn[e.name[:40]]+=1
for n_,v_ in fk.most_common(): print(f'  fused {n_}: {v_/5/1000:.2f} ms/step  calls {fn[n_]//5}  per call {v_/fn[n_]/1000:.3f} ms')
