"""Compiled versus eager gradients of two consecutive traced blocks (training settings: BF16 autocast).

Usage: python docs/research/2026-10-12/scripts/check_phase_trace_compiled.py [eta=0.2|none] [kernel=torch|triton]

Two blocks so that the second block's timing term reads a trace built from the first block's phases,
and the gradient flows through the trace. Reports the relative L2 error and cosine per parameter.
"""
import json, math, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.experiment_free_phase_windows import configuration, model_class

eta = (None if sys.argv[1] == 'none' else float(sys.argv[1])) if len(sys.argv) > 1 else 0.2
kernel = sys.argv[2] if len(sys.argv) > 2 else 'torch'
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
import torch._inductor.config as ic
ic.triton.persistent_reductions = False
cfg = configuration()
cfg.update(batch_size=4, seq_len=81, num_puzzle_identifiers=1, amp_dtype='bfloat16')
torch.manual_seed(3391)
with torch.device('cuda'):
    inner = model_class('pairangle', modes=1, qk_l2=True, write_sum=True, phase_frame='unrotated', dc_hebbian=True,
                        dc_alpha_init=1.0, boundary_ffn='swiglu', feature_precision='bfloat16', kernel=kernel,
                        phase_trace_eta=eta)(t.LTConfig.from_dict(cfg))
layer = inner.layers[0]
h = torch.randn(4, 81, 832, device='cuda', requires_grad=True)
inj = (torch.randn_like(h) / math.sqrt(832)).requires_grad_()
ek0 = torch.randn(4, 8, 81, 104, device='cuda') * .3
ev0 = torch.randn(4, 8, 81, 104, device='cuda') * .3
fresh = torch.tensor([True, False, False, False], device='cuda')
cot = torch.randn_like(h)
named = [('hidden', h), ('injection', inj)] + list(inner.named_parameters())


def fn(h, inj):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h1, _, ek, ev = inner.block(layer, h, inj, None, ek0, ev0, fresh)
        h2, _, ek, ev = inner.block(layer, h1, inj, None, ek, ev, None)
    return h2


eager = fn(h, inj)
eg = torch.autograd.grad((eager * cot).sum(), [p for _, p in named], allow_unused=True)
compiled = torch.compile(fn, fullgraph=True, dynamic=False)
actual = compiled(h, inj)
ag = torch.autograd.grad((actual * cot).sum(), [p for _, p in named], allow_unused=True)
rows = {}
for (name, _), a, b in zip(named, ag, eg):
    assert (a is None) == (b is None), name
    if a is None:
        continue
    a, b = a.float().flatten(), b.float().flatten()
    rows[name] = dict(relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-12)),
                      cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)), norm=float(b.norm()))
out = dict(eta=eta, kernel=kernel, output_relative_l2=float((actual.float() - eager.float()).norm() / eager.float().norm()),
           min_cosine=min(r['cosine'] for r in rows.values()), max_relative_l2=max(r['relative_l2'] for r in rows.values()),
           gradients=rows)
print(json.dumps({k: v for k, v in out.items() if k != 'gradients'}, indent=1))
for name, r in sorted(rows.items(), key=lambda kv: kv[1]['cosine'])[:5]:
    print(name, r)
dest = Path(__file__).resolve().parents[1] / f'results/phase_trace_compiled_check_eta{eta}_{kernel}.json'
dest.write_text(json.dumps(out, indent=1))
