"""Compiled versus eager gradients of two consecutive cumulative-phase blocks (BF16 autocast, fused kernel).

Usage: python docs/research/2026-10-12/scripts/check_cumphase_compiled.py [head|pair]
omega parameters are set nonzero so that the offset path carries gradient.
"""
import json, math, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t
from lt.cumulative_phase import offset_model_class
from lt.experiment_free_phase_windows import configuration

mode = sys.argv[1] if len(sys.argv) > 1 else 'head'
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
import torch._inductor.config as ic
ic.triton.persistent_reductions = False
cfg = configuration()
cfg.update(batch_size=4, seq_len=81, num_puzzle_identifiers=1, amp_dtype='bfloat16')
torch.manual_seed(3391)
with torch.device('cuda'):
    inner = offset_model_class(mode, math.pi, 'triton', 'bfloat16')(t.LTConfig.from_dict(cfg))
layer = inner.layers[0]
with torch.no_grad():
    layer.omega_proj.normal_(0, .05)
    layer.omega_bias.normal_(0, .3)
h = torch.randn(4, 81, 832, device='cuda', requires_grad=True)
inj = (torch.randn_like(h) / math.sqrt(832)).requires_grad_()
width = 1 if mode == 'head' else 52
off0 = (torch.rand(4, 8, 81, width, device='cuda') * 2 - 1) * 3
fresh = torch.tensor([True, False, False, False], device='cuda')
cot = torch.randn_like(h)
named = [('hidden', h), ('injection', inj)] + list(inner.named_parameters())

def fn(h, inj):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h1, _, off, _ = inner.block(layer, h, inj, None, off0, None, fresh)
        h2, _, off, _ = inner.block(layer, h1, inj, None, off, None, None)
    return h2

eager = fn(h, inj)
eg = torch.autograd.grad((eager * cot).sum(), [p for _, p in named], allow_unused=True)
compiled = torch.compile(fn, fullgraph=True, dynamic=False)
actual = compiled(h, inj)
ag = torch.autograd.grad((actual * cot).sum(), [p for _, p in named], allow_unused=True)
rows = {}
for (name, _), a, b in zip(named, ag, eg):
    assert (a is None) == (b is None), name
    if a is None or float(b.float().norm()) == 0:
        continue
    a, b = a.float().flatten(), b.float().flatten()
    rows[name] = dict(relative_l2=float((a - b).norm() / b.norm()), cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
out = dict(mode=mode, output_relative_l2=float((actual.float() - eager.float()).detach().norm() / eager.float().detach().norm()),
           min_cosine=min(r['cosine'] for r in rows.values()), max_relative_l2=max(r['relative_l2'] for r in rows.values()),
           omega=rows.get('layers.0.omega_proj'), gradients=rows)
print(json.dumps({k: v for k, v in out.items() if k != 'gradients'}, indent=1))
(Path(__file__).resolve().parents[1] / f'results/cumphase_compiled_check_{mode}.json').write_text(json.dumps(out, indent=1))
