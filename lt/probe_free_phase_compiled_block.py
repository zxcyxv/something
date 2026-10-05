"""Compiled versus eager full-block gradients, including phase/activity feedback.

This is a correctness audit, not a GPU timing benchmark.
"""
import json
import math

import torch

from . import train as t
from .experiment_free_phase_windows import configuration, model_class
from .research_free_phase_windows import DEST


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions=False
    cfg=configuration()
    cfg.update(batch_size=2,seq_len=81,num_puzzle_identifiers=1,amp_dtype='bfloat16')
    report=[]
    for window in ('exponential','fourier','biexponential'):
        torch.manual_seed(3391)
        with torch.device('cuda'):
            inner=model_class(window,True,4,.5,'diagonal')(t.LTConfig.from_dict(cfg))
        layer=inner.layers[0]
        with torch.no_grad():layer.phase_local_gain.normal_(0,.03)
        h=torch.randn(2,81,832,device='cuda',requires_grad=True)
        inj=(torch.randn_like(h)/math.sqrt(832)).requires_grad_()
        cot=torch.randn_like(h)
        named=[('hidden',h),('injection',inj)]+list(inner.named_parameters())
        def fn(h,inj):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return inner.block(layer,h,inj,None,None,None,None)[0]
        print('COMPILE',window,flush=True)
        compiled=torch.compile(fn,fullgraph=True,dynamic=False)
        eager=fn(h,inj)
        eg=torch.autograd.grad((eager*cot).sum(),[p for _,p in named],allow_unused=True)
        actual=compiled(h,inj)
        ag=torch.autograd.grad((actual*cot).sum(),[p for _,p in named],allow_unused=True)
        errors={}
        for (name,p),a,b in zip(named,ag,eg):
            assert (a is None)==(b is None),name
            if a is None:continue
            diff=(a.float()-b.float())
            relative=float(diff.norm()/b.float().norm().clamp_min(1e-12))
            errors[name]=dict(relative_l2=relative,max_abs=float(diff.abs().max()),reference_norm=float(b.float().norm()))
        out_error=float((actual.float()-eager.float()).norm()/eager.float().norm())
        row=dict(window=window,amp='bfloat16',output_relative_l2=out_error,gradient_errors=errors)
        report.append(row)
        (DEST/'compiled_block_gradient_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(window,'output',out_error,'max_gradient_relative',max(r['relative_l2'] for r in errors.values()),flush=True)
        assert out_error < .01,row
        assert max(r['relative_l2'] for r in errors.values()) < .025,row
        del inner,layer,eager,actual,ag,eg,compiled
        torch.cuda.empty_cache()


if __name__=='__main__':main()
