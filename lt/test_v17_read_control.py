"""Verify that the v1.7 read ablation preserves writes and initialization."""
from pathlib import Path
import unittest

import torch

from .research_v17_normalization import load_trainer,raw_addresses


SOURCE=Path('runs/kv_collapse_20261003/reference/train_v17.py')
if not SOURCE.exists():
    SOURCE=Path('docs/research/2026-10-03/reference/train_v17.py')


@unittest.skipUnless(SOURCE.exists(),'Recovered historical v1.7 source required')
class ReadControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.models={}
        for mode in ('mixed','memory'):
            t=load_trainer(SOURCE,mode)
            t.LT_Inner._unit=raw_addresses
            cfg=dict(t.CFG,hidden_size=16,num_heads=2,grid=3,seq_len=9,
                     batch_size=2,num_puzzle_identifiers=1,puzzle_emb_ndim=16,
                     amp=False,blocks_per_seg=2)
            torch.manual_seed(0)
            self.models[mode]=t.LT(cfg).double()

    def test_parameter_initialization_is_identical(self):
        a,b=(self.models[k].state_dict() for k in ('mixed','memory'))
        self.assertEqual(set(a),set(b))
        for key in a:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_only_read_changes_with_existing_memory_and_fresh_lane(self):
        torch.manual_seed(3)
        h=torch.randn(2,9,16,dtype=torch.float64)
        old_w=torch.randn(2,2,9,9,dtype=torch.float64)
        trace=torch.randn(2,9,2,4,2,dtype=torch.float64)
        fresh=torch.tensor([True,False])
        outputs={}
        for mode,model in self.models.items():
            inner=model.inner;layer=inner.layers[0]
            outputs[mode]=inner.step(layer,h,inner.W_C(layer),inner.kernel(layer),
                old_w,fresh,inner.kernel(layer,layer.beta),trace,apply_phi=False)
        for a,b in zip(outputs['mixed'][1:],outputs['memory'][1:]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        memory_model=self.models['memory'];layer=memory_model.inner.layers[0]
        w=outputs['memory'][1]
        v=torch.einsum('btd,hcd->bthc',h,layer.w_sh)
        transported=torch.einsum('bhtn,bnhc->bthc',w,v)
        expected=h+torch.einsum('bthc,hcd->btd',transported,layer.w_sh)
        torch.testing.assert_close(outputs['memory'][0],expected,rtol=0,atol=0)
        self.assertFalse(torch.equal(outputs['memory'][0],outputs['mixed'][0]))
        outputs['memory'][0].square().mean().backward()
        self.assertIsNone(layer.lam_raw.grad)
        self.assertIsNone(layer.psi.grad)
        self.assertGreater(float(layer.w_sh.grad.norm()),0)
        self.assertGreater(float(layer.beta.grad.norm()),0)


if __name__=='__main__':
    unittest.main()
