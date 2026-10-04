"""Exact phase STDP window, including its zero-lag convention."""
import math
import torch
from .kv_stability import ExponentialPhaseCurrentReadInner
from .test_phase_current import PhaseCurrentTests
from .test_kv_stability import config


class ExponentialPhaseTests(PhaseCurrentTests):
    def setUp(self):
        super().setUp()
        self.inner=ExponentialPhaseCurrentReadInner(config()).double()
        self.layer=self.inner.layers[0]

    def test_complex_outer_and_direct_phase_difference(self):
        inner,layer=self.inner,self.layer
        read,g,_,_=inner.memory_step(layer,self.q,self.k,self.v)
        qr,kr=[inner.apply_rope(x,layer) for x in (self.q,self.k)]
        pk,pv=inner.phases(layer)
        delta=pv[:,:,None]-pk[:,None,:]
        window=delta.sign()*torch.exp(-delta.abs()/inner.phase_tau)
        expected=torch.einsum('bhti,bhtj,hij->bhij',self.v,kr,window)/9
        torch.testing.assert_close(g,expected)
        torch.testing.assert_close(read,qr@expected.transpose(-1,-2))
        read.square().mean().backward()
        for x in (self.q,self.k,self.v,layer.theta_k_raw,layer.theta_v_raw):
            self.assertTrue(x.grad.isfinite().all())
            self.assertGreater(x.grad.norm().item(),0)

    def test_zero_lag_and_one_sided_limits_and_decay(self):
        inner,layer=self.inner,self.layer
        with torch.no_grad():
            layer.theta_k_raw.zero_();layer.theta_v_raw.zero_()
        self.assertEqual(inner.phase_window(layer,torch.float64).abs().max().item(),0)
        for delta in [-.1,-1e-6,1e-6,.1,1.]:
            with torch.no_grad():
                layer.theta_v_raw.fill_(math.atanh(delta/inner.phase_limit))
            w=inner.phase_window(layer,torch.float64)
            torch.testing.assert_close(w,torch.full_like(w,math.copysign(math.exp(-abs(delta)),delta)))

    def test_block_two_norms_and_optimizer_update(self):
        # Same block path as the sine variant, but exercise this window's gradients.
        from unittest.mock import patch
        inner=self.inner.float(); layer=self.layer
        h,inj=torch.randn(2,9,16),torch.randn(2,9,16)
        with patch.object(inner,'phi',wraps=inner.phi) as norm:
            output,*_=inner.block(layer,h,inj,None,None,None,None)
            self.assertEqual(norm.call_count,2)
        before=layer.theta_k_raw.detach().clone()
        opt=torch.optim.Adam(inner.parameters(),lr=1e-3)
        output[...,0].square().mean().backward();opt.step()
        self.assertFalse(torch.equal(before,layer.theta_k_raw))
        self.assertTrue(output.isfinite().all())
