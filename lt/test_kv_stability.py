"""Check the invariants the experimental stability controls promise."""
import math
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch

from . import train as t
from .kv_stability import (ORIGINAL_INNER,BoundedReadInner,PreFFNPhiInner,
                          UnitQKActivityInner,InterpolatedReadInner,CurrentReadInner,
                          ComplexCurrentReadInner,CurrentPlusSTDPInner,QuarterReadInner)


def config():
    return t.LTConfig.from_dict(dict(t.CFG,hidden_size=16,num_heads=2,grid=3,
        seq_len=9,batch_size=2,num_puzzle_identifiers=1,puzzle_emb_ndim=16,
        amp=False,activation_checkpoint=False,blocks_per_seg=2,nograd_blocks=0))


class KVStabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)

    def test_bounded_read_preserves_stdp_write_and_traces(self):
        original=ORIGINAL_INNER(config())
        bounded=BoundedReadInner(config())
        bounded.load_state_dict(original.state_dict())
        q,k,v,ek,ev=[torch.randn(2,2,9,8) for _ in range(5)]
        memory=torch.randn(2,2,8,8)
        a=original.memory_step(original.layers[0],q,k,v,memory,ek,ev)
        b=bounded.memory_step(bounded.layers[0],q,k,v,memory,ek,ev)
        for x,y in zip(a[1:],b[1:]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertFalse(torch.equal(a[0],b[0]))

    def test_quarter_read_preserves_writes_traces_and_initialization(self):
        torch.manual_seed(91); original=ORIGINAL_INNER(config()).double()
        torch.manual_seed(91); scaled=QuarterReadInner(config()).double()
        for name,value in original.state_dict().items():
            torch.testing.assert_close(scaled.state_dict()[name],value,rtol=0,atol=0)
        q,k,v,ek,ev=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(5)]
        memory=torch.randn(2,2,8,8,dtype=torch.float64)
        fresh=torch.tensor([False,True])
        a=original.memory_step(original.layers[0],q,k,v,memory,ek,ev,fresh)
        b=scaled.memory_step(scaled.layers[0],q,k,v,memory,ek,ev,fresh)
        torch.testing.assert_close(b[0],0.25*a[0],rtol=0,atol=0)
        for x,y in zip(a[1:],b[1:]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_read_gain_is_bounded_even_for_large_stored_memory(self):
        inner=BoundedReadInner(config())
        q=torch.randn(2,2,9,8)
        zero=torch.zeros_like(q)
        memory=torch.randn(2,2,8,8)*1e10
        read,m,_,_=inner.memory_step(inner.layers[0],q,zero,zero,memory)
        torch.testing.assert_close(m,memory,rtol=0,atol=0)
        limit=4*math.sqrt(inner.dh)*q.norm(dim=-1)
        self.assertTrue((read.norm(dim=-1)<=limit+1e-4).all())
        self.assertTrue(read.isfinite().all())
        self.assertGreater(float(read.detach().norm()),0)

    def test_pre_ffn_phi_bounds_input_and_keeps_finite_gradient(self):
        inner=PreFFNPhiInner(config())
        layer=inner.layers[0]
        with torch.no_grad():
            layer.b_down.weight.normal_(0,.02)
        captured=[]
        handle=layer.b_gate_up.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
        h=(torch.randn(2,9,16)*1000).requires_grad_()
        result=inner.boundary(layer,h)
        result.square().mean().backward()
        handle.remove()
        self.assertLessEqual(float(captured[0].detach().norm(dim=-1).max()),math.sqrt(inner.d))
        self.assertTrue(h.grad.isfinite().all())
        self.assertGreater(float(h.grad.norm()),0)

    def test_read_normalizer_has_finite_gradient_at_zero_memory(self):
        inner=BoundedReadInner(config())
        q,k,v=[torch.randn(2,2,9,8,requires_grad=True) for _ in range(3)]
        memory=torch.zeros(2,2,8,8,requires_grad=True)
        read,_,_,_=inner.memory_step(inner.layers[0],q,k,v,memory)
        read.sum().backward()
        for x in (q,k,v,memory):
            self.assertTrue(x.grad.isfinite().all())

    def test_unit_addresses_keep_pair_stdp_in_normalized_activity_space(self):
        cfg=config();cfg.kv_write_reduction='sum'
        inner=UnitQKActivityInner(cfg).double()
        layer=inner.layers[0]
        memory=ek=ev=None
        history=[]
        expected=torch.zeros(2,2,8,8,dtype=torch.float64)
        lam=layer.trace_decay_channels[None,:,None,:]
        for step in range(4):
            q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
            kn=k/(k.norm(dim=-1,keepdim=True)+cfg.eps)
            if history:
                past_k=sum((1-lam)*lam**(step-j-1)*old_k for j,(old_k,_) in enumerate(history))
                past_v=sum((1-lam)*lam**(step-j-1)*old_v for j,(_,old_v) in enumerate(history))
                expected=expected+v.transpose(-1,-2)@inner.apply_rope(past_k,layer)-past_v.transpose(-1,-2)@inner.apply_rope(kn,layer)
            _,memory,ek,ev=inner.memory_step(layer,q,k,v,memory,ek,ev)
            torch.testing.assert_close(memory,expected,rtol=1e-10,atol=1e-10)
            history.append((kn,v))

    def test_unit_addresses_remove_cubic_activity_scaling_of_memory_read(self):
        inner=UnitQKActivityInner(config()).double()
        samples=[[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)] for _ in range(4)]
        def run(scale):
            m=ek=ev=None
            for q,k,v in samples:
                read,m,ek,ev=inner.memory_step(inner.layers[0],scale*q,scale*k,scale*v,m,ek,ev)
            return read
        torch.testing.assert_close(run(3),3*run(1),rtol=5e-4,atol=5e-4)

    def test_unit_addresses_zero_activity_has_finite_gradient(self):
        inner=UnitQKActivityInner(config())
        q,k,v=[torch.zeros(2,2,9,8,requires_grad=True) for _ in range(3)]
        read,m,ek,ev=inner.memory_step(inner.layers[0],q,k,v)
        (read.sum()+m.sum()+ek.sum()+ev.sum()).backward()
        for x in (q,k,v):
            self.assertTrue(x.grad.isfinite().all())

    def test_unit_activity_rejects_independent_trace_renormalization(self):
        cfg=config();cfg.kv_qk_l2norm=True
        with self.assertRaisesRegex(ValueError,'cannot be combined'):
            UnitQKActivityInner(cfg)

    def test_interpolated_read_preserves_initialization_and_all_stdp_states(self):
        torch.manual_seed(7);original=ORIGINAL_INNER(config()).double()
        torch.manual_seed(7);mixed=InterpolatedReadInner(config()).double()
        for key,value in original.state_dict().items():
            torch.testing.assert_close(mixed.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(tuple(mixed.layers[0].read_lam_raw.shape),(2,1,1))
        q,k,v,ek,ev=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(5)]
        m=torch.randn(2,2,8,8,dtype=torch.float64)
        fresh=torch.tensor([False,True])
        a=original.memory_step(original.layers[0],q,k,v,m,ek,ev,fresh)
        b=mixed.memory_step(mixed.layers[0],q,k,v,m,ek,ev,fresh)
        for x,y in zip(a[1:],b[1:]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        layer=mixed.layers[0]
        # Independent token x token computation verifies the channel contraction.
        qr=mixed.apply_rope(q,layer);kr=mixed.apply_rope(k,layer)
        now=(qr@kr.transpose(-1,-2))@v/k.shape[-2]
        lam=layer.read_lam_raw.sigmoid()[None]
        torch.testing.assert_close(b[0],(1-lam)*now+lam*a[0],rtol=1e-12,atol=1e-12)
        b[0].square().sum().backward()
        self.assertTrue(layer.read_lam_raw.grad.isfinite().all())
        self.assertGreater(float(layer.read_lam_raw.grad.norm()),0)

    def test_interpolation_endpoints_select_each_read_and_keep_first_write_zero(self):
        inner=InterpolatedReadInner(config()).double();layer=inner.layers[0]
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
        with torch.no_grad():layer.read_lam_raw.fill_(-100)
        read,m,_,_=inner.memory_step(layer,q,k,v)
        expected=(inner.apply_rope(q,layer)@inner.apply_rope(k,layer).transpose(-1,-2))@v/9
        torch.testing.assert_close(m,torch.zeros_like(m),rtol=0,atol=0)
        torch.testing.assert_close(read,expected,rtol=1e-12,atol=1e-12)
        with torch.no_grad():layer.read_lam_raw.fill_(100)
        read,m,_,_=inner.memory_step(layer,q,k,v)
        torch.testing.assert_close(read,torch.zeros_like(read),rtol=0,atol=0)

    def test_current_read_matches_token_attention_and_ignores_all_history(self):
        torch.manual_seed(7);original=ORIGINAL_INNER(config()).double()
        torch.manual_seed(7);inner=CurrentReadInner(config()).double()
        self.assertEqual(set(inner.state_dict()),set(original.state_dict()))
        for key,value in original.state_dict().items():
            torch.testing.assert_close(inner.state_dict()[key],value,rtol=0,atol=0)
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        m=torch.full((2,2,8,8),float('nan'),dtype=torch.float64,requires_grad=True)
        ek,ev=[torch.full_like(k,float('nan'),requires_grad=True) for _ in range(2)]
        fresh=torch.tensor([False,True])
        read,current,new_ek,new_ev=inner.memory_step(inner.layers[0],q,k,v,m,ek,ev,fresh)
        no_history=inner.memory_step(inner.layers[0],q,k,v)
        torch.testing.assert_close(read,no_history[0],rtol=0,atol=0)
        torch.testing.assert_close(current,no_history[1],rtol=0,atol=0)
        self.assertIsNone(new_ek);self.assertIsNone(new_ev)
        expected=(inner.apply_rope(q,inner.layers[0])@
                  inner.apply_rope(k,inner.layers[0]).transpose(-1,-2))@v/9
        torch.testing.assert_close(read,expected,rtol=1e-12,atol=1e-12)
        actual_grads=torch.autograd.grad(read.square().sum(),(q,k,v,m,ek,ev),
                                        retain_graph=True,allow_unused=True)
        expected_grads=torch.autograd.grad(expected.square().sum(),(q,k,v))
        for actual,wanted in zip(actual_grads[:3],expected_grads):
            torch.testing.assert_close(actual,wanted,rtol=1e-11,atol=1e-11)
            self.assertTrue(actual.isfinite().all())
            self.assertGreater(float(actual.norm()),0)
        self.assertEqual(actual_grads[3:],(None,None,None))

    def test_current_read_full_recurrence_has_no_memory_or_trace_dependence(self):
        cfg=config()
        with patch.object(t,'KVSTDPInner',CurrentReadInner):
            model=t.LT(vars(cfg))
        batch=dict(inputs=torch.randint(1,cfg.vocab_size,(2,9)),
                   labels=torch.randint(1,cfg.vocab_size,(2,9)),
                   puzzle_identifiers=torch.zeros(2,dtype=torch.int32))
        carry,output=model(model.initial_carry(batch),batch)
        self.assertIsNone(carry.key_trace);self.assertIsNone(carry.value_trace)
        dirty=replace(carry,coupling=torch.full_like(carry.coupling,float('nan')),
                      key_trace=torch.full((2,2,9,8),float('nan')),
                      value_trace=torch.full((2,2,9,8),float('nan')))
        clean_result,clean_output=model(carry,batch)
        dirty_result,dirty_output=model(dirty,batch)
        torch.testing.assert_close(clean_output['logits'],dirty_output['logits'],rtol=0,atol=0)
        torch.testing.assert_close(clean_result.coupling,dirty_result.coupling,rtol=0,atol=0)
        dirty_output['logits'].square().mean().backward()
        layer=model.inner.layers[0]
        self.assertIsNone(layer.trace_lam_raw.grad)
        for projection in [layer.q_proj,layer.k_proj,layer.v_proj,layer.out_proj]:
            self.assertTrue(projection.weight.grad.isfinite().all())
            self.assertGreater(float(projection.weight.grad.norm()),0)

    def test_complex_current_read_preserves_initialization_traces_and_complex_product(self):
        torch.manual_seed(7);original=ORIGINAL_INNER(config()).double()
        torch.manual_seed(7);inner=ComplexCurrentReadInner(config()).double()
        for key,value in original.state_dict().items():
            torch.testing.assert_close(inner.state_dict()[key],value,rtol=0,atol=0)
        q,k,v,ek,ev=[torch.randn(2,2,9,8,dtype=torch.float64,requires_grad=True) for _ in range(5)]
        m=torch.full((2,2,8,8),float('nan'),dtype=torch.float64,requires_grad=True)
        fresh=torch.tensor([False,True])
        layer=inner.layers[0]
        read,g,new_ek,new_ev=inner.memory_step(layer,q,k,v,m,ek,ev,fresh)
        _,_,orig_ek,orig_ev=original.memory_step(original.layers[0],q,k,v,None,ek,ev,fresh)
        torch.testing.assert_close(new_ek,orig_ek,rtol=0,atol=0)
        torch.testing.assert_close(new_ev,orig_ev,rtol=0,atol=0)
        old_ek=torch.where(fresh[:,None,None,None],torch.zeros_like(ek),ek)
        old_ev=torch.where(fresh[:,None,None,None],torch.zeros_like(ev),ev)
        kc=torch.complex(inner.apply_rope(old_ek,layer),inner.apply_rope(k,layer))
        vc=torch.complex(old_ev,v)
        expected_g=(vc.transpose(-1,-2)@kc.conj()).imag/9
        expected_read=inner.apply_rope(q,layer)@expected_g.transpose(-1,-2)
        torch.testing.assert_close(g,expected_g,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(read,expected_read,rtol=1e-12,atol=1e-12)
        gradients=torch.autograd.grad(read.square().sum(),(q,k,v,ek,ev,m),allow_unused=True)
        for grad in gradients[:-1]:
            self.assertTrue(grad.isfinite().all())
            self.assertGreater(float(grad.norm()),0)
        self.assertIsNone(gradients[-1])

    def test_complex_current_read_keeps_temporal_trace_gradients_without_accumulation(self):
        inner=ComplexCurrentReadInner(config()).double()
        layer=inner.layers[0]
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
        first,g,ek,ev=inner.memory_step(layer,q,k,v)
        torch.testing.assert_close(first,torch.zeros_like(first),rtol=0,atol=0)
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
        read,g,_,_=inner.memory_step(layer,q,k,v,torch.randn_like(g)*1e8,ek,ev)
        other,other_g,_,_=inner.memory_step(layer,q,k,v,None,ek,ev)
        torch.testing.assert_close(read,other,rtol=0,atol=0)
        torch.testing.assert_close(g,other_g,rtol=0,atol=0)
        read.square().mean().backward()
        self.assertTrue(layer.trace_lam_raw.grad.isfinite().all())
        self.assertGreater(float(layer.trace_lam_raw.grad.norm()),0)

    def test_current_plus_stdp_has_exact_three_terms_and_preserves_traces(self):
        torch.manual_seed(7);original=ORIGINAL_INNER(config()).double()
        torch.manual_seed(7);inner=CurrentPlusSTDPInner(config()).double()
        self.assertEqual(set(original.state_dict()),set(inner.state_dict()))
        for key,value in original.state_dict().items():
            torch.testing.assert_close(inner.state_dict()[key],value,rtol=0,atol=0)
        q,k,v,ek,ev=[torch.randn(2,2,9,8,dtype=torch.float64,requires_grad=True) for _ in range(5)]
        m=torch.full((2,2,8,8),float('nan'),dtype=torch.float64,requires_grad=True)
        fresh=torch.tensor([False,True])
        layer=inner.layers[0]
        read,operator,new_ek,new_ev=inner.memory_step(layer,q,k,v,m,ek,ev,fresh)
        _,_,orig_ek,orig_ev=original.memory_step(original.layers[0],q,k,v,None,ek,ev,fresh)
        torch.testing.assert_close(new_ek,orig_ek,rtol=0,atol=0)
        torch.testing.assert_close(new_ev,orig_ev,rtol=0,atol=0)
        old_ek=torch.where(fresh[:,None,None,None],torch.zeros_like(ek),ek)
        old_ev=torch.where(fresh[:,None,None,None],torch.zeros_like(ev),ev)
        qr,kr,past_kr=[inner.apply_rope(x,layer) for x in (q,k,old_ek)]
        complex_g=(torch.complex(old_ev,v).transpose(-1,-2)@
                   torch.complex(past_kr,kr).conj()).imag/9
        expected=complex_g+v.transpose(-1,-2)@kr/9
        torch.testing.assert_close(operator,expected,rtol=1e-12,atol=1e-12)
        token_read=((qr@kr.transpose(-1,-2))@v+
                    (qr@past_kr.transpose(-1,-2))@v-
                    (qr@kr.transpose(-1,-2))@old_ev)/9
        torch.testing.assert_close(read,token_read,rtol=1e-12,atol=1e-12)
        gradients=torch.autograd.grad(read.square().sum(),(q,k,v,ek,ev,m),allow_unused=True)
        for grad in gradients[:-1]:
            self.assertTrue(grad.isfinite().all())
            self.assertGreater(float(grad.norm()),0)
        self.assertIsNone(gradients[-1])

    def test_current_plus_stdp_first_read_is_real_kv_and_later_trace_gradient_survives(self):
        inner=CurrentPlusSTDPInner(config()).double()
        layer=inner.layers[0]
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
        first,operator,ek,ev=inner.memory_step(layer,q,k,v)
        reference=CurrentReadInner(config()).double()
        reference.load_state_dict(inner.state_dict())
        real_read,real_matrix,_,_=reference.memory_step(reference.layers[0],q,k,v)
        torch.testing.assert_close(first,real_read,rtol=0,atol=0)
        torch.testing.assert_close(operator,real_matrix,rtol=0,atol=0)
        q,k,v=[torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
        read,_,_,_=inner.memory_step(layer,q,k,v,operator,ek,ev)
        read.square().mean().backward()
        self.assertTrue(layer.trace_lam_raw.grad.isfinite().all())
        self.assertGreater(float(layer.trace_lam_raw.grad.norm()),0)


if __name__=='__main__':
    unittest.main()
