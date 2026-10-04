import unittest
from dataclasses import replace
import torch
from .test_kv_stability import config
from .urm_full_bptt import URMFullBPTTInner


class URMFullBPTTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.cfg=replace(config(),memory_type='address',loops=1,num_layers=2,
                         blocks_per_seg=8,puzzle_emb_ndim=16)
        self.batch=dict(inputs=torch.randint(0,11,(2,9)),
                        puzzle_identifiers=torch.zeros(2,dtype=torch.long))

    def carry(self,m):
        return m.reset_carry(torch.ones(2,dtype=torch.bool),m.empty_carry(2))

    def test_all_16_layer_outputs_receive_gradient(self):
        m=URMFullBPTTInner(self.cfg)
        outputs=[]
        def hook(module,args,out):
            self.assertTrue(torch.is_grad_enabled())
            out.retain_grad();outputs.append(out)
        handles=[layer.register_forward_hook(hook) for layer in m.layers]
        carry,logits=m(self.carry(m),self.batch)
        logits.square().mean().backward()
        self.assertEqual(len(outputs),16)
        for out in outputs:
            self.assertIsNotNone(out.grad)
            self.assertTrue(out.grad.isfinite().all())
            self.assertGreater(out.grad.norm().item(),0)
        self.assertFalse(carry.current_hidden.requires_grad)
        self.assertEqual(logits.shape,(2,9,11))
        for handle in handles:handle.remove()

    def test_checkpoint_preserves_full_graph(self):
        a=URMFullBPTTInner(self.cfg)
        b=URMFullBPTTInner(replace(self.cfg,activation_checkpoint=True))
        b.load_state_dict(a.state_dict())
        ya=a(self.carry(a),self.batch)[1];yb=b(self.carry(b),self.batch)[1]
        torch.testing.assert_close(ya,yb)
        ya.square().mean().backward();yb.square().mean().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad)

    def test_rejects_invalid_loops_and_no_grad(self):
        for cfg in [replace(self.cfg,loops=0),replace(self.cfg,nograd_blocks=1)]:
            with self.assertRaises(ValueError):URMFullBPTTInner(cfg)


class URMSwiGLUTests(URMFullBPTTTests):
    def test_plain_formula_and_matching_projections(self):
        from .urm_full_bptt import URMSwiGLUFullBPTTInner
        from .urm_vendor.layers import SwiGLU
        torch.manual_seed(55); conv=URMFullBPTTInner(self.cfg)
        torch.manual_seed(55); plain=URMSwiGLUFullBPTTInner(self.cfg)
        for a,b in zip(conv.layers,plain.layers):
            self.assertIsInstance(b.mlp,SwiGLU)
            self.assertFalse(hasattr(b.mlp,'dwconv'))
            torch.testing.assert_close(a.mlp.gate_up_proj.weight,b.mlp.gate_up_proj.weight)
            torch.testing.assert_close(a.mlp.down_proj.weight,b.mlp.down_proj.weight)
            x=torch.randn(2,10,16)
            gate,up=b.mlp.gate_up_proj(x).chunk(2,-1)
            torch.testing.assert_close(b.mlp(x),b.mlp.down_proj(torch.nn.functional.silu(gate)*up))
        outputs=[]
        def hook(module,args,out):
            out.retain_grad();outputs.append(out)
        handles=[l.register_forward_hook(hook) for l in plain.layers]
        nc,logits=plain(self.carry(plain),self.batch)
        logits.square().mean().backward()
        self.assertEqual(len(outputs),16)
        self.assertTrue(all(o.grad is not None and o.grad.isfinite().all() and o.grad.norm()>0 for o in outputs))
        for h in handles:h.remove()

    def test_act_disabled_and_all_lanes_complete_fixed_iterations(self):
        from .urm_full_bptt import install
        from . import train as t
        original_inner,original_id=t.LT_Inner,t.model_id_of
        try:
            install('swiglu')
            model=t.LT(dict(t.CFG,hidden_size=16,num_heads=2,grid=3,seq_len=9,
                batch_size=2,num_puzzle_identifiers=1,puzzle_emb_ndim=16,amp=False,
                activation_checkpoint=False,num_layers=2,blocks_per_seg=8,
                loops=16,memory_type='address',nograd_blocks=0))
            carry=model.initial_carry(self.batch)
            calls=[]
            handle=model.inner.layers[0].register_forward_hook(lambda *a:calls.append(1))
            with torch.no_grad():model.inner.urm.q_head.bias.fill_(100)
            nc,out=model(carry,self.batch)
            self.assertEqual(len(calls),8)
            self.assertFalse(nc.halted.any())
            self.assertTrue((out['q_halt_logits']==-5).all())
            out['logits'].square().mean().backward()
            self.assertIsNone(model.inner.urm.q_head.weight.grad)
            handle.remove()
            second=dict(self.batch,inputs=(self.batch['inputs']+1)%11)
            for step in range(2,17):
                nc,out=model(nc,second)
                torch.testing.assert_close(nc.current_data['inputs'],self.batch['inputs'])
                self.assertTrue((nc.steps==step).all())
                self.assertEqual(bool(nc.halted.all()),step==16)
                self.assertFalse(nc.current_hidden.requires_grad)
            nc,_=model(nc,second)
            torch.testing.assert_close(nc.current_data['inputs'],second['inputs'])
            self.assertTrue((nc.steps==1).all())
        finally:t.LT_Inner,t.model_id_of=original_inner,original_id


if __name__=='__main__':unittest.main()
