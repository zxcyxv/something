"""Channel phase attention: factorization, geometry, gradients and training."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import train as t
from .kv_stability import (ChannelAttentionExponentialPhaseCurrentReadInner,
                           ExponentialPhaseCurrentReadInner)
from .test_kv_stability import config


def norm(x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                  + 1e-5)).to(x.dtype)


class ChannelPhaseExponentialTests(unittest.TestCase):
    inner_cls = ChannelAttentionExponentialPhaseCurrentReadInner

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(106)
        self.inner = self.inner_cls(config())
        self.layer = self.inner.layers[0]

    def activate_phase_branch(self):
        with torch.no_grad():
            self.layer.phase_attention.readout.weight.normal_(std=.03)

    def test_zero_readout_matches_baseline_and_opens_branch_gradients(self):
        torch.manual_seed(106)
        fixed = ExponentialPhaseCurrentReadInner(config())
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(self.inner.state_dict()[name], value, rtol=0, atol=0)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        actual = self.inner.block(self.layer, h, inj, None, None, None, None)
        expected = fixed.block(fixed.layers[0], h, inj, None, None, None, None)
        for a, b in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        cotangent = torch.randn_like(actual[0])
        (actual[0] * cotangent).sum().backward()
        branch = self.layer.phase_attention
        self.assertEqual(branch.readout.weight.count_nonzero().item(), 0)
        self.assertGreater(branch.readout.weight.grad.norm().item(), 0)
        for name, parameter in branch.named_parameters():
            if name != 'readout.weight':
                self.assertEqual(parameter.grad.norm().item(), 0)
        opt = torch.optim.Adam(self.inner.parameters(), lr=1e-3)
        opt.step(); opt.zero_grad(set_to_none=True)
        updated = self.inner.block(self.layer, h, inj, None, None, None, None)
        (updated[0] * cotangent).sum().backward()
        for name, p in branch.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(p.grad.isfinite().all())
                self.assertGreater(p.grad.norm().item(), 0)

    def test_direct_pair_write_and_full_phase_gradients_match_one_gemm(self):
        inner = self.inner.double()
        self.activate_phase_branch()
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64, requires_grad=True)
                   for _ in range(3)]
        read, g, ek, ev = inner.memory_step(self.layer, q, k, v)
        qr, kr = [inner.apply_rope(x, self.layer) for x in (q, k)]
        pk, pv = inner.phases(self.layer, kr, v)
        delta = pv[..., None, :, None] - pk[..., None, None, :]
        window = delta.sign() * (-delta.abs()).exp()
        direct = (v[..., :, None] * kr[..., None, :] * window).mean(-3)
        expected = qr @ direct.transpose(-1, -2)
        torch.testing.assert_close(g, direct, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(read, expected, rtol=1e-12, atol=1e-12)
        inputs = (q, k, v, self.layer.theta, self.layer.theta_k_raw,
                  self.layer.theta_v_raw, *self.layer.phase_attention.parameters())
        cotangent = torch.randn_like(read)
        grads = torch.autograd.grad((read * cotangent).sum(), inputs, retain_graph=True)
        refs = torch.autograd.grad((expected * cotangent).sum(), inputs)
        for a, b in zip(grads, refs):
            # Residual RMSNorm intentionally accumulates in FP32 even here.
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
            self.assertTrue(a.isfinite().all())
            self.assertGreater(a.norm().item(), 0)
        dirty = inner.memory_step(self.layer, q, k, v, torch.full_like(g, float('nan')),
                                  torch.full_like(k, float('nan')), torch.full_like(v, float('nan')))
        torch.testing.assert_close(dirty[0], read, rtol=0, atol=0)
        self.assertIsNone(ek); self.assertIsNone(ev)

    def test_equal_means_different_positions_and_joint_role_attention(self):
        self.activate_phase_branch()
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        pk, pv = self.inner.phases(self.layer, k, v)
        self.assertEqual(pk.shape, (2, 2, 8))
        changed = k.clone()
        changed[:, :, 0, 0] += .75
        changed[:, :, 1, 0] -= .75
        torch.testing.assert_close(changed.mean(-2), k.mean(-2))
        altered = self.inner.phases(self.layer, changed, v)
        self.assertGreater((altered[0]-pk).abs().max().item(), 1e-5)
        permuted = self.inner.phases(self.layer, k.flip(-2), v.flip(-2))
        self.assertGreater((permuted[0]-pk).abs().max().item(), 1e-5)
        # V-channel state can change K phases through the joint attention.
        value_changed = self.inner.phases(self.layer, k, -v)
        self.assertGreater((value_changed[0]-pk).abs().max().item(), 1e-5)
        only_other = v.clone(); only_other[1, 1] *= -3
        independent = self.inner.phases(self.layer, k, only_other)
        for a, b in zip((pk, pv), independent):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(a[1, 0], b[1, 0], rtol=0, atol=0)

    def test_branch_norm_positions_and_amp_precision(self):
        self.activate_phase_branch()
        branch = self.layer.phase_attention
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        profiles = torch.cat((k.transpose(-1,-2), v.transpose(-1,-2)), -2)
        z = branch.profile_proj(profiles) + branch.channel_embedding
        q, kk, vv = branch.qkv(z).chunk(3, -1)
        attention = (q @ kk.transpose(-1,-2) / math.sqrt(16)).softmax(-1) @ vv
        first_residual = z + branch.out_proj(attention)
        u = norm(first_residual)
        second_residual = u + branch.ffn_down(torch.nn.functional.gelu(branch.ffn_up(u)))
        with patch.object(branch, 'norm', wraps=branch.norm) as rms:
            actual = branch(k, v)
            self.assertEqual(rms.call_count, 2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], first_residual)
            torch.testing.assert_close(rms.call_args_list[1].args[0], second_residual)
        expected = branch.readout(norm(second_residual)).squeeze(-1).split(8, -1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        plain = self.inner.phases(self.layer, k, v)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp = self.inner.phases(self.layer, k, v)
        for a, b in zip(plain, amp):
            self.assertEqual(b.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            x = torch.randn(2, 2, 16, 16).to(dtype)
            torch.testing.assert_close(branch.norm(x), norm(x), rtol=0, atol=0)
        zero = torch.zeros_like(k, requires_grad=True)
        sum(x.sum() for x in self.inner.phases(self.layer, zero, zero)).backward()
        self.assertTrue(zero.grad.isfinite().all())

    def test_main_block_norms_and_rotated_key_profiles(self):
        self.activate_phase_branch()
        inner, layer = self.inner, self.layer
        with torch.no_grad():
            layer.b_down.weight.normal_(std=.015)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        x = h + inner.embed_scale * inj
        heads = lambda y: y.reshape(2,9,2,8).transpose(1,2)
        q,k,v = [heads(p(x)) for p in (layer.q_proj,layer.k_proj,layer.v_proj)]
        read, g, _, _ = inner.memory_step(layer,q,k,v)
        first_residual = x + layer.out_proj(read.transpose(1,2).reshape(2,9,16))
        u = norm(first_residual)
        gate,value = layer.b_gate_up(u).chunk(2,-1)
        second_residual = u + layer.b_down(.5*gate*value)
        with patch.object(inner, 'phi', wraps=inner.phi) as rms, \
             patch.object(inner, 'phases', wraps=inner.phases) as phases:
            actual = inner.block(layer,h,inj,None,None,None,None)
            self.assertEqual(rms.call_count,2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], first_residual)
            torch.testing.assert_close(rms.call_args_list[1].args[0], second_residual)
            torch.testing.assert_close(phases.call_args.args[1], inner.apply_rope(k,layer))
            torch.testing.assert_close(phases.call_args.args[2], v)
        torch.testing.assert_close(actual[0], norm(second_residual))
        torch.testing.assert_close(actual[1], g)
        self.assertFalse(t._is_no_decay('inner.layers.0.phase_attention.readout.weight',
                                       layer.phase_attention.readout.weight))
        self.assertTrue(t._is_no_decay('inner.layers.0.theta_k_raw',layer.theta_k_raw))

    def test_training_recompute_checkpoint_resume_and_retention(self):
        cfg = dict(t.CFG, **vars(config()))
        cfg.update(global_batch_size=2,nograd_fixed=0,nograd_every=0,lr_warmup_steps=0,
                   activation_checkpoint=True,compile=False,keep_last=3,
                   data_fingerprint='channel-phase-unit-test')
        device = torch.device('cpu')
        with patch.object(t,'KVSTDPInner',self.inner_cls):
            base = t.ACTLossHead(t.LT(cfg))
            opts,lrs = t.create_optimizers(base,cfg,1)
            ema=t.EMAHelper(.999);ema.register(base)
            state=t.TrainState()
            batch=dict(inputs=torch.randint(1,10,(2,9)),labels=torch.randint(1,10,(2,9)),
                       puzzle_identifiers=torch.zeros(2,dtype=torch.int32))
            with tempfile.TemporaryDirectory() as directory:
                for i in range(16):
                    metrics=t.train_batch(base,base,state,batch,cfg,opts,lrs,100,0,1,device)
                    ema.update(base);state.batch_in_iter+=1
                    self.assertTrue(math.isfinite(metrics['lm_loss']))
                    if i in (1,2,3,15):
                        path=t.save_training_checkpoint(directory,state,base,opts,ema,cfg,0,1,device)
                self.assertEqual(len(list(Path(directory).glob('step_*.pt'))),3)
                self.assertEqual(metrics['_count_raw'],2)
                self.assertIsNone(state.carry.key_trace)
                self.assertFalse(state.carry.current_hidden.requires_grad)
                branch=base.model.inner.layers[0].phase_attention
                self.assertGreater(branch.readout.weight.norm().item(),0)
                for p in branch.parameters():
                    # train_batch clears gradients after the optimizer update.
                    momentum = opts[-1].state[p]['m']
                    self.assertTrue(momentum.isfinite().all())
                    self.assertGreater(momentum.norm().item(),0)
                restored=t.ACTLossHead(t.LT(cfg))
                ropts,_=t.create_optimizers(restored,cfg,1)
                rema=t.EMAHelper(.999);rema.register(restored)
                rstate=t.load_training_checkpoint(path,restored,ropts,rema,cfg,0,1,device)
                a=t.train_batch(base,base,state,batch,cfg,opts,lrs,100,0,1,device)
                b=t.train_batch(restored,restored,rstate,batch,cfg,ropts,lrs,100,0,1,device)
                self.assertAlmostEqual(a['lm_loss'],b['lm_loss'],places=6)
                for name,tensor in base.state_dict().items():
                    torch.testing.assert_close(tensor,restored.state_dict()[name],rtol=2e-6,atol=2e-7)


if __name__=='__main__':
    unittest.main()
