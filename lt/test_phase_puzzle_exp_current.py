"""Puzzle-conditioned phases: exact contraction, gradients, norms and resume."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import train as t
from .kv_stability import (ExponentialPhaseCurrentReadInner,
                           PuzzleStateExponentialPhaseCurrentReadInner)
from .test_kv_stability import config


class PuzzlePhaseExponentialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(105)
        self.inner = PuzzleStateExponentialPhaseCurrentReadInner(config())
        self.layer = self.inner.layers[0]

    def test_zero_init_matches_baseline_and_projection_can_learn(self):
        torch.manual_seed(105)
        fixed = ExponentialPhaseCurrentReadInner(config())
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(self.inner.state_dict()[name], value, rtol=0, atol=0)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        actual = self.inner.block(self.layer, h, inj, None, None, None, None)
        expected = fixed.block(fixed.layers[0], h, inj, None, None, None, None)
        for a, b in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        actual[0][..., 0].square().mean().backward()
        for proj in (self.layer.phase_k_proj, self.layer.phase_v_proj):
            self.assertEqual(proj.weight.count_nonzero().item(), 0)
            self.assertTrue(proj.weight.grad.isfinite().all())
            self.assertGreater(proj.weight.grad.norm().item(), 0)
        opt = torch.optim.Adam(self.inner.parameters(), lr=1e-3)
        opt.step()
        a = self.inner.phases(self.layer, h)
        b = self.inner.phases(self.layer, -h)
        for x, y in zip(a, b):
            self.assertFalse(torch.equal(x, y))

    def test_direct_pairs_and_gradients_match_single_gemm(self):
        inner = self.inner.double()
        with torch.no_grad():
            self.layer.phase_k_proj.weight.normal_(std=.03)
            self.layer.phase_v_proj.weight.normal_(std=.03)
        hidden = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64, requires_grad=True)
                   for _ in range(3)]
        pk, pv = inner.phases(self.layer, hidden)
        read, g, ek, ev = inner.memory_step(self.layer, q, k, v, phases=(pk, pv))
        qr, kr = [inner.apply_rope(x, self.layer) for x in (q, k)]
        delta = pv[:, :, None, :, None] - pk[:, :, None, None, :]
        window = delta.sign() * (-delta.abs()).exp()
        direct = (v[..., :, None] * kr[..., None, :] * window).mean(-3)
        expected = qr @ direct.transpose(-1, -2)
        torch.testing.assert_close(g, direct, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(read, expected, rtol=1e-12, atol=1e-12)
        inputs = (q, k, v, hidden, self.layer.phase_k_proj.weight,
                  self.layer.phase_v_proj.weight, self.layer.theta_k_raw, self.layer.theta_v_raw)
        cotangent = torch.randn_like(read)
        grads = torch.autograd.grad((read * cotangent).sum(), inputs, retain_graph=True)
        refs = torch.autograd.grad((expected * cotangent).sum(), inputs)
        for a, b in zip(grads, refs):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
            self.assertTrue(a.isfinite().all())
            self.assertGreater(a.norm().item(), 0)
        dirty = inner.memory_step(self.layer, q, k, v, torch.full_like(g, float('nan')),
                                  torch.full_like(k, float('nan')), torch.full_like(v, float('nan')),
                                  phases=(pk, pv))
        torch.testing.assert_close(dirty[0], read, rtol=0, atol=0)
        self.assertIsNone(ek); self.assertIsNone(ev)

    def test_pool_scope_summary_norm_and_fp32_autocast(self):
        with torch.no_grad():
            self.layer.phase_k_proj.weight.normal_(std=.03)
            self.layer.phase_v_proj.weight.normal_(std=.03)
        hidden = torch.randn(2, 9, 16)
        pk, pv = self.inner.phases(self.layer, hidden)
        self.assertEqual(pk.shape, (2, 2, 8))
        summary = hidden.mean(1)
        summary = summary * torch.rsqrt(summary.square().mean(-1, keepdim=True) + 1e-5)
        expected = (math.pi/2) * (self.layer.phase_k_proj(summary).reshape(2, 2, 8)
                                 + self.layer.theta_k_raw).tanh()
        torch.testing.assert_close(pk, expected, rtol=0, atol=0)
        permuted = self.inner.phases(self.layer, hidden.flip(1))
        for a, b in zip((pk, pv), permuted):
            torch.testing.assert_close(a, b)
        other = hidden.clone(); other[1] *= -3
        changed = self.inner.phases(self.layer, other)
        for a, b in zip((pk, pv), changed):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            self.assertFalse(torch.equal(a[1], b[1]))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp_phases = self.inner.phases(self.layer, hidden)
        for a, b in zip((pk, pv), amp_phases):
            self.assertEqual(b.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        zero = torch.zeros_like(hidden, requires_grad=True)
        sum(p.sum() for p in self.inner.phases(self.layer, zero)).backward()
        self.assertTrue(zero.grad.isfinite().all())

    def test_exact_zero_signed_limits_and_channel_specific_rotations(self):
        pk = torch.zeros(2, 2, 8, dtype=torch.float64)
        for delta in (0., -1e-6, 1e-6, -.5, .5, 1.):
            pv = torch.full_like(pk, delta)
            actual = self.inner.phase_window(self.layer, torch.float64, phases=(pk, pv))
            value = 0. if delta == 0 else math.copysign(math.exp(-abs(delta)), delta)
            torch.testing.assert_close(actual, torch.full_like(actual, value))
        pk, pv = torch.randn_like(pk), torch.randn_like(pk)
        carrier = torch.randn(2, 2, 1, dtype=torch.float64)
        a = self.inner.phase_window(self.layer, torch.float64, phases=(pk, pv))
        b = self.inner.phase_window(self.layer, torch.float64, phases=(pk+carrier, pv+carrier))
        torch.testing.assert_close(a, b)

    def test_block_has_norm_at_each_residual_and_injected_state_summary(self):
        inner, layer = self.inner, self.layer
        with torch.no_grad():
            layer.b_down.weight.normal_(std=.015)
            layer.phase_k_proj.weight.normal_(std=.03)
            layer.phase_v_proj.weight.normal_(std=.03)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        x = h + inner.embed_scale * inj
        heads = lambda y: y.reshape(2, 9, 2, 8).transpose(1, 2)
        q, k, v = [heads(p(x)) for p in (layer.q_proj, layer.k_proj, layer.v_proj)]
        read, g, _, _ = inner.memory_step(layer, q, k, v, phases=inner.phases(layer, x))
        attention_residual = x + layer.out_proj(read.transpose(1, 2).reshape(2, 9, 16))
        norm = lambda y: (y.float() * torch.rsqrt(y.float().square().mean(-1, keepdim=True)+1e-5)).to(y.dtype)
        u = norm(attention_residual)
        gate, value = layer.b_gate_up(u).chunk(2, -1)
        ffn_residual = u + layer.b_down(.5 * gate * value)
        with patch.object(inner, 'phi', wraps=inner.phi) as rms, \
             patch.object(inner, 'phases', wraps=inner.phases) as phases:
            actual = inner.block(layer, h, inj, None, None, None, None)
            self.assertEqual(rms.call_count, 2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], attention_residual)
            torch.testing.assert_close(rms.call_args_list[1].args[0], ffn_residual)
            torch.testing.assert_close(phases.call_args.args[1], x)
        torch.testing.assert_close(actual[0], norm(ffn_residual))
        torch.testing.assert_close(actual[1], g)
        for dtype in (torch.float32, torch.bfloat16):
            z = torch.randn(2, 9, 16).to(dtype)
            torch.testing.assert_close(inner.phi(z), norm(z), rtol=0, atol=0)
        self.assertFalse(t._is_no_decay('inner.layers.0.phase_k_proj.weight', layer.phase_k_proj.weight))
        self.assertTrue(t._is_no_decay('inner.layers.0.theta_k_raw', layer.theta_k_raw))

    def test_training_checkpoint_recompute_resume_and_keep_three(self):
        cfg = dict(t.CFG, **vars(config()))
        cfg.update(global_batch_size=2, nograd_fixed=0, nograd_every=0,
                   lr_warmup_steps=0, activation_checkpoint=True, compile=False,
                   keep_last=3, data_fingerprint='puzzle-phase-unit-test')
        device = torch.device('cpu')
        with patch.object(t, 'KVSTDPInner', PuzzleStateExponentialPhaseCurrentReadInner):
            base = t.ACTLossHead(t.LT(cfg))
            opts, lrs = t.create_optimizers(base, cfg, 1)
            ema = t.EMAHelper(.999); ema.register(base)
            state = t.TrainState()
            batch = dict(inputs=torch.randint(1, 10, (2, 9)), labels=torch.randint(1, 10, (2, 9)),
                         puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
            with tempfile.TemporaryDirectory() as directory:
                for i in range(16):
                    metrics = t.train_batch(base, base, state, batch, cfg, opts, lrs, 100, 0, 1, device)
                    ema.update(base)
                    state.batch_in_iter += 1
                    self.assertTrue(math.isfinite(metrics['lm_loss']))
                    if i in (1, 2, 3, 15):
                        path = t.save_training_checkpoint(directory, state, base, opts, ema, cfg, 0, 1, device)
                self.assertEqual(sorted(p.name for p in Path(directory).glob('step_*.pt')),
                                 ['step_16.pt', 'step_3.pt', 'step_4.pt'])
                self.assertEqual(metrics['_count_raw'], 2)
                self.assertIsNone(state.carry.key_trace)
                self.assertFalse(state.carry.current_hidden.requires_grad)
                layer = base.model.inner.layers[0]
                self.assertGreater(layer.phase_k_proj.weight.norm().item(), 0)
                restored = t.ACTLossHead(t.LT(cfg))
                ropts, rlrs = t.create_optimizers(restored, cfg, 1)
                rema = t.EMAHelper(.999); rema.register(restored)
                rstate = t.load_training_checkpoint(path, restored, ropts, rema, cfg, 0, 1, device)
                a = t.train_batch(base, base, state, batch, cfg, opts, lrs, 100, 0, 1, device)
                b = t.train_batch(restored, restored, rstate, batch, cfg, ropts, rlrs, 100, 0, 1, device)
                self.assertAlmostEqual(a['lm_loss'], b['lm_loss'], places=6)
                for name, tensor in base.state_dict().items():
                    torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
