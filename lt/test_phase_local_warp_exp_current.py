"""Local monotone phase warp: algebra, locality, order, real training/resume."""
from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import train as t
from .kv_stability import ExponentialPhaseCurrentReadInner, LocalWarpExponentialPhaseCurrentReadInner
from .probe_local_phase_warp import warp as reference_warp
from .test_kv_stability import config
from .test_phase_channel_exp_current import norm


class LocalWarpExponentialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(106)
        self.inner = LocalWarpExponentialPhaseCurrentReadInner(config())
        self.layer = self.inner.layers[0]

    def activate(self):
        with torch.no_grad():
            self.layer.phase_warp.projection.weight.normal_(std=.3)

    def contributions(self, k, v):
        phases = self.inner.phases(self.layer, k, v)
        window = self.inner.phase_window(self.layer, k.dtype, phases=phases)
        return v[..., :, None] * k[..., None, :] * window

    def test_identity_recovers_baseline_and_local_head_receives_gradients(self):
        torch.manual_seed(106)
        fixed = ExponentialPhaseCurrentReadInner(config())
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(self.inner.state_dict()[name], value, rtol=0, atol=0)
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        actual = self.inner.phases(self.layer, k, v)
        for phase, base in zip(actual, fixed.phases(fixed.layers[0])):
            torch.testing.assert_close(phase, base[None, :, None].expand_as(phase), rtol=0, atol=0)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        actual = self.inner.block(self.layer, h, inj, None, None, None, None)
        expected = fixed.block(fixed.layers[0], h, inj, None, None, None, None)
        for a, b in zip(actual[:2], expected[:2]):
            # The two-GEMM write reassociates products, even at identity warp.
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        (actual[0] * torch.randn_like(actual[0])).sum().backward()
        p = self.layer.phase_warp.projection.weight
        self.assertEqual(p.count_nonzero().item(), 0)
        self.assertTrue(p.grad.isfinite().all())
        self.assertGreater(p.grad.norm().item(), 0)

    def test_two_gemm_matches_direct_pair_outputs_and_all_gradients(self):
        self.inner.double()
        self.activate()
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        read, g, ek, ev = self.inner.memory_step(self.layer, q, k, v)
        qr, kr = [self.inner.apply_rope(x, self.layer) for x in (q, k)]
        pk, pv = self.inner.phases(self.layer, kr, v)
        delta = pv[..., :, None] - pk[..., None, :]
        window = delta.sign() * (-delta.abs()).exp()
        direct = (v[..., :, None] * kr[..., None, :] * window).mean(-3)
        expected = qr @ direct.transpose(-1, -2)
        torch.testing.assert_close(g, direct, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(read, expected, rtol=1e-11, atol=1e-12)
        inputs = (q, k, v, self.layer.theta, self.layer.theta_k_raw,
                  self.layer.theta_v_raw, self.layer.phase_warp.projection.weight)
        cotangent = torch.randn_like(read)
        gradients = torch.autograd.grad((read * cotangent).sum(), inputs, retain_graph=True)
        references = torch.autograd.grad((expected * cotangent).sum(), inputs)
        for a, b in zip(gradients, references):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
            self.assertTrue(a.isfinite().all())
            self.assertGreater(a.norm().item(), 0)
        dirty = self.inner.memory_step(self.layer, q, k, v, torch.full_like(g, float('nan')),
                                      torch.full_like(k, float('nan')), torch.full_like(v, float('nan')))
        torch.testing.assert_close(dirty[0], read, rtol=0, atol=0)
        self.assertIsNone(ek); self.assertIsNone(ev)

    def test_locality_additivity_permutation_and_cross_head_independence(self):
        self.activate()
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        original = self.contributions(k, v)
        changed = v.clone()
        changed[0, 0, 0, 0] += 1
        altered = self.contributions(k, changed)
        torch.testing.assert_close(original[:, :, 1:], altered[:, :, 1:], rtol=0, atol=0)
        torch.testing.assert_close(original[1], altered[1], rtol=0, atol=0)
        torch.testing.assert_close(original[0, 1], altered[0, 1], rtol=0, atol=0)
        self.assertGreater((original[0, 0, 0] - altered[0, 0, 0]).norm().item(), 0)
        phases = self.inner.phases(self.layer, k, v)
        reverse = self.inner.phases(self.layer, k.flip(-2), v.flip(-2))
        for a, b in zip(phases, reverse):
            torch.testing.assert_close(a.flip(-2), b, rtol=0, atol=0)
        split = (self.contributions(k[:, :, :4], v[:, :, :4]).sum(-3)
                 + self.contributions(k[:, :, 4:], v[:, :, 4:]).sum(-3))
        torch.testing.assert_close(original.sum(-3), split)
        k.requires_grad_(); v.requires_grad_()
        phase = self.inner.phases(self.layer, k, v)[0][0, 0, 0].sum()
        gradients = torch.autograd.grad(phase, (k, v))
        for g in gradients:
            self.assertGreater(g[0, 0, 0].norm().item(), 0)
            self.assertEqual(g[:, :, 1:].count_nonzero().item(), 0)
            self.assertEqual(g[1].count_nonzero().item(), 0)
            self.assertEqual(g[0, 1].count_nonzero().item(), 0)

    def test_warp_matches_independent_construction_and_preserves_order(self):
        warp = self.layer.phase_warp.double()
        base = torch.linspace(-math.pi / 2, math.pi / 2, 129, dtype=torch.float64).repeat(2, 1)
        logits = 3 * torch.randn(2, 2, 9, 8, dtype=torch.float64)
        actual = warp.warp(base, logits)
        expected = reference_warp(base, logits)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue((actual.diff(dim=-1) > 0).all())
        self.assertGreater((actual - base[None, :, None]).abs().max().item(), .5)
        for index in (0, -1):
            torch.testing.assert_close(actual[..., index], base[None, :, None, index].expand_as(actual[..., index]),
                                       rtol=0, atol=0)
        identity = warp.warp(base, torch.zeros_like(logits))
        torch.testing.assert_close(identity, base[None, :, None].expand_as(identity), rtol=0, atol=0)

    def test_ties_and_baseline_order_changes(self):
        self.activate()
        with torch.no_grad():
            self.layer.theta_v_raw[:, 0] = self.layer.theta_k_raw[:, 0]
            # Distinct raw values can saturate to equal bounded baseline phases.
            self.layer.theta_k_raw[:, 1] = 40
            self.layer.theta_v_raw[:, 1] = 50
        q, k, v = [torch.randn(2, 2, 9, 8) for _ in range(3)]
        read, g, _, _ = self.inner.memory_step(self.layer, q, k, v)
        for index in (0, 1):
            self.assertEqual(g[..., index, index].count_nonzero().item(), 0)
        pk, pv = self.inner.phases(self.layer, k, v)
        bk, bv = self.inner.base_phases(self.layer)
        delta = pv[..., :, None] - pk[..., None, :]
        expected_order = (bv[..., :, None] - bk[..., None, :]).sign()[None, :, None]
        self.assertTrue(((delta.sign() == expected_order) | (delta.abs() < 1e-6)).all())
        read.sum().backward()
        self.assertTrue(self.layer.phase_warp.projection.weight.grad.isfinite().all())
        with torch.no_grad():
            self.layer.theta_v_raw[:, 0] += .2
        _, altered, _, _ = self.inner.memory_step(self.layer, q, k, v)
        self.assertGreater(altered[..., 0, 0].abs().sum().item(), 0)

    def test_main_norm_positions_rotated_input_and_phase_precision(self):
        self.activate()
        inner, layer = self.inner, self.layer
        with torch.no_grad():
            layer.b_down.weight.normal_(std=.015)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        x = h + inner.embed_scale * inj
        heads = lambda y: y.reshape(2, 9, 2, 8).transpose(1, 2)
        q, k, v = [heads(p(x)) for p in (layer.q_proj, layer.k_proj, layer.v_proj)]
        read, g, _, _ = inner.memory_step(layer, q, k, v)
        first_residual = x + layer.out_proj(read.transpose(1, 2).reshape(2, 9, 16))
        u = norm(first_residual)
        gate, value = layer.b_gate_up(u).chunk(2, -1)
        second_residual = u + layer.b_down(.5 * gate * value)
        with patch.object(inner, 'phi', wraps=inner.phi) as rms, \
             patch.object(inner, 'phases', wraps=inner.phases) as phase_call:
            actual = inner.block(layer, h, inj, None, None, None, None)
            self.assertEqual(rms.call_count, 2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], first_residual)
            torch.testing.assert_close(rms.call_args_list[1].args[0], second_residual)
            torch.testing.assert_close(phase_call.call_args.args[1], inner.apply_rope(k, layer))
            torch.testing.assert_close(phase_call.call_args.args[2], v)
        torch.testing.assert_close(actual[0], norm(second_residual))
        torch.testing.assert_close(actual[1], g)
        plain = inner.phases(layer, k, v)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp = inner.phases(layer, k, v)
        for a, b in zip(plain, amp):
            self.assertEqual(b.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            z = torch.randn(2, 9, 16).to(dtype)
            torch.testing.assert_close(inner.phi(z), norm(z), rtol=0, atol=0)
        self.assertFalse(t._is_no_decay('inner.layers.0.phase_warp.projection.weight', layer.phase_warp.projection.weight))
        self.assertTrue(t._is_no_decay('inner.layers.0.theta_k_raw', layer.theta_k_raw))

    def test_training_recompute_checkpoint_resume_and_retention(self):
        cfg = dict(t.CFG, **vars(config()))
        cfg.update(global_batch_size=2, nograd_fixed=0, nograd_every=0, lr_warmup_steps=0,
                   activation_checkpoint=True, compile=False, keep_last=3,
                   data_fingerprint='local-warp-unit-test', research_variant='phase_local_warp_exp_current_only')
        device = torch.device('cpu')
        with patch.object(t, 'KVSTDPInner', LocalWarpExponentialPhaseCurrentReadInner):
            base = t.ACTLossHead(t.LT(cfg))
            opts, lrs = t.create_optimizers(base, cfg, 1)
            ema = t.EMAHelper(.999); ema.register(base)
            state = t.TrainState()
            batch = dict(inputs=torch.randint(1, 10, (2, 9)), labels=torch.randint(1, 10, (2, 9)),
                         puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
            with tempfile.TemporaryDirectory() as directory:
                for i in range(16):
                    metrics = t.train_batch(base, base, state, batch, cfg, opts, lrs, 100, 0, 1, device)
                    ema.update(base); state.batch_in_iter += 1
                    self.assertTrue(math.isfinite(metrics['lm_loss']))
                    if i in (1, 2, 3, 15):
                        path = t.save_training_checkpoint(directory, state, base, opts, ema, cfg, 0, 1, device)
                self.assertEqual(len(list(Path(directory).glob('step_*.pt'))), 3)
                self.assertEqual(metrics['_count_raw'], 2)
                self.assertIsNone(state.carry.key_trace)
                self.assertFalse(state.carry.current_hidden.requires_grad)
                p = base.model.inner.layers[0].phase_warp.projection.weight
                self.assertGreater(p.norm().item(), 0)
                momentum = opts[-1].state[p]['m']
                self.assertTrue(momentum.isfinite().all())
                self.assertGreater(momentum.norm().item(), 0)
                restored = t.ACTLossHead(t.LT(cfg))
                ropts, _ = t.create_optimizers(restored, cfg, 1)
                rema = t.EMAHelper(.999); rema.register(restored)
                rstate = t.load_training_checkpoint(path, restored, ropts, rema, cfg, 0, 1, device)
                a = t.train_batch(base, base, state, batch, cfg, opts, lrs, 100, 0, 1, device)
                b = t.train_batch(restored, restored, rstate, batch, cfg, ropts, lrs, 100, 0, 1, device)
                self.assertAlmostEqual(a['lm_loss'], b['lm_loss'], places=6)
                for name, tensor in base.state_dict().items():
                    torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
