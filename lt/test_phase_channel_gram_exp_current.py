"""Full-profile coactivity phases: no absolute-slot parameters, exact write."""
from unittest.mock import patch
import unittest

import torch

from . import test_phase_channel_exp_current as shared
from .kv_stability import (CoactivityChannelExponentialPhaseCurrentReadInner,
                           CoactivityChannelPhaseAttention)


class CoactivityPhaseExponentialTests(shared.ChannelPhaseExponentialTests):
    # Inherit baseline equivalence/gradient opening, pair-vs-GEMM full gradients,
    # main residual norm placement, and actual training/EMA/checkpoint/resume.
    inner_cls = CoactivityChannelExponentialPhaseCurrentReadInner

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
        self.assertGreater((altered[0] - pk).abs().max().item(), 1e-5)
        # A common permutation preserves coactivity; a V-only permutation does
        # not, even though every channel's mean and RMS are preserved by both.
        permuted = self.inner.phases(self.layer, k.flip(-2), v.flip(-2))
        for a, b in zip((pk, pv), permuted):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
        value_changed = self.inner.phases(self.layer, k, v.flip(-2))
        self.assertGreater((value_changed[0] - pk).abs().max().item(), 1e-5)
        only_other = v.clone()
        only_other[1, 1] *= -3
        independent = self.inner.phases(self.layer, k, only_other)
        for a, b in zip((pk, pv), independent):
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(a[1, 0], b[1, 0], rtol=0, atol=0)

    def test_branch_norm_positions_and_amp_precision(self):
        self.activate_phase_branch()
        branch = self.layer.phase_attention
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        profiles = torch.cat((k.transpose(-1, -2), v.transpose(-1, -2)), -2)
        # Independently form pairwise products before reducing tokens.
        scores = (profiles[..., :, None, :] * profiles[..., None, :, :]).mean(-1)
        embedding = branch.channel_embedding
        first_residual = embedding + branch.out_proj(
            scores.softmax(-1) @ branch.value_proj(embedding))
        u = shared.norm(first_residual)
        second_residual = u + branch.ffn_down(torch.nn.functional.gelu(branch.ffn_up(u)))
        with patch.object(branch, 'norm', wraps=branch.norm) as rms:
            actual = branch(k, v)
            self.assertEqual(rms.call_count, 2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], first_residual)
            torch.testing.assert_close(rms.call_args_list[1].args[0], second_residual)
        expected = branch.readout(shared.norm(second_residual)).squeeze(-1).split(8, -1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
        plain = self.inner.phases(self.layer, k, v)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp = self.inner.phases(self.layer, k, v)
        for a, b in zip(plain, amp):
            self.assertEqual(b.dtype, torch.float32)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            x = torch.randn(2, 2, 16, 16).to(dtype)
            torch.testing.assert_close(branch.norm(x), shared.norm(x), rtol=0, atol=0)
        zero = torch.zeros_like(k, requires_grad=True)
        sum(x.sum() for x in self.inner.phases(self.layer, zero, zero)).backward()
        self.assertTrue(zero.grad.isfinite().all())

    def test_token_permutation_preserves_phases_and_permutates_activity_gradients(self):
        self.activate_phase_branch()
        # Verify full 81-position inputs, including gradients that flow only
        # through the phase generator, not through the direct KV write path.
        k, v = [torch.randn(2, 2, 81, 8, requires_grad=True) for _ in range(2)]
        permutation = torch.randperm(81)
        kp = k.detach()[:, :, permutation].requires_grad_()
        vp = v.detach()[:, :, permutation].requires_grad_()
        phases = self.inner.phases(self.layer, k, v)
        permuted = self.inner.phases(self.layer, kp, vp)
        cotangents = [torch.randn_like(p) for p in phases]
        params = tuple(self.layer.phase_attention.parameters())
        gradients = torch.autograd.grad(sum((a * c).sum() for a, c in zip(phases, cotangents)),
                                        (k, v, *params))
        refs = torch.autograd.grad(sum((a * c).sum() for a, c in zip(permuted, cotangents)),
                                   (kp, vp, *params))
        for a, b in zip(phases, permuted):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
        for index, (a, b) in enumerate(zip(gradients, refs)):
            self.assertTrue(a.isfinite().all())
            self.assertGreater(a.norm().item(), 0)
            if index < 2:
                a = a[:, :, permutation]
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)

    def test_no_trainable_spatial_slots_at_any_sequence_length(self):
        torch.manual_seed(170)
        short = CoactivityChannelPhaseAttention(9, 2, 8)
        torch.manual_seed(170)
        full = CoactivityChannelPhaseAttention(81, 2, 8)
        self.assertFalse(hasattr(full, 'profile_proj'))
        self.assertFalse(hasattr(full, 'qkv'))
        for name, value in short.state_dict().items():
            torch.testing.assert_close(full.state_dict()[name], value, rtol=0, atol=0)
        full.readout.weight.data.normal_(std=.03)
        k, v = [torch.randn(2, 2, 9, 8) for _ in range(2)]
        original = full(k, v)
        # Dividing by T preserves coactivity when the same records are repeated.
        repeated = full(k.repeat(1, 1, 9, 1), v.repeat(1, 1, 9, 1))
        for a, b in zip(original, repeated):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
