"""CPU/FP64 checks of the plain linear KV memory (kv_ema=True): W is the EMA of sum_t v k~^T and the read is q~ W^T,
and the learned eta receives a gradient.
Run: python -m unittest discover tests"""
import unittest

import torch

from lt.model import LT
from tests.test_model import batch, tiny


def kv_tiny(**kw):
    return tiny(**dict(dict(kv_ema=True, tag_bound=0.0, outer_eta=0.0625, outer_eta_learn=True), **kw))


class KVEMA(unittest.TestCase):
    def test_read_is_ema_of_hebbian_write(self):
        torch.manual_seed(0)
        m = LT(kv_tiny()).double().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.3)
            b = batch(0)
            state = m.initial_state(b['inputs'].shape[0], torch.device('cpu'), torch.float64)
            eta = torch.sigmoid(m.block.eta_raw)[None, :, None, None]
            h = m.init_hidden.expand(b['inputs'].shape[0], m.config.seq_len, m.d)
            angles = m.rope_angles()
            tables = angles.cos()[None], angles.sin()[None]
            unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
            w = torch.zeros_like(state[3])
            for _ in range(3):
                u = h + m.embed_scale * m.injection(b)
                B, T, _ = u.shape
                heads = lambda x: x.reshape(B, T, m.H, m.dh).transpose(1, 2)
                q, k, v = heads(m.block.q_proj(u)), heads(m.block.k_proj(u)), heads(m.block.v_proj(u))
                kr, qr = m.rope(unit(k), tables), m.rope(unit(q), tables)
                w = (1 - eta) * w + eta * (v.transpose(-1, -2) @ kr)
                read, state = m.memory_step(q, k, v, state)
                torch.testing.assert_close(read, qr @ w.transpose(-1, -2), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(state[3], w, rtol=1e-12, atol=1e-12)
                h, _ = m.block_step(h, m.injection(b), m.initial_state(B, torch.device('cpu'), torch.float64))

    def test_current_read_adds_hebb_at_gain_one(self):
        """kv_ema_current: read = q~ (sum_t v k~^T + alpha W)^T, W the same EMA as plain kv_ema."""
        torch.manual_seed(0)
        m = LT(kv_tiny(kv_ema_current=True)).double().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.3)
            B, T, D = 2, m.config.seq_len, m.dh
            g = torch.Generator().manual_seed(1)
            q, k, v = (torch.randn(B, m.H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
            state = m.initial_state(B, torch.device('cpu'), torch.float64)
            w0 = torch.randn(state[3].shape, generator=g, dtype=torch.float64)
            state = (*state[:3], w0, state[4])
            angles = m.rope_angles()
            tables = angles.cos()[None], angles.sin()[None]
            unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
            kr, qr = m.rope(unit(k), tables), m.rope(unit(q), tables)
            hebb = v.transpose(-1, -2) @ kr
            eta = torch.sigmoid(m.block.eta_raw)[None, :, None, None]
            w = (1 - eta) * w0 + eta * hebb
            read, state = m.memory_step(q, k, v, state)
            alpha = m.block.alpha[None, :, None, None]
            torch.testing.assert_close(read, qr @ (hebb + alpha * w).transpose(-1, -2), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(state[3], w, rtol=1e-12, atol=1e-12)

    def test_eta_gets_gradient_and_no_alpha(self):
        torch.manual_seed(0)
        m = LT(kv_tiny()).double()
        self.assertFalse(hasattr(m.block, 'alpha'))
        b = batch(0)
        carry, logits = m(m.initial_carry(b), b)
        carry, logits = m(carry, b)
        logits.square().sum().backward()
        self.assertTrue(m.block.eta_raw.grad.abs().sum() > 0)
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, n)

    def test_config_checks(self):
        with self.assertRaises(ValueError):
            tiny(kv_ema=True, tag_bound=0.0)                      # needs outer_eta


if __name__ == '__main__':
    unittest.main()
