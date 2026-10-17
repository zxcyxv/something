"""CPU/FP64 checks of the event-STDP control (event_stdp=True): alpha = 0 is the Hebbian-only model, no change means no
event read, no event in a puzzle's first block, the pair sum over changes, the other modes untouched, and gradients.
Run: python -m unittest discover tests"""
import cmath
import unittest

import torch

from lt.model import LT
from tests.test_model import batch, tiny


def event_tiny(**kw):
    return tiny(**dict(dict(event_stdp=True, tag_bound=0.0), **kw))


def perturbed(cfg, seed=0):
    torch.manual_seed(seed)
    m = LT(cfg).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return m


def qkv(m, carry, b):
    """The projections of the next block (fresh lanes start from init_hidden)."""
    h = torch.where(carry.halted.view(-1, 1, 1), m.init_hidden, carry.hidden) if carry.clock is not None else \
        m.init_hidden.expand(b['inputs'].shape[0], m.config.seq_len, m.d)
    u = h + m.embed_scale * m.injection(b)
    B, T, _ = u.shape
    heads = lambda x: x.reshape(B, T, m.H, m.dh).transpose(1, 2)
    return heads(m.block.q_proj(u)), heads(m.block.k_proj(u)), heads(m.block.v_proj(u))


class EventSTDP(unittest.TestCase):
    def test_alpha_zero_is_hebbian_only(self):
        """Same seed -> same initial weights as the phase-clock model (no extra random draw); with alpha = 0 both reduce
        to the Hebbian read, so the logits agree over two segments."""
        for kw in (dict(), dict(outer_eta=0.125, event_theta=0.5, event_even=True)):
            torch.manual_seed(0)
            ref = LT(tiny(tag_bound=0.0, **{k: v for k, v in kw.items() if k == 'outer_eta'})).double().eval()
            torch.manual_seed(0)
            ev = LT(event_tiny(**kw)).double().eval()
            base = {n: p for n, p in ev.named_parameters()}
            for n, p in ref.named_parameters():
                torch.testing.assert_close(base[n], p, msg=n)
            with torch.no_grad():
                for m in (ref, ev):
                    m.block.alpha.zero_()
                    m.block.ffn_down.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(1))
                cr, ce = ref.initial_carry(batch(0)), ev.initial_carry(batch(0))
                for _ in range(2):
                    cr, lr = ref(cr, batch(0))
                    ce, le = ev(ce, batch(0))
                    torch.testing.assert_close(le, lr, rtol=1e-12, atol=1e-12)

    def test_no_change_no_event_read(self):
        """With q, k, v equal to the previous block's, the event read is exactly 0 (read = Hebbian read) and W only
        decays; in a puzzle's first block every event is 0."""
        for eta in (0.0, 0.25):
            m = perturbed(event_tiny(outer_eta=eta, event_even=True))
            with torch.no_grad():
                m.block.gamma_even.fill_(0.7)
            carry = m.initial_carry(batch(0))
            with torch.no_grad():
                carry, _ = m(carry, batch(0))
                q, k, v = qkv(m, carry, batch(0))
                state = (carry.clock, carry.key_trace, carry.value_trace, carry.stdp_w, carry.prev)
                read1, s1 = m.memory_step(q, k, v, state)          # changes against the carried prev
                read2, s2 = m.memory_step(q, k, v, s1)             # no change
                hebb_only = LT(m.config.__class__(**dict(m.config.__dict__, event_stdp=False, event_theta=0.0,
                                                         event_even=False))).double()
                hebb_only.load_state_dict({n: t for n, t in m.state_dict().items() if n != 'block.gamma_even'})
                with torch.no_grad():
                    hebb_only.block.alpha.zero_()
                ref, _ = hebb_only.memory_step(q, k, v, hebb_only.initial_state(q.shape[0], q.device, q.dtype))
                torch.testing.assert_close(read2, ref, rtol=1e-12, atol=1e-12)
                self.assertGreater(float((read1 - ref).abs().max()), 1e-6)
                if eta:
                    torch.testing.assert_close(s2[3], (1 - eta) * s1[3])
                # first block of a puzzle: no event, even with a nonzero prev
                fresh = m.initial_state(q.shape[0], q.device, q.dtype)
                fresh = fresh[:4] + (torch.randn_like(fresh[4]),)
                read0, s0 = m.memory_step(q, k, v, fresh)
                torch.testing.assert_close(read0, ref, rtol=1e-12, atol=1e-12)
                self.assertEqual(float(s0[1].abs().max()), 0.0)

    def test_pair_sum_over_changes(self):
        """Sum over blocks of P = Im S (+ gamma Re S) = explicit sum over every (V event @ a, K event @ b) pair of a
        cell with weight (1 - lam) lam^|a-b| e^{i omega (a-b)} (same-block pairs once), events = changes."""
        m = perturbed(event_tiny(event_even=True, outer_rho=0.0))
        cfg = m.config
        with torch.no_grad():
            m.block.gamma_even.fill_(0.3)
        writes, events = [], []
        orig = m.accumulate

        def spy(w, p, dtype):
            writes.append(p)
            return orig(w, p, dtype)
        m.accumulate = spy
        carry, seq = m.initial_carry(batch(3)), []
        with torch.no_grad():
            for _ in range(2):
                carry, _ = m(carry, batch(3))
        # recompute the per-block q^, k^, v by replaying the blocks
        m.accumulate = orig
        with torch.no_grad():
            b = batch(3)
            h = m.init_hidden.expand(2, cfg.seq_len, m.d)
            inj = m.injection(b)
            state = m.initial_state(2, h.device, h.dtype)
            for _ in range(2 * cfg.blocks_per_seg):
                u = h + m.embed_scale * inj
                heads = lambda x: x.reshape(2, cfg.seq_len, m.H, m.dh).transpose(1, 2)
                k = heads(m.block.k_proj(u))
                v = heads(m.block.v_proj(u))
                seq.append((k / (k.norm(dim=-1, keepdim=True) + cfg.eps), v))
                h, state = m.block_step(h, inj, state)
        angles = m.rope_angles()
        rope = lambda x: m.rope(x, (angles.cos()[None], angles.sin()[None]))
        e_k = [torch.zeros_like(seq[0][0])] + [seq[i][0] - seq[i - 1][0] for i in range(1, len(seq))]
        e_v = [torch.zeros_like(seq[0][1])] + [seq[i][1] - seq[i - 1][1] for i in range(1, len(seq))]
        lam, om, n = cfg.trace_lambda, cfg.block_omega, len(seq)
        total = torch.zeros(2, m.H, m.dh, m.dh, dtype=torch.complex128)
        for a in range(n):
            for bb in range(n):
                wgt = (1 - lam) * lam ** abs(a - bb) * cmath.exp(1j * om * (a - bb))
                total = total + wgt * torch.einsum('bhna,bhnj->bhaj', e_v[a].to(torch.complex128),
                                                   rope(e_k[bb]).to(torch.complex128))
        got = torch.stack(writes).sum(0)
        torch.testing.assert_close(got, total.imag + 0.3 * total.real, rtol=1e-9, atol=1e-10)

    def test_other_modes_unchanged(self):
        """event_stdp=False: the carry has no prev and the configuration checks hold."""
        m = perturbed(tiny(outer_eta=0.125))
        carry, _ = m(m.initial_carry(batch(0)), batch(0))
        self.assertIsNone(carry.prev)
        with self.assertRaises(ValueError):
            tiny(event_stdp=True)                                 # tag_bound must be 0
        with self.assertRaises(ValueError):
            tiny(event_theta=0.5)                                 # needs event_stdp
        with self.assertRaises(ValueError):
            event_tiny(kernel='triton')

    def test_gradients(self):
        """alpha, event_theta_raw, gamma_even and eta_raw receive finite, nonzero gradients."""
        torch.manual_seed(0)
        m = LT(event_tiny(outer_eta=0.25, outer_eta_learn=True, event_theta=0.5, event_even=True)).double()
        with torch.no_grad():
            m.block.ffn_down.weight.normal_(0, 0.05)
            m.block.gamma_even.fill_(0.2)
        carry = m.initial_carry(batch(1))
        for _ in range(2):
            carry, logits = m(carry, batch(1))
        logits.square().mean().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        for name in ('alpha', 'event_theta_raw', 'gamma_even', 'eta_raw'):
            self.assertGreater(float(getattr(m.block, name).grad.abs().sum()), 0, name)


if __name__ == '__main__':
    unittest.main()
