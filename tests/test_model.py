"""CPU/FP64 checks of the model: the trace recursion against the explicit pair sum, the clock, model B's W,
gradients, and the order/latency split. Run: python -m unittest discover tests"""
import math
import unittest

import torch

from lt.model import LT, ModelConfig, clock_step, rep2, stdp_step, wrap_phase


def tiny(**kw):
    return ModelConfig(**dict(dict(batch_size=2, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, blocks_per_seg=3,
                                   loops=4, amp=False, forward_dtype='float64', kernel='torch'), **kw))


def rope_fn(H, T, P, g):
    angles = torch.rand(1, H, T, P, generator=g, dtype=torch.float64) * 2 * math.pi
    cos, sin = angles.cos(), angles.sin()

    def rope(x):
        x0, x1 = x.reshape(*x.shape[:-1], P, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape(x.shape)
    return rope


def batch(seed, n=2):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 11, (n, 81), generator=g), labels=torch.randint(2, 11, (n, 81), generator=g),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32))


class TraceRecursion(unittest.TestCase):
    def test_equals_explicit_pair_sum(self):
        """Im S of the recursion = Im of the sum over every (post V, pre K) pair of a cell since the lane's reset,
        weight (1 - lam) lam^k e^{i(tau_post - tau_pre)}, same-block pairs once (lane 0 is reset at block 3)."""
        torch.set_default_dtype(torch.float64)
        try:
            g = torch.Generator().manual_seed(0)
            B, H, T, P, blocks, lam = 2, 2, 5, 3, 6, 0.4
            D = 2 * P
            rope = rope_fn(H, T, P, g)
            seq = [(torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g), dim=-1),
                    torch.randn(B, H, T, D, generator=g),
                    torch.randn(B, H, T, P, generator=g) * 3, torch.randn(B, H, T, P, generator=g) * 3)
                   for _ in range(blocks)]
            reset = {0: torch.tensor([True, True]), 3: torch.tensor([True, False])}
            traces, got = tuple(torch.zeros(B, H, T, D) for _ in range(4)), []
            for i, (k, v, tk, tv) in enumerate(seq):
                if i in reset:
                    m = reset[i].view(-1, 1, 1, 1)
                    traces = tuple(torch.where(m, torch.zeros_like(x), x) for x in traces)
                hebb, im_s, traces = stdp_step(k, rope(k), v, rope, tk, tv, lam, traces)
                got.append((hebb, im_s))
            start = torch.zeros(B, dtype=torch.long)
            for i in range(blocks):
                if i in reset:
                    start = torch.where(reset[i], torch.full_like(start, i), start)
                k_i, v_i, tk_i, tv_i = seq[i]
                post_now = v_i * torch.exp(1j * rep2(tv_i))
                pre_now = rope(k_i) * torch.exp(-1j * rep2(tk_i))
                s = torch.zeros(B, H, D, D, dtype=torch.complex128)
                for j in range(i + 1):
                    active = (j >= start).view(-1, 1, 1, 1).double()
                    w = (1 - lam) * lam ** (i - j)
                    k_j, v_j, tk_j, tv_j = seq[j]
                    s = s + active * w * torch.einsum('bhna,bhnj->bhaj', post_now, rope(k_j) * torch.exp(-1j * rep2(tk_j)))
                    if j < i:
                        s = s + active * w * torch.einsum('bhna,bhnj->bhaj', v_j * torch.exp(1j * rep2(tv_j)), pre_now)
                torch.testing.assert_close(got[i][1], s.imag, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(got[i][0], torch.einsum('bhna,bhnj->bhaj', v_i, rope(k_i)))
        finally:
            torch.set_default_dtype(torch.float32)


class Clock(unittest.TestCase):
    def test_tags_and_gauge(self):
        """tau = c sin(sum of wrapped pair-angle steps since the reset) + n omegaT, n = 0 in the reset block; a constant
        rotation of each pair's basis leaves the tags unchanged; gradients are finite."""
        g = torch.Generator().manual_seed(1)
        B, H, T, P, blocks, c, om = 2, 2, 4, 3, 7, 0.14, 0.34
        D = 2 * P
        ks = [torch.randn(B, H, T, D, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(blocks)]
        vs = [torch.randn(B, H, T, D, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(blocks)]
        gauge = torch.rand(1, H, 1, P, generator=g, dtype=torch.float64) * 2 * math.pi
        angle = lambda x: torch.atan2(x[..., 1::2], x[..., 0::2])

        def rotate(x):
            x0, x1 = x[..., 0::2], x[..., 1::2]
            return torch.stack((x0 * gauge.cos() - x1 * gauge.sin(), x0 * gauge.sin() + x1 * gauge.cos()), -1).reshape(x.shape)
        runs = []
        for rotated in (False, True):
            clock = torch.zeros(B, H, T, 6 * P + 1, dtype=torch.float64)
            ref_k, ref_v, n = (torch.zeros(*s, dtype=torch.float64) for s in ((B, H, T, P), (B, H, T, P), (B, 1, 1, 1)))
            prev = None
            tags = []
            for i in range(blocks):
                k, v = (rotate(ks[i]), vs[i]) if rotated else (ks[i], vs[i])
                if i == 3:                                            # lane 0 starts a new puzzle
                    m = torch.tensor([True, False]).view(-1, 1, 1, 1)
                    clock = torch.where(m, torch.zeros_like(clock), clock)
                    ref_k, ref_v, n = (torch.where(m, torch.zeros_like(x), x) for x in (ref_k, ref_v, n))
                    prev = tuple(torch.where(m, torch.full_like(x, float('nan')), x) for x in prev)
                tk, tv, count, clock = clock_step(clock, k, v, c, om)
                if prev is not None:
                    ref_k = ref_k + torch.nan_to_num(wrap_phase(angle(k) - prev[0]))
                    ref_v = ref_v + torch.nan_to_num(wrap_phase(angle(v) - prev[1]))
                prev = (angle(k), angle(v))
                torch.testing.assert_close(count.double(), n.expand_as(count).double())
                torch.testing.assert_close(tk, c * torch.sin(ref_k) + n * om)
                torch.testing.assert_close(tv, c * torch.sin(ref_v) + n * om)
                n = n + 1
                tags.append(tk + tv)
            runs.append(tags)
        for a, b in zip(*runs):
            torch.testing.assert_close(a, b)
        sum(t.sum() for t in runs[0]).backward()
        self.assertTrue(all(bool(torch.isfinite(x.grad).all()) for x in ks + vs))

    def test_reduces_to_relative_angle(self):
        """sin of the accumulated wrapped displacement equals sin(theta_b - theta_0): the latency depends only on the
        pair's rotation since the puzzle's first block (the winding number is discarded)."""
        g = torch.Generator().manual_seed(3)
        B, H, T, P = 2, 2, 5, 4
        clock = torch.zeros(B, H, T, 6 * P + 1, dtype=torch.float64)
        xs = [torch.randn(B, H, T, 2 * P, generator=g, dtype=torch.float64) * (0.1 + 3 * i % 7) for i in range(40)]
        angle = lambda x: torch.atan2(x[..., 1::2], x[..., 0::2])
        for n, x in enumerate(xs):
            tk, _, _, clock = clock_step(clock, x, x, 0.14, 0.34)
            torch.testing.assert_close(tk, 0.14 * torch.sin(angle(x) - angle(xs[0])) + n * 0.34, rtol=0, atol=1e-12)

    def test_kappa_is_poisson_sum(self):
        """kappa = (1-lam)(1-lam^2)/(1 - 2 lam cos wT + lam^2) = (1-lam)[1 + 2 sum_k lam^k cos(k wT)]."""
        from lt.analysis.common import kappa
        for lam, om in ((0.5, 0.34), (0.8, 0.2), (0.3, 1.0)):
            series = (1 - lam) * (1 + 2 * sum(lam ** k * math.cos(k * om) for k in range(1, 400)))
            self.assertAlmostEqual(kappa(lam, om), series, places=12)

    def test_time_shift_invariance(self):
        """D1: the STDP term depends on the block counter only through differences; starting the counter at 1000
        instead of 0 leaves Im S unchanged."""
        g = torch.Generator().manual_seed(4)
        B, H, T, P = 2, 2, 5, 4
        D = 2 * P
        rope = rope_fn(H, T, P, g)
        seq = [(torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g, dtype=torch.float64), dim=-1),
                torch.randn(B, H, T, D, generator=g, dtype=torch.float64)) for _ in range(12)]
        runs = []
        for n0 in (0.0, 1000.0):
            clock = torch.zeros(B, H, T, 6 * P + 1, dtype=torch.float64)
            clock[..., -1] = n0
            traces, out = tuple(torch.zeros(B, H, T, D, dtype=torch.float64) for _ in range(4)), []
            for k, v in seq:
                tk, tv, _, clock = clock_step(clock, k, v, 0.14, 0.34)
                _, im_s, traces = stdp_step(k, rope(k), v, rope, tk, tv, 0.5, traces)
                out.append(im_s)
            runs.append(out)
        for a, b in zip(*runs):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-11)

    def test_sign_follows_block_order(self):
        """With omegaT > 2c and K omegaT + 2c < pi, sin(tau_post - tau_pre) > 0 for every pair 1..K blocks apart."""
        c, om, K = 0.14, 0.34, 8
        self.assertTrue(om > 2 * c and K * om + 2 * c < math.pi)
        lat = torch.linspace(-c, c, 41)
        for k in range(1, K + 1):
            diff = k * om + lat[:, None] - lat[None]
            self.assertTrue(bool((torch.sin(diff) > 0).all()))


class Model(unittest.TestCase):
    def test_outer_integral(self):
        """Model B: W_b = rho W_{b-1} + Im S_b, or the EMA form W_b = (1 - eta) W_{b-1} + eta Im S_b (over segments too,
        reset for fresh lanes); rho = eta = 0 is model A."""
        torch.manual_seed(0)
        for rho, eta in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.25)):
            m = LT(tiny(outer_rho=rho, outer_eta=eta)).double().eval()
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.3)
            seen = []
            m.probe = lambda r: seen.append(r.im_s) or None
            carry = m.initial_carry(batch(0))
            with torch.no_grad():
                for _ in range(2):
                    carry, _ = m(carry, batch(0))
            if rho or eta:
                w = torch.zeros_like(seen[0])
                for s in seen:
                    w = rho * w + s if rho else (1 - eta) * w + eta * s
                torch.testing.assert_close(carry.stdp_w, w)
            else:
                self.assertIsNone(carry.stdp_w)
            self.assertEqual(int(carry.clock[..., -1].max()), 2 * m.config.blocks_per_seg)
        with self.assertRaises(ValueError):
            tiny(outer_rho=0.5, outer_eta=0.25)
        # learned eta starts exactly at the fixed value and receives a gradient
        torch.manual_seed(0)
        fixed = LT(tiny(outer_eta=0.25)).double()
        torch.manual_seed(0)
        learn = LT(tiny(outer_eta=0.25, outer_eta_learn=True)).double()
        self.assertEqual(learn.block.eta_raw.shape, (learn.H,))
        with torch.no_grad():
            for pf, pl in zip(fixed.parameters(), [p for n, p in learn.named_parameters() if n != 'block.eta_raw']):
                pl.copy_(pf)
        carry_f, logits_f = fixed(fixed.initial_carry(batch(0)), batch(0))
        carry_l, logits_l = learn(learn.initial_carry(batch(0)), batch(0))
        torch.testing.assert_close(logits_l, logits_f)
        torch.testing.assert_close(carry_l.stdp_w, carry_f.stdp_w)
        logits_l.square().mean().backward()
        self.assertGreater(float(learn.block.eta_raw.grad.abs().max()), 0.0)

    def test_without_hebb(self):
        """hebb=False: the write is alpha Im S only, so with alpha = 0 the memory read is zero and the value projection
        has no effect; with alpha != 0 the Hebbian and no-Hebbian models differ."""
        torch.manual_seed(0)
        outs = {}
        for hebb in (True, False):
            torch.manual_seed(0)
            m = LT(tiny(hebb=hebb)).double().eval()
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.3)
            outs[hebb] = m(m.initial_carry(batch(2)), batch(2))[1]
            if not hebb:
                with torch.no_grad():
                    m.block.alpha.zero_()
                    ref = m(m.initial_carry(batch(2)), batch(2))[1]
                    m.block.v_proj.weight.normal_()
                    torch.testing.assert_close(m(m.initial_carry(batch(2)), batch(2))[1], ref)
        self.assertGreater(float((outs[True] - outs[False]).abs().max()), 1e-3)

    def test_activation_checkpointing(self):
        """act_ckpt recomputes each block in backward: same logits and gradients (EMA form, two segments)."""
        res = {}
        for ck in (False, True):
            torch.manual_seed(0)
            m = LT(tiny(outer_eta=0.125, act_ckpt=ck)).double()
            with torch.no_grad():
                m.block.ffn_down.weight.normal_(0, 0.05)
            carry = m.initial_carry(batch(1))
            for _ in range(2):
                carry, logits = m(carry, batch(1))
            logits.square().mean().backward()
            res[ck] = (logits.detach(), {n: p.grad.clone() for n, p in m.named_parameters()})
        torch.testing.assert_close(res[True][0], res[False][0])
        for n in res[False][1]:
            torch.testing.assert_close(res[True][1][n], res[False][1][n], msg=n)

    def test_initial_step_and_gradients(self):
        """Every trainable parameter gets a finite gradient; alpha does once the clock has moved."""
        torch.manual_seed(0)
        m = LT(tiny()).double()
        with torch.no_grad():
            m.block.ffn_down.weight.normal_(0, 0.05)
        carry = m.initial_carry(batch(1))
        for _ in range(2):
            carry, logits = m(carry, batch(1))
        logits.square().mean().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        self.assertGreater(float(m.block.alpha.grad.abs().sum()), 0)

    def test_order_latency_split(self):
        """Shadow traces with tags shifted by -2 n omegaT give Im S(+) - Im S(-) = 2 O, odd in omegaT, and
        Im S(+) + Im S(-) = 2 L; at a fixed point (constant events) O -> 0 and L -> kappa SB."""
        from lt.analysis.common import OrderLatencyProbe
        g = torch.Generator().manual_seed(2)
        B, H, T, P, lam, om, c = 1, 1, 4, 2, 0.5, 0.34, 0.14
        D = 2 * P
        rope = rope_fn(H, T, P, g)
        k = torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g, dtype=torch.float64), dim=-1)
        v = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
        lat_k, lat_v = (c * torch.sin(torch.randn(B, H, T, P, generator=g, dtype=torch.float64)) for _ in range(2))
        probe = OrderLatencyProbe(om, lam)
        traces = tuple(torch.zeros(B, H, T, D, dtype=torch.float64) for _ in range(4))
        for n in range(60):
            count = torch.full((B, H, T, 1), float(n), dtype=torch.float64)
            tk, tv = lat_k + n * om, lat_v + n * om
            hebb, im_s, new = stdp_step(k, rope(k), v, rope, tk, tv, lam, traces)
            from lt.model import StepRecord
            probe(StepRecord(k, rope(k), None, v, tk, tv, count, traces, new, hebb, im_s, rope))
            traces = new
        O, L = probe.last['O'], probe.last['L']
        torch.testing.assert_close(O + L, im_s)
        self.assertLess(float(O.norm() / hebb.norm()), 1e-10)
        a_k, a_v = rep2(lat_k), rep2(lat_v)
        kr = rope(k)
        sb = (v * a_v.sin()).transpose(-1, -2) @ (kr * a_k.cos()) - (v * a_v.cos()).transpose(-1, -2) @ (kr * a_k.sin())
        kappa = (1 - lam) * (1 - lam ** 2) / (1 - 2 * lam * math.cos(om) + lam ** 2)
        torch.testing.assert_close(L, kappa * sb)


if __name__ == '__main__':
    unittest.main()
