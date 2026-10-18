"""Harness: resuming from a checkpoint reproduces the uninterrupted run exactly (CPU, tiny model, real data)."""
import contextlib
import io
import tempfile
import unittest

import torch

from lt import train


def tiny_run(out, steps, **over):
    cfg = dict(out_dir=out, max_steps=steps, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, blocks_per_seg=2,
               loops=3, global_batch_size=4, test_size=8, epochs=2, eval_interval=1, lr_warmup_steps=2, compile=False,
               dataloader_workers=0, save_every_steps=0, log_every=1000, amp=False, kernel='torch')
    cfg.update(over)
    with contextlib.redirect_stdout(io.StringIO()):
        return train.main(cfg)


class Resume(unittest.TestCase):
    def test_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_run(a, 7)
            tiny_run(b, 4)                       # stops at step 4 with a checkpoint (mid-puzzle: carry is saved)
            tiny_run(b, 7)                       # resumes from it
            ca = torch.load(f'{a}/step_7.pt', weights_only=False)
            cb = torch.load(f'{b}/step_7.pt', weights_only=False)
            for key in ('raw_model', 'model', 'ema'):
                for name, value in ca[key].items():
                    self.assertTrue(torch.equal(value, cb[key][name]), f'{key}.{name}')
            for name in ('hidden', 'clock', 'key_trace', 'value_trace'):
                self.assertTrue(torch.equal(ca['rank_states'][0]['carry'][name], cb['rank_states'][0]['carry'][name]))


    def test_resume_is_exact_with_act(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_run(a, 7, act=True, act_explore=0.5)
            tiny_run(b, 4, act=True, act_explore=0.5)
            tiny_run(b, 7, act=True, act_explore=0.5)
            ca = torch.load(f'{a}/step_7.pt', weights_only=False)
            cb = torch.load(f'{b}/step_7.pt', weights_only=False)
            for name, value in ca['raw_model'].items():
                self.assertTrue(torch.equal(value, cb['raw_model'][name]), name)
            for name in ('hidden', 'steps', 'halted'):
                self.assertTrue(torch.equal(ca['rank_states'][0]['carry'][name], cb['rank_states'][0]['carry'][name]))


class GradAccum(unittest.TestCase):
    def test_micro_batches_equal_one_batch(self):
        """grad_accum 2 x 2 lanes = 4 lanes in one batch: same weights and carry (up to the summation order)."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_run(a, 5)
            tiny_run(b, 5, grad_accum=2)
            ca = torch.load(f'{a}/step_5.pt', weights_only=False)
            cb = torch.load(f'{b}/step_5.pt', weights_only=False)
            for name, value in ca['raw_model'].items():
                torch.testing.assert_close(value, cb['raw_model'][name], atol=1e-5, rtol=1e-4, msg=name)
            for name in ('hidden', 'key_trace', 'value_trace', 'steps'):
                torch.testing.assert_close(ca['rank_states'][0]['carry'][name], cb['rank_states'][0]['carry'][name],
                                           atol=1e-4, rtol=1e-4)

    def test_no_carry_checkpoint_resumes_fresh(self):
        with tempfile.TemporaryDirectory() as b:
            tiny_run(b, 4, save_carry=False)
            ck = torch.load(f'{b}/step_4.pt', weights_only=False)
            self.assertIsNone(ck['rank_states'][0]['carry'])
            self.assertEqual(tiny_run(b, 6, save_carry=False), 6)

    @unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
    def test_offloaded_bank_round_trip(self):
        from lt.model import Carry
        dev = torch.device('cuda')
        bank = train.CarryBank(2, True, dev)
        mk = lambda s: Carry(hidden=torch.randn(3, 5, 4, device=dev) + s, steps=torch.arange(3, device=dev, dtype=torch.int32),
                             halted=torch.tensor([True, False, True], device=dev),
                             data=dict(inputs=torch.randint(0, 9, (3, 5), device=dev)), key_trace=torch.randn(3, 2, 5, 8, device=dev))
        c0, c1 = mk(0), mk(10)
        bank.put(0, c0)
        bank.put(1, c1)
        for _ in range(2):
            bank.prefetch(0)
            t0 = bank.take(0)
            bank.prefetch(1)
            t1 = bank.take(1)
            for x, y in ((t0, c0), (t1, c1)):
                self.assertTrue(torch.equal(x.hidden, y.hidden) and torch.equal(x.key_trace, y.key_trace)
                                and torch.equal(x.data['inputs'], y.data['inputs']) and x.clock is None)
            bank.put(0, t0)
            bank.put(1, t1)
        self.assertTrue(torch.equal(bank.all().hidden, torch.cat([c0.hidden, c1.hidden]).cpu()))


class Act(unittest.TestCase):
    def test_halting(self):
        from lt.model import LT, ModelConfig
        torch.manual_seed(0)
        m = LT(ModelConfig(batch_size=3, seq_len=16, grid=4, vocab_size=6, hidden_size=16, num_heads=2, puzzle_emb_ndim=16,
                           puzzle_emb_mode='prefix', blocks_per_seg=2, loops=4, amp=False, kernel='torch', act=True,
                           act_explore=0.0, forward_dtype='float64')).double()
        batch = dict(inputs=torch.randint(1, 6, (3, 16)), labels=torch.randint(1, 6, (3, 16)),
                     puzzle_identifiers=torch.zeros(3, dtype=torch.int32))
        m.train()
        carry, _ = m(m.initial_carry(batch), batch)
        self.assertFalse(bool(carry.halted.any()))                    # bias -5: no early halt
        self.assertTrue(torch.allclose(m.q_halt_logits, torch.full((3,), -5.0)))
        with torch.no_grad():
            m.q_head.bias.fill_(5.0)
        carry, _ = m(m.initial_carry(batch), batch)
        self.assertTrue(bool(carry.halted.all()))                     # q_halt > 0 halts after one segment
        m.config.act_explore = 1.0                                    # exploration: at least 2 segments
        carry, _ = m(m.initial_carry(batch), batch)
        self.assertFalse(bool(carry.halted.any()))
        m.eval()                                                      # evaluation runs `loops` segments
        carry = m.initial_carry(batch)
        for s in range(4):
            carry, _ = m(carry, batch)
            self.assertEqual(bool(carry.halted.all()), s == 3)


class FastFFN(unittest.TestCase):
    def model(self, fast, seed=0):
        from lt.model import LT, ModelConfig
        torch.manual_seed(seed)
        return LT(ModelConfig(batch_size=3, seq_len=16, grid=4, vocab_size=6, hidden_size=16, num_heads=2,
                              puzzle_emb_ndim=16, blocks_per_seg=3, loops=4, amp=False, kernel='torch', bands=2,
                              band_periods=(2.0, 6.0), tag_bound=0.0, outer_eta=0.125, fast_ffn=fast,
                              forward_dtype='float64')).double()

    def batch(self):
        g = torch.Generator().manual_seed(1)
        return dict(inputs=torch.randint(1, 6, (3, 16), generator=g), labels=torch.randint(1, 6, (3, 16), generator=g),
                    puzzle_identifiers=torch.zeros(3, dtype=torch.int32))

    def test_starts_as_the_model_without_it(self):
        a, b = self.model(0), self.model(8)
        with torch.no_grad():                       # nonzero FFN so the FFN path matters
            for m in (a, b):
                m.block.ffn_down.weight.copy_(torch.randn_like(m.block.ffn_down.weight, generator=torch.Generator().manual_seed(2)))
        batch = self.batch()
        ca, cb = a.initial_carry(batch), b.initial_carry(batch)
        for _ in range(2):
            ca, la = a(ca, batch)
            cb, lb = b(cb, batch)
            self.assertTrue(torch.allclose(la, lb, atol=1e-12))
        self.assertGreater(float(cb.fast_w.abs().sum()), 0.0)          # dW is written even while W_fo = 0

    def test_delta_step_lowers_the_inner_loss(self):
        m = self.model(8)
        B, T, I, r = 3, 16, m.block.ffn_down.in_features, 8
        g = torch.Generator().manual_seed(3)
        h = torch.randn(B, T, 16, generator=g, dtype=torch.float64)
        mm = torch.randn(B, T, I, generator=g, dtype=torch.float64)
        w0 = torch.randn(B, r, I, generator=g, dtype=torch.float64)
        with torch.no_grad():
            m.block.ffw_decay_raw.fill_(-30.0)                         # lam ~ 0: the step is pure gradient descent
            _, w1 = m.fast_ffn_step(h, mm, torch.zeros(B, T, 16, dtype=torch.float64), w0)
        phi = mm / (mm.norm(dim=-1, keepdim=True) + m.config.eps)
        vf = h @ m.block.ffw_v.T
        loss = lambda w: ((vf - phi @ w.transpose(-1, -2)) ** 2).sum()
        self.assertLess(float(loss(w1)), float(loss(w0)))

    def test_gradient_reaches_the_new_parameters(self):
        m = self.model(8)
        with torch.no_grad():
            m.block.ffn_down.weight.normal_(0, 0.1)
        batch = self.batch()
        c = m.initial_carry(batch)
        c, _ = m(c, batch)
        _, logits = m(c, batch)
        logits.square().sum().backward()
        self.assertGreater(float(m.block.ffw_out.grad.abs().sum()), 0.0)

    def test_resume_is_exact_and_warm_start_runs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, tempfile.TemporaryDirectory() as c:
            over = dict(fast_ffn=4, bands=2, band_periods=[2.0, 6.0], tag_bound=0.0, outer_eta=0.125)
            tiny_run(a, 6, **over)
            tiny_run(b, 3, **over)
            tiny_run(b, 6, **over)
            ca = torch.load(f'{a}/step_6.pt', weights_only=False)
            cb = torch.load(f'{b}/step_6.pt', weights_only=False)
            for name, value in ca['raw_model'].items():
                self.assertTrue(torch.equal(value, cb['raw_model'][name]), name)
            self.assertTrue(torch.equal(ca['rank_states'][0]['carry']['fast_w'], cb['rank_states'][0]['carry']['fast_w']))
            base = dict(bands=2, band_periods=[2.0, 6.0], tag_bound=0.0, outer_eta=0.125)
            tiny_run(c + '/base', 2, **base)
            self.assertEqual(tiny_run(c + '/warm', 2, init_from=c + '/base/step_2.pt', **over), 2)
            warm = torch.load(c + '/warm/step_2.pt', weights_only=False)
            self.assertIn('block.ffw_out', warm['raw_model'])


if __name__ == '__main__':
    unittest.main()
