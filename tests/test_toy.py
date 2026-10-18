"""CPU/FP64 checks of the 4x4 toy path: linear_attn reads q~ (sum_t v k~^T)^T, mlp_inter sets the FFN width, and the
data loader accepts 4x4 boards (augmentation keeps solutions valid).
Run: python -m unittest discover tests"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from lt.data import _apply_aug, _check_boards, _draw_aug_params, eval_batches
from lt.model import LT, ModelConfig


def toy(**kw):
    return ModelConfig(**dict(dict(batch_size=2, hidden_size=12, num_heads=1, puzzle_emb_ndim=0, grid=4, seq_len=16,
                                   vocab_size=6, blocks_per_seg=4, loops=2, mlp_inter=32, linear_attn=True,
                                   tag_bound=0.0, amp=False, forward_dtype='float64', kernel='torch'), **kw))


GRID = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]], dtype=np.uint8)


class Toy(unittest.TestCase):
    def test_linear_attn_read(self):
        torch.manual_seed(0)
        m = LT(toy()).double().eval()
        g = torch.Generator().manual_seed(1)
        q, k, v = (torch.randn(2, 1, 16, 12, generator=g, dtype=torch.float64) for _ in range(3))
        state = m.initial_state(2, torch.device('cpu'), torch.float64)
        read, new = m.memory_step(q, k, v, state)
        a = m.rope_angles()
        tab = a.cos()[None], a.sin()[None]
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
        qr, kr = m.rope(unit(q), tab), m.rope(unit(k), tab)
        torch.testing.assert_close(read, qr @ (v.transpose(-1, -2) @ kr).transpose(-1, -2), rtol=1e-12, atol=1e-12)
        self.assertFalse(hasattr(m.block, 'alpha'))
        self.assertEqual(m.block.ffn_down.in_features, 32)

    def test_forward_backward(self):
        torch.manual_seed(0)
        m = LT(toy()).double()
        g = torch.Generator().manual_seed(2)
        batch = dict(inputs=torch.randint(1, 6, (2, 16), generator=g), labels=torch.randint(2, 6, (2, 16), generator=g),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        carry = m.initial_carry(batch)
        carry, logits = m(carry, batch)
        self.assertEqual(tuple(logits.shape), (2, 16, 6))
        logits.sum().backward()
        self.assertGreater(float(m.block.q_proj.weight.grad.abs().sum()), 0.0)

    def test_data_4x4(self):
        x = GRID.copy()
        x[0, 0] = x[1, 2] = 0
        cx, cy = _check_boards(x[None], GRID[None], 'toy')
        self.assertEqual(cx.shape, (1, 4, 4))
        for i in range(20):
            prm = _draw_aug_params(np.random.Generator(np.random.Philox(key=np.array([3, i], dtype=np.uint64))), 4)
            _check_boards(_apply_aug(x, *prm)[None], _apply_aug(GRID, *prm)[None], 'aug')
        b = next(eval_batches(cx, cy, 4))
        self.assertEqual(tuple(b['inputs'].shape), (1, 16))

    def test_ema_puzzle_emb(self):
        """with_puzzle_emb averages the puzzle-embedding buffer too and swaps it in and out with the parameters."""
        from lt.train import EMA
        torch.manual_seed(0)
        m = LT(toy(puzzle_emb_ndim=12, num_puzzle_identifiers=3))
        ema = EMA(m, 0.5, with_puzzle_emb=True)
        with torch.no_grad():
            m.puzzle_emb.weights.add_(2.0)
        ema.update(m)
        raw = m.puzzle_emb.weights.clone()
        with ema.swap(m):
            torch.testing.assert_close(m.puzzle_emb.weights, raw - 1.0)
        torch.testing.assert_close(m.puzzle_emb.weights, raw)
        self.assertNotIn('puzzle_emb.weights', EMA(m, 0.5).shadow)

    def test_prefix_puzzle_embedding(self):
        """prefix mode: the embedding occupies ceil(ndim / d) positions in front of the cells (angle-0 RoPE) and the
        logits cover the cells only; different identifiers change the output."""
        torch.manual_seed(0)
        m = LT(toy(puzzle_emb_ndim=12, num_puzzle_identifiers=3, puzzle_emb_mode='prefix')).double().eval()
        self.assertEqual((m.prefix_len, m.n_pos), (1, 17))
        with torch.no_grad():
            m.puzzle_emb.weights.normal_()
        self.assertTrue(bool((m.rope_angles()[:, 0] == 0).all()))
        g = torch.Generator().manual_seed(4)
        inputs = torch.randint(1, 6, (2, 16), generator=g)
        outs = []
        for pid in (0, 1):
            batch = dict(inputs=inputs, labels=torch.full((2, 16), 2), puzzle_identifiers=torch.full((2,), pid, dtype=torch.int32))
            carry = m.initial_carry(batch)
            self.assertEqual(tuple(carry.hidden.shape), (2, 17, 12))
            carry, logits = m(carry, batch)
            self.assertEqual(tuple(logits.shape), (2, 16, 6))
            outs.append(logits)
        self.assertGreater(float((outs[0] - outs[1]).abs().max()), 1e-6)
        with self.assertRaises(ValueError):
            toy(puzzle_emb_mode='concat')

    def test_task_film(self):
        """gamma = 0 equals the model without FiLM; a nonzero row changes only puzzles with that identifier."""
        torch.manual_seed(0)
        ref = LT(toy(num_puzzle_identifiers=3)).double().eval()
        torch.manual_seed(0)
        m = LT(toy(num_puzzle_identifiers=3, task_film=True)).double().eval()
        with torch.no_grad():
            ref.block.ffn_down.weight.normal_()          # zero at initialization, which would hide the gate
        m.load_state_dict({**ref.state_dict(), 'block.task_film': torch.zeros(3, 32, dtype=torch.float64)})
        g = torch.Generator().manual_seed(5)
        batch = dict(inputs=torch.randint(1, 6, (2, 16), generator=g), labels=torch.full((2, 16), 2),
                     puzzle_identifiers=torch.tensor([0, 2], dtype=torch.int32))
        out = lambda model: model(model.initial_carry(batch), batch)[1]
        torch.testing.assert_close(out(m), out(ref), rtol=0, atol=0)
        with torch.no_grad():
            m.block.task_film[2].normal_()
        diff = (out(m) - out(ref)).abs().amax((1, 2))
        self.assertEqual(float(diff[0]), 0.0)
        self.assertGreater(float(diff[1]), 0.0)

    def test_task_lowrank(self):
        """U = 0 equals the model without the change; a nonzero U row changes only puzzles with that identifier."""
        torch.manual_seed(0)
        ref = LT(toy(num_puzzle_identifiers=3)).double().eval()
        torch.manual_seed(0)
        m = LT(toy(num_puzzle_identifiers=3, task_lowrank=2)).double().eval()
        with torch.no_grad():
            ref.block.ffn_down.weight.normal_()
        sd = {**ref.state_dict(), 'block.task_u': torch.zeros(3, 2, 32, dtype=torch.float64),
              'block.task_v': m.block.task_v.detach().clone()}
        m.load_state_dict(sd)
        self.assertGreater(float(m.block.task_v.abs().sum()), 0.0)
        g = torch.Generator().manual_seed(6)
        batch = dict(inputs=torch.randint(1, 6, (2, 16), generator=g), labels=torch.full((2, 16), 2),
                     puzzle_identifiers=torch.tensor([0, 1], dtype=torch.int32))
        out = lambda model: model(model.initial_carry(batch), batch)[1]
        torch.testing.assert_close(out(m), out(ref), rtol=0, atol=0)
        with torch.no_grad():
            m.block.task_u[1].normal_()
        diff = (out(m) - out(ref)).abs().amax((1, 2))
        self.assertEqual(float(diff[0]), 0.0)
        self.assertGreater(float(diff[1]), 0.0)

    def test_film_from_embedding(self):
        """puzzle_emb_mode film: a zero embedding table equals the model without task information; the shared
        projection turns a nonzero row into gate scales for that identifier only, and the embedding gets a gradient."""
        torch.manual_seed(0)
        ref = LT(toy(num_puzzle_identifiers=3)).double().eval()
        torch.manual_seed(0)
        m = LT(toy(num_puzzle_identifiers=3, puzzle_emb_ndim=12, puzzle_emb_mode='film')).double().eval()
        with torch.no_grad():
            ref.block.ffn_down.weight.normal_()
        own = {k: v for k, v in m.state_dict().items() if 'task_film_proj' in k or 'puzzle_emb' in k}
        m.load_state_dict({**ref.state_dict(), **own})
        self.assertGreater(float(m.block.task_film_proj.weight.abs().sum()), 0.0)
        g = torch.Generator().manual_seed(7)
        batch = dict(inputs=torch.randint(1, 6, (2, 16), generator=g), labels=torch.full((2, 16), 2),
                     puzzle_identifiers=torch.tensor([0, 1], dtype=torch.int32))
        out = lambda model: model(model.initial_carry(batch), batch)[1]
        torch.testing.assert_close(out(m), out(ref), rtol=0, atol=0)
        with torch.no_grad():
            m.puzzle_emb.weights[1].normal_()
        diff = (out(m) - out(ref)).abs().amax((1, 2))
        self.assertEqual(float(diff[0]), 0.0)
        self.assertGreater(float(diff[1]), 0.0)

    def test_ignore_label0(self):
        """ignore_label0: cells whose raw label is 0 (token 1) leave the loss and the exact count unaffected."""
        from lt.train import segment_loss
        torch.manual_seed(0)
        logits = torch.randn(2, 6, 4, dtype=torch.float64)
        labels = torch.tensor([[1, 1, 2, 3, 2, 3], [1, 1, 3, 3, 2, 2]])
        halted = torch.ones(2, dtype=torch.bool)
        loss, met = segment_loss(logits, labels, halted, True)
        ref, _ = segment_loss(logits[:, 2:], labels[:, 2:], halted)
        torch.testing.assert_close(loss, ref)
        logits2 = logits.clone()
        logits2[..., 2:4] = 0
        logits2[:, 2:, :] = -10
        for b in range(2):
            for t in range(2, 6):
                logits2[b, t, labels[b, t]] = 10
        _, met2 = segment_loss(logits2, labels, halted, True)
        self.assertEqual(int(met2['exact_accuracy']), 2)

    def test_task_film_up(self):
        """Up-side scale: w = 0 equals the model without it; with only slot i scaled by 1 + w the FFN output moves by
        w times slot i's own contribution (linear in each slot)."""
        torch.manual_seed(0)
        ref = LT(toy(num_puzzle_identifiers=3)).double().eval()
        torch.manual_seed(0)
        m = LT(toy(num_puzzle_identifiers=3, task_film_up=True)).double().eval()
        with torch.no_grad():
            ref.block.ffn_down.weight.normal_()
        m.load_state_dict({**ref.state_dict(), 'block.task_film_up': torch.zeros(3, 32, dtype=torch.float64)})
        g = torch.Generator().manual_seed(8)
        batch = dict(inputs=torch.randint(1, 6, (2, 16), generator=g), labels=torch.full((2, 16), 2),
                     puzzle_identifiers=torch.tensor([0, 2], dtype=torch.int32))
        out = lambda model: model(model.initial_carry(batch), batch)[1]
        torch.testing.assert_close(out(m), out(ref), rtol=0, atol=0)
        L = m.block
        h = torch.randn(2, 16, 12, dtype=torch.float64, generator=g)
        gg, aa = L.ffn_gate_up(h).chunk(2, dim=-1)
        slot = F.silu(gg) * aa
        with torch.no_grad():
            L.task_film_up[2, 5] = 0.7
        m.task_ids = torch.tensor([0, 2])
        scaled = slot * (1 + L.task_film_up[m.task_ids][:, None, :])
        diff = L.ffn_down(scaled) - L.ffn_down(slot)
        torch.testing.assert_close(diff[1], 0.7 * slot[1, :, 5:6] * L.ffn_down.weight[:, 5])
        torch.testing.assert_close(diff[0], torch.zeros_like(diff[0]))

    def test_config_checks(self):
        for kw in (dict(outer_eta=0.125), dict(tag_bound=0.14), dict(hebb=False), dict(mlp_inter=-1)):
            with self.assertRaises(ValueError, msg=str(kw)):
                toy(**kw)


if __name__ == '__main__':
    unittest.main()
