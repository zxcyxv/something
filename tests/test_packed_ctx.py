"""ctx_packed (padding-free in-context ARC): lt.data.ctx_pack layout, dense = packed when nothing is dropped, padding
writes nothing, lanes take a new length together, and the packed dataset stream. Run: python -m unittest discover tests"""
import unittest

import numpy as np
import torch

from lt.data import ArcContextDataset, ctx_episode, ctx_pack, ctx_packed_length
from lt.model import LT
from tests.test_frames import cfg as frame_cfg
from tests.test_arc_ctx import _store

S, G = 3, 4


def cfg(**kw):
    return frame_cfg(**dict(dict(local_conv=True, frame_rope=True), **kw))


def full_episodes(n, seed):
    """Raw dense episodes whose slots 0..G-2 have no pad cell: packing drops nothing."""
    g = np.random.default_rng(seed)
    x = np.zeros((n, G * S, S), np.uint8)
    x[:, :(G - 1) * S] = g.integers(1, 12, (n, (G - 1) * S, S))
    y = np.zeros_like(x)
    y[:, (G - 1) * S:] = g.integers(0, 12, (n, S, S))
    return x, y


def randomize(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.block.local_conv_w.copy_(torch.randn(m.block.local_conv_w.shape, generator=g, dtype=torch.float64) * 0.3)
        m.block.frame_gate.copy_(torch.rand(m.block.frame_gate.shape, generator=g, dtype=torch.float64))
        m.block.rope_theta.uniform_(-1.0, 1.0, generator=g)


def segments(m, b, n):
    carry, out = m.initial_carry(b), []
    for _ in range(n):
        carry, logits = m(carry, b)
        out.append(logits)
    return carry, out


class Layout(unittest.TestCase):
    def test_full_slots_keep_the_dense_order(self):
        x, y = full_episodes(2, 0)
        b = ctx_pack(x, y, S, G)
        T = G * S * S
        self.assertEqual(b['inputs'].shape[1] % 256, 0)
        np.testing.assert_array_equal(b['inputs'][:, :T].numpy(), x.reshape(2, -1).astype(np.int64) + 1)
        np.testing.assert_array_equal(b['labels'][:, :T].numpy(), y.reshape(2, -1).astype(np.int64) + 1)
        self.assertTrue((b['valid'][:, :T] == 1).all() and (b['valid'][:, T:] == 0).all())
        self.assertTrue((b['labels'][:, T:] == 1).all())
        t = np.arange(T)
        np.testing.assert_array_equal(b['pos_row'][0, :T].numpy(), t // S)
        np.testing.assert_array_equal(b['pos_col'][0, :T].numpy(), t % S)

    def test_dropped_cells_and_neighbours(self):
        demo = (np.array([[1, 2]]), np.array([[3]]))
        x, y = ctx_episode([demo], (np.array([[4]]), np.array([[5, 6]])), S, 1)
        b = ctx_pack(x[None], y[None], S, G)
        L = int(b['valid'].sum())
        self.assertEqual(L, int(ctx_packed_length(x[None], S, G)[0]))
        self.assertEqual(L, 2 * 3 + 2 * 2 + 2 * 2 + S * S)     # (h+1)(w+1) cells per grid (EOS incl. corner) + query slot
        rows, cols, nbr, T = b['pos_row'][0], b['pos_col'][0], b['nbr'][0], b['inputs'].shape[1]
        where = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows[:L], cols[:L]))}
        for i in range(L):
            for k in range(9):
                r, c = int(rows[i]) + k // 3 - 1, int(cols[i]) + k % 3 - 1
                same_slot = r // S == int(rows[i]) // S and 0 <= c < S and r >= 0
                want = where.get((r, c), T) if same_slot else T
                self.assertEqual(int(nbr[i, k]), want)


class Model(unittest.TestCase):
    def test_packed_equals_dense_when_nothing_is_dropped(self):
        x, y = full_episodes(2, 1)
        dense = dict(inputs=torch.from_numpy(x.reshape(2, -1).astype(np.int64) + 1),
                     labels=torch.from_numpy(y.reshape(2, -1).astype(np.int64) + 1),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        packed = dict(ctx_pack(x, y, S, G), puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        torch.manual_seed(0)
        md = LT(cfg()).double().eval()
        torch.manual_seed(0)
        mp = LT(cfg(ctx_packed=True)).double().eval()
        randomize(md, 3)
        randomize(mp, 3)
        T = G * S * S
        with torch.no_grad():
            _, ld = segments(md, dense, 3)
            _, lp = segments(mp, packed, 3)
        for a, b in zip(ld, lp):
            torch.testing.assert_close(b[:, :T], a, rtol=1e-10, atol=1e-10)

    def test_padding_writes_nothing(self):
        demo = (np.array([[1, 2], [3, 0]]), np.array([[3]]))
        x, y = ctx_episode([demo], (np.array([[4, 1]]), np.array([[5, 6]])), S, 1)
        b = dict(ctx_pack(np.stack([x, x]), np.stack([y, y]), S, G), puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        b2 = {k: v.clone() for k, v in b.items()}
        pad = b2['valid'] == 0
        b2['inputs'][pad] = 7
        b2['pos_row'][pad] = 5
        torch.manual_seed(0)
        m = LT(cfg(ctx_packed=True)).double().eval()
        randomize(m, 4)
        with torch.no_grad():
            _, l1 = segments(m, b, 2)
            _, l2 = segments(m, b2, 2)
        live = b['valid'] == 1
        for a, c in zip(l1, l2):
            torch.testing.assert_close(c[live], a[live], rtol=1e-12, atol=1e-12)

    def test_lanes_take_a_new_length_together(self):
        torch.manual_seed(0)
        m = LT(cfg(ctx_packed=True, loops=2)).double().eval()
        x1, y1 = full_episodes(2, 5)
        small = ctx_episode([(np.array([[1]]), np.array([[2]]))], (np.array([[3]]), np.array([[4]])), S, 1)
        b1 = dict(ctx_pack(x1, y1, S, G), puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        b2 = dict(ctx_pack(np.stack([small[0]] * 2), np.stack([small[1]] * 2), S, G),
                  puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        self.assertNotEqual(b1['inputs'].shape, b2['inputs'].shape) if b1['inputs'].shape[1] != b2['inputs'].shape[1] else None
        with torch.no_grad():
            carry = m.initial_carry(b1)
            carry, _ = m(carry, b1)
            carry, _ = m(carry, b2)                     # not halted: b2 ignored
            self.assertTrue(bool(carry.halted.all()))
            carry, logits = m(carry, b2)                # every lane takes b2
        self.assertEqual(logits.shape[1], b2['inputs'].shape[1])


class Stream(unittest.TestCase):
    def test_packed_stream(self):
        rng = np.random.default_rng(0)
        grid = lambda: rng.integers(0, 10, (int(rng.integers(1, 3)), int(rng.integers(1, 3))))
        tasks = [[(grid(), grid()) for _ in range(3)] for _ in range(12)]
        store = _store(tasks, S=S, kmax=1)
        args = dict(seed=1, global_batch_size=2, rank=0, world_size=1, epochs_per_iter=2, start_iter=0, total_iters=2,
                    packed=True, bucket_window=4)
        full = [(it, {k: v.clone() for k, v in b.items()}) for it, b in ArcContextDataset(store, **args)]
        self.assertEqual(len(full), 2 * (12 * 2 // 2))
        for _, b in full:
            self.assertEqual(b['inputs'].shape[1] % 256, 0)
            lab = b['labels'][b['valid'] == 1]
            self.assertTrue(int((lab != 1).sum()) > 0)
        resumed = list(ArcContextDataset(store, **dict(args, start_iter=1, skip_batches=5)))
        for (i1, b1), (i2, b2) in zip(full[12 + 5:], resumed):
            self.assertEqual(i1, i2)
            self.assertTrue(all(bool((b1[k] == b2[k]).all()) for k in b1))


if __name__ == '__main__':
    unittest.main()
