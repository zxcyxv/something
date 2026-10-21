"""URM puzzle embedding in the in-context ARC setting: prefix tokens under ctx_packed, ArcContextDataset num_aug versions
and rule_prior_pe (the prior synapse from the puzzle embedding). Run: python -m unittest discover tests"""
import unittest

import numpy as np
import torch

from lt.data import ArcContextDataset, ArcTrainDataset, ctx_episode, ctx_pack
from lt.model import LT
from tests.test_arc_ctx import _store
from tests.test_packed_ctx import S, G, cfg as packed_cfg, full_episodes, randomize, segments

PE = dict(puzzle_emb_ndim=16, puzzle_emb_mode='prefix', num_puzzle_identifiers=5)


def cfg(**kw):
    return packed_cfg(**dict(PE, **kw))


def leaf(m):
    """.double() turns the SignSGD buffer local_weights into a non-leaf; make it a leaf again (as in FP32 training)."""
    m.puzzle_emb.local_weights = m.puzzle_emb.local_weights.detach().requires_grad_(True)
    return m


def fill_table(m, seed):
    g = torch.Generator().manual_seed(seed)
    m.puzzle_emb.weights.copy_(torch.randn(m.puzzle_emb.weights.shape, generator=g, dtype=torch.float64))


class PackedPrefix(unittest.TestCase):
    def test_packed_equals_dense_with_a_prefix_token(self):
        x, y = full_episodes(2, 1)
        ids = torch.tensor([3, 1], dtype=torch.int32)
        dense = dict(inputs=torch.from_numpy(x.reshape(2, -1).astype(np.int64) + 1),
                     labels=torch.from_numpy(y.reshape(2, -1).astype(np.int64) + 1), puzzle_identifiers=ids)
        packed = dict(ctx_pack(x, y, S, G), puzzle_identifiers=ids)
        models = []
        for packed_mode in (False, True):
            torch.manual_seed(0)
            m = LT(cfg(local_conv=False, ctx_packed=packed_mode)).double().eval()
            self.assertEqual(m.prefix_len, 1)
            with torch.no_grad():
                fill_table(m, 7)
                m.block.frame_gate.uniform_(0.0, 1.0)
                m.block.rope_theta.uniform_(-1.0, 1.0)
            models.append(m)
        with torch.no_grad():
            _, ld = segments(models[0], dense, 3)
            _, lp = segments(models[1], packed, 3)
        T = G * S * S
        for a, b in zip(ld, lp):
            torch.testing.assert_close(b[:, :T], a, rtol=1e-10, atol=1e-10)
        self.assertFalse(torch.allclose(ld[0][0], ld[0][1]))        # the identifiers matter

    def test_local_conv_ignores_the_prefix(self):
        demo = (np.array([[1, 2], [3, 0]]), np.array([[3]]))
        x, y = ctx_episode([demo], (np.array([[4, 1]]), np.array([[5, 6]])), S, 1)
        b = dict(ctx_pack(np.stack([x, x]), np.stack([y, y]), S, G), puzzle_identifiers=torch.tensor([1, 2]))
        torch.manual_seed(0)
        m = LT(cfg(ctx_packed=True)).double().eval()
        randomize(m, 4)
        with torch.no_grad():
            m(m.initial_carry(b), b)                  # sets the prefixed per-token fields
            T, I = b['inputs'].shape[1], m.block.local_conv_w.shape[0]
            mm = torch.randn(2, T + 1, I, dtype=torch.float64)
            with_prefix = m.local_conv(mm)
            m.cur_nbr = b['nbr']
            cells = m.local_conv(mm[:, 1:])
        torch.testing.assert_close(with_prefix[:, 1:], cells, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(with_prefix[:, :1], mm[:, :1], rtol=0, atol=0)

    def test_prefix_learns_through_signsgd_path(self):
        x, y = full_episodes(2, 2)
        b = dict(ctx_pack(x, y, S, G), puzzle_identifiers=torch.tensor([1, 4], dtype=torch.int32))
        torch.manual_seed(0)
        m = leaf(LT(cfg(ctx_packed=True)).double().train())
        randomize(m, 5)
        carry = m.initial_carry(b)
        _, logits = m(carry, b)
        logits[b['valid'] == 1].square().sum().backward()
        self.assertGreater(float(m.puzzle_emb.local_weights.grad.abs().sum()), 0.0)


class Versions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        grid = lambda: rng.integers(0, 10, (2, 3))
        self.tasks = [[(grid(), grid()) for _ in range(3)] for _ in range(4)]
        self.store = _store(self.tasks, S=4, kmax=2)

    def ds(self, **kw):
        args = dict(seed=3, global_batch_size=4, rank=0, world_size=1, epochs_per_iter=3, start_iter=0, total_iters=2,
                    num_aug=5)
        return ArcContextDataset(self.store, **dict(args, **kw))

    def test_identifiers_and_fixed_transforms(self):
        ds = self.ds()
        seen = {}
        for _, b in ds:
            for x, pid in zip(b['inputs'], b['puzzle_identifiers']):
                t, v = divmod(int(pid), 6)
                self.assertTrue(0 <= t < 4)
                seen.setdefault((t, v), set())
        self.assertGreater(len({v for _, v in seen}), 1)
        rng = np.random.Generator(np.random.Philox(key=np.array([1, 2], np.uint64)))
        for _ in range(30):
            x, _, pid = ds.episode(2, rng)
            t, v = divmod(pid, 6)
            self.assertEqual(t, 2)
            k, perm = ds.version(t, v)
            self.assertEqual(int(perm[0]), 0)
            self.assertEqual(sorted(perm.tolist()), list(range(10)))
            want = {ArcTrainDataset._dihedral(perm[np.asarray(a, np.uint8)], k).tobytes() for a, _ in self.tasks[2]}
            slots = x.reshape(2 * ds.kmax + 2, ds.S, ds.S)
            demo_in = slots[0]
            h, w = ArcTrainDataset._dihedral(np.zeros((2, 3)), k).shape
            self.assertIn((demo_in[:h, :w].astype(np.int64) - 1).astype(np.uint8).tobytes(), want)
        (k1, p1), (k2, p2) = ds.version(2, 3), ds.version(2, 3)
        self.assertEqual(k1, k2)
        np.testing.assert_array_equal(p1, p2)
        k0, p0 = ds.version(2, 0)
        self.assertEqual(k0, 0)
        np.testing.assert_array_equal(p0, np.arange(10))

    def test_packed_stream_carries_the_identifiers(self):
        for _, b in self.ds(packed=True, bucket_window=2):
            self.assertTrue(bool((b['puzzle_identifiers'] < 4 * 6).all()))
            self.assertTrue(bool((b['puzzle_identifiers'] >= 0).all()))

    def test_without_versions_the_stream_is_unchanged(self):
        args = dict(seed=3, global_batch_size=4, rank=0, world_size=1, epochs_per_iter=1, start_iter=0, total_iters=1)
        for _, b in ArcContextDataset(self.store, **args):
            self.assertTrue(bool((b['puzzle_identifiers'] == 0).all()))


class RulePriorPE(unittest.TestCase):
    def test_prior_synapse_from_the_puzzle_embedding(self):
        x, y = full_episodes(2, 3)
        b = dict(ctx_pack(x, y, S, G), puzzle_identifiers=torch.tensor([1, 4], dtype=torch.int32))
        torch.manual_seed(0)
        m = leaf(LT(cfg(ctx_packed=True, rule_path=True, rule_prior_emb=3, rule_prior_pe=True)).double().train())
        randomize(m, 6)
        L = m.block
        with torch.no_grad():
            m.puzzle_emb.weights.normal_()
            L.rp_prior_map.normal_(0, 0.1)
            L.rp_out.weight.normal_(0, 0.1)
        self.assertFalse(hasattr(L, 'rp_prior_table'))
        carry = m.initial_carry(b)
        carry, logits = m(carry, b)
        want = (m.puzzle_emb.weights[b['puzzle_identifiers'].long()] @ L.rp_prior_proj.weight.T
                @ L.rp_prior_map.T).view(2, m.H, 2 * m.dh, m.dh)
        torch.testing.assert_close(m.cur_R0, want.float())
        logits[b['valid'] == 1].square().sum().backward()
        for p in (L.rp_prior_proj.weight, L.rp_prior_map, m.puzzle_emb.local_weights):
            self.assertGreater(float(p.grad.abs().sum()), 0.0)

    def test_needs_a_puzzle_embedding(self):
        with self.assertRaises(ValueError):
            cfg(ctx_packed=True, rule_path=True, rule_prior_emb=3, rule_prior_pe=True, puzzle_emb_ndim=0)


if __name__ == '__main__':
    unittest.main()
