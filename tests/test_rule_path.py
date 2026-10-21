"""rule_path (learned rule synapse on bands, no hand wiring) and grad_blocks (last-block gradients).
Run: python -m unittest discover tests"""
import unittest

import numpy as np
import torch

from lt.data import ctx_episode, ctx_pack
from lt.model import LT
from tests.test_packed_ctx import S, G, cfg as packed_cfg


def cfg(**kw):
    return packed_cfg(**dict(dict(ctx_packed=True, rule_path=True), **kw))


def batch(n=2, seed=0):
    g = np.random.default_rng(seed)
    eps = []
    for _ in range(n):
        demos = [(g.integers(0, 10, (2, 2)), g.integers(0, 10, (2, 3))) for _ in range(1)]
        eps.append(ctx_episode(demos, (g.integers(0, 10, (2, 2)), g.integers(0, 10, (3, 2))), S, 1))
    b = ctx_pack(np.stack([e[0] for e in eps]), np.stack([e[1] for e in eps]), S, G)
    b['puzzle_identifiers'] = torch.zeros(n, dtype=torch.int32)
    return b


def run(m, b, segs=2):
    carry, out = m.initial_carry(b), []
    for _ in range(segs):
        carry, logits = m(carry, b)
        out.append(logits)
    return carry, out


class RulePath(unittest.TestCase):
    def test_runs_and_learns_every_part(self):
        torch.manual_seed(0)
        m = LT(cfg()).double()
        with torch.no_grad():
            m.block.rp_out.weight.normal_(0, 0.1)
        b = batch()
        carry = m.initial_carry(b)
        carry, logits = m(carry, b)
        self.assertEqual(tuple(carry.prev.shape), (2, m.H, 2 * m.dh, m.dh))
        self.assertGreater(float(carry.prev.abs().sum()), 0.0)
        logits[b['valid'] == 1].square().sum().backward()
        L = m.block
        for name in ('rp_q', 'rp_k', 'rp_v', 'rp_key', 'rp_write', 'rp_out'):
            self.assertGreater(float(getattr(L, name).weight.grad.abs().sum()), 0.0, name)
        for p in (L.rp_eta_raw, L.rp_role, L.rp_pair):
            self.assertGreater(float(p.grad.abs().sum()), 0.0)

    def test_delta_write_converges_to_the_targets(self):
        """theta ~ 0, fixed reads and keys: the gated delta steps descend the least-squares fit of R kappa_t to a_t."""
        torch.manual_seed(1)
        m = LT(cfg()).double()
        b = batch()
        m.cur_valid, m.cur_pos = b['valid'], (b['pos_row'], b['pos_col'])
        with torch.no_grad():
            m.block.rp_theta_raw.fill_(-40.0)
            m.block.rp_eta_raw.fill_(4.0)
            m.block.rp_write.bias.fill_(-40.0)
            m.block.rp_write.weight.zero_()
            T = b['inputs'].shape[1]
            u = torch.randn(2, T, m.d, dtype=torch.float64)
            R = torch.zeros(2, m.H, 2 * m.dh, m.dh, dtype=torch.float64)
            m.block.rp_write.bias[0] = 40.0                       # head 0 writes from every token
            for _ in range(300):
                out, R = m.rule_path_step(u, R)
            # recompute the read and keys exactly as rule_path_step does
            L = m.block
            heads = lambda x: x.reshape(2, T, m.H, m.dh).transpose(1, 2)
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + m.config.eps)
            kappa = unit(heads(u @ L.rp_key.weight.T))
            live = b['valid'].bool()
            # with more tokens than key dims the fit is least squares: the residual is orthogonal to the keys
            ang = m.rope_angles()
            tables = ang.cos(), ang.sin()
            lv = b['valid'].double()[:, None, :, None]
            q = m.rope(unit(heads(u @ L.rp_q.weight.T)), tables)
            k = m.rope(unit(heads(u @ L.rp_k.weight.T)), tables) * lv
            psi = torch.stack(tables, -1).reshape(*tables[0].shape[:-1], m.dh)
            v = torch.cat((heads(u @ L.rp_v.weight.T), psi), -1) * lv
            a = unit(q @ (k.transpose(-1, -2) @ v))
            gnorm = lambda RR: float(((a - kappa @ RR.transpose(-1, -2))[:, 0] * lv[:, 0]).transpose(-1, -2)
                                     .__matmul__(kappa[:, 0]).norm())
            self.assertLess(gnorm(R), 0.05 * gnorm(torch.zeros_like(R)))   # least-squares gradient mostly gone
            self.assertLess(float(R[:, 1:].abs().max()), 1e-12)   # heads that (almost) never write stay ~0

    def test_soft_threshold_prunes(self):
        torch.manual_seed(2)
        m = LT(cfg()).double()
        b = batch()
        m.cur_valid, m.cur_pos = b['valid'], (b['pos_row'], b['pos_col'])
        with torch.no_grad():
            m.block.rp_theta_raw.fill_(10.0)                      # theta ~ 10 >> any entry
            T = b['inputs'].shape[1]
            _, R = m.rule_path_step(torch.randn(2, T, m.d, dtype=torch.float64),
                                    torch.zeros(2, m.H, 2 * m.dh, m.dh, dtype=torch.float64))
        self.assertEqual(float(R.abs().max()), 0.0)


class GradBlocks(unittest.TestCase):
    def test_same_forward_last_block_gradient(self):
        b = batch()
        torch.manual_seed(3)
        full = LT(cfg()).double()
        torch.manual_seed(3)
        last = LT(cfg(grad_blocks=1)).double()
        for m in (full, last):
            with torch.no_grad():
                m.block.rp_out.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(5))
        _, l1 = run(full, b, 1)
        _, l2 = run(last, b, 1)
        torch.testing.assert_close(l2[0], l1[0], rtol=1e-12, atol=1e-12)
        l1[0].square().sum().backward()
        l2[0].square().sum().backward()
        g1, g2 = full.block.q_proj.weight.grad, last.block.q_proj.weight.grad
        self.assertGreater(float(g2.abs().sum()), 0.0)
        self.assertGreater(float((g1 - g2).abs().max()), 1e-9)   # earlier blocks no longer contribute


if __name__ == '__main__':
    unittest.main()


class TaskPrior(unittest.TestCase):
    def test_zero_map_is_the_plain_rule_path_and_the_prior_seeds_r(self):
        b = batch()
        b['puzzle_identifiers'] = torch.tensor([1, 2], dtype=torch.int32)
        torch.manual_seed(7)
        plain = LT(cfg(num_puzzle_identifiers=3, puzzle_emb_ndim=0)).double().eval()
        torch.manual_seed(7)
        prior = LT(cfg(num_puzzle_identifiers=3, puzzle_emb_ndim=0, rule_prior_emb=4)).double().eval()
        with torch.no_grad():
            for m in (plain, prior):
                m.block.rp_out.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(8))
            prior.block.rope_theta.copy_(plain.block.rope_theta)
            _, l1 = run(plain, b, 1)
            _, l2 = run(prior, b, 1)
            torch.testing.assert_close(l2[0], l1[0], rtol=1e-12, atol=1e-12)     # M = 0: R0 = 0
            prior.block.rp_prior_map.normal_(0, 0.5, generator=torch.Generator().manual_seed(9))
            prior.block.rp_theta_raw.fill_(-40.0)
            prior.block.rp_write.bias.fill_(-40.0)                                   # nothing written
            prior.block.rp_write.weight.zero_()
            carry, l3 = run(prior, b, 1)
        self.assertLess(float(carry.prev.abs().max()), 1e-9)                          # state = R - R0 stays 0
        self.assertGreater(float((l3[0] - l1[0]).abs().max()), 1e-6)                  # but the read sees R0


class TaskIds(unittest.TestCase):
    def test_ids_and_dropout(self):
        from lt.data import ArcContextDataset
        from tests.test_arc_ctx import _store
        rng = np.random.default_rng(0)
        grid = lambda: rng.integers(0, 10, (2, 2))
        store = _store([[(grid(), grid()) for _ in range(3)] for _ in range(8)], S=S, kmax=1)
        args = dict(seed=1, global_batch_size=4, rank=0, world_size=1, epochs_per_iter=50, start_iter=0, total_iters=1,
                    packed=True, bucket_window=4, task_ids=True)
        ids = torch.cat([b['puzzle_identifiers'] for _, b in ArcContextDataset(store, **dict(args, id_dropout=0.5))])
        self.assertTrue(((ids >= 0) & (ids <= 8)).all())
        self.assertTrue(0.35 < float((ids == 0).float().mean()) < 0.65)
        none = torch.cat([b['puzzle_identifiers'] for _, b in ArcContextDataset(store, **dict(args, id_dropout=0.0))])
        self.assertTrue((none >= 1).all())
