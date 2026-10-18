"""CPU/FP64 checks of the packed bands forward (lt/packed.py): with a crop that covers the whole dense grid the packed
segments reproduce the dense ones (logits, hidden, traces, W, loss, gradients); a lane's result does not depend on
the other lanes of the pack, on the chunk size or on the bucket padding. Run: python -m unittest discover tests"""
import unittest

import torch

from lt.model import LT
from lt.packed import PackedLT, build_pack, packed_loss, unpack
from lt.train import segment_loss
from tests.test_model import tiny

G = 4


def band_cfg(**kw):
    return tiny(**dict(dict(bands=2, band_periods=(3.0, 11.0), tag_bound=0.0, outer_eta=0.125, outer_eta_learn=True,
                            grid=G, seq_len=G * G, batch_size=3, num_puzzle_identifiers=5, puzzle_emb_mode='prefix',
                            vocab_size=13), **kw))


def make_model(**kw):
    torch.manual_seed(0)
    m = LT(band_cfg(**kw)).double()
    with torch.no_grad():
        m.puzzle_emb.weights.normal_()
        m.block.ffn_down.weight.normal_(0, 0.05)
        m.block.band_coef.normal_()
        m.block.band_pole.add_(torch.randn_like(m.block.band_pole) * 0.3)
    return m


def dense_batch(seed, B=3, T=G * G):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 13, (B, T), generator=g), labels=torch.randint(1, 13, (B, T), generator=g),
                puzzle_identifiers=torch.randint(0, 5, (B,), generator=g).int())


def full_boxes(B):
    return [(0, 0, G, G)] * B


class PackedEqualsDense(unittest.TestCase):
    def run_both(self, m, chunk, bucket, segs=2):
        b = dense_batch(1)
        B = len(b['inputs'])
        pack = build_pack(b, G, prefix_len=m.prefix_len, chunk=chunk, bucket=bucket, boxes=full_boxes(B))
        pm = PackedLT(m)
        dc, pc = m.initial_carry(b), None
        out = []
        for _ in range(segs):
            dc, dl = m(dc, b)
            pc, pl = pm(pc, pack)
            out.append((dc, dl, pc, pl))
        return b, pack, out

    def test_logits_and_state(self):
        m = make_model().eval()
        for chunk, bucket in ((1, 0), (8, 0), (8, 64)):
            b, pack, out = self.run_both(m, chunk, bucket)
            for dc, dl, pc, pl in out:
                torch.testing.assert_close(unpack(pl, pack), dl, rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(unpack(pc.hidden, pack), dc.hidden[:, m.prefix_len:], rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(pc.stdp_w, dc.stdp_w, rtol=1e-10, atol=1e-11)
                for name in ('key_trace', 'value_trace'):
                    dense = getattr(dc, name).transpose(1, 2)                  # [B, T, H, C]
                    torch.testing.assert_close(unpack(getattr(pc, name), pack), dense[:, m.prefix_len:],
                                               rtol=1e-10, atol=1e-11)
                pre = pack.prefix.nonzero().flatten()
                torch.testing.assert_close(pc.hidden[pre], dc.hidden[:, 0], rtol=1e-10, atol=1e-11)
                self.assertTrue(torch.equal(pc.halted, dc.halted))

    def test_loss_and_gradients(self):
        m = make_model(hebb=False, puzzle_emb_mode='add', puzzle_emb_ndim=12, act_ckpt=True).train()
        b = dense_batch(2)
        pack = build_pack(b, G, prefix_len=m.prefix_len, chunk=8, bucket=32, boxes=full_boxes(3))
        grads = []
        for packed in (False, True):
            m.zero_grad()
            if packed:
                carry, logits = PackedLT(m)(None, pack)
                loss, met = packed_loss(logits, pack, carry.halted)
            else:
                carry, logits = m(m.initial_carry(b), b)
                loss, met = segment_loss(logits, b['labels'], carry.halted, ignore0=True)
            loss.backward()
            grads.append((loss.detach(), {k: met[k].double() for k in met},
                          {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-11, atol=1e-12)
        for k in grads[0][1]:
            torch.testing.assert_close(grads[1][1][k], grads[0][1][k])
        self.assertEqual(set(grads[0][2]), set(grads[1][2]))
        for n in grads[0][2]:
            torch.testing.assert_close(grads[1][2][n], grads[0][2][n], rtol=1e-9, atol=1e-12, msg=n)


class LaneIndependence(unittest.TestCase):
    def test_lane_does_not_see_other_lanes(self):
        """Lane 0 (crop 3x2 at offset (1, 5) of a 30-wide canvas) gives the same logits and W packed alone or with
        two other lanes of different crops, for any chunk size and bucket."""
        m = make_model().eval()
        W = 30
        g = torch.Generator().manual_seed(3)
        b = dict(inputs=torch.randint(1, 13, (3, W * W), generator=g), labels=torch.randint(1, 13, (3, W * W), generator=g),
                 puzzle_identifiers=torch.tensor([1, 2, 3], dtype=torch.int32))
        boxes = [(1, 5, 3, 2), (0, 0, 7, 9), (20, 25, 10, 5)]
        alone = {k: v[:1] for k, v in b.items()}
        ref_pack = build_pack(alone, W, m.prefix_len, chunk=1, bucket=0, boxes=boxes[:1])
        pm = PackedLT(m)
        ref, c = [], None
        for _ in range(3):
            c, logits = pm(c, ref_pack)
            ref.append((unpack(logits, ref_pack)[0], c.stdp_w[0]))
        for chunk, bucket in ((4, 0), (8, 128), (16, 64)):
            other = dict(b, inputs=b['inputs'].clone())
            other['inputs'][1:] = torch.randint(1, 13, (2, W * W), generator=g)
            pack = build_pack(other, W, m.prefix_len, chunk=chunk, bucket=bucket, boxes=boxes)
            c = None
            for logits_ref, w_ref in ref:
                c, logits = pm(c, pack)
                lane0 = unpack(logits, pack)[0]
                cells = ref_pack.src[ref_pack.src >= 0]
                torch.testing.assert_close(lane0.reshape(-1, lane0.shape[-1])[cells],
                                           logits_ref.reshape(-1, lane0.shape[-1])[cells], rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(c.stdp_w[0], w_ref, rtol=1e-10, atol=1e-11)

    def test_crop_boxes(self):
        """The default crop is the bounding box of the grids and their EOS (token > 1 in input or label)."""
        from lt.data import _arc_canvas
        x = _arc_canvas(torch.ones(2, 3, dtype=torch.uint8).numpy() + 1, 2, 3, 4, 7)
        y = _arc_canvas(torch.ones(5, 1, dtype=torch.uint8).numpy() + 1, 5, 1, 4, 7)
        b = dict(inputs=torch.from_numpy(x.reshape(1, -1)).int() + 1, labels=torch.from_numpy(y.reshape(1, -1)).int() + 1,
                 puzzle_identifiers=torch.zeros(1, dtype=torch.int32))
        pack = build_pack(b, 30, 1, chunk=8, bucket=0)
        cells = pack.src[pack.src >= 0]
        rows, cols = cells // 30, cells % 30
        self.assertEqual((int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())), (4, 9, 7, 10))
        self.assertEqual(pack.num_tokens % 8, 0)


if __name__ == '__main__':
    unittest.main()
