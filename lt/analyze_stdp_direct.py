"""Direct readout of the STDP write of the general-window pairangle model.

Each block writes G = sum_n v_n kr_n^T (1 + alpha_h W(phiV_n,A - phiK_n,B)) and reads
read_t = G q_t. Channels are neurons (pre = K pair B, post = V pair A), the 81 tokens of a
block are its co-firing observations, the pair angles are spike timings, and the gain
1 + alpha W amplifies (delta in (0, pi): LTP) or suppresses (delta in (-pi, 0): LTD) the
Hebbian association of that observation. Every quantity is read from the forward pass of
the evaluated (EMA) weights; the recomputed write and read are checked against the
model's own at every block.

  timing      co-firing weighted phase differences, LTP/LTD mass, gain distribution
  synapses    per synapse order consistency R (token-permutation null), net polarity,
              G_STDP against G_Hebb, sharing across puzzles
  dynamics    the above per block over the 16 x 8 recurrence, order flips, G_STDP change
  sources     exact split of each cell's read into source tokens (self, peers, others)
  elimination first-order effect of each source on the final logits (last block only)

python -m lt.analyze_stdp_direct --run runs/<run> --checkpoint runs/<run>/step_20000.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import train as t
from .experiment_free_phase_windows import model_class

GROUPS = ('solved', 'unsolved')
CLASSES = ('self', 'peer_given', 'peer_empty', 'other_given', 'other_empty')
SHARED_SEGMENTS = (1, 4, 8, 16)
SCALARS = ('m', 'm_ltp', 'm_absw', 'm_gneg', 'm_glo', 'm_ghi', 'absz',
           'm_null', 'm_ltp_null', 'm_absw_null', 'absz_null', 'm_flip',
           'ratio', 'cos', 'pot', 'read_ratio', 'count', 'gs_change', 'change_count')
HISTS = dict(delta=(36, -math.pi, math.pi), gain=(30, -0.5, 2.5), R=(20, 0.0, 1.0), pol=(20, -1.0, 1.0))
CHUNK, SUB = 32, 8


def peer_mask():
    r, c = torch.arange(81) // 9, torch.arange(81) % 9
    box = (r // 3) * 3 + c // 3
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (box[:, None] == box[None])
    return same & ~torch.eye(81, dtype=torch.bool)


def make_model(protocol, cfg, device, state=None, seed=None):
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
                                protocol['generator'], protocol.get('feature_precision', 'float32'), 1.0,
                                protocol.get('tie_qk', False), protocol.get('tie_vo', False),
                                protocol.get('qk_l2', False), protocol.get('write_sum', False),
                                protocol.get('tau_phi', 2.0), protocol.get('phase_floor', 0.5),
                                protocol.get('v_norm', 'none'), protocol.get('tie_all', False),
                                protocol.get('phase_kappa', 1.0), protocol.get('phase_omega', 0.0),
                                protocol.get('phase_frame', 'rotated'), protocol.get('dc_hebbian', False),
                                protocol.get('dc_alpha_init', 0.0),
                               protocol.get('boundary_ffn', 'bilinear'),
                               protocol.get('kernel', 'torch'))
    if seed is not None:
        torch.manual_seed(seed)
    model_cfg = dict(cfg, batch_size=cfg['global_batch_size'], seq_len=81, num_puzzle_identifiers=1)
    with torch.device(device):
        base = t.ACTLossHead(t.LT(model_cfg), q_weight=cfg['q_weight'])
    if state is not None:
        base.load_state_dict(state, strict=True)
    base.eval()
    return base


@torch.no_grad()
def predict(base, x, y, cfg, device):
    """Per-segment predictions, and the trainer's own end-of-horizon metrics."""
    preds = torch.empty(cfg['loops'], len(x), 81, dtype=torch.uint8)
    totals, start = dict(count=0.0, accuracy=0.0, exact=0.0), 0
    for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        carry, n = base.initial_carry(batch), batch['inputs'].shape[0]
        for s in range(cfg['loops']):
            carry, _, metrics, out, _ = base(carry=carry, batch=batch, return_keys={'preds'})
            preds[s, start:start + n] = out['preds'].to(torch.uint8).cpu()
        totals['count'] += float(metrics['count'])
        totals['accuracy'] += float(metrics['accuracy'])
        totals['exact'] += float(metrics['exact_accuracy'])
        start += n
    return preds, totals


class Probe:
    def __init__(self, inner, device, full):
        self.inner, self.full, self.device = inner, full, device
        H, D = inner.H, inner.dh
        self.H, self.D, self.P, self.G = H, D, D // 2, len(GROUPS)
        segments, blocks = inner.config.loops, inner.config.blocks_per_seg
        self.blocks_per_seg = blocks
        f64 = dict(dtype=torch.float64, device=device)
        self.sc = {n: torch.zeros(self.G, segments * blocks, H, **f64) for n in SCALARS}
        self.hist = {n: torch.zeros(self.G, segments, H, nb, **f64) for n, (nb, _, _) in HISTS.items()}
        J = len(SHARED_SEGMENTS)
        self.shared = dict(GS=torch.zeros(self.G, J, H, D, D, **f64), GH=torch.zeros(self.G, J, H, D, D, **f64),
                           pol=torch.zeros(self.G, J, H, self.P, self.P, **f64),
                           GS_sq=torch.zeros(self.G, J, H, **f64), GH_sq=torch.zeros(self.G, J, H, **f64),
                           pol_sq=torch.zeros(self.G, J, H, **f64), cnt=torch.zeros(self.G, J, **f64))
        self.src = dict(share=torch.zeros(self.G, segments, 2, len(CLASSES), **f64),
                        frac=torch.zeros(self.G, segments, len(CLASSES), **f64), cnt=torch.zeros(self.G, segments, **f64))
        self.elim = dict(e=torch.zeros(self.G, 2, 4, **f64), e_cnt=torch.zeros(self.G, 4, **f64),
                         press=torch.zeros(self.G, 2, 2, **f64), sup=torch.zeros(self.G, 2, len(CLASSES), **f64),
                         tgt=torch.zeros(self.G, **f64), agree=torch.zeros(2, **f64))
        self.checks = dict(current=0.0, read=0.0, readS=0.0)
        self.peer = peer_mask().to(device)
        gen = torch.Generator().manual_seed(0)
        while True:
            perm = torch.randperm(81, generator=gen)
            if not (perm == torch.arange(81)).any():
                break
        self.perm = perm.to(device)
        self.wpairs = list(zip(inner.window_frequencies.tolist(), inner.window_coefficients.tolist()))

    # ---------------------------------------------------------------- wiring
    def install(self):
        inner = self.inner
        self.orig_block, self.orig_memory = inner.block, inner.memory_step

        def block(L, h, inj, *args):
            self.h_pre = h + inner.embed_scale * inj
            return self.orig_block(L, h, inj, *args)

        def memory_step(L, q, k, v, *args, **kwargs):
            out = self.orig_memory(L, q, k, v, *args, **kwargs)
            self.on_block(L, q, k, v, out[0], out[1])
            return out
        inner.block, inner.memory_step = block, memory_step

    def remove(self):
        del self.inner.block, self.inner.memory_step

    def begin_batch(self, groups, inputs, labels, pred16):
        self.groups, self.inputs, self.labels, self.pred16 = groups, inputs, labels, pred16
        self.prev = {}

    def begin_segment(self, seg):
        self.seg, self.k = seg, 0

    # ---------------------------------------------------------------- helpers
    def window(self, delta):
        return sum(c * torch.sin(f * delta) for f, c in self.wpairs)

    def check(self, name, value, ref):
        self.checks[name] = max(self.checks[name], float((value - ref).norm() / ref.norm()))

    def add(self, name, blk, g, value):
        self.sc[name][:, blk].index_add_(0, g, value.double())

    def hist_add(self, name, values, weights, g):
        nb, lo, hi = HISTS[name]
        b, H = values.shape[:2]
        tail = [1] * (values.dim() - 2)
        idx = ((values - lo) * (nb / (hi - lo))).floor().clamp_(0, nb - 1).long()
        key = (g.view(b, 1, *tail) * H + torch.arange(H, device=values.device).view(1, H, *tail)) * nb + idx
        counts = torch.bincount(key.flatten(), weights=weights.flatten().double(), minlength=self.G * H * nb)
        self.hist[name][:, self.seg - 1] += counts.view(self.G, H, nb)

    def class_masks(self, given):
        b, T = given.shape
        eye = torch.eye(T, dtype=torch.bool, device=given.device)
        gn, other = given[:, None, :], ~self.peer & ~eye
        return torch.stack([eye.expand(b, T, T), self.peer & gn, self.peer & ~gn, other & gn, other & ~gn], 1)

    # ---------------------------------------------------------------- per block
    def on_block(self, L, q, k, v, read, current):
        blk = (self.seg - 1) * self.blocks_per_seg + self.k
        last = self.k == self.blocks_per_seg - 1
        final = last and self.seg == self.inner.config.loops
        with torch.autocast('cuda', enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            J = self.jacobian(L, read) if (final and self.full) else None
            for s0 in range(0, q.shape[0], CHUNK):
                sl = slice(s0, s0 + CHUNK)
                self.chunk(L, q[sl], k[sl], v[sl], read[sl], current[sl], s0, blk, last, J)
        self.k += 1

    def jacobian(self, L, read):
        """d logits_t / d read_t at the last block; the map after the read is token-local."""
        inner = self.inner
        B, H, T, D = read.shape
        with torch.enable_grad():
            r = read.detach().transpose(1, 2).reshape(B, T, H * D).float().requires_grad_(True)
            out = inner.w_cls(inner.phi(inner.boundary(L, self.h_pre.detach().float() + L.out_proj(r))))
            pred = self.pred16.to(out.device)
            self.elim['agree'] += torch.tensor([float((out.argmax(-1) == pred).sum()), float(pred.numel())],
                                               dtype=torch.float64, device=out.device)
            rows = [torch.autograd.grad(out[..., d].sum(), r, retain_graph=d < out.shape[-1] - 1)[0]
                    for d in range(out.shape[-1])]
        return torch.stack(rows, 2)                                    # [B, T, vocab, H*D]

    def chunk(self, L, q, k, v, read_m, cur_m, s0, blk, last, J):
        inner = self.inner
        b, H, T, D = v.shape
        P = D // 2
        g = self.groups[s0:s0 + b]
        eps = inner.config.eps
        qU = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        kU = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        tables = inner.rope_tables(L)
        qr, kr = inner.apply_rope(qU, L, tables), inner.apply_rope(kU, L, tables)
        vp, kp, qp, krp = (x.reshape(b, H, T, P, 2) for x in (v, kU, qr, kr))
        vmag, kmag = vp.norm(dim=-1), kp.norm(dim=-1)
        phiV, phiK = torch.atan2(vp[..., 1], vp[..., 0]), torch.atan2(kp[..., 1], kp[..., 0])
        alpha = L.stdp_alpha.detach().float()
        a4, a5 = alpha.view(1, H, 1, 1), alpha.view(1, H, 1, 1, 1)
        delta = phiV[..., :, None] - phiK[..., None, :]                # [b,H,T(n),A,B]
        W = self.window(delta)
        m = vmag[..., :, None] * kmag[..., None, :]                    # co-firing weight |v_A||k_B|
        GH = torch.einsum('bhna,bhnc->bhac', v, kr)
        GS = torch.einsum('bhnAi,bhnAB,bhnBj->bhAiBj', vp, W, krp).reshape(b, H, D, D)
        readH, readS = qr @ GH.transpose(-1, -2), qr @ GS.transpose(-1, -2)
        self.check('current', GH + a4 * GS, cur_m)
        self.check('read', readH + a4 * readS, read_m)

        # timing, LTP/LTD and gain
        ltp = torch.sin(delta) > 0                                     # sign W = sign sin delta (c1 > 2 c2)
        gain = 1 + a5 * W
        red = lambda x: x.sum(dim=(2, 3, 4))
        self.add('m', blk, g, red(m))
        self.add('m_ltp', blk, g, red(m * ltp))
        self.add('m_absw', blk, g, red(m * (a5 * W).abs()))
        self.add('m_gneg', blk, g, red(m * (gain < 0)))
        self.add('m_glo', blk, g, red(m * (gain < 0.5)))
        self.add('m_ghi', blk, g, red(m * (gain > 1.5)))
        self.hist_add('delta', torch.remainder(delta + math.pi, 2 * math.pi) - math.pi, m, g)
        self.hist_add('gain', gain, m, g)

        # per synapse: order consistency R = |sum_n m e^{i delta}| / sum_n m and net polarity
        vx, vy, kx, ky = vp[..., 0], vp[..., 1], kp[..., 0], kp[..., 1]
        e2 = lambda x, y: torch.einsum('bhnA,bhnB->bhAB', x, y)
        den = e2(vmag, kmag)
        absz = torch.hypot(e2(vx, kx) + e2(vy, ky), e2(vy, kx) - e2(vx, ky))
        pol = (m * W).sum(2) / den.clamp_min(1e-12)
        self.add('absz', blk, g, absz.sum((2, 3)))
        self.hist_add('R', absz / den.clamp_min(1e-12), den, g)
        self.hist_add('pol', pol, den, g)
        # null: V phases from another token of the same puzzle (co-firing broken, marginals kept)
        pv = self.perm
        vxn, vyn, vmn = vx[:, :, pv], vy[:, :, pv], vmag[:, :, pv]
        mn = vmn[..., :, None] * kmag[..., None, :]
        dn = phiV[:, :, pv][..., :, None] - phiK[..., None, :]
        self.add('m_null', blk, g, red(mn))
        self.add('m_ltp_null', blk, g, red(mn * (torch.sin(dn) > 0)))
        self.add('m_absw_null', blk, g, red(mn * (a5 * self.window(dn)).abs()))
        self.add('absz_null', blk, g, torch.hypot(e2(vxn, kx) + e2(vyn, ky), e2(vyn, kx) - e2(vxn, ky)).sum((2, 3)))
        del mn, dn

        # synaptic matrices
        nH, nS = GH.norm(dim=(-2, -1)), GS.norm(dim=(-2, -1))
        dot = (GH * GS).sum((-2, -1))
        ones = torch.ones_like(nH)
        self.add('ratio', blk, g, alpha * nS / nH)
        self.add('cos', blk, g, dot / (nS * nH))
        self.add('pot', blk, g, alpha * dot / nH ** 2)
        self.add('read_ratio', blk, g, alpha * readS.norm(dim=(-2, -1)) / readH.norm(dim=(-2, -1)))
        self.add('count', blk, g, ones)
        prev = self.prev.get(s0)
        if prev is not None:
            self.add('m_flip', blk, g, red(m * (ltp != prev['ltp'])))
            self.add('gs_change', blk, g, (GS - prev['GS']).norm(dim=(-2, -1)) / nS)
            self.add('change_count', blk, g, ones)
        self.prev[s0] = dict(ltp=ltp, GS=GS)

        if last and self.seg in SHARED_SEGMENTS:
            j = SHARED_SEGMENTS.index(self.seg)
            sh = self.shared
            sh['GS'][:, j].index_add_(0, g, GS.double())
            sh['GH'][:, j].index_add_(0, g, GH.double())
            sh['pol'][:, j].index_add_(0, g, pol.double())
            sh['GS_sq'][:, j].index_add_(0, g, nS.double() ** 2)
            sh['GH_sq'][:, j].index_add_(0, g, nH.double() ** 2)
            sh['pol_sq'][:, j].index_add_(0, g, pol.double().pow(2).sum((2, 3)))
            sh['cnt'][:, j].index_add_(0, g, torch.ones(b, dtype=torch.float64, device=g.device))
        if not (self.full and last):
            return

        # exact source split of the read: read_t = sum_n (r^H_tn + r^S_tn)
        given = self.inputs[s0:s0 + b] > 1
        s_tn = torch.einsum('bhta,bhna->bhtn', qr, kr)
        spair = torch.einsum('bhtBj,bhnBj->bhtnB', qp, krp)            # per-pair q.k
        u = torch.einsum('bhnAB,bhtnB->bhtnA', W, spair)               # r^S_tn = alpha v_n * u_tn
        readSp = readS.reshape(b, H, T, P, 2)
        self.check('readS', torch.einsum('bhnAi,bhtnA->bhtAi', vp, u), readSp)
        dH = (s_tn * torch.einsum('bhna,bhta->bhtn', v, readH)).sum(1)
        dS = (a4 ** 2 * (u * torch.einsum('bhnAi,bhtAi->bhtnA', vp, readSp)).sum(-1)).sum(1)
        NH, NS = readH.pow(2).sum((1, 3)), (a4 ** 2 * readS.pow(2)).sum((1, 3))
        masks = self.class_masks(given)
        empty = (~given).double()
        for branch, (Dx, Nx) in enumerate(((dH, NH), (dS, NS))):
            share = (Dx[:, None] * masks).sum(-1) / Nx[:, None]       # [b,5,T]
            self.src['share'][:, self.seg - 1, branch].index_add_(0, g, (share.double() * empty[:, None]).sum(-1))
        self.src['frac'][:, self.seg - 1].index_add_(0, g, (masks.sum(-1).double() / T * empty[:, None]).sum(-1))
        self.src['cnt'][:, self.seg - 1].index_add_(0, g, empty.sum(-1))
        if J is None:
            return
        for e0 in range(0, b, SUB):
            es = slice(e0, e0 + SUB)
            bb = v[es].shape[0]
            rH = s_tn[es, ..., None] * v[es, :, None]                  # [bb,H,T,T,D]
            rS = (a5[..., None] * vp[es, :, None] * u[es, ..., None]).reshape(bb, H, T, T, D)
            Jc = J[s0 + e0:s0 + e0 + bb].reshape(bb, T, -1, H, D)
            for branch, r in enumerate((rH, rS)):
                self.eliminate(torch.einsum('btdha,bhtna->btnd', Jc, r), branch, s0 + e0, bb)

    def eliminate(self, c, branch, off, bb):
        """c[b,t,n,d]: first-order change of logit d of cell t carried by source n."""
        inputs, labels = self.inputs[off:off + bb], self.labels[off:off + bb]
        pred, g = self.pred16[off:off + bb], self.groups[off:off + bb]
        T = inputs.shape[1]
        given = inputs > 1
        empty = ~given
        ref = c[..., 2:11].mean(-1)                                    # digits are tokens 2..10
        src = torch.where(given, inputs, pred)
        valid = (src >= 2) & (src <= 10)
        e = c.gather(-1, src.clamp(0, c.shape[-1] - 1)[:, None, :, None].expand(bb, T, T, 1)).squeeze(-1) - ref
        sup = c.gather(-1, labels[:, :, None, None].expand(bb, T, T, 1)).squeeze(-1) - ref
        masks = self.class_masks(given)
        tgt = empty[:, None, :, None]
        em = masks[:, 1:] & tgt & valid[:, None, None, :]
        self.elim['e'][:, branch].index_add_(0, g, (e[:, None] * em).sum((2, 3)).double())
        press = torch.stack([(e * em[:, 0]).sum(-1), (e * em[:, 1]).sum(-1)], 1)
        self.elim['press'][:, branch].index_add_(0, g, (press * empty[:, None]).sum(-1).double())
        self.elim['sup'][:, branch].index_add_(0, g, (sup[:, None] * (masks & tgt)).sum((2, 3)).double())
        if branch == 0:
            self.elim['e_cnt'].index_add_(0, g, em.sum((2, 3)).double())
            self.elim['tgt'].index_add_(0, g, empty.sum(-1).double())


@torch.no_grad()
def run_probe(base, probe, x, y, cfg, device, groups, inputs, labels, preds):
    probe.install()
    mismatch, start, began = 0, 0, time.monotonic()
    try:
        for i, batch in enumerate(t.eval_batches(x, y, cfg['global_batch_size'], 0, 1)):
            batch = {k: v.to(device) for k, v in batch.items()}
            n = batch['inputs'].shape[0]
            sl = slice(start, start + n)
            probe.begin_batch(groups[sl].to(device), inputs[sl].to(device), labels[sl].to(device),
                              preds[-1, sl].long().to(device))
            carry = base.initial_carry(batch)
            for s in range(cfg['loops']):
                probe.begin_segment(s + 1)
                carry, _, _, out, _ = base(carry=carry, batch=batch, return_keys={'preds'})
                mismatch += int((out['preds'].cpu().to(torch.uint8) != preds[s, sl]).sum())
            start += n
            print(f'PROBE batch {i + 1} puzzles {start}/{len(x)} {time.monotonic() - began:.0f}s', flush=True)
    finally:
        probe.remove()
    return mismatch


def tolist(x):
    return x.detach().cpu().tolist()


def summarizer(probe):
    """agg(group, segment[, head]): ratios of the accumulated sums over one segment's blocks."""
    sc = probe.sc
    nb = probe.blocks_per_seg

    def seg_slice(s):
        return slice((s - 1) * nb, s * nb)

    def agg(g, s, head=None):
        sl = seg_slice(s)
        pick = (lambda x: x[g, sl].sum(0)) if head is None else (lambda x: x[g, sl, head].sum())
        tot = lambda name: pick(sc[name]).sum() if head is None else pick(sc[name])
        m, mn, cnt = tot('m'), tot('m_null'), tot('count')
        out = dict(ltp=tot('m_ltp') / m, abs_alphaW=tot('m_absw') / m, gain_neg=tot('m_gneg') / m,
                   gain_lt_05=tot('m_glo') / m, gain_gt_15=tot('m_ghi') / m, R=tot('absz') / m,
                   R_null=tot('absz_null') / mn, ltp_null=tot('m_ltp_null') / mn,
                   abs_alphaW_null=tot('m_absw_null') / mn, flip=tot('m_flip') / m,
                   ratio=tot('ratio') / cnt, read_ratio=tot('read_ratio') / cnt, cos=tot('cos') / cnt,
                   pot=tot('pot') / cnt, gs_change=tot('gs_change') / tot('change_count').clamp_min(1))
        return {k: float(v) for k, v in out.items()}
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N held-out puzzles (0: all)')
    ap.add_argument('--init-n', type=int, default=256, help='puzzles for the seed-0 initialisation baseline (0: skip)')
    ap.add_argument('--out', type=Path, default=None)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    _, _, x, y, _, fingerprint = t.load_data(cfg)
    assert fingerprint == cfg['data_fingerprint'], 'Evaluation data changed'
    if opt.limit:
        x, y = x[:opt.limit], y[:opt.limit]
    step = int(ck['step'])
    out_path = opt.out or run / 'diagnostics' / f'stdp_direct_step{step}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = make_model(protocol, cfg, device, ck['model_state_dict'])
    began = time.monotonic()
    preds, official = predict(base, x, y, cfg, device)
    inputs = torch.from_numpy(x.reshape(-1, 81).astype(np.int64) + 1)
    labels = torch.from_numpy(y.reshape(-1, 81).astype(np.int64) + 1)
    solved = (preds[-1].long() == labels).all(-1)
    groups = (~solved).long()
    empty = inputs == 1
    correct = torch.stack([((preds[s].long() == labels) & empty).sum(-1).double() / empty.sum(-1)
                           for s in range(cfg['loops'])])              # [S, N]
    print(f"OFFICIAL count {official['count']:.0f} acc {official['accuracy'] / official['count']:.4f} "
          f"exact {official['exact']:.0f}  ({time.monotonic() - began:.0f}s)", flush=True)

    probe = Probe(base.model.inner, device, full=True)
    mismatch = run_probe(base, probe, x, y, cfg, device, groups, inputs, labels, preds)
    alpha = tolist(base.model.inner.layers[0].stdp_alpha)
    result = dict(checkpoint=str(opt.checkpoint), step=step, puzzles=len(x), solved=int(solved.sum()),
                  official=official, alpha=alpha, pass2_pred_mismatch=mismatch, checks=probe.checks,
                  readout_agreement=float(probe.elim['agree'][0] / probe.elim['agree'][1]),
                  empty_correct={GROUPS[gi]: tolist(correct[:, groups == gi].mean(1)) for gi in range(2)})
    agg = summarizer(probe)
    result['summary'] = {GROUPS[gi]: {s: agg(gi, s) for s in range(1, cfg['loops'] + 1)} for gi in range(2)
                         if int((groups == gi).sum())}
    result['per_head_seg16'] = {GROUPS[gi]: [agg(gi, cfg['loops'], h) for h in range(probe.H)] for gi in range(2)
                                if int((groups == gi).sum())}
    sh = probe.shared
    frac = lambda S, sq, cnt: (((S / cnt[..., None, None, None]) ** 2).sum((-2, -1)) / (sq / cnt[..., None]))
    result['shared_fraction'] = dict(segments=SHARED_SEGMENTS,
                                     GS=tolist(frac(sh['GS'], sh['GS_sq'], sh['cnt'])),
                                     GH=tolist(frac(sh['GH'], sh['GH_sq'], sh['cnt'])),
                                     pol=tolist(frac(sh['pol'], sh['pol_sq'], sh['cnt'])))
    result['mean_polarity_seg16_solved'] = tolist(sh['pol'][0, -1] / sh['cnt'][0, -1].clamp_min(1))
    src = probe.src
    result['sources'] = dict(classes=CLASSES,
                             share=tolist(src['share'] / src['cnt'][..., None, None].clamp_min(1)),
                             frac=tolist(src['frac'] / src['cnt'][..., None].clamp_min(1)))
    el = probe.elim
    result['elimination'] = dict(classes=CLASSES[1:],
                                 e_per_pair=tolist(el['e'] / el['e_cnt'][:, None].clamp_min(1)),
                                 peer_pressure_per_target=tolist(el['press'] / el['tgt'][:, None, None].clamp_min(1)),
                                 support_per_target=tolist(el['sup'] / el['tgt'][:, None, None].clamp_min(1)),
                                 pairs=tolist(el['e_cnt']), targets=tolist(el['tgt']))
    result['hist'] = {k: dict(edges=list(HISTS[k][1:]), bins=HISTS[k][0], values=tolist(v)) for k, v in probe.hist.items()}
    del base, probe
    torch.cuda.empty_cache()

    if opt.init_n:
        n0 = min(opt.init_n, len(x))
        init = make_model(protocol, cfg, device, None, seed=cfg['seed'])
        p0, _ = predict(init, x[:n0], y[:n0], cfg, device)
        probe0 = Probe(init.model.inner, device, full=False)
        run_probe(init, probe0, x[:n0], y[:n0], cfg, device, torch.zeros(n0, dtype=torch.long),
                  inputs[:n0], labels[:n0], p0)
        agg0 = summarizer(probe0)
        result['init'] = dict(puzzles=n0, alpha=tolist(init.model.inner.layers[0].stdp_alpha),
                              summary={s: agg0(0, s) for s in (1, 16)},
                              shared_fraction=dict(GS=tolist(frac(probe0.shared['GS'], probe0.shared['GS_sq'], probe0.shared['cnt'])[0]),
                                                   GH=tolist(frac(probe0.shared['GH'], probe0.shared['GH_sq'], probe0.shared['cnt'])[0])))
    out_path.write_text(json.dumps(result, indent=1, allow_nan=True) + '\n')
    print(f'WROTE {out_path} ({time.monotonic() - began:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
