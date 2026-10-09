"""Which cell-to-cell couplings does STDP strengthen or weaken, at which lags, when, and does
the puzzle logic explain it? (general-window pairangle model)

Through one head, target cell t reads source cell n with a Hebbian message r^H_tn = s_tn v_n
(s_tn = q_t . k_n) and an STDP message r^S_tn = alpha v_n * u_tn, u_tnA = sum_B W(delta_nAB)
(q_t . k_n)_B. Their ratio along the Hebbian message, summed over heads,

    kappa_tn = 1 + sum_h alpha_h s_tn sum_A u_tnA |v_nA|^2 / sum_h s_tn^2 |v_n|^2,

is the STDP gain W(delta_nAB) of the source's spike pairs averaged with the weights
s_tn (q_t . k_n)_B |v_nA|^2 of the synapses (A <- B) that carry this message:
> 1 strengthened, < 1 weakened, < 0 reversed. The weight energy |r^H_tn|^2 weights averages.
Logit effect of a message (first order through the block's read, logit readout of the block):
c_tn[d]; elimination of the source digit e_n at t: c[e_n] - mean_d c[d]; support of t's answer:
c[y_t] - mean_d c[d]. States are those entering the block (output of the previous block):
given; committed (correct from here to the end); open-correct; open-wrong.
Phases per puzzle: seg1 (blocks 1-8), solving (until the block from which the whole grid stays
correct), solved. Lags: source spike-pair lags weighted by their weight in peer messages.

python -m lt.stdp_message_flow --run runs/<run> --checkpoint runs/<run>/step_20000.pt --case 30
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
from .analyze_stdp_direct import make_model, peer_mask

STATUS = ('given', 'committed', 'open_correct', 'open_wrong')
PHASES = ('seg1', 'solving', 'solved')
RELS = ('peer', 'other')
TARGETS = ('open', 'committed')
OFF = 16
NBINS = 36
CH = 4
SUMS = ('energy', 'ek', 'estr', 'pairs', 'eH', 'eS', 'evalid', 'sH', 'sS')


@torch.no_grad()
def trajectory(base, batch, cfg):
    """argmax w_cls(h) after every block, for the whole batch: [blocks, B, 81]."""
    inner = base.model.inner
    orig, preds = inner.block, []

    def block(L, h, inj, *args):
        out = orig(L, h, inj, *args)
        with torch.autocast('cuda', enabled=False):
            preds.append(inner.w_cls(out[0].float()).argmax(-1))
        return out
    inner.block = block
    try:
        carry = base.initial_carry(batch)
        for _ in range(cfg['loops']):
            carry, *_ = base(carry=carry, batch=batch, return_keys=set())
    finally:
        del inner.block
    return torch.stack(preds)


def stays(ok):
    """ok [NB, ...] -> first block index from which ok holds to the end (NB if never)."""
    stay = ok.int().flip(0).cumprod(0).flip(0).bool()
    return torch.where(stay.any(0), stay.int().argmax(0), torch.full_like(stay[0], ok.shape[0], dtype=torch.long))


class FlowProbe:
    def __init__(self, inner, device, nblocks):
        self.inner, self.device, self.NB = inner, device, nblocks
        H, P = inner.H, inner.dh // 2
        f64 = dict(dtype=torch.float64, device=device)
        self.acc = {n: torch.zeros(len(PHASES), len(RELS), len(STATUS), len(TARGETS), **f64) for n in SUMS}
        self.src = {n: torch.zeros(2 * OFF + 1, **f64) for n in ('energy', 'ek', 'eS', 'eH', 'evalid')}
        self.tgt = {n: torch.zeros(2 * OFF + 1, **f64) for n in ('supS', 'supH', 'setS', 'setH', 'opnS', 'opnH', 'n')}
        self.lag = torch.zeros(len(PHASES), len(STATUS), NBINS, **f64)
        self.lag_sw = torch.zeros(len(PHASES), len(STATUS), 2, **f64)          # strengthen / weaken mass
        self.syn_w = torch.zeros(len(PHASES), len(STATUS), H, P, P, **f64)      # message weight per synapse
        self.syn_wW = torch.zeros(len(PHASES), len(STATUS), H, P, P, **f64)     # ... times alpha W
        self.check = 0.0
        self.peer = peer_mask().to(device)
        self.wpairs = list(zip(inner.window_frequencies.tolist(), inner.window_coefficients.tolist()))
        self.case = None

    def install(self):
        inner = self.inner
        self.orig_block, self.orig_memory = inner.block, inner.memory_step

        def block(L, h, inj, *args):
            self.h_pre = h + inner.embed_scale * inj
            return self.orig_block(L, h, inj, *args)

        def memory_step(L, q, k, v, *args, **kwargs):
            out = self.orig_memory(L, q, k, v, *args, **kwargs)
            if self.b > 0 and len(self.sel):
                with torch.autocast('cuda', enabled=False):
                    self.on_block(L, q, k, v, out[0])
            self.b += 1
            return out
        inner.block, inner.memory_step = block, memory_step

    def remove(self):
        del self.inner.block, self.inner.memory_step

    def begin_batch(self, sel, given, inputs, labels, preds, commit, solve, case_pos):
        self.sel, self.given, self.inputs, self.labels = sel, given, inputs, labels
        self.preds, self.commit, self.solve, self.case_pos, self.b = preds, commit, solve, case_pos, 0

    def jacobian(self, L, read, h_pre):
        inner = self.inner
        S, H, T, D = read.shape
        with torch.enable_grad():
            r = read.transpose(1, 2).reshape(S, T, H * D).requires_grad_(True)
            out = inner.w_cls(inner.phi(inner.boundary(L, h_pre + L.out_proj(r))))
            rows = [torch.autograd.grad(out[..., d].sum(), r, retain_graph=d < out.shape[-1] - 1)[0]
                    for d in range(out.shape[-1])]
        return torch.stack(rows, 2)

    def on_block(self, L, q, k, v, read):
        sel = self.sel
        q, k, v, read = (x[sel].float() for x in (q, k, v, read))
        J = self.jacobian(L, read, self.h_pre[sel].float())
        for c0 in range(0, len(sel), CH):
            self.chunk(L, q[c0:c0 + CH], k[c0:c0 + CH], v[c0:c0 + CH], read[c0:c0 + CH], J[c0:c0 + CH], c0)

    def chunk(self, L, q, k, v, read, J, c0):
        inner, b = self.inner, self.b
        c, H, T, D = v.shape
        P = D // 2
        eps = inner.config.eps
        qU = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        kU = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        tables = inner.rope_tables(L)
        qr, kr = inner.apply_rope(qU, L, tables), inner.apply_rope(kU, L, tables)
        vp, kp, qp, krp = (x.reshape(c, H, T, P, 2) for x in (v, kU, qr, kr))
        delta = torch.atan2(vp[..., 1], vp[..., 0])[..., :, None] - torch.atan2(kp[..., 1], kp[..., 0])[..., None, :]
        W = sum(cf * torch.sin(f * delta) for f, cf in self.wpairs)            # [c,H,n,A,B]
        alpha = L.stdp_alpha.detach().float()
        s = torch.einsum('bhta,bhna->bhtn', qr, kr)
        spair = torch.einsum('bhtBj,bhnBj->bhtnB', qp, krp)
        u = torch.einsum('bhnAB,bhtnB->bhtnA', W, spair)
        vA2 = vp.pow(2).sum(-1)
        num = (alpha.view(1, H, 1, 1) * s * torch.einsum('bhtnA,bhnA->bhtn', u, vA2)).sum(1)
        energy = (s.pow(2) * v.pow(2).sum(-1)[:, :, None, :]).sum(1)              # sum_h |r^H_tn|^2
        kappa = 1 + num / energy.clamp_min(1e-30)
        rS = (alpha.view(1, H, 1, 1, 1, 1) * vp[:, :, None] * u[..., None]).reshape(c, H, T, T, D)
        recon = torch.einsum('bhtn,bhna->bhta', s, v) + rS.sum(3)
        self.check = max(self.check, float((recon - read).norm() / read.norm()))
        Jc = J.reshape(c, T, -1, H, D)
        cH = torch.einsum('bhtn,bhtdn->btnd', s, torch.einsum('btdha,bhna->bhtdn', Jc, v))
        cS = torch.einsum('bhtna,btdha->btnd', rS, Jc)
        del rS

        sl = slice(c0, c0 + c)
        given, inp, lab = self.given[sl], self.inputs[sl], self.labels[sl]
        commit, solve, prev = self.commit[sl], self.solve[sl], self.preds[b - 1, sl]
        empty = ~given
        committed = empty & (commit <= b - 1)
        correct = prev == lab
        src = torch.stack([given, committed, empty & ~committed & correct, empty & ~committed & ~correct], 1)
        tgt = torch.stack([empty & ~committed, committed], 1)
        digit = torch.where(given, inp, prev)
        valid = (digit >= 2) & (digit <= 10)
        ref = lambda cx: cx[..., 2:11].mean(-1)
        pick = lambda cx, idx: cx.gather(-1, idx.expand(c, T, T, 1)).squeeze(-1)
        E = {x: pick(cx, digit[:, None, :, None]) - ref(cx) for x, cx in (('H', cH), ('S', cS))}
        Sup = {x: pick(cx, lab[:, :, None, None]) - ref(cx) for x, cx in (('H', cH), ('S', cS))}
        phase = torch.where(torch.full_like(solve, b) < 8, 0, torch.where(b < solve, 1, 2))
        eye = torch.eye(T, dtype=torch.bool, device=v.device)
        for r, R in enumerate((self.peer, ~self.peer & ~eye)):
            for si in range(len(STATUS)):
                for ti in range(len(TARGETS)):
                    m = R[None] & tgt[:, ti, :, None] & src[:, si, None, :]
                    mv = m & valid[:, None, :]
                    vals = dict(energy=(energy * m).sum((1, 2)), ek=(energy * kappa * m).sum((1, 2)),
                                estr=(energy * (kappa > 1) * m).sum((1, 2)), pairs=m.sum((1, 2)),
                                eH=(E['H'] * mv).sum((1, 2)), eS=(E['S'] * mv).sum((1, 2)), evalid=mv.sum((1, 2)),
                                sH=(Sup['H'] * m).sum((1, 2)), sS=(Sup['S'] * m).sum((1, 2)))
                    for name, val in vals.items():
                        self.acc[name][:, r, si, ti].index_add_(0, phase, val.double())

        # aligned to the block where a source becomes committed (first block reading a committed state)
        tm = self.peer[None] & empty[:, :, None]                                  # empty peer targets
        o = b - (commit + 1)
        ev = empty & (commit >= 8) & (commit < self.NB) & (o.abs() <= OFF)
        if ev.any():
            idx = (o + OFF)[ev]
            per = dict(energy=(energy * tm).sum(1), ek=(energy * kappa * tm).sum(1),
                       eS=(E['S'] * tm).sum(1) * valid, eH=(E['H'] * tm).sum(1) * valid,
                       evalid=tm.sum(1) * valid)
            for name, val in per.items():
                self.src[name].index_add_(0, idx, val[ev].double())
        # aligned to the block where a target becomes committed
        settled = (given | committed)[:, None, :] & self.peer[None] & valid[:, None, :]
        opened = (empty & ~committed)[:, None, :] & self.peer[None] & valid[:, None, :]
        evt = empty & (commit >= 8) & (commit < self.NB) & (o.abs() <= OFF)
        if evt.any():
            idx = (o + OFF)[evt]
            per = dict(supS=(Sup['S'] * self.peer).sum(2), supH=(Sup['H'] * self.peer).sum(2),
                       setS=(E['S'] * settled).sum(2), setH=(E['H'] * settled).sum(2),
                       opnS=(E['S'] * opened).sum(2), opnH=(E['H'] * opened).sum(2), n=torch.ones_like(o, dtype=torch.float))
            for name, val in per.items():
                self.tgt[name].index_add_(0, idx, val[evt].double())

        # lags carried by peer messages, per source state
        sp = s[..., None] * spair                                                  # [c,H,t,n,B]
        tmh = tm[:, None, :, :, None]
        Mp, Mn = (sp.clamp_min(0) * tmh).sum(2), ((-sp).clamp_min(0) * tmh).sum(2)  # [c,H,n,B]
        weight = vA2[..., :, None] * (Mp + Mn)[..., None, :]                       # [c,H,n,A,B]
        aw = alpha.view(1, H, 1, 1, 1) * W
        strong = vA2[..., :, None] * aw.abs() * torch.where(aw > 0, Mp[..., None, :], Mn[..., None, :])
        weak = vA2[..., :, None] * aw.abs() * torch.where(aw > 0, Mn[..., None, :], Mp[..., None, :])
        dw = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
        bins = ((dw + math.pi) * (NBINS / (2 * math.pi))).floor().clamp_(0, NBINS - 1).long()
        for si in range(len(STATUS)):
            sm = src[:, si][:, None, :, None, None].float()
            for ci in range(c):
                p = int(phase[ci])
                wsm = (weight[ci] * sm[ci])
                self.lag[p, si] += torch.bincount(bins[ci].flatten(), weights=wsm.flatten().double(), minlength=NBINS)
                self.lag_sw[p, si, 0] += (strong[ci] * sm[ci]).sum().double()
                self.lag_sw[p, si, 1] += (weak[ci] * sm[ci]).sum().double()
                self.syn_w[p, si] += wsm.sum(1).double()
                self.syn_wW[p, si] += (wsm * aw[ci]).sum(1).double()
        if self.case_pos is not None and c0 <= self.case_pos < c0 + c:
            i = self.case_pos - c0
            row = dict(kappa=kappa[i], energy=energy[i], EH=E['H'][i], ES=E['S'][i], SH=Sup['H'][i], SS=Sup['S'][i])
            for name, val in row.items():
                self.case[name][b] = val.cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--case', type=int, default=30)
    ap.add_argument('--batches', type=int, default=0, help='number of eval batches (0: all)')
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
    _, _, x, y, _, _ = t.load_data(cfg)
    base = make_model(protocol, cfg, device, ck['model_state_dict'])
    NB = cfg['loops'] * cfg['blocks_per_seg']
    probe = FlowProbe(base.model.inner, device, NB)
    gbs, began = cfg['global_batch_size'], time.monotonic()
    probe.case = {n: torch.zeros(NB, 81, 81) for n in ('kappa', 'energy', 'EH', 'ES', 'SH', 'SS')}
    case_info, n_sel = None, 0
    for bi, batch in enumerate(t.eval_batches(x, y, gbs, 0, 1)):
        if opt.batches and bi >= opt.batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = trajectory(base, batch, cfg)                                      # [NB,B,T]
        labels, inputs = batch['labels'].long(), batch['inputs'].long()
        commit = stays(preds == labels[None])                                     # [B,T]
        solve = stays((preds == labels[None]).all(-1))                            # [B]
        sel = torch.nonzero(solve < NB).flatten()
        case_pos = None
        if bi * gbs <= opt.case < (bi + 1) * gbs:
            local = opt.case - bi * gbs
            assert int(solve[local]) < NB, 'case puzzle is not solved'
            case_pos = int((sel == local).nonzero()[0, 0])
            case_info = dict(puzzle=opt.case, commit=commit[local].tolist(), solve_block=int(solve[local]),
                             preds=preds[:, local].tolist())
        probe.begin_batch(sel, (inputs > 1)[sel], inputs[sel], labels[sel], preds[:, sel], commit[sel], solve[sel], case_pos)
        probe.install()
        try:
            with torch.no_grad():
                carry = base.initial_carry(batch)
                for _ in range(cfg['loops']):
                    carry, *_ = base(carry=carry, batch=batch, return_keys=set())
        finally:
            probe.remove()
        n_sel += len(sel)
        print(f'batch {bi + 1}: solved {len(sel)} (total {n_sel}) {time.monotonic() - began:.0f}s check {probe.check:.2e}', flush=True)
    out = opt.out or run / 'diagnostics' / f"stdp_message_flow_step{int(ck['step'])}"
    out.mkdir(parents=True, exist_ok=True)
    tl = lambda d: {k: v.cpu().tolist() for k, v in d.items()}
    (out / 'aggregate.json').write_text(json.dumps(dict(
        puzzles=n_sel, phases=PHASES, relations=RELS, status=STATUS, targets=TARGETS, offsets=list(range(-OFF, OFF + 1)),
        recon_check=probe.check, acc=tl(probe.acc), source_event=tl(probe.src), target_event=tl(probe.tgt),
        lag_hist=probe.lag.cpu().tolist(), lag_strong_weak=probe.lag_sw.cpu().tolist(),
        alpha=base.model.inner.layers[0].stdp_alpha.tolist())) + '\n')
    torch.save(dict(syn_w=probe.syn_w.cpu(), syn_wW=probe.syn_wW.cpu()), out / 'synapses.pt')
    if case_info is not None:
        np.savez_compressed(out / f'case{opt.case}.npz', **{k: v.numpy() for k, v in probe.case.items()})
        (out / f'case{opt.case}.json').write_text(json.dumps(case_info) + '\n')
    print(f'WROTE {out} ({time.monotonic() - began:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
