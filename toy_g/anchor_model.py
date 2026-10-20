"""Version C with a learned content-dependent part: the anchor (docs/claude_2026-10-20_arc_logic_foundations.md
section 9, definition 4; toy toy_g/anchor_toy.py).

Fixed (architecture): reads by cell codes, values are colours; the demonstrations write a Hebbian colour-pair synapse;
a rule is a word over shared primitive operators, optionally followed by the anchor part; posterior over programs by
explained cells minus a cost per symbol; background stays background. Learned: everything inside the parts.
    cell codes      G = exp(i Phi) [16, d]      phases learned (complex, unit modulus per channel)
    primitives      A_1..A_K [d, d]             unitary (matrix_exp of a skew-Hermitian matrix), no names
    anchor part     alpha_x = softmax_x(e[colour of x])     e [6] learned: which content marks the anchor
                    P = sum_x alpha_x G_x                    pooled anchor code
                    query  q_o = (U_w G_o) * P * B            B = exp(i b) learned binding constant (* elementwise)
    programs        (anchored in {0, 1}) x (every word of length <= N); cost |w| + anchored
    reads           softmax_x(tau Re<q_o, G_x> / d)          (fixed programs: q_o = U_w G_o)
    colour map, score, posterior, prediction: as toy_g/learned_structure.FreeStructureModel (colours 1..5)
With G the characters of the torus (G_{z+m} = G_z * G_m, G_0 = 1), alpha on the marker and B = 1, the anchored read
of a word realising z -> R^-1 (z - tau) is exactly the anchored rule (p, tau): see designed().
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import group as G
from .anchor_toy import MARKER, NCOL6


class AnchorStructureModel(nn.Module):
    def __init__(self, d=16, K=4, N=5, tau=3.0, beta=0.1, beta_c=1.0, seed=0, lam=0.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.d, self.K, self.N, self.lam = d, K, N, lam
        self.phase = nn.Parameter(2 * math.pi * torch.rand(G.CELLS, d, generator=gen))
        self.A_re = nn.Parameter(torch.randn(K, d, d, generator=gen) / math.sqrt(d))
        self.A_im = nn.Parameter(torch.randn(K, d, d, generator=gen) / math.sqrt(d))
        self.col_score = nn.Parameter(torch.zeros(NCOL6))
        self.bind_phase = nn.Parameter(torch.zeros(d))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        self.log_beta_c = nn.Parameter(torch.tensor(math.log(beta_c)))
        self.words = [()] + [w for L in range(1, N + 1) for w in itertools.product(range(K), repeat=L)]
        lens = torch.tensor([float(len(w)) for w in self.words])
        self.register_buffer('lens2', torch.cat([lens, lens + 1]), persistent=False)
        self.fixed_ops = None                                                    # designed(): exact operators

    def codes(self):
        return torch.polar(torch.ones_like(self.phase), self.phase)

    def primitives(self):
        A = torch.complex(self.A_re, self.A_im)
        return torch.matrix_exp(A - A.conj().transpose(-1, -2))

    def operators(self):
        if self.fixed_ops is not None:
            return self.fixed_ops
        A = self.primitives()
        level = [torch.eye(self.d, dtype=A.dtype, device=A.device)[None]]
        for _ in range(self.N):
            level.append(torch.einsum('jde,pef->pjdf', A, level[-1]).reshape(-1, self.d, self.d))
        return torch.cat(level)

    def addresses(self):
        """[W, 16, d]: U_w G_o for every fixed program."""
        return torch.einsum('wde,oe->wod', self.operators(), self.codes())

    def anchor_code(self, grids):
        """grids [M, 16] -> pooling weights [M, 16] and the bound anchor P * B [M, d]."""
        Gc = self.codes()
        alpha = self.col_score[grids].softmax(-1)
        P = torch.einsum('mx,xd->md', alpha.to(Gc.dtype), Gc)
        B = torch.polar(torch.ones_like(self.bind_phase), self.bind_phase)
        return alpha, P * B[None]

    def weights(self, grids=None):
        """Fixed reads [W, 16(o), 16(x)], or anchored reads for grids [M, 16] -> [M, W, 16, 16]."""
        Gc = self.codes()
        addr = self.addresses()                                                   # [W, 16, d]
        tau = self.log_tau.exp()
        if grids is None:
            return (tau * torch.einsum('wod,xd->wox', addr, Gc.conj()).real / self.d).softmax(-1)
        _, PB = self.anchor_code(grids)
        key = Gc[None] * PB.conj()[:, None, :]                                    # q.conj(G_x) = addr.conj(key_x)
        return (tau * torch.einsum('wod,mxd->mwox', addr, key.conj()).real / self.d).softmax(-1)

    def forward(self, demo_in, demo_out, demo_mask, query_in):
        Bn, k = demo_in.shape[:2]
        din = demo_in.reshape(Bn, k, G.CELLS).long()
        dout = demo_out.reshape(Bn, k, G.CELLS).long()
        q = query_in.reshape(Bn, G.CELLS).long()
        wf = self.weights()
        W = wf.shape[0]
        wa = self.weights(torch.cat([din.reshape(-1, G.CELLS), q]))
        wa_d, wa_q = wa[:Bn * k].reshape(Bn, k, W, G.CELLS, G.CELLS), wa[Bn * k:]
        X = F.one_hot(din, NCOL6).float()[..., 1:]
        Y = F.one_hot(dout, NCOL6).float()[..., 1:] * demo_mask.float()[:, :, None, None]
        S_f = torch.einsum('baxco,wox->bwac', torch.einsum('bkxa,bkoc->baxco', X, Y), wf)
        S_a = torch.einsum('bkwoa,bkoc->bwac', torch.einsum('bkwox,bkxa->bkwoa', wa_d, X), Y)
        S = torch.cat([S_f, S_a], 1)                                              # [B, 2W, 5, 5]
        kap = (self.log_beta_c.exp() * S).softmax(-1)
        score = (kap * S).sum((-1, -2))
        lens = self.lens2[:2 * W] if self.fixed_ops is None else torch.cat([torch.zeros(W), torch.ones(W)]).to(S)
        log_post = (self.log_beta.exp() * (score - self.lam * lens)).log_softmax(-1)
        Xq = F.one_hot(q, NCOL6).float()
        reads = torch.cat([torch.einsum('wox,bxa->bwoa', wf, Xq), torch.einsum('bwox,bxa->bwoa', wa_q, Xq)], 1)
        post = log_post.exp()
        bg = torch.einsum('bw,bwo->bo', post, reads[..., 0])
        col = torch.einsum('bw,bwoa,bwac->boc', post, reads[..., 1:], kap)
        return dict(probs=torch.cat([bg[..., None], col], -1), log_post=log_post, scores=score,
                    anchored_mass=post[:, W:].sum(1), exp_len=(post * lens).sum(-1))

    @torch.no_grad()
    def diagnostics(self, heldout_fixed=()):
        """Anchor pooling per colour; homomorphism of the codes (cos(G_{z+m}, G_z G_m conj(G_0)), torus addition);
        transformations realised by the fixed reads of the words."""
        Gc = self.codes()
        z = torch.arange(G.CELLS, device=Gc.device)
        zr, zc = z // G.N, z % G.N
        add = ((zr[:, None] + zr[None]) % G.N) * G.N + (zc[:, None] + zc[None]) % G.N  # [16, 16]: z + m
        pred = Gc[:, None, :] * Gc[None, :, :] * Gc[0].conj()[None, None, :]
        hom = (pred * Gc[add].conj()).real.sum(-1) / self.d                       # cos for unit-modulus codes
        maps = self.weights().argmax(-1).cpu()
        rows = {tuple(r): t for t, r in enumerate(G.SRC.tolist())}
        hits = [rows.get(tuple(m), -1) for m in maps.tolist()]
        valid = {t for t in hits if t >= 0}
        pool = self.col_score.softmax(-1).tolist()
        B = torch.polar(torch.ones_like(self.bind_phase), self.bind_phase)
        return dict(pool_softmax=[round(v, 3) for v in pool], hom_cos_mean=float(hom.mean()), hom_cos_min=float(hom.min()),
                    bind_vs_conjG0=float((B * Gc[0]).real.mean()), words_valid=sum(t >= 0 for t in hits),
                    distinct=len(valid), heldout_fixed_realised=sum(int(t) in valid for t in heldout_fixed),
                    tau=float(self.log_tau.exp()), beta=float(self.log_beta.exp()), beta_c=float(self.log_beta_c.exp()))


def designed(model):
    """Hand-set parameters for the existence check only: characters of the torus as codes (d = 16), the 128 exact
    operators as the program set, the marker as the anchor, B = 1, sharp temperatures."""
    assert model.d == 16
    dev = model.phase.device
    z = torch.tensor([[r, c] for r in range(G.N) for c in range(G.N)], dtype=torch.float64)
    phase = 2 * math.pi * (z @ z.T) / G.N                                        # rows: cells, columns: frequencies
    model.phase.data = phase.float().to(dev)
    Gc = model.codes().to(torch.complex128)
    src = torch.as_tensor(G.SRC, device=dev)
    model.fixed_ops = torch.stack([Gc[src[t]].T @ Gc.conj() / G.CELLS for t in range(G.NUM_T)]).to(torch.complex64)
    model.col_score.data = torch.tensor([0, 0, 0, 0, 0, 30.0], device=dev)
    model.bind_phase.data.zero_()
    model.log_tau.data.fill_(math.log(30.0))
    model.log_beta.data.fill_(math.log(5.0))
    model.log_beta_c.data.fill_(math.log(5.0))
    return model


class PhaseAnchorModel(AnchorStructureModel):
    """Programs = (word over K learned unitary primitives, length <= N: the non-commutative part)
                 x (offset cell v: the commutative part, binding with the position code G_v conj(G_0))
                 x (anchored or not: binding with the content-pooled code P conj(G_0)).
    q_o = (U_w G_o) * G_v * conj(G_0) [* P * conj(G_0)], cost |w| + [v != 0] + anchored; program index w * 16 + v.
    Translations exist only as bindings, so a translation rule is fitted only if G_{z+v} = G_z * G_v * conj(G_0)
    (characters of the torus, learned); with such codes the anchored binding moves by the pooled cell."""

    def __init__(self, d=16, K=2, N=4, tau=3.0, beta=0.1, beta_c=1.0, seed=0, lam=0.0):
        super().__init__(d=d, K=K, N=N, tau=tau, beta=beta, beta_c=beta_c, seed=seed, lam=lam)
        lw = torch.tensor([float(len(w)) for w in self.words])
        lens = (lw[:, None] + (torch.arange(G.CELLS) != 0).float()[None]).reshape(-1)
        self.register_buffer('lens2', torch.cat([lens, lens + 1]), persistent=False)
        del self.bind_phase                                                      # the binding constant is conj(G_0)
        self.register_buffer('bind_phase', torch.zeros(d), persistent=False)     # kept for diagnostics() only

    def addresses(self):
        Gc = self.codes()
        base = torch.einsum('wde,oe->wod', self.operators(), Gc)                  # [W, 16(o), d]
        shift = Gc * Gc[0].conj()[None]                                           # [16(v), d]
        return (base[:, None] * shift[None, :, None, :]).reshape(-1, G.CELLS, self.d)

    def anchor_code(self, grids):
        Gc = self.codes()
        alpha = self.col_score[grids].softmax(-1)
        P = torch.einsum('mx,xd->md', alpha.to(Gc.dtype), Gc)
        return alpha, P * Gc[0].conj()[None]


def designed_phase(model):
    """Existence check only: characters as codes, the 8 linear point operations (about cell 0) as the words, the
    marker as the anchor, sharp temperatures."""
    assert model.d == 16
    dev = model.phase.device
    z = torch.tensor([[r, c] for r in range(G.N) for c in range(G.N)], dtype=torch.float64)
    model.phase.data = (2 * math.pi * (z @ z.T) / G.N).float().to(dev)
    Gc = model.codes().to(torch.complex128)
    mats = [((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
            ((0, 1), (1, 0)), ((1, 0), (0, -1)), ((0, -1), (-1, 0)), ((-1, 0), (0, 1))]
    zi = z.long()
    ops = []
    for R in mats:
        img = (zi @ torch.tensor(R).T) % G.N
        perm = (img[:, 0] * G.N + img[:, 1]).to(dev)
        ops.append(Gc[perm].T @ Gc.conj() / G.CELLS)
    model.fixed_ops = torch.stack(ops).to(torch.complex64)
    model.col_score.data = torch.tensor([0, 0, 0, 0, 0, 30.0], device=dev)
    model.log_tau.data.fill_(math.log(30.0))
    model.log_beta.data.fill_(math.log(5.0))
    model.log_beta_c.data.fill_(math.log(5.0))
    return model
