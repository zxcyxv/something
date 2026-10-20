"""Version C: in-context rule inference with a learned structure (no grammar, no rule list, no colour-permutation list).

Kept (architectural): addresses and contents are separate (reads are by cell codes, values are colours); the
demonstrations write a Hebbian colour-pair rule synapse; a rule is a product of shared primitive operators; background
stays background. Learned or inferred: everything else.
    cell codes      G [16, d]                 free parameters (no generator construction)
    primitives      A_1..A_K [d, d]           orthogonal (matrix_exp of a skew-symmetric matrix), no names
    candidate rules every word of length <= N over the primitives (sum_L K^L words), U_w = A_{w_L} ... A_{w_1}
    reads           weights_w[o, x] = softmax_x(tau cos(U_w G_o, G_x))
    rule synapse    R_ac[x, o] = sum_k 1[I_k(x) = a] 1[O_k(o) = c]      (colours 1..4)
    colour map      any function, per word: kappa_w(c | a) = softmax_c(beta_c S_ac(w)),  S_ac(w) = <R_ac, weights_w^T>
    score           s(w) = sum_{a,c} kappa_w(c | a) S_ac(w)          (expected explained cells)
    posterior       softmax(beta (s(w) - lam |w|))   lam = description cost per symbol, in explained cells (0: none)
    prediction      p(o, y) = sum_w pi_w sum_a read_w(o, a) kappa_w(y | a), background -> background
"""
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import group as G

NCOL = G.COLORS + 1


def skew(a):
    return a - a.transpose(-1, -2)


class FreeStructureModel(nn.Module):
    def __init__(self, d=32, K=4, N=6, tau=3.0, beta=0.1, beta_c=1.0, seed=0, lam=0.0, read='cos', ls_eps=1e-2):
        super().__init__()
        if read not in ('cos', 'ls'):
            raise ValueError("read is 'cos' (Hebbian: cosine with each cell code) or 'ls' (delta rule: least squares)")
        self.lam, self.read, self.ls_eps = lam, read, ls_eps
        gen = torch.Generator().manual_seed(seed)
        self.d, self.K, self.N = d, K, N
        self.G = nn.Parameter(torch.randn(G.CELLS, d, generator=gen) / math.sqrt(d))
        self.A_raw = nn.Parameter(torch.randn(K, d, d, generator=gen) / math.sqrt(d))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        self.log_beta_c = nn.Parameter(torch.tensor(math.log(beta_c)))
        words = [()] + [w for L in range(1, N + 1) for w in itertools.product(range(K), repeat=L)]
        self.words = words                                                       # application order
        self.register_buffer('lens', torch.tensor([float(len(w)) for w in words]), persistent=False)
        self.register_buffer('src', torch.as_tensor(G.SRC), persistent=False)

    @property
    def n_words(self):
        return len(self.words)

    def primitives(self):
        return torch.matrix_exp(skew(self.A_raw))

    def operators(self):
        """[n_words, d, d] in the order of self.words: built level by level (each level = one more primitive)."""
        A = self.primitives()
        level = [torch.eye(self.d, device=A.device, dtype=A.dtype)[None]]
        for _ in range(self.N):
            prev = level[-1]                                                    # [K^(L-1), d, d], itertools order
            level.append(torch.einsum('jde,pef->pjdf', A, prev).reshape(-1, self.d, self.d))
        return torch.cat(level)

    def read_weights(self):
        """'cos': softmax_x(tau cos(U_w g_o, g_x)). 'ls': the delta-rule fixed point, coefficients of the address in the
        dual basis of the cell codes, c = (Gn Gn^T + eps I)^-1 Gn a, then softmax_x(tau c_x) (equal to 'cos' for orthonormal
        codes; ill-conditioned when two codes coincide)."""
        U = self.operators()
        Gn = F.normalize(self.G, dim=-1)
        addr = F.normalize(torch.einsum('wde,oe->wod', U, self.G), dim=-1)
        sim = torch.einsum('wod,xd->wox', addr, Gn)
        if self.read == 'ls':
            inv = torch.linalg.inv(Gn @ Gn.T + self.ls_eps * torch.eye(G.CELLS, device=Gn.device, dtype=Gn.dtype))
            sim = sim @ inv                                                      # inv is symmetric
        return (self.log_tau.exp() * sim).softmax(-1)

    def separation_loss(self, thresh=0.9):
        """Distinct addresses: mean over cell pairs of relu(cos(g_x, g_y) - thresh)^2."""
        Gn = F.normalize(self.G, dim=-1)
        off = (Gn @ Gn.T)[~torch.eye(G.CELLS, dtype=torch.bool, device=Gn.device)]
        return F.relu(off - thresh).square().mean()

    def forward(self, demo_in, demo_out, demo_mask, query_in):
        B, K = demo_in.shape[:2]
        weights = self.read_weights()                                            # [W, 16(o), 16(x)]
        X = F.one_hot(demo_in.reshape(B, K, G.CELLS).long(), NCOL).float()[..., 1:]
        Y = F.one_hot(demo_out.reshape(B, K, G.CELLS).long(), NCOL).float()[..., 1:] * demo_mask.float()[:, :, None, None]
        R = torch.einsum('bkxa,bkoc->baxco', X, Y)                               # Hebbian colour-pair synapse
        S = torch.einsum('baxco,wox->bwac', R, weights)                          # [B, W, 4(a), 4(c)]
        kap = (self.log_beta_c.exp() * S).softmax(-1)                            # colour map per word: P(c | a)
        score = (kap * S).sum((-1, -2))                                          # [B, W]
        log_post = (self.log_beta.exp() * (score - self.lam * self.lens)).log_softmax(-1)
        Xq = F.one_hot(query_in.reshape(B, G.CELLS).long(), NCOL).float()
        reads = torch.einsum('wox,bxa->bwoa', weights, Xq)                       # [B, W, 16, 5]
        post = log_post.exp()
        bg = torch.einsum('bw,bwo->bo', post, reads[..., 0])                     # background -> background
        col = torch.einsum('bw,bwoa,bwac->boc', post, reads[..., 1:], kap)
        probs = torch.cat([bg[..., None], col], -1)
        return dict(probs=probs, log_post=log_post, scores=score, exp_len=(post * self.lens).sum(-1))

    @torch.no_grad()
    def realised(self, heldout_t=()):
        """Argmax read map of every word -> which of the 128 transformations it realises (or none)."""
        maps = self.read_weights().argmax(-1)                                    # [W, 16]: output o reads x
        rows = {tuple(r): t for t, r in enumerate(G.SRC.tolist())}
        hits = [rows.get(tuple(m), -1) for m in maps.cpu().tolist()]
        valid = {t for t in hits if t >= 0}
        lens = [len(w) for w in self.words]
        shortest = {}
        for t, L in zip(hits, lens):
            if t >= 0:
                shortest[t] = min(shortest.get(t, 99), L)
        return dict(words=len(hits), words_valid=sum(t >= 0 for t in hits), distinct=len(valid),
                    mean_shortest_len=float(sum(shortest.values()) / max(len(shortest), 1)),
                    heldout_realised=sum(t in valid for t in heldout_t), heldout_total=len(heldout_t),
                    tau=float(self.log_tau.exp()), beta=float(self.log_beta.exp()), beta_c=float(self.log_beta_c.exp()))
