"""Rule-synapse models of docs/claude_2026-10-20_rule_synapse_architecture.md on the 4x4 torus toy (in context, no
task identifier). Both return dict(probs [B, 16, 5], log_post [B, 128]) like toy_g.model.StructureModel.

RuleSynapseModel (version A, the structure pathway). The demonstrations write one Hebbian colour-pair synapse
    R_ab[x, o] = sum_k 1[I_k(x) = a] 1[O_k(o) = b]            (a, b = colours 1..4, masked demonstrations excluded)
and every candidate (t, c) is scored against it through the read weights of StructureModel (designed or learned
generators, code- or cell-space composition):
    s(t, c) = sum_a <R_{a, kappa_c(a)}, M_t>,   M_t[x, o] = weights[t, o, x]    (= explained output cells when exact)
posterior softmax(beta s) over the 3,072 (t, c); the prediction is StructureModel's posterior mixture of query reads.
Only the score differs from StructureModel (a linear synapse statistic instead of the read log-likelihood).

BandsRuleModel (version B, the window learned; exact operators so that only the window is in question). Tokens are
the demonstration output cells; an imposed block schedule gives their states, and the synapse is the bands window of
lt/model.py summed over blocks and tokens (closed form of the recursion, see window_synapse):
    '2block'   s(0) = p(o) = e(O(o)) (x) e_o,  s(1) = V_k = sum_x e(I_k(x)) (x) e_x          (key = value = state)
    'persist'  s(0) = p(o),  s(1) = s(2) = V_k                                               (the state stays)
    'residual' s(0) = [p(o); 0],  s(1) = [p(o); V_k]   key = W_k s, value = W_v s (learned 64 x 128 projections)
The value-by-key synapse W [64, 64] is read as R_ab[x, o] = W[(a, x), (b, o)] and scored as above with M_t = P_t.
Window: R bands, poles z_r = exp(-e^nu_r + i theta_r), complex a_r (value at or after the key), b_r (value before):
    l(D) = sum_r gamma_r Re(a_r z_r^D) (D >= 0),  l(-D) = sum_r gamma_r Re(b_r z_r^D) (D > 0),  gamma_r = sqrt(1-|z_r|^2)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import group as G
from .model import NCOL, StructureModel

C = G.COLORS


def colour_onehot(grids, drop_background=True):
    """[..., 16] colours 0..4 -> [..., 16, 4] (colours 1..4) or [..., 16, 5]."""
    oh = F.one_hot(grids.long(), NCOL).float()
    return oh[..., 1:] if drop_background else oh


def score_and_predict(R, M, perms, beta, query_reads):
    """R [B, 4, 16(x), 4, 16(o)] synapse; M [128, 16(x), 16(o)] operator read matrices; perms [24, 5, 5];
    query_reads [B, 128, 16(o), 5]. Returns probs [B, 16, 5], log_post_t [B, 128], score [B, 128, 24]."""
    B = R.shape[0]
    S = torch.einsum('baxco,txo->bact', R, M)                                    # [B, 4, 4, 128] = <R_ac, M_t>
    kap = perms[:, 1:, 1:].argmax(-1)                                            # [24, 4]: kappa_c(a) - 1
    score = sum(S[:, a, kap[:, a], :] for a in range(C)).transpose(1, 2)         # [B, 128, 24]
    logits = beta * score
    log_post = logits.reshape(B, -1).log_softmax(-1).reshape(logits.shape)
    cmaps = torch.einsum('btc,cay->btay', log_post.exp(), perms)                 # pi_t * E[P_c | t]
    probs = torch.einsum('btoa,btay->boy', query_reads, cmaps)
    return probs, log_post.logsumexp(-1), score


class RuleSynapseModel(StructureModel):
    """StructureModel's codes, generators, read weights and colour maps ('perm'), with the rule-synapse score."""

    def forward(self, demo_in, demo_out, demo_mask, query_in):
        if self.cmap != 'perm':
            raise ValueError("RuleSynapseModel scores (t, c) jointly: cmap='perm'")
        B, K = demo_in.shape[:2]
        weights = self.structure()[3]                                            # [128, 16(o), 16(x)]
        X = colour_onehot(demo_in.reshape(B, K, G.CELLS))
        Y = colour_onehot(demo_out.reshape(B, K, G.CELLS)) * demo_mask.float()[:, :, None, None]
        R = torch.einsum('bkxa,bkoc->baxco', X, Y)                               # Hebbian colour-pair synapse
        Xq = colour_onehot(query_in.reshape(B, G.CELLS), drop_background=False)
        query = torch.einsum('tox,bxa->btoa', weights, Xq)
        probs, log_post, score = score_and_predict(R, weights.transpose(1, 2), self.perms, self.beta, query)
        return dict(probs=probs, log_post=log_post, scores=score.max(-1).values)


def perm_ops(device=None):
    """[128, 16(x), 16(o)] exact read matrices P_t[x, o] = 1[x = SRC[t, o]]."""
    P = torch.zeros(G.NUM_T, G.CELLS, G.CELLS, device=device)
    P[torch.arange(G.NUM_T)[:, None], torch.as_tensor(G.SRC), torch.arange(G.CELLS)[None, :]] = 1
    return P


def window_synapse(k_states, v_states, lag_weight):
    """Closed form of the bands recursion summed over blocks: W = sum_{n >= m} l(n - m) C(n, m) + sum_{m < n}
    l(-(n - m)) C'(m, n), with C(n, m) = sum_tokens v(n) k(m)^T and C'(m, n) = sum_tokens v(m) k(n)^T.
    k_states, v_states [blocks, B, tokens, dk / dv]; lag_weight(D) -> scalar tensor. Returns [B, dv, dk]."""
    N = k_states.shape[0]
    W = 0
    for n in range(N):
        for m in range(N):
            W = W + lag_weight(n - m) * torch.einsum('btv,btk->bvk', v_states[n], k_states[m])
    return W


class BandsRuleModel(nn.Module):
    def __init__(self, schedule='2block', bands=4, init='odd', beta=0.1, seed=0):
        super().__init__()
        if schedule not in ('2block', 'persist', 'residual') or init not in ('odd', 'causal', 'even', 'random'):
            raise ValueError(f'bad schedule/init: {schedule} {init}')
        self.schedule = schedule
        gen = torch.Generator().manual_seed(seed)
        # poles: |z| from nu (|z| = exp(-e^nu)), angles spread over (0, pi); start near |z| = 0.5
        self.nu = nn.Parameter(torch.full((bands,), math.log(math.log(2.0))))
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, bands + 2)[1:-1].clone())
        a = torch.randn(bands, 2, generator=gen) * 0.5
        self.a = nn.Parameter(a)                                                  # (Re, Im) per band
        self.b = nn.Parameter(torch.zeros(bands, 2))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)))
        self.register_buffer('P', perm_ops(), persistent=False)
        self.register_buffer('perms', F.one_hot(torch.as_tensor(G.CMAPS), NCOL).float(), persistent=False)
        if schedule == 'residual':
            mix = torch.cat([torch.eye(C * G.CELLS), torch.eye(C * G.CELLS)], 1) / math.sqrt(2)
            self.Wk = nn.Parameter(mix.clone())
            self.Wv = nn.Parameter(mix.clone())
        with torch.no_grad():
            self.set_window(init)

    # ------------------------------------------------------------------ window
    def poles(self):
        mag = torch.exp(-torch.exp(self.nu))
        z = torch.complex(mag * torch.cos(self.theta), mag * torch.sin(self.theta))
        gamma = torch.sqrt(-torch.expm1(-2 * torch.exp(self.nu)))
        return z, gamma

    def lag(self, D):
        z, gamma = self.poles()
        coef = self.a if D >= 0 else self.b
        c = torch.complex(coef[:, 0], coef[:, 1])
        return (gamma * c * z ** abs(D)).real.sum()

    def set_window(self, init):
        """Least-squares coefficients for target lags [l(-2), ..., l(2)] at the current poles."""
        if init == 'random':
            return
        target = {'odd': [0, -1, 0, 1, 0], 'causal': [0, 0, 0, 1, 0], 'even': [0, 1, 0, 1, 0]}[init]
        z, gamma = self.poles()
        Ds = [0, 1, 2]
        Apos = torch.stack([gamma * z ** D for D in Ds])                          # [3, R] complex
        # Re(c z^D) = Re(c) Re(z^D) - Im(c) Im(z^D): real system in (Re c, Im c)
        def solve(targets):
            M = torch.cat([Apos.real, -Apos.imag], 1)                             # [3, 2R]
            sol = torch.linalg.lstsq(M, torch.tensor(targets, dtype=M.dtype)[:, None]).solution[:, 0]
            R = len(self.nu)
            return torch.stack([sol[:R], sol[R:]], 1)
        self.a.copy_(solve([target[2], target[3], target[4]]))
        self.b.copy_(solve([0.0, target[1], target[0]]))                          # b: D = 0 unused, then -1, -2

    def window(self):
        return {D: float(self.lag(D)) for D in (-2, -1, 0, 1, 2)}

    # ------------------------------------------------------------------ forward
    def states(self, demo_in, demo_out, demo_mask):
        """[blocks, B, K*16, dim] key and value states of the demonstration output tokens under the schedule."""
        B, K = demo_in.shape[:2]
        m = demo_mask.float()[:, :, None, None]
        Y = colour_onehot(demo_out.reshape(B, K, G.CELLS)) * m                    # [B, K, 16(o), 4]
        own = torch.einsum('bkoc,op->bkocp', Y, torch.eye(G.CELLS, device=Y.device)).reshape(B, K * G.CELLS, -1)
        X = colour_onehot(demo_in.reshape(B, K, G.CELLS)) * m                     # [B, K, 16(x), 4]
        V = X.transpose(2, 3).reshape(B, K, 1, -1).expand(B, K, G.CELLS, -1)      # every output token of pair k
        V = V.reshape(B, K * G.CELLS, -1)
        if self.schedule == '2block':
            s = torch.stack([own, V])
            return s, s
        if self.schedule == 'persist':
            s = torch.stack([own, V, V])
            return s, s
        s = torch.stack([torch.cat([own, torch.zeros_like(V)], -1), torch.cat([own, V], -1)])
        return s @ self.Wk.T, s @ self.Wv.T

    def forward(self, demo_in, demo_out, demo_mask, query_in):
        B = demo_in.shape[0]
        ks, vs = self.states(demo_in, demo_out, demo_mask)
        W = window_synapse(ks, vs, self.lag)                                       # [B, 64 (a,x), 64 (b,o)]
        R = W.reshape(B, C, G.CELLS, C, G.CELLS)
        Xq = colour_onehot(query_in.reshape(B, G.CELLS), drop_background=False)   # [B, 16(x), 5]
        query = torch.einsum('txo,bxa->btoa', self.P, Xq)                         # exact reads
        probs, log_post, score = score_and_predict(R, self.P, self.perms, self.log_beta.exp(), query)
        return dict(probs=probs.clamp_min(0), log_post=log_post, scores=score.max(-1).values)
