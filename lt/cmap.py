"""Cognitive-map rule synapse, spec v2 (docs/claude_2026-10-21_cognitive_map_rule_synapse.md section 12).

Structure (forced by format or logic): observation and inference streams kept apart; grids talk to each other only through
the fast rule synapse (slow attention stays inside a grid, plus a pair's input -> output); the rule is the sparse-LTD
(proximal-gradient) fit of the demonstrations started at the puzzle embedding's prior centre; recurrence; exact
equivariance to the augmentation group (D4 by group averaging, colour permutations by using colours only as relations).
Everything else (what a frame picks, where things stop, what is counted) is learned.

Shapes: B lanes, G = 2 D + 2 grids (demo in/out x D, query in, query out), N x N region per grid, O = N^2 cells,
P = D + 1 pairs, F = 1 + Kc + Kg frames, J rule terms, code width C2 = 2 * (number of frequencies), 10 colours.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import rms_norm

NC = 10                                        # colours


def op_t(x, k, dims):
    """D4 element k on two axes (same convention as lt.cmap_data.op_apply)."""
    x = torch.rot90(x, k % 4, dims)
    return x.transpose(*dims) if k >= 4 else x


FAST = ('A', 'Pi', 'Gam', 'Bh')       # fast blocks: one lasso over all of them


def d4_tables(N):
    """Matrices M_k (p' = M_k p on centred coordinates), inverse indices and product table of the 8 region ops."""
    m = (N - 1) / 2
    R, C = np.meshgrid(np.arange(N) - m, np.arange(N) - m, indexing='ij')
    mats = []
    for k in range(8):
        old = np.stack([op_t(torch.tensor(R), k, (0, 1)).numpy(), op_t(torch.tensor(C), k, (0, 1)).numpy()], -1)
        new = np.stack([R, C], -1)                     # new location of the cell whose old coordinate is `old`
        sol, *_ = np.linalg.lstsq(old.reshape(-1, 2), new.reshape(-1, 2), rcond=None)
        mats.append(np.round(sol.T).astype(np.int64))
    find = lambda M: next(i for i, A in enumerate(mats) if (A == M).all())
    inv = [find(np.round(np.linalg.inv(M)).astype(np.int64)) for M in mats]
    mul = [[find(mats[a] @ mats[b]) for b in range(8)] for a in range(8)]
    return np.stack(mats), inv, mul


@dataclass
class CmapConfig:
    N: int = 5
    D: int = 8                   # demonstration slots
    n_tasks: int = 149
    d: int = 128
    heads: int = 4
    base_freqs: int = 2          # code width = 2 * 8 * base_freqs (D4 orbits)
    layers: int = 2              # perception layers per recurrence step
    Kc: int = 4                  # cell frames
    Kg: int = 4                  # grid frames
    J: int = 2                   # rule terms
    steps: int = 16              # recurrence
    grad_steps: int = 2          # phantom gradient: last steps with gradients
    emb: int = 64                # puzzle embedding
    emb_dropout: float = 0.3
    use_emb: bool = True         # False: no puzzle embedding at all (every episode starts at the shared prior)
    frame_impl: str = 'sigmoid'  # 'sdpa': a cell frame's anchor and colours are a softmax-weighted average over its key grid
                                 # (an attention output, via SDPA: no [cells x cells] mask tensor); 'sigmoid': membership
                                 # masks m in [0, 1] materialised per cell pair (needs group_average for exact D4 equivariance)
    attn_impl: str = 'einsum'    # 'sdpa': perception attention by scaled_dot_product_attention (flash on CUDA); the colour
                                 # bias and the key mask ride in extra q/k channels, so no mask tensor is passed
    group_average: bool = True   # exact D4 equivariance of perception and frames
    patch: int = 1               # perception tokens are p x p cell patches (lossless rearrangement + linear, ViT-style);
                                 # the cell stream, frames, rule synapse and output stay per cell. 1 = per-cell perception
    tau_init: float = 1e-3       # sparse-LTD threshold
    eta_init: float = 0.1        # write step size (learned per recurrence step, LISTA-style) ...
    eta_max: float = 0.15        # ... capped: by construction L <~ 12 (frame colours <= 9, other terms ~1 each, mean
                                 # over cells), so eta L < 2 and the step cannot diverge (measured L: 2.1-7.8)


class Cmap(nn.Module):
    def __init__(self, cfg: CmapConfig):
        super().__init__()
        self.cfg = c = cfg
        N, D = c.N, c.D
        self.G, self.O, self.P = 2 * D + 2, N * N, D + 1
        self.F = 1 + c.Kc + c.Kg
        self.nf = 8 * c.base_freqs
        self.C2 = 2 * self.nf
        assert c.d % c.heads == 0 and c.d // c.heads == self.C2, 'head width must equal the code width (RoPE = code)'
        mats, inv, mul = d4_tables(N)
        self.op_inv, self.op_mul = inv, mul
        self.register_buffer('d4', torch.tensor(mats, dtype=torch.float32), persistent=False)
        # channel (b, d) has frequency M_d^T theta_b; position op k sends channel (b, d) to (b, mul[d][k])
        perm = torch.tensor([[b * 8 + mul[dd][k] for b in range(c.base_freqs) for dd in range(8)] for k in range(8)])
        pair_perm = torch.stack([torch.stack([2 * perm[k], 2 * perm[k] + 1], -1).reshape(-1) for k in range(8)])
        self.register_buffer('code_perm', pair_perm, persistent=False)          # [8, C2]: code(M_k x) = code(x)[perm]
        self.theta = nn.Parameter(torch.randn(c.base_freqs, 2))
        m = (N - 1) / 2
        rr, cc = torch.meshgrid(torch.arange(N) - m, torch.arange(N) - m, indexing='ij')
        self.register_buffer('pos', torch.stack([rr, cc], -1).reshape(-1, 2).float(), persistent=False)   # [O, 2]
        # grid bookkeeping
        G = self.G
        role = torch.tensor([i % 2 for i in range(2 * D)] + [2, 3])
        pair = torch.tensor([i // 2 for i in range(2 * D)] + [D, D])
        key_grid = torch.tensor([i if i % 2 == 0 else i - 1 for i in range(2 * D)] + [G - 2, G - 2])
        self.register_buffer('role', role, persistent=False)
        self.register_buffer('pair', pair, persistent=False)
        self.register_buffer('key_grid', key_grid, persistent=False)
        self.in_grids = [2 * i for i in range(D)] + [G - 2]
        self.out_grids = [2 * i + 1 for i in range(D)] + [G - 1]
        allow = torch.zeros(G, G, dtype=torch.bool)
        for g in range(G):
            allow[g, g] = True
            if g in self.out_grids:
                allow[g, key_grid[g]] = True
        self.register_buffer('grid_allow', allow, persistent=False)
        # observation embedding: valid, background, role (4), pair (D + 1)
        self.obs = nn.Linear(2 + 4 + D + 1, c.d)
        self.h0 = nn.Parameter(torch.zeros(c.d))
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(c.d, 3 * c.d, bias=False), out=nn.Linear(c.d, c.d, bias=False),
            gate_up=nn.Linear(c.d, 4 * c.d, bias=False), down=nn.Linear(2 * c.d, c.d, bias=False)))
            for _ in range(c.layers)])
        self.color_bias = nn.Parameter(torch.zeros(c.layers, c.heads))
        if c.patch > 1:
            p = c.patch
            n = -(-N // p)
            self.Np, self.T = n * p, n * n
            self.p_in = nn.Linear(p * p * c.d, c.d, bias=False)
            self.p_obs = nn.Linear(p * p * c.d, c.d, bias=False)
            self.p_out = nn.Linear(c.d, p * p * c.d, bias=False)
            ctr = (torch.arange(n) * p + (p - 1) / 2) - (N - 1) / 2             # patch centres in cell coordinates
            pr, pc = torch.meshgrid(ctr, ctr, indexing='ij')
            self.register_buffer('ppos', torch.stack([pr, pc], -1).reshape(-1, 2).float(), persistent=False)
        # frames
        self.fq = nn.Linear(c.d, c.Kc * self.C2, bias=False)
        self.fk = nn.Linear(c.d, c.Kc * self.C2, bias=False)
        self.f_color = nn.Parameter(torch.zeros(c.Kc))
        self.f_bias = nn.Parameter(torch.zeros(c.Kc))
        self.gq = nn.Parameter(torch.randn(c.Kg, self.C2) / math.sqrt(self.C2))
        self.gk = nn.Linear(c.d, self.C2, bias=False)
        self.g_bias = nn.Parameter(torch.zeros(c.Kg))
        # fast-state shapes and the puzzle embedding's prior centre
        J, C2, Fn = c.J, self.C2, self.F
        self.nc = 3 + 3 * c.Kg                            # size candidates: 1, h_in, w_in, (std_r, std_c, mass) per grid frame
        # fast (per task, inferred from the demonstrations): spatial operators A_j (colour kept), a colour map Pi,
        # frame-colour weights Gam, a constant output Bh; all enter the read linearly, so the write is one lasso
        self.shapes = dict(A=(J, C2, C2), Pi=(NC, NC), Gam=(Fn,), Bh=(NC, C2), beta=(2, self.nc))
        # slow (shared): which frame pairs each spatial term reads (omega) and the colour term's spatial read (omega_c)
        om = torch.zeros(J, Fn, Fn)
        om[0, 0, 0] = 1.0
        om[1:] = 0.1 * torch.randn(J - 1, Fn, Fn)
        self.omega = nn.Parameter(om)
        omc = torch.zeros(Fn, Fn)
        omc[0, 0] = 1.0
        self.omega_c = nn.Parameter(omc)
        self.sizes = {k: int(np.prod(v)) for k, v in self.shapes.items()}
        total = sum(self.sizes.values())
        self.task_emb = nn.Parameter(torch.randn(c.n_tasks, c.emb) * 0.02)
        self.dec = nn.Linear(c.emb, total, bias=False)
        nn.init.normal_(self.dec.weight, std=0.01)
        d0 = dict(A=torch.zeros(J, C2, C2), Pi=torch.zeros(NC, NC), Gam=torch.zeros(Fn), Bh=torch.zeros(NC, C2),
                  beta=torch.zeros(2, self.nc))
        d0['A'][0] = torch.eye(C2)
        d0['beta'][0, 1] = d0['beta'][1, 2] = 1.0
        self.d0 = nn.ParameterDict({k: nn.Parameter(v) for k, v in d0.items()})
        self.tau = nn.ParameterDict({k: nn.Parameter(torch.full((), math.log(math.expm1(c.tau_init))))
                                     for k in self.shapes})
        self.eta_raw = nn.Parameter(torch.full((c.steps,), math.log(c.eta_init / (c.eta_max - c.eta_init))))
        # output
        self.src_w = nn.Linear(c.d, 3 + c.Kc)
        self.fb = nn.Linear(7, c.d)
        self.temp = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------ code
    def freqs(self):
        return torch.einsum('dij,bi->bdj', self.d4, self.theta).reshape(-1, 2)        # [nf, 2]: M_d^T theta_b

    def code(self, x):
        """Phase code of displacements x [..., 2] -> [..., C2] (cos, sin interleaved per frequency)."""
        a = x @ self.freqs().T
        return torch.stack([a.cos(), a.sin()], -1).flatten(-2) / math.sqrt(self.nf)      # unit norm

    def rope(self, v, x):
        """Rotate pairs of v [..., C2] by the angles of positions x [..., 2]."""
        a = x @ self.freqs().T
        cos, sin = a.cos(), a.sin()
        v0, v1 = v[..., 0::2], v[..., 1::2]
        return torch.stack([v0 * cos - v1 * sin, v0 * sin + v1 * cos], -1).flatten(-2)

    # ------------------------------------------------------------------ group averaging
    def averaged(self, fn, cells, pairwise=()):
        """(1/8) sum_k op_k fn(op_k^-1 inputs). cells: tensors [B, G, N, N, ...]; outputs of fn are tensors whose dims
        2,3 (and, for names in pairwise, also the two dims after them) are cell grids."""
        if not self.cfg.group_average:
            return fn(*cells)
        B = cells[0].shape[0]
        stacked = [torch.cat([op_t(x, self.op_inv[k], (2, 3)) for k in range(8)], 0) for x in cells]
        amp = cells[0].is_cuda and cells[0].dtype == torch.float32
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            outs = fn(*stacked)                                                   # one call on the 8 copies
        outs = [y.float() if amp and y.is_floating_point() else y for y in outs]
        res = []
        for i, y in enumerate(outs):
            acc = 0
            for k, yk in enumerate(y.split(B, 0)):
                yk = op_t(yk, k, (2, 3))
                if i in pairwise:
                    yk = op_t(yk, k, (-2, -1))
                acc = acc + yk
            res.append(acc / 8)
        return res

    # ------------------------------------------------------------------ perception (slow, inside a grid + pair)
    def perceive(self, h, obs, cv, live):
        """h [B,G,N,N,d], obs [B,G,N,N,d], cv colour vectors [B,G,N,N,10], live [B,G,N,N] (cells that exist as keys).
        Every token attends to the tokens of its own grid and, for an output grid, of its pair's input grid. With
        patch p > 1 a token is a p x p patch (the region padded to a multiple of p): token = linear(concat of its cells),
        "same colour" between tokens = <colour histograms> (still colour-permutation equivariant), position = patch centre;
        the result goes back to the cells through a linear map, added to the cell stream."""
        c = self.cfg
        B, G, N = h.shape[0], self.G, c.N
        if c.patch == 1:
            x = self.attend(h.reshape(B, G, self.O, -1), obs.reshape(B, G, self.O, -1), cv.reshape(B, G, self.O, NC),
                            live.reshape(B, G, self.O), self.pos)
            return x.view(h.shape)
        p, Np, n = c.patch, self.Np, self.Np // c.patch

        def to_patches(t):                                                         # [B,G,N,N,e] -> [B,G,T,p*p*e]
            t = F.pad(t, (0, 0, 0, Np - N, 0, Np - N))
            return t.view(B, G, n, p, n, p, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, G, self.T, -1)

        x = rms_norm(self.p_in(to_patches(h)))
        o = self.p_obs(to_patches(obs))
        hist = to_patches(cv).view(B, G, self.T, p * p, NC).sum(3) / (p * p)
        lv = to_patches(live[..., None].to(h.dtype)).amax(-1) > 0
        x = self.attend(x, o, hist, lv, self.ppos)
        dx = self.p_out(x).view(B, G, n, n, p, p, -1).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, G, Np, Np, -1)
        return rms_norm(h + dx[:, :, :N, :N])

    def attend(self, x, o, cvf, lv, pos):
        """Perception layers on tokens. x, o [B,G,T,d], cvf colour vectors [B,G,T,10], lv [B,G,T], pos [T,2]."""
        c = self.cfg
        B, G, T, H, dh = x.shape[0], self.G, x.shape[2], c.heads, self.C2
        pos = pos[None, None, :, None]                                            # [1,1,T,1,2]
        is_out = (self.role % 2 == 1)
        kcv = torch.cat([cvf, cvf[:, self.key_grid]], 2)                           # [B,G,2T,10]
        same = torch.einsum('bgtx,bgsx->bgts', cvf, kcv)
        klive = torch.cat([lv, lv[:, self.key_grid] & is_out[None, :, None]], 2)  # second half only for outputs
        eye = torch.cat([torch.eye(T, dtype=torch.bool, device=x.device),
                         torch.zeros(T, T, dtype=torch.bool, device=x.device)], 1)
        keymask = klive[:, :, None, :] | eye[None, None]                           # [B,G,T,2T]
        for li, L in enumerate(self.layers):
            u = x + o
            q, k, v = L['qkv'](u).view(B, G, T, 3, H, dh).unbind(3)
            q, k = self.rope(q, pos), self.rope(k, pos)
            k = torch.cat([k, k[:, self.key_grid]], 2)
            v = torch.cat([v, v[:, self.key_grid]], 2)
            if c.attn_impl == 'sdpa':
                a = self.sdpa(q, k, v, cvf, kcv, klive, li).reshape(B, G, T, -1)
                x = rms_norm(x + L['out'](a))
                gt, up = L['gate_up'](x).chunk(2, -1)
                x = rms_norm(x + L['down'](F.silu(gt) * up))
                continue
            sc = torch.einsum('bgthd,bgshd->bghts', q, k) / math.sqrt(dh)
            sc = sc + self.color_bias[li][None, None, :, None, None] * same[:, :, None]
            sc = sc.masked_fill(~keymask[:, :, None], float('-inf'))
            a = torch.einsum('bghts,bgshd->bgthd', sc.softmax(-1), v).reshape(B, G, T, -1)
            x = rms_norm(x + L['out'](a))
            gt, up = L['gate_up'](x).chunk(2, -1)
            x = rms_norm(x + L['down'](F.silu(gt) * up))
        return x

    def sdpa(self, q, k, v, cvf, kcv, klive, li):
        """Perception attention through scaled_dot_product_attention. Score = q.k / sqrt(dh) + b_h <cv_t, cv_s>
        - 1e4 [key s not live]: the colour bias and the mask are extra q/k channels (q: b_h cv_t and 1; k: cv_s and
        0 or -1e4), so no mask tensor is passed and the flash kernel applies. Rows with no live key (cells outside an
        input grid, never used) get uniform weights instead of the einsum path's self-attention.
        q [B,G,O,H,dh], k, v [B,G,2O,H,dh] -> [B,G,O,H,dh]."""
        B, G, O, H, dh = q.shape
        S = k.shape[2]
        cb = self.color_bias[li].to(q.dtype)
        qc = cb[None, None, None, :, None] * cvf[:, :, :, None, :]                       # [B,G,O,H,10]
        kc = kcv[:, :, :, None, :].expand(B, G, S, H, NC)
        qv = torch.ones(B, G, O, H, 1, dtype=q.dtype, device=q.device)
        kv = torch.where(klive, 0.0, -1e4).to(q.dtype)[:, :, :, None, None].expand(B, G, S, H, 1)
        E = dh + NC + 1
        Ep = -(-E // 8) * 8                                                               # head dim multiple of 8
        pad = lambda t, n: F.pad(t, (0, Ep - n))
        Q = pad(torch.cat([q / math.sqrt(dh), qc, qv], -1), E)
        K = pad(torch.cat([k, kc, kv], -1), E)
        V = pad(v, dh)
        Q, K, V = (t.reshape(B * G, t.shape[2], H, Ep).transpose(1, 2) for t in (Q, K, V))
        if Q.is_cuda and Q.dtype == torch.float32:
            out = F.scaled_dot_product_attention(Q.bfloat16(), K.bfloat16(), V.bfloat16(), scale=1.0).float()
        else:
            out = F.scaled_dot_product_attention(Q, K, V, scale=1.0)
        return out.transpose(1, 2)[..., :dh].reshape(B, G, O, H, dh)

    # ------------------------------------------------------------------ frames (learned masks and their moments)
    def masks(self, h, cv, live, cells=True):
        """Cell masks [B,G,N,N,Kc,N,N] (each cell over its key grid; None if not cells) and grid masks [B,G,N,N,Kg]."""
        c = self.cfg
        B, G, N, C2 = h.shape[0], self.G, c.N, self.C2
        if not cells:
            gs = torch.einsum('kx,bgrcx->bgrck', self.gq, self.gk(h)) / math.sqrt(C2) + self.g_bias
            return None, torch.sigmoid(gs) * live[..., None]
        hk, cvk, lk = h[:, self.key_grid], cv[:, self.key_grid], live[:, self.key_grid]
        pos = self.pos.view(N, N, 2)
        q = self.rope(self.fq(h).view(B, G, N, N, c.Kc, C2), pos[None, None, :, :, None])
        k = self.rope(self.fk(hk).view(B, G, N, N, c.Kc, C2), pos[None, None, :, :, None])
        s = torch.einsum('bgrckx,bguvkx->bgrckuv', q, k) / math.sqrt(C2)
        same = torch.einsum('bgrcx,bguvx->bgrcuv', cv, cvk)
        s = s + self.f_color[:, None, None] * same[:, :, :, :, None] + self.f_bias[:, None, None]
        cell = torch.sigmoid(s) * lk[:, :, None, None, None]
        gs = torch.einsum('kx,bgrcx->bgrck', self.gq, self.gk(h)) / math.sqrt(C2) + self.g_bias
        grid = torch.sigmoid(gs) * live[..., None]
        return cell, grid

    def cell_frames_sdpa(self, h, cv, live, colk):
        """frame_impl 'sdpa': for every cell s and cell frame k, softmax over the cells t of s's key grid of
        q_k(s).k_k(t) / sqrt(C2) (learned RoPE) + f_color_k <cv_s, cv_t> - 1e4 [t not live]; the anchor and the frame
        colours are the weighted averages of t's position and colour, i.e. an attention output with values
        [position, colour one-hot]. Returns cmean [B,G,O,Kc,2], ckap [B,G,O,Kc,10]."""
        c = self.cfg
        B, G, O, C2, Kc = h.shape[0], self.G, self.O, self.C2, c.Kc
        hk = h[:, self.key_grid].reshape(B, G, O, -1)
        lk = live[:, self.key_grid].reshape(B, G, O)
        cvs, cvk = cv.reshape(B, G, O, NC), cv[:, self.key_grid].reshape(B, G, O, NC)
        pos = self.pos[None, None, :, None]
        q = self.rope(self.fq(h.reshape(B, G, O, -1)).view(B, G, O, Kc, C2), pos) / math.sqrt(C2)
        k = self.rope(self.fk(hk).view(B, G, O, Kc, C2), pos)
        qc = self.f_color.to(q.dtype)[None, None, None, :, None] * cvs[:, :, :, None, :]
        kc = cvk[:, :, :, None, :].expand(B, G, O, Kc, NC)
        qv = torch.ones(B, G, O, Kc, 1, dtype=q.dtype, device=q.device)
        kv = torch.where(lk, 0.0, -1e4).to(q.dtype)[:, :, :, None, None].expand(B, G, O, Kc, 1)
        val = torch.cat([self.pos.to(q.dtype)[None, None].expand(B, G, O, 2), colk.to(q.dtype)], -1)
        val = val[:, :, :, None].expand(B, G, O, Kc, 2 + NC)
        E = C2 + NC + 1
        Ep = -(-E // 8) * 8
        pad = lambda t: F.pad(t, (0, Ep - t.shape[-1]))
        Q, K, V = (pad(t).reshape(B * G, O, Kc, Ep).transpose(1, 2)
                   for t in (torch.cat([q, qc, qv], -1), torch.cat([k, kc, kv], -1), val))
        if Q.is_cuda and Q.dtype == torch.float32:
            out = F.scaled_dot_product_attention(Q.bfloat16(), K.bfloat16(), V.bfloat16(), scale=1.0).float()
        else:
            out = F.scaled_dot_product_attention(Q, K, V, scale=1.0)
        out = out.transpose(1, 2).reshape(B, G, O, Kc, Ep)
        return out[..., :2], out[..., 2:2 + NC]

    @staticmethod
    def moments(m, pos, colors):
        """m [..., O] weights over cells with positions pos [O, 2] and colour one-hots colors [..., O, 10]."""
        mass = m.sum(-1)
        w = m / (mass[..., None] + 1e-6)
        mean = w @ pos
        var = (w[..., None] * (pos - mean[..., None, :]).square()).sum(-2)
        kappa = (w[..., None] * colors).sum(-2)
        return mass, mean, var, kappa

    def frames(self, h, cv, live, color1h):
        """Codes U [B,G,O,F,C2] of every cell in every frame, frame colours kappa [B,G,O,F,10], size candidates
        cand [B,G,nc] (meaningful on input grids)."""
        c = self.cfg
        B, G, N, O = h.shape[0], self.G, c.N, self.O
        colk = color1h.reshape(B, G, O, NC)[:, self.key_grid]                    # key-grid colours [B,G,O,10]
        if c.frame_impl == 'sdpa':
            if c.group_average:
                raise ValueError("frame_impl 'sdpa' does not support group_average (anchors are vectors)")
            cmean, ckap = self.cell_frames_sdpa(h, cv, live, colk)
            _, grid = self.masks(h, cv, live, cells=False)
        else:
            cell, grid = self.averaged(lambda a, b, d: self.masks(a, b, d), [h, cv, live], pairwise=(0,))
            cell = cell.reshape(B, G, O, c.Kc, O)
            _, cmean, _, ckap = self.moments(cell, self.pos, colk[:, :, None, None])
        grid = grid.reshape(B, G, O, c.Kg).transpose(2, 3)                       # [B,G,Kg,O]
        gmass, gmean, gvar, gkap = self.moments(grid, self.pos, color1h.reshape(B, G, 1, O, NC))
        # grid frames of an output grid are those of its paired input
        gmass, gmean, gvar, gkap = (t[:, self.key_grid] for t in (gmass, gmean, gvar, gkap))
        gmemb = grid[:, self.key_grid].transpose(2, 3)                           # [B,G,O,Kg] membership (inputs)
        is_in = (self.role % 2 == 0)[None, :, None, None]
        gate = torch.where(is_in, gmemb, torch.ones_like(gmemb))                 # outputs read ungated
        pos = self.pos[None, None]
        U0 = self.code(pos).expand(B, G, O, self.C2)[:, :, :, None]
        Uc = self.code(pos[:, :, :, None] - cmean)
        Ug = self.code(pos[:, :, :, None] - gmean[:, :, None]) * gate[..., None]
        U = torch.cat([U0, Uc, Ug], 3)
        allk = self.moments(live[:, self.key_grid].reshape(B, G, O).to(h.dtype), self.pos.to(h.dtype), colk)[3]
        kap = torch.cat([allk[:, :, None, None].expand(B, G, O, 1, NC), ckap,
                         gkap[:, :, None].expand(B, G, O, c.Kg, NC)], 3)
        hw = self.grid_hw(live)[:, self.key_grid].to(h.dtype)
        cand = torch.cat([torch.ones_like(hw[..., :1]), hw, torch.stack([(gvar[..., 0] + 1e-4).sqrt(), (gvar[..., 1] + 1e-4).sqrt(),
                                                                         gmass], -1).flatten(-2)], -1)
        return U, kap, cand

    @staticmethod
    def grid_hw(live):
        """Extent (h, w) of each grid from its live cells [B,G,N,N]."""
        return torch.stack([live.any(3).sum(2), live.any(2).sum(2)], -1)

    # ------------------------------------------------------------------ rule synapse
    def read(self, st, M, Uo, kap):
        """c_hat [B,P,O,10]: M [B,P,F,10,C2] pair-input memories, Uo [B,P,O,F,C2] output-cell codes,
        kap [B,P,O,F,10] frame colours. Linear in the fast state (A, Pi, Gam, Bh)."""
        AU = torch.einsum('bjde,bpoqe->bjpoqd', st['A'], Uo)                  # spatial terms (colour kept)
        om = self.omega / (self.omega.abs().sum((1, 2), keepdim=True) + 1e-6)  # unit l1 per term: the read's scale
        AU = torch.einsum('bjpoqd,jkq->bpokd', AU, om)                        # is fixed by construction
        out = torch.einsum('bpkcd,bpokd->bpoc', M, AU)
        omc = self.omega_c / (self.omega_c.abs().sum() + 1e-6)
        V = torch.einsum('bpoqd,kq->bpokd', Uo, omc)                          # colour term: slow spatial read,
        out = out + torch.einsum('bcx,bpox->bpoc', st['Pi'], torch.einsum('bpkcd,bpokd->bpoc', M, V))  # fast Pi
        out = out + torch.einsum('bk,bpokc->bpoc', st['Gam'], kap)
        return out + torch.einsum('bcd,bpod->bpoc', st['Bh'], Uo[:, :, :, 0])

    def centre(self, task, op, perm, drop):
        """Prior centre per lane: decoder(e_task) + shared d0 (symmetrised), conjugated into the lane's version."""
        B = task.shape[0]
        keep = (~drop) & torch.full_like(drop, self.cfg.use_emb)
        e = self.task_emb[task] * keep[:, None].to(self.task_emb.dtype)
        flat = self.dec(e)
        st, i = {}, 0
        for k, shp in self.shapes.items():
            st[k] = flat[:, i:i + self.sizes[k]].view(B, *shp) + self.shared_prior(k)[None]
            i += self.sizes[k]
        return self.conjugate(st, op, perm)

    def shared_prior(self, k):
        """d0 projected on the augmentation-invariant subspace (the prior of an unseen task)."""
        x = self.d0[k]
        if k == 'A':
            return sum(x[:, self.code_perm[o]][:, :, self.code_perm[o]] for o in range(8)) / 8
        if k == 'Pi':
            bg_bg, bg_r, r_bg = x[0, 0], x[0, 1:].mean(), x[1:, 0].mean()
            eye = torch.eye(NC - 1, device=x.device)
            diag = (x[1:, 1:] * eye).sum() / (NC - 1)
            off = (x[1:, 1:] * (1 - eye)).sum() / ((NC - 1) * (NC - 2))
            y = torch.zeros_like(x)
            y[0, 0], y[0, 1:], y[1:, 0] = bg_bg, bg_r, r_bg
            y[1:, 1:] = diag * eye + off * (1 - eye)
            return y
        if k == 'Bh':
            return torch.zeros_like(x)
        if k == 'beta':
            return (x + self.swap_beta(x[None])[0]) / 2
        return x

    def swap_beta(self, b):
        """beta [B, 2, nc] under an axis swap: rows swap, candidate pairs (h_in, w_in), (std_r, std_c) swap."""
        idx = [0, 2, 1] + [j for g in range(self.cfg.Kg) for j in (3 + 3 * g + 1, 3 + 3 * g, 3 + 3 * g + 2)]
        return b.flip(1)[:, :, idx]

    def conjugate(self, st, op, perm):
        """Fast state of version (op, colour perm) from that of the original (proposition 3)."""
        B = op.shape[0]
        P = self.code_perm[op]                                                  # [B, C2]
        S = F.one_hot(perm, NC).to(st['Pi'].dtype).transpose(1, 2)                          # S[new, old]
        A = st['A']
        A = torch.gather(A, 2, P[:, None, :, None].expand_as(A))
        A = torch.gather(A, 3, P[:, None, None, :].expand_as(A))
        Q = F.one_hot(P, self.C2).to(st['A'].dtype)                                        # code(Mx) = Q code(x)
        swap = torch.tensor([((k % 4) % 2 == 1) != (k >= 4) for k in range(8)], device=op.device)[op]
        beta = torch.where(swap[:, None, None], self.swap_beta(st['beta']), st['beta'])
        return dict(A=A, Pi=S @ st['Pi'] @ S.transpose(-1, -2), Gam=st['Gam'], Bh=S @ st['Bh'] @ Q.transpose(1, 2),
                    beta=beta)

    def prox_step(self, st, ctr, step, M, Uo, kap, y, ymask, grad):
        """One proximal-gradient step of the lasso over all fast blocks at once:
        mean_demo-cells 0.5 |y - read(st)|^2 + sum_k tau_k |st_k - ctr_k|_1 (per lane). The read is linear in st and
        its scale is fixed by construction (unit codes, mean memories, l1-normalised frame selection, mean over
        cells), so the step size is a learned constant per recurrence step (LISTA) instead of an estimated 1 / L."""
        with torch.enable_grad():
            xl = {k: st[k].detach().requires_grad_(True) for k in FAST}
            r = self.read(xl, M, Uo, kap)
            n = ymask.flatten(1).sum(1).clamp_min(1)
            loss = (0.5 * ((y - r).square() * ymask[..., None]).flatten(1).sum(1) / n).sum()
            gs = torch.autograd.grad(loss, [xl[k] for k in FAST], create_graph=grad)
        eta = self.cfg.eta_max * torch.sigmoid(self.eta_raw[min(step, len(self.eta_raw) - 1)])   # extrapolated
                                                                       # steps reuse the last learned step size
        new = dict(st)
        for k, g in zip(FAST, gs):
            z = st[k] - eta * g - ctr[k]
            new[k] = ctr[k] + torch.sign(z) * F.relu(z.abs() - eta * F.softplus(self.tau[k]))
            if not grad:
                new[k] = new[k].detach()
        return new

    def size_step(self, beta, ctr, cand, target, tmask, grad):
        """Proximal step for the size rule: target [B,P,2] = beta [B,2,nc] cand [B,P,nc] on demonstrations."""
        w = tmask[..., None].to(cand.dtype)
        Cm = (cand * w).transpose(1, 2) @ cand                                   # [B, nc, nc]
        L = torch.diagonal(Cm.detach(), dim1=1, dim2=2).sum(-1).clamp_min(1e-6)[:, None, None]   # trace >= lambda_max
        r = target - cand @ beta.transpose(1, 2)                                 # [B, P, 2]
        g = -((r * w).transpose(1, 2) @ cand)                                    # [B, 2, nc]
        z = beta - (0.9 / L) * g - ctr
        tau = F.softplus(self.tau['beta'])
        out = ctr + torch.sign(z) * F.relu(z.abs() - (0.9 / L) * tau)
        return out if grad else out.detach()

    # ------------------------------------------------------------------ forward
    def forward(self, batch, train=True, steps=None):
        """steps: recurrence length (default cfg.steps; longer at inference = extrapolation)."""
        c = self.cfg
        n_steps = steps or c.steps
        color = batch['color']
        B, G, N, O, P = color.shape[0], self.G, c.N, self.O, self.P
        dev = color.device
        dt = self.h0.dtype
        live = color >= 0
        col1h = F.one_hot(color.clamp_min(0), NC).to(dt) * live[..., None]
        demo = batch['demo']
        pair_ok = torch.cat([demo, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)          # [B, P]
        live_keys = live.clone()
        live_keys[:, -1] = True                                                  # query output cells exist as keys
        obs = torch.cat([live.to(dt)[..., None], ((color == 0) & live).to(dt)[..., None],
                         F.one_hot(self.role, 4).to(dt)[None, :, None, None].expand(B, G, N, N, 4),
                         F.one_hot(self.pair, c.D + 1).to(dt)[None, :, None, None].expand(B, G, N, N, c.D + 1)], -1)
        obs = self.obs(obs)
        drop = (torch.rand(B, device=dev) < c.emb_dropout) if train else torch.zeros(B, dtype=torch.bool, device=dev)
        ctr = self.centre(batch['task'], batch['op'], batch['perm'], drop)
        st = {k: v.detach() for k, v in ctr.items() if k != 'beta'}
        beta = ctr['beta'].detach()
        h = self.h0.expand(B, G, N, N, c.d).clone()
        cv = col1h.clone()
        ins, outs = torch.tensor(self.in_grids, device=dev), torch.tensor(self.out_grids, device=dev)
        y = col1h[:, outs[:-1]].reshape(B, c.D, O, NC)
        y = torch.cat([y, torch.zeros(B, 1, O, NC, device=dev, dtype=dt)], 1)              # query output: never a target
        ymask = live[:, outs].reshape(B, P, O).to(dt) * pair_ok[..., None].to(dt)
        ymask[:, -1] = 0
        size_t = batch['size'][:, outs].to(dt)
        logits = size = None
        for step in range(n_steps):
            grad = train and step >= n_steps - c.grad_steps
            if grad and step == n_steps - c.grad_steps:
                st = {k: ctr[k] + (v - ctr[k]).detach() for k, v in st.items()}   # re-attach to the centre
                beta = ctr['beta'] + (beta - ctr['beta']).detach()
            with torch.set_grad_enabled(grad):
                h = self.averaged(lambda a, b, d, e: [self.perceive(a, b, d, e)], [h, obs, cv, live_keys])[0]
                U, kap, cand = self.frames(h, cv, live_keys, col1h)
                n_in = live[:, ins].reshape(B, P, O).sum(-1).clamp_min(1).to(dt)
                M = torch.einsum('bpoc,bpokd->bpkcd', col1h[:, ins].reshape(B, P, O, NC), U[:, ins]) \
                    / n_in[:, :, None, None, None]                                  # mean over the input grid
                Uo, kapo = U[:, outs], kap[:, outs]
                st = self.prox_step(st, ctr, step, M, Uo, kapo, y, ymask, grad)
                candp = cand[:, outs]
                beta = self.size_step(beta, ctr['beta'], candp[:, :-1], size_t[:, :-1], pair_ok[:, :-1], grad)
                chat = self.read(st, M, Uo, kapo)                                  # [B, P, O, 10]
                xpair = col1h[:, ins].reshape(B, P, O, NC)
                bg = F.one_hot(torch.zeros((), dtype=torch.long, device=dev), NC).to(dt).expand(B, P, O, NC)
                ho = h[:, outs].reshape(B, P, O, -1)
                w = self.src_w(ho)
                src = torch.stack([chat, xpair, bg], 3)
                src = torch.cat([src, kapo[:, :, :, 1:1 + c.Kc]], 3)              # [B,P,O,3+Kc,10]
                logits = (w[..., None] * src).sum(3) * self.temp
                p = logits.softmax(-1)
                fb = torch.stack([(p * xpair).sum(-1), p[..., 0], p.max(-1).values,
                                  -(p * p.clamp_min(1e-9).log()).sum(-1), (chat * xpair).sum(-1), chat[..., 0],
                                  chat.norm(dim=-1)], -1)
                h = h.clone()
                h[:, outs] = h[:, outs] + self.fb(fb).view(B, P, N, N, -1)
                cv = cv.clone()
                cv[:, -1] = p[:, -1].view(B, N, N, NC)
                size = (candp[:, -1:] @ beta.transpose(1, 2))[:, 0]               # [B, 2]
        return logits[:, -1].view(B, N, N, NC), size, st, beta
