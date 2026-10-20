"""Structure-code model of the 4x4 torus toy (Tolman-Eichenbaum style in-context rule inference).

Per sample: k demonstration pairs and a query input; the query output is predicted by inference over the 128 candidate
transformations t, each a transition operator on a position code.

- Position codes G [16, d], one row per cell (row-major, as group.SRC).
- Generator matrices W_gen (rot, flip, down, right): W_gen G_x ~ G_{DEST[GENERATORS[gen], x]}.
- Candidate operator U_t = product of W_gen along word(inverse(t)) in application order, so U_t G_o ~ G_{SRC[t, o]}
  (DEST[inverse(t)] = SRC[t]).
- Memory of a grid: keys = position codes, values = one-hot colours (TEM-t: addressing depends only on positions).
  Read at address a = U_t G_o: softmax_x(tau <G_x, a>) @ onehot(grid), a distribution over the 5 colours.
- Colour map per candidate (background 0 -> 0 fixed), from the demonstrations:
    'sinkhorn' / 'row': Sinkhorn- or row-normalised soft co-occurrence M_t[a, b] = sum_{k,o} read_t[k,o,a] out[k,o,b];
    'perm': posterior over the 24 colour permutations of the task family given t, C_t = E[P_c | t] (Sinkhorn computed
            exactly: the permutations are enumerated, so the posterior runs over all 3,072 (t, c)).
- Score s_t = log-likelihood of the demonstration outputs under (U_t, C_t); posterior softmax(beta s);
  prediction = posterior mixture of the query reads mapped by C_t.

Modes:
  designed      d = 16, G = identity, W_gen = exact permutation matrices; only tau and beta learn (sanity check).
  learned       g0 and the four generators learn; G[(r, c)] = W_down^r W_right^c g0 (g0 normalised to unit length).
                Generators are orthogonal (matrix_exp of a skew-symmetric matrix) or free matrices.
  learned_free  ablation: the same codes (g0, W_down, W_right), memory, voting and colour maps, but the 128 operators
                U_t are independent learned matrices with no generator structure.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import group as G

GEN_NAMES = ('rot', 'flip', 'down', 'right')
NCOL = G.COLORS + 1                                   # 5 colours, 0 = background
MODES = ('designed', 'learned', 'learned_free')
CMAP_METHODS = ('perm', 'sinkhorn', 'row')


def word_table():
    """[128, L] generator indices (into GEN_NAMES) of word(inverse(t)) in application order, padded with -1."""
    words = [[GEN_NAMES.index(g) for g in G.word(G.inverse(t))] for t in range(G.NUM_T)]
    tab = np.full((G.NUM_T, max(len(w) for w in words)), -1, dtype=np.int64)
    for t, w in enumerate(words):
        tab[t, :len(w)] = w
    return tab


def perm_matrices(maps):
    """[n, 16] cell maps m -> [n, 16, 16] matrices P with P[m[x], x] = 1, i.e. P e_x = e_{m[x]}."""
    maps = np.asarray(maps)
    P = np.zeros((len(maps), G.CELLS, G.CELLS), dtype=np.float32)
    P[np.arange(len(maps))[:, None], maps, np.arange(G.CELLS)[None, :]] = 1
    return P


def skew(a):
    return a - a.transpose(-1, -2)


def random_rotation_params(n, d, init, gen):
    """Raw parameters whose matrix_exp(skew(.)) is a random rotation (entries N(0, (init / sqrt d)^2))."""
    return torch.randn(n, d, d, generator=gen) * (init / np.sqrt(d))


class StructureModel(nn.Module):
    def __init__(self, mode='learned', d=16, generators='orthogonal', cmap='perm', sinkhorn_iters=20, cmap_eps=1e-3,
                 floor=1e-6, tau=3.0, beta=0.1, gen_init=1.0, seed=0, compose='code'):
        super().__init__()
        if mode not in MODES or cmap not in CMAP_METHODS or generators not in ('orthogonal', 'free'):
            raise ValueError(f'bad mode/cmap/generators: {mode} {cmap} {generators}')
        if mode == 'designed' and d != G.CELLS:
            raise ValueError('designed mode uses d = 16 (identity codes)')
        if compose not in ('code', 'cell') or (compose == 'cell' and mode != 'learned'):
            raise ValueError("compose is 'code' or 'cell' (cell: learned mode only)")
        self.mode, self.d, self.generators, self.cmap, self.compose = mode, d, generators, cmap, compose
        self.sinkhorn_iters, self.cmap_eps, self.floor = sinkhorn_iters, cmap_eps, floor
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(beta))))
        self.register_buffer('src', torch.as_tensor(G.SRC), persistent=False)
        self.register_buffer('words', torch.as_tensor(word_table()), persistent=False)
        self.register_buffer('perms', F.one_hot(torch.as_tensor(G.CMAPS), NCOL).float(), persistent=False)  # [24,5,5]
        e00 = torch.zeros(NCOL, NCOL)
        e00[0, 0] = 1
        self.register_buffer('e00', e00, persistent=False)
        gen = torch.Generator().manual_seed(seed)
        if mode == 'designed':
            dest = G.DEST[[G.GENERATORS[g] for g in GEN_NAMES]]
            self.register_buffer('gen_fixed', torch.as_tensor(perm_matrices(dest)), persistent=False)
            return
        self.g0 = nn.Parameter(torch.randn(d, generator=gen))
        raw = random_rotation_params(len(GEN_NAMES), d, gen_init, gen)
        self.gen_raw = nn.Parameter(raw if generators == 'orthogonal' else torch.matrix_exp(skew(raw)))
        if mode == 'learned_free':                     # rot/flip generators stay unused; codes use down/right only
            raw = random_rotation_params(G.NUM_T, d, gen_init, gen)
            self.op_raw = nn.Parameter(raw if generators == 'orthogonal' else torch.matrix_exp(skew(raw)))

    # ---------------------------------------------------------------- structure
    @property
    def tau(self):
        return self.log_tau.exp()

    @property
    def beta(self):
        return self.log_beta.exp()

    def set_tau_beta(self, tau=None, beta=None):
        with torch.no_grad():
            if tau is not None:
                self.log_tau.fill_(float(np.log(tau)))
            if beta is not None:
                self.log_beta.fill_(float(np.log(beta)))

    def gen_matrices(self):
        """[4, d, d] generator matrices in GEN_NAMES order."""
        if self.mode == 'designed':
            return self.gen_fixed
        return torch.matrix_exp(skew(self.gen_raw)) if self.generators == 'orthogonal' else self.gen_raw

    def codes(self, W=None):
        """[16, d] position codes, row (r, c) = W_down^r W_right^c g0 (identity in designed mode)."""
        if self.mode == 'designed':
            return torch.eye(G.CELLS, device=self.log_tau.device)
        W = self.gen_matrices() if W is None else W
        w_down, w_right = W[GEN_NAMES.index('down')], W[GEN_NAMES.index('right')]
        row = [self.g0 / self.g0.norm()]
        for _ in range(G.N - 1):
            row.append(w_right @ row[-1])                             # W_right^c g0
        rows = [torch.stack(row)]
        for _ in range(G.N - 1):
            rows.append(rows[-1] @ w_down.T)                          # W_down^r W_right^c g0
        return torch.stack(rows).reshape(G.CELLS, self.d)

    def operators(self, W=None):
        """[128, d, d] candidate operators U_t (U_t G_o should be G_{SRC[t, o]})."""
        if self.mode == 'learned_free':
            return torch.matrix_exp(skew(self.op_raw)) if self.generators == 'orthogonal' else self.op_raw
        W = self.gen_matrices() if W is None else W
        U = torch.eye(self.d, device=W.device, dtype=W.dtype).expand(G.NUM_T, self.d, self.d)
        for j in range(self.words.shape[1]):                          # product along the word, application order
            idx = self.words[:, j]
            U = torch.where((idx >= 0)[:, None, None], W[idx.clamp(min=0)] @ U, U)
        return U

    def structure(self):
        """codes [16, d], operators [128, d, d], addresses U_t G_o [128, 16, d], read weights [128, 16(o), 16(x)].

        compose 'code': read weights softmax_x(tau <G_x, U_t G_o>), the whole word multiplied in code space.
        compose 'cell': clean-up after every generator step. Each generator becomes a soft cell map
        A_g[x, y] = softmax_y(tau <G_y, W_g G_x>) and the word is composed in cell space, M_t = A_{g_1} ... A_{g_L}
        (row o: where output cell o reads), so a step's error is not carried into the next step."""
        W = self.gen_matrices()
        codes, U = self.codes(W), self.operators(W)
        addr = torch.einsum('tde,oe->tod', U, codes)
        if self.compose == 'code':
            weights = (self.tau * torch.einsum('tod,xd->tox', addr, codes)).softmax(-1)
            return codes, U, addr, weights
        A = (self.tau * torch.einsum('gde,xe,yd->gxy', W, codes, codes)).softmax(-1)      # [4, 16(x), 16(y)]
        M = torch.eye(G.CELLS, device=W.device, dtype=W.dtype).expand(G.NUM_T, G.CELLS, G.CELLS)
        for j in range(self.words.shape[1]):
            idx = self.words[:, j]
            M = torch.where((idx >= 0)[:, None, None], M @ A[idx.clamp(min=0)], M)
        return codes, U, addr, M

    def closure_loss(self):
        """Label-free structural prior that transitions move positions to positions: every transition operator
        (the 4 generators in learned mode, all 128 operators in learned_free) maps each position code onto one position
        code (cosine 1 with its best match) and distinct cells onto distinct cells (column sums of the soft assignment
        equal 1). With exact generator permutations every word is the composed permutation, held-out words included."""
        if self.mode == 'designed':
            return torch.zeros((), device=self.log_tau.device)
        W = self.gen_matrices()
        codes = self.codes(W)
        ops = W if self.mode == 'learned' else self.operators(W)
        moved = F.normalize(torch.einsum('gde,xe->gxd', ops, codes), dim=-1)
        cos = torch.einsum('gxd,yd->gxy', moved, codes)
        hit = 1 - cos.max(-1).values
        col = ((self.tau * cos).softmax(-1).sum(1) - 1).square()
        return hit.mean() + col.mean()

    # ---------------------------------------------------------------- colour maps
    def colour_map(self, M):
        """Soft co-occurrence M [..., 5, 5] -> row-stochastic colour map [..., 5, 5] with background 0 -> 0 fixed."""
        block = M[..., 1:, 1:] + self.cmap_eps
        if self.cmap == 'row':
            block = block / block.sum(-1, keepdim=True)
        else:
            lk = block.log()
            for _ in range(self.sinkhorn_iters):
                lk = lk - lk.logsumexp(-2, keepdim=True)              # columns
                lk = lk - lk.logsumexp(-1, keepdim=True)              # rows last: C is row-stochastic
            block = lk.exp()
        return F.pad(block, (1, 0, 1, 0)) + self.e00

    # ---------------------------------------------------------------- inference
    def forward(self, demo_in, demo_out, demo_mask, query_in):
        """demo_in, demo_out [B, K, 4, 4] or [B, K, 16] colours 0..4; demo_mask [B, K] bool (which demos exist);
        query_in [B, 4, 4] or [B, 16]. Returns probs [B, 16, 5], log_post [B, 128] over t, scores [B, 128]."""
        B, K = demo_in.shape[:2]
        weights = self.structure()[3]
        X = F.one_hot(demo_in.reshape(B, K, G.CELLS).long(), NCOL).float()
        Y = F.one_hot(demo_out.reshape(B, K, G.CELLS).long(), NCOL).float() * demo_mask.float()[:, :, None, None]
        Xq = F.one_hot(query_in.reshape(B, G.CELLS).long(), NCOL).float()
        reads = torch.einsum('tox,bkxa->bktoa', weights, X)                       # [B, K, T, 16, 5]
        query = torch.einsum('tox,bxa->btoa', weights, Xq)                        # [B, T, 16, 5]
        beta = self.beta
        if self.cmap == 'perm':
            # s[t, c] = sum_{k,o} log read_t[k, o, c^-1(out)]  (exact: P_c is a permutation)
            target = torch.einsum('bkoy,cay->bkoca', Y, self.perms)               # [B, K, 16, 24, 5]
            s_tc = torch.einsum('bktoa,bkoca->btc', torch.log(reads + self.floor), target)
            logits = beta * s_tc
            log_post_tc = logits.reshape(B, -1).log_softmax(-1).reshape(logits.shape)
            cmaps = torch.einsum('btc,cay->btay', log_post_tc.exp(), self.perms)  # pi_t * E[P_c | t]
            probs = torch.einsum('btoa,btay->boy', query, cmaps)
            return dict(probs=probs, log_post=log_post_tc.logsumexp(-1), scores=logits.logsumexp(-1) / beta)
        M = torch.einsum('bktoa,bkoy->btay', reads, Y)                            # soft co-occurrence [B, T, 5, 5]
        C = self.colour_map(M)
        lik = (torch.einsum('bktoa,btay->bktoy', reads, C) * Y[:, :, None]).sum(-1)   # [B, K, T, 16]
        scores = (torch.log(lik + self.floor) * demo_mask.float()[:, :, None, None]).sum((1, 3))
        log_post = (beta * scores).log_softmax(-1)
        probs = torch.einsum('bt,btoa,btay->boy', log_post.exp(), query, C)
        return dict(probs=probs, log_post=log_post, scores=scores)


def nll(probs, target, eps=1e-9):
    """Mean per-cell negative log-likelihood of target [B, 16] (or [B, 4, 4]) under probs [B, 16, 5]."""
    p = probs.gather(-1, target.reshape(len(probs), G.CELLS, 1).long()).squeeze(-1)
    return -(p + eps).log().mean()


@torch.no_grad()
def diagnostics(model, heldout_t=()):
    """Learned-structure checks: relation errors on the code orbit, read exactness by train / held-out t, code
    orthogonality, read margins."""
    W = model.gen_matrices()
    codes, U, addr, weights = model.structure()
    d = codes.shape[1]
    eye = torch.eye(d, device=codes.device)
    norms = codes.norm(dim=1)

    def orbit(R):                                     # mean_x |R G_x| / |G_x|
        return float(((codes @ R.T).norm(dim=1) / norms).mean())

    def full(R):                                      # |R|_F / sqrt(d)
        return float(R.norm() / np.sqrt(d))

    mp = torch.linalg.matrix_power
    w = {name: W[i] for i, name in enumerate(GEN_NAMES)}
    rels = dict(down4=mp(w['down'], 4) - eye, right4=mp(w['right'], 4) - eye,
                comm_right_down=w['right'] @ w['down'] - w['down'] @ w['right'])
    if model.mode != 'learned_free':
        rels.update(rot4=mp(w['rot'], 4) - eye, flip2=mp(w['flip'], 2) - eye)
    out = dict(mode=model.mode, d=d, tau=float(model.tau), beta=float(model.beta),
               relation_orbit_err={k: orbit(v) for k, v in rels.items()},
               relation_full_err={k: full(v) for k, v in rels.items()})
    if model.mode != 'designed' and model.generators == 'orthogonal':
        out['code_norm'] = float(norms.mean())
    sim = torch.einsum('tod,xd->tox', addr, codes)                             # [T, 16, 16]
    if getattr(model, 'compose', 'code') == 'cell':                            # the reads actually used: log M_t
        sim = weights.clamp_min(1e-30).log()
    src = model.src
    hit = sim.argmax(-1) == src
    true_sim = sim.gather(-1, src[..., None]).squeeze(-1)
    other = sim.masked_fill(F.one_hot(src, G.CELLS).bool(), -float('inf')).max(-1).values
    margin = (true_sim - other) / norms.pow(2).mean()                          # read margin in units of |G|^2
    cos = F.cosine_similarity(addr, codes[src], dim=-1)                        # U_t G_o vs G_{SRC[t, o]}
    held = torch.zeros(G.NUM_T, dtype=torch.bool, device=codes.device)
    held[list(heldout_t)] = True
    for name, sel in (('all', torch.ones_like(held)), ('train_t', ~held), ('heldout_t', held)):
        if sel.any():
            out[f'read_exact_{name}'] = float(hit[sel].float().mean())
            out[f'op_exact_{name}'] = float(hit[sel].all(1).float().mean())   # fraction of t with all 16 reads right
            out[f'addr_cos_{name}'] = float(cos[sel].mean())
            out[f'margin_min_{name}'] = float(margin[sel].min())
            out[f'margin_mean_{name}'] = float(margin[sel].mean())
    # Up to relabelling: which group element does each candidate's read map (o -> argmax_x) realise?
    read_map = sim.argmax(-1)                                                  # [T, 16]
    match = (read_map[:, None, :] == src[None, :, :]).all(-1)                  # [T, T']: read map of t is SRC[t']
    realised = torch.where(match.any(1), match.float().argmax(1), torch.full_like(read_map[:, 0], -1))
    valid = realised >= 0
    out['realised'] = realised.tolist()                                        # element realised by candidate t
    out['realised_valid'] = float(valid.float().mean())                        # read map is a group element
    out['realised_distinct'] = int(realised[valid].unique().numel())
    out['realised_label_exact'] = float((realised == torch.arange(G.NUM_T, device=src.device)).float().mean())
    if len(heldout_t):
        out['heldout_elements_realised'] = float(np.isin(np.asarray(heldout_t), realised.cpu().numpy()).mean())
    top2 = sim.topk(2, -1).values
    out['relabel_margin_min'] = float(((top2[..., 0] - top2[..., 1]) / norms.pow(2).mean()).min())
    dest_rows = {tuple(r): i for i, r in enumerate(G.DEST)}
    gen_action = {}
    for i, name in enumerate(GEN_NAMES):
        if model.mode == 'learned_free' and name in ('rot', 'flip'):
            continue
        img = codes @ W[i].T                                                   # W_gen G_x
        near = (img @ codes.T).argmax(1)
        e = dest_rows.get(tuple(near.tolist()), -1)                            # element whose DEST is the code map
        gen_action[name] = dict(element=e, decoded=list(G.decode_t(e)) if e >= 0 else None,
                                true=G.GENERATORS[name],
                                cos_min=float(F.cosine_similarity(img, codes[near], dim=-1).min()))
    out['generator_action'] = gen_action
    unit = codes / norms[:, None]
    gram = unit @ unit.T
    off = gram[~torch.eye(G.CELLS, dtype=torch.bool, device=codes.device)]
    sv = torch.linalg.svdvals(unit)
    p = sv.pow(2) / sv.pow(2).sum()
    out.update(code_cos_abs_mean=float(off.abs().mean()), code_cos_max=float(off.max()),
               code_eff_rank=float(torch.exp(-(p * (p + 1e-12).log()).sum())))
    return out
