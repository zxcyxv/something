"""Coupled model: bands perception + cmap head, reconstructed on 2026-10-23 from the 10-22 conversation record (the
10-22 code was lost; docs/claude_2026-10-23_coupled_reconstruction.md lists what the record fixes and what was chosen).

A lane is one pair: the input canvas and the output canvas stacked along rows (LT num_grids = 2), C x C each with
C = N + 1 (EOS row and column, lt.data / data/prep_arc1.py conventions: tokens 1 pad, 2..11 colours, 12 EOS). The output
canvas enters as the 'unknown' token 0 everywhere, for demonstrations and the query alike, so the answer never leaks
and the size has to be produced as EOS cells. Lanes never mix in perception (each lane is its own LT sample); lanes of
one task form a group. Training: a group is up to D + 1 train pairs of a task, every lane is a demonstration for the
others and a query for itself (leave-one-out), so a batch of 64 lanes is 64 queries with up to D demonstrations each
(cmap's episodes). Evaluation: a group is the task's first D train pairs (demonstrations) and one test pair (query).

Every block i (one segment of `blocks` blocks, gradients through the last `grad_blocks` only):
    1. perception: the LT bands block (lt.model.LT.block_step, unchanged); injection = token embedding + W_fb r_{i-1}
    2. features: phi(o) = unit(W_phi h(o)) on the output-canvas cells, so tr G = 1 below
    3. write (every write_every blocks): per lane, statistics of the OTHER demonstration lanes of its group,
           G = sum phi phi^T / n,   S = sum onehot(y) phi^T / n          (cells with a label: grid cells and EOS)
       and one proximal step from the previous Theta: Theta <- soft(Theta - s (Theta G - S), s lam), s = 1 (valid
       since lambda_max(G) <= tr G = 1) or learned per block (step_learn: s = 2 sigmoid, initialised at 1)
    4. read: r(o) = Theta phi(o) (13 symbols); a demonstration lane reads its leave-one-out Theta, the query the full one
Output: logits = classifier(h) + tau r. With no demonstrations Theta stays 0 and the model is the bands model.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_arc_ctx
from .model import LT, ModelConfig

VOCAB = 13                    # 0 unknown (output canvas input), 1 pad, 2..11 colours 0..9, 12 EOS
PAD, EOS, UNKNOWN = 1, 12, 0


@dataclass
class CoupledConfig:
    N: int = 10                       # largest grid side; canvas C = N + 1
    D: int = 5                        # demonstration lanes per episode (cmap's max_demos)
    d: int = 512
    heads: int = 8
    blocks: int = 32
    grad_blocks: int = 4
    write_every: int = 1              # 10-22: 2 (cmap's 2 perception layers per write), then 1 (every block)
    phi: int = 64
    lam_init: float = 1e-3
    tau_init: float = 1.0
    step_learn: bool = False          # 10-22 'learned step' variant
    bands: int = 4
    band_periods: tuple = (2.0, 6.0, 20.0, 64.0)
    band_coef_std: float = 0.1
    outer_eta: float = 0.125
    mlp_expansion: float = 4.0
    forward_dtype: str = 'float32'

    @property
    def C(self):
        return self.N + 1


def soft(x, t):
    return torch.sign(x) * F.relu(x.abs() - t)


class Coupled(nn.Module):
    def __init__(self, cfg: CoupledConfig):
        super().__init__()
        self.cfg = c = cfg
        C = c.C
        self.T, self.O = 2 * C * C, C * C
        self.lt = LT(ModelConfig(batch_size=1, seq_len=self.T, grid=C, num_grids=2, vocab_size=VOCAB,
                                 puzzle_emb_ndim=0, hidden_size=c.d, num_heads=c.heads, blocks_per_seg=c.blocks, loops=1,
                                 grad_blocks=c.grad_blocks, mlp_expansion=c.mlp_expansion, bands=c.bands,
                                 band_periods=tuple(c.band_periods), band_coef_std=c.band_coef_std, tag_bound=0.0,
                                 outer_eta=c.outer_eta, outer_eta_learn=True, outer_eta_cap=True, kernel='torch',
                                 forward_dtype=c.forward_dtype))
        self.w_phi = nn.Linear(c.d, c.phi, bias=False)
        self.w_fb = nn.Linear(VOCAB, c.d, bias=False)
        nn.init.zeros_(self.w_fb.weight)                       # the feedback path opens by learning
        self.lam_raw = nn.Parameter(torch.tensor(math.log(math.expm1(c.lam_init))))
        self.tau = nn.Parameter(torch.tensor(c.tau_init))
        if c.step_learn:
            self.step_raw = nn.Parameter(torch.zeros(c.blocks))  # s = 2 sigmoid(raw): 1 at init, < 2 always
        self.block_fn = None                                   # compiled lt.block_step (trainer)

    def project_(self):
        self.lt.block.project_eta_(self.lt.config)

    def features(self, h):
        """phi [B,L,O,phi] of the output-canvas cells (unit length) from h [B*L,T,d]."""
        x = self.w_phi(h[:, self.O:].to(self.w_phi.weight.dtype))
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def statistics(self, phi, y, m, demo, group, ng):
        """Leave-one-out statistics per lane: G [L,phi,phi], S [L,13,phi] of the demonstration lanes of the lane's group
        other than the lane itself. phi [L,O,p], y one-hot [L,O,13], m [L,O] labelled cells, demo [L], group [L]."""
        w = m * demo[:, None]
        Gl = torch.einsum('lo,lop,loq->lpq', w, phi, phi)
        Sl = torch.einsum('lo,loc,lop->lcp', w, y, phi)
        nl = w.sum(-1)
        Gg = Gl.new_zeros(ng, *Gl.shape[1:]).index_add_(0, group, Gl)
        Sg = Sl.new_zeros(ng, *Sl.shape[1:]).index_add_(0, group, Sl)
        Ng = nl.new_zeros(ng).index_add_(0, group, nl)
        n = (Ng[group] - nl).clamp_min(1)[:, None, None]       # own lane removed (a non-demonstration adds nothing)
        return (Gg[group] - Gl) / n, (Sg[group] - Sl) / n

    def step(self, theta, G, S, i):
        s = 2 * torch.sigmoid(self.step_raw[i]) if self.cfg.step_learn else 1.0
        return soft(theta - s * (theta @ G - S), s * F.softplus(self.lam_raw))

    def forward(self, batch, head=True):
        """batch: inputs, labels [L,T] (labels only on output canvases, pad elsewhere), group [L] (0..ng-1), demo [L]
        (lanes whose labels enter their group's statistics). Returns logits [L,O,13] of the output canvases, Theta."""
        c, lt = self.cfg, self.lt
        x, labels, group = batch['inputs'].long(), batch['labels'], batch['group'].long()
        L, T = x.shape
        ng = int(group.max()) + 1
        dev, dt = x.device, self.w_phi.weight.dtype
        demo = batch['demo'].to(dt)
        yl = labels[:, self.O:].long()
        m = (yl != PAD).to(dt)
        y = F.one_hot(yl, VOCAB).to(dt)
        h = lt.init_hidden.expand(L, T, c.d).clone()
        state = lt.initial_state(L, dev, dt, T)
        amp = lt._amp(dev)
        theta = torch.zeros(L, VOCAB, c.phi, device=dev, dtype=dt)
        r = torch.zeros(L, self.O, VOCAB, device=dev, dtype=dt)
        first_grad = c.blocks - c.grad_blocks
        fn = self.block_fn or lt.block_step
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            emb = lt.injection(dict(inputs=x))
        for i in range(c.blocks):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= first_grad):
                if i == first_grad:                            # Theta of the no-grad blocks enters as a constant
                    theta, r = theta.detach(), r.detach()
                inj = emb
                if head:
                    fb = self.w_fb(r)
                    inj = torch.cat([emb[:, :self.O], emb[:, self.O:] + fb.to(emb.dtype)], 1)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                    h, state = fn(h, inj, state)
                if not head:
                    continue
                with torch.autocast('cuda', enabled=False):
                    phi = self.features(h)
                    if (i + 1) % c.write_every == 0:
                        G, S = self.statistics(phi, y, m, demo, group, ng)
                        theta = self.step(theta, G, S, i)
                    r = torch.einsum('lcp,lop->loc', theta, phi)
        with torch.autocast('cuda', enabled=False):
            logits = lt.classifier(h[:, self.O:].to(dt))
            if head:
                logits = logits + self.tau * r
        return logits, theta


# ---------------------------------------------------------------------------------------------------- data
class CoupledEpisodes:
    """Episodes of the in-context pair store (data/prep_arc_ctx.py) as lanes. URM/HRM augmentation (lt.data
    ArcTrainDataset): version v of a task (0 identity, 1..num_aug: a dihedral transform and a permutation of colours
    1..9, fixed by (seed, task, v)) for every pair of the episode, and in training an independent random translation of
    each pair inside the canvas (input and output together). Episodes as lt.cmap_data: the query is one trainable pair,
    the demonstrations up to D of the others; evaluation: every test pair, the task's train pairs in order, version 0."""
    def __init__(self, path, *, seed=0, num_aug=1000, D=5, N=None):
        s = load_arc_ctx(path)
        self.cells, self.off, self.hw, self.task = s['cells'], s['off'], s['hw'], s['task']
        self.trainable = np.asarray(s['trainable'], bool)
        self.N = int(N or self.hw.max())
        self.C, self.D, self.nv, self.seed = self.N + 1, D, num_aug + 1, seed
        self.n_tasks = int(self.task.max()) + 1
        self.train_pairs = {t: np.flatnonzero((self.task == t) & self.trainable) for t in range(self.n_tasks)}
        self.test_pairs = {t: np.flatnonzero((self.task == t) & ~self.trainable) for t in range(self.n_tasks)}
        self.tasks = np.array([t for t in range(self.n_tasks) if len(self.train_pairs[t]) >= 2])
        self._key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def grids(self, p):
        hi, wi, ho, wo = (int(x) for x in self.hw[p])
        o = int(self.off[p])
        return (self.cells[o:o + hi * wi].reshape(hi, wi).astype(np.int64),
                self.cells[o + hi * wi:o + hi * wi + ho * wo].reshape(ho, wo).astype(np.int64))

    def version(self, t, v):
        if v == 0:
            return 0, np.arange(10)
        key = np.array([self._key0, np.uint64(int(t) * self.nv + int(v))], dtype=np.uint64)
        g = np.random.Generator(np.random.Philox(key=key))
        return int(g.integers(0, 8)), np.concatenate([[0], 1 + g.permutation(9)])

    def canvas(self, grid, r, c):
        """Tokens of a C x C canvas: colour + 2 at offset (r, c), EOS on the row and column just outside, pad elsewhere."""
        C = self.C
        out = np.full((C, C), PAD, np.int64)
        h, w = grid.shape
        out[r:r + h, c:c + w] = grid + 2
        if r + h < C:
            out[r + h, c:min(c + w + 1, C)] = EOS
        if c + w < C:
            out[r:min(r + h + 1, C), c + w] = EOS
        return out

    def lane(self, p, k, perm, rng):
        gi, go = self.grids(p)
        gi, go = (np.rot90(perm[g], k % 4) for g in (gi, go))
        if k >= 4:
            gi, go = gi.T, go.T
        r = c = 0
        if rng is not None:                                     # URM: one translation per example
            r = int(rng.integers(0, self.C - max(gi.shape[0], go.shape[0]) + 1))
            c = int(rng.integers(0, self.C - max(gi.shape[1], go.shape[1]) + 1))
        x = np.concatenate([self.canvas(gi, r, c), np.full((self.C, self.C), UNKNOWN, np.int64)]).reshape(-1)
        y = np.concatenate([np.full((self.C, self.C), PAD, np.int64), self.canvas(go, r, c)]).reshape(-1)
        return x, y

    def group(self, t, pairs, v, rng, demo, query):
        """Lanes of one task in version v: (inputs, labels, demo flag, query flag) per pair."""
        k, perm = self.version(t, v)
        return [(*self.lane(p, k, perm, rng), d, q) for p, d, q in zip(pairs, demo, query)]

    @staticmethod
    def collate(groups):
        lanes = [(*x, g) for g, grp in enumerate(groups) for x in grp]
        col = lambda i, dt=None: torch.as_tensor(np.stack([l[i] for l in lanes]) if dt is None
                                                 else np.array([l[i] for l in lanes], dt))
        return dict(inputs=col(0), labels=col(1), demo=col(2, bool), query=col(3, bool), group=col(4, np.int64))

    def train_batch(self, n_lanes, rng):
        """n_lanes lanes: groups of up to D + 1 train pairs of random tasks (the last group cut to fit); every lane is
        a demonstration for its group and a query for itself."""
        groups, n = [], 0
        while n < n_lanes:
            t = int(rng.choice(self.tasks))
            pairs = rng.permutation(self.train_pairs[t])[:min(self.D + 1, n_lanes - n)]
            groups.append(self.group(t, pairs, int(rng.integers(self.nv)), rng, [True] * len(pairs),
                                     [True] * len(pairs)))
            n += len(pairs)
        return self.collate(groups)

    def eval_batches(self, n_groups):
        """Every test pair: the task's first D train pairs (demonstrations) + the test pair (query), version 0."""
        items = [(t, p) for t in range(self.n_tasks) for p in self.test_pairs[t]]
        for i in range(0, len(items), n_groups):
            groups = []
            for t, p in items[i:i + n_groups]:
                demos = list(self.train_pairs[t][:self.D])
                groups.append(self.group(t, demos + [p], 0, None, [True] * len(demos) + [False],
                                         [False] * len(demos) + [True]))
            yield self.collate(groups)


def scores(logits, labels, query):
    """Query lanes: exact (every labelled cell, EOS included, so the size must be right), cell accuracy inside the
    true grid (colour cells only, cmap's criterion) and with EOS cells (counts per batch)."""
    y = labels[query, -logits.shape[1]:]
    pred = logits[query].argmax(-1)
    lab = y != PAD
    grid = lab & (y != EOS)
    ok = pred == y
    return dict(exact=int(((ok | ~lab).all(-1)).sum()), n=len(y), cell=int((ok & grid).sum()), n_cell=int(grid.sum()),
                cell_eos=int((ok & lab).sum()), n_eos=int(lab.sum()))
