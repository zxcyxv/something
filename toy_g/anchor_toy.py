"""Anchor-rule toy on the 4x4 torus (docs/claude_2026-10-20_arc_logic_foundations.md section 9, definition 4):
content-dependent addresses a(o; x) = A_theta(x)(o) with theta = the position of a marker.

Colours 0 (background), 1..4, 5 = marker (exactly one cell of every input). Tasks:
    fixed     (t, c)          out = CMAP6[c][ in at SRC[t] ]                                   3,072 (version-C toy)
    anchored  (p, tau, c)     D_p, then the translation that puts the marker at cell tau, then CMAP6[c]   3,072
CMAP6[c] permutes 1..4 as group.CMAPS[c] and fixes 0 and 5. Task id = kind * 3072 + rule * 24 + c, rule = t (fixed)
or 16 p + tau (anchored). An anchored rule is the fixed transformation T_ANC[rule, m] for an input whose marker is at
cell m. A sample is kept only if its query output is the same under every one of the 6,144 tasks consistent with its
demonstrations (so an ideal in-context learner is exact).
Split: fixed as data.make_split (16 held-out transformations -> group 2, 10% of the other pairs -> group 1); anchored
the same with 16 held-out rules (p, tau), two per point operation, 16 distinct targets.

    python -m toy_g.anchor_toy      # writes data/anchor_instances.npz (one determined instance per task, 3 demos)
"""
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import group as G

NCOL6 = 6
MARKER = 5
NUM_TASKS = 2 * G.NUM_TASKS
CMAP6 = np.concatenate([G.CMAPS, np.full((G.NUM_C, 1), MARKER)], 1)                       # [24, 6]
DP = np.stack([G.DEST[G.encode_t(p, 0, 0)] for p in range(8)])                            # [8, 16]: cell i -> D_p(i)


def _anchored_table():
    """T_ANC[16 p + tau, m]: the transformation an anchored rule applies to an input whose marker is at cell m."""
    tab = np.zeros((128, G.CELLS), dtype=np.int64)
    for p in range(8):
        for tau in range(G.CELLS):
            for m in range(G.CELLS):
                r, c = divmod(int(DP[p, m]), G.N)
                tr, tc = divmod(tau, G.N)
                tab[16 * p + tau, m] = G.encode_t(p, (tr - r) % G.N, (tc - c) % G.N)
    return tab


T_ANC = _anchored_table()


def apply_numpy(grid, task):
    """Reference: one flat grid [16], one task id -> output [16]."""
    kind, rest = divmod(int(task), G.NUM_TASKS)
    rule, c = divmod(rest, G.NUM_C)
    t = rule if kind == 0 else T_ANC[rule, int(np.flatnonzero(grid == MARKER)[0])]
    return CMAP6[c][grid[G.SRC[t]]]


def make_split(seed=0, combo_frac=0.1):
    fixed = D.make_split(seed)
    rng = np.random.default_rng(seed + 7)
    taus = rng.permutation(G.CELLS)
    held = sorted(16 * p + int(taus[2 * p + j]) for p in range(8) for j in range(2))
    rest = np.array([G.NUM_TASKS + r * G.NUM_C + c for r in range(128) if r not in held for c in range(G.NUM_C)])
    combo = np.sort(rest[rng.permutation(len(rest))[:int(round(combo_frac * len(rest)))]])
    ht = np.array([G.NUM_TASKS + r * G.NUM_C + c for r in held for c in range(G.NUM_C)])
    group = np.concatenate([fixed['group'], np.zeros(G.NUM_TASKS, dtype=np.int64)])
    group[combo], group[ht] = 1, 2
    train = np.flatnonzero(group == 0)
    return dict(train=train, group=group, heldout_fixed=fixed['heldout_t_elements'], heldout_anchored=np.array(held))


class AnchorSampler:
    """Fresh determined samples on a torch device. Consistency through pair sets as train_g.TorchSampler: a task maps a
    demonstration to its output iff every (moved colour, output colour) pair under its transformation is a pair of its
    colour map; an anchored rule uses, for each demonstration, the transformation fixed by that input's marker."""

    def __init__(self, tasks, device='cpu', seed=0):
        self.device = torch.device(device)
        self.tasks = torch.as_tensor(np.asarray(tasks), device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.src = torch.as_tensor(G.SRC, device=self.device)
        self.cmap = torch.as_tensor(CMAP6, device=self.device)
        self.tanc = torch.as_tensor(T_ANC, device=self.device)
        pairs = torch.nn.functional.one_hot(self.cmap, NCOL6).reshape(G.NUM_C, NCOL6 * NCOL6)
        self.not_pairs = (1 - pairs).float().T.contiguous()                                  # [36, 24]

    def _randint(self, lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=self.gen, device=self.device)

    def random_inputs(self, n, sparsity=(3, 8)):
        """[n, 16]: lo..hi coloured cells (colours 1..4) and one marker cell (5) on another cell."""
        lo, hi = sparsity
        counts = self._randint(lo, hi + 1, (n,))
        order = torch.rand(n, G.CELLS, generator=self.gen, device=self.device).argsort(1)
        colours = self._randint(1, G.COLORS + 1, (n, G.CELLS))
        rank = torch.arange(G.CELLS, device=self.device)[None]
        values = torch.where(rank < counts[:, None], colours, 0)
        values = torch.where(rank == G.CELLS - 1, MARKER, values)                          # last cell in the order
        return torch.zeros(n, G.CELLS, dtype=torch.long, device=self.device).scatter_(1, order, values)

    def transformation(self, grids, task):
        """grids [n, m, 16], task [n] -> transformation index per grid [n, m]."""
        kind, rest = task // G.NUM_TASKS, task % G.NUM_TASKS
        rule = rest // G.NUM_C
        marker = (grids == MARKER).float().argmax(-1)                                       # [n, m]
        anc = self.tanc[rule[:, None].clamp(max=127), marker]
        return torch.where(kind[:, None] == 0, rule[:, None].expand_as(marker), anc)

    def apply(self, grids, task):
        t = self.transformation(grids, task)
        moved = grids.gather(-1, self.src[t])
        c = task % G.NUM_C
        return self.cmap[c][:, None, :].expand(-1, grids.shape[1], -1).gather(-1, moved)

    def _present(self, grids, outs):
        """[n, m, 128, 36]: pair types present under every transformation, per grid."""
        moved = grids[:, :, self.src]                                                       # [n, m, 128, 16]
        idx = moved * NCOL6 + outs[:, :, None, :]
        return torch.zeros(*moved.shape[:3], NCOL6 * NCOL6, device=self.device).scatter_(3, idx, 1.0)

    def _per_task(self, present, grids):
        """present [n, m, 128, 36] -> pairs over the m grids under each fixed rule and each anchored rule [n, 2, 128, 36]."""
        n, m = present.shape[:2]
        fixed = present.amax(1)
        marker = (grids == MARKER).float().argmax(-1)                                       # [n, m]
        t = self.tanc[:, marker].permute(1, 2, 0)                                           # [n, m, 128 rules]
        anc = present.gather(2, t[..., None].expand(-1, -1, -1, NCOL6 * NCOL6)).amax(1)
        return torch.stack([fixed, anc], 1)

    def consistent(self, demo_in, demo_out):
        """[n, 6144] bool."""
        v = self._per_task(self._present(demo_in, demo_out), demo_in) @ self.not_pairs     # [n, 2, 128, 24]
        return (v == 0).reshape(len(demo_in), NUM_TASKS)

    def determined(self, demo_in, demo_out, query_in, query_out, chunk=1024):
        res = []
        for s in range(0, len(query_in), chunk):
            e = s + chunk
            cons = self.consistent(demo_in[s:e], demo_out[s:e])
            q, qo = query_in[s:e, None], query_out[s:e, None]
            bad = (self._per_task(self._present(q, qo), q) @ self.not_pairs > 0).reshape(len(q), NUM_TASKS)
            res.append(~(cons & bad).any(1))
        return torch.cat(res)

    def sample(self, n, k, sparsity=(3, 8), max_rounds=200):
        keep, have = [], 0
        for _ in range(max_rounds):
            m = max(2 * (n - have), 256)
            task = self.tasks[self._randint(0, len(self.tasks), (m,))]
            din = self.random_inputs(m * k, sparsity).reshape(m, k, G.CELLS)
            qin = self.random_inputs(m, sparsity)[:, None]
            dout, qout = self.apply(din, task), self.apply(qin, task)
            ok = self.determined(din, dout, qin[:, 0], qout[:, 0])
            keep.append((din[ok], dout[ok], qin[ok, 0], qout[ok, 0], task[ok]))
            have += int(ok.sum())
            if have >= n:
                break
        else:
            raise RuntimeError('could not draw enough determined samples')
        din, dout, qin, qout, task = (torch.cat([x[i] for x in keep])[:n] for i in range(5))
        return dict(demo_in=din, demo_out=dout, query_in=qin, query_out=qout, task=task)

    def batch(self, n, ks=(1, 2, 3), sparsity=(3, 8)):
        """A training batch with a random number of demonstrations per sample (padded to max(ks), masked)."""
        kmax = max(ks)
        k = int(ks[int(self._randint(0, len(ks), (1,)))])
        s = self.sample(n, k, sparsity)
        pad = kmax - k
        if pad:
            z = torch.zeros(n, pad, G.CELLS, dtype=torch.long, device=self.device)
            s['demo_in'], s['demo_out'] = torch.cat([s['demo_in'], z], 1), torch.cat([s['demo_out'], z], 1)
        s['demo_mask'] = torch.arange(kmax, device=self.device)[None].expand(n, -1) < k
        return s


def instances(seed=0, n_demo=3, device='cuda'):
    """One instance per task (6,144): n_demo demonstrations and a test pair determined by them; determined_k[:, k-1]
    says whether the first k demonstrations already determine it."""
    S = AnchorSampler(np.arange(NUM_TASKS), device, seed=seed + 1000)
    got = {}
    need = torch.arange(NUM_TASKS, device=S.device)
    while len(need):
        task = need.repeat_interleave(4)
        m = len(task)
        din = S.random_inputs(m * n_demo).reshape(m, n_demo, G.CELLS)
        qin = S.random_inputs(m)[:, None]
        dout, qout = S.apply(din, task), S.apply(qin, task)
        ok = S.determined(din, dout, qin[:, 0], qout[:, 0]).cpu().numpy()
        for i in np.flatnonzero(ok):
            got.setdefault(int(task[i]), (din[i], dout[i], qin[i, 0], qout[i, 0]))
        need = torch.as_tensor([t for t in range(NUM_TASKS) if t not in got], dtype=torch.long, device=S.device)
    out = {key: torch.stack([got[t][j] for t in range(NUM_TASKS)]).cpu().numpy().astype(np.uint8)
           for j, key in enumerate(('demo_in', 'demo_out', 'test_in', 'test_out'))}
    det = np.zeros((NUM_TASKS, n_demo), dtype=bool)
    for k in range(1, n_demo + 1):
        det[:, k - 1] = S.determined(*(torch.as_tensor(out[key].astype(np.int64), device=S.device)[:, :k] if key.startswith(
            'demo') else torch.as_tensor(out[key].astype(np.int64), device=S.device) for key in
            ('demo_in', 'demo_out', 'test_in', 'test_out'))).cpu().numpy()
    out['determined_k'] = det
    out['task'] = np.arange(NUM_TASKS)
    return out


INSTANCES = Path(__file__).resolve().parents[1] / 'data' / 'anchor_instances.npz'


def load_instances(path=INSTANCES):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


if __name__ == '__main__':
    inst = instances()
    split = make_split()
    np.savez_compressed(INSTANCES, group=split['group'], **inst)
    det = inst['determined_k']
    print(f'wrote {INSTANCES}: {NUM_TASKS} tasks; determined by the first k = 1, 2, 3 demonstrations: '
          f'{det[:, 0].mean():.3f} {det[:, 1].mean():.3f} {det[:, 2].mean():.3f}; groups '
          f'{np.bincount(split["group"]).tolist()}')
