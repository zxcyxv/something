"""Episodes for the cognitive-map rule synapse (lt.cmap, docs/claude_2026-10-21_cognitive_map_rule_synapse.md section 12).

Every grid of an episode sits in an N x N region (N = the store's largest grid side), anchored top-left for the original
version. A version v of task t is one D4 element (applied to the whole region, so a grid lands in another corner) and one
permutation of colors 1..9 (color 0 fixed), both fixed by (seed, t, v); v = 0 is the identity. The whole episode (all
demonstrations and the query) gets the same version, so the rule is conjugated, not changed.

A batch holds G = 2 D + 2 grids per episode in the order (demo input, demo output) x D, query input, query output:
    color [B, G, N, N]   observed color 0..9, -1 outside the grid; the query output is all -1 (nothing observed)
    label [B, N, N]      query output colors, -1 outside its grid
    size  [B, G, 2]      grid extents (h, w) after the version's D4 element; the query output's is the target
    demo  [B, D]         which demonstration slots hold a pair
    task, op [B]         task index, D4 element index (lt.cmap.OPS), perm [B, 10] the color permutation
    pid [B]              puzzle identifier task * (num_aug + 1) + version (URM)
"""
import numpy as np
import torch

from .data import load_arc_ctx


def op_apply(a, k):
    """D4 element k on the last two axes: rot90 k % 4 times, then transpose when k >= 4 (lt.data.ArcTrainDataset)."""
    a = np.rot90(a, k % 4, axes=(-2, -1))
    return np.swapaxes(a, -1, -2) if k >= 4 else a


class CmapEpisodes:
    def __init__(self, path, *, seed=0, num_aug=100, max_demos=8):
        s = load_arc_ctx(path)
        self.cells, self.off, self.hw, self.task = s['cells'], s['off'], s['hw'], s['task']
        self.trainable = np.asarray(s['trainable'], bool)
        self.N = int(max(self.hw.max(), 1))
        self.D, self.nv, self.seed = max_demos, num_aug + 1, seed
        self.n_tasks = int(self.task.max()) + 1
        self.train_pairs = {t: np.flatnonzero((self.task == t) & self.trainable) for t in range(self.n_tasks)}
        self.test_pairs = {t: np.flatnonzero((self.task == t) & ~self.trainable) for t in range(self.n_tasks)}
        self.tasks = np.array([t for t in range(self.n_tasks) if len(self.train_pairs[t]) >= 2])
        self._key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def grids(self, p):
        hi, wi, ho, wo = (int(x) for x in self.hw[p])
        o = int(self.off[p])
        c = self.cells
        return c[o:o + hi * wi].reshape(hi, wi), c[o + hi * wi:o + hi * wi + ho * wo].reshape(ho, wo)

    def version(self, t, v):
        if v == 0:
            return 0, np.arange(10)
        key = np.array([self._key0 ^ np.uint64(0x5A), np.uint64(int(t) * self.nv + int(v))], dtype=np.uint64)
        g = np.random.Generator(np.random.Philox(key=key))
        k = int(g.integers(0, 8))
        return k, np.concatenate([[0], 1 + g.permutation(9)])

    def _region(self, g, k, perm):
        N = self.N
        r = np.full((N, N), -1, np.int64)
        r[:g.shape[0], :g.shape[1]] = perm[g]
        return op_apply(r, k), op_apply(g, k).shape

    def episode(self, t, demos, query, v):
        """Arrays of one episode: demos and query are pair indices."""
        k, perm = self.version(t, v)
        G, N = 2 * self.D + 2, self.N
        color = np.full((G, N, N), -1, np.int64)
        size = np.zeros((G, 2), np.int64)
        demo = np.zeros(self.D, bool)
        for i, p in enumerate(demos[:self.D]):
            for j, g in enumerate(self.grids(p)):
                color[2 * i + j], size[2 * i + j] = self._region(g, k, perm)
            demo[i] = True
        qi, qo = self.grids(query)
        color[-2], size[-2] = self._region(qi, k, perm)
        label, size[-1] = self._region(qo, k, perm)
        return dict(color=color, label=label, size=size, demo=demo, task=t, op=k, perm=perm, pid=t * self.nv + v)

    @staticmethod
    def collate(eps):
        out = {}
        for key in eps[0]:
            out[key] = torch.as_tensor(np.stack([np.asarray(e[key]) for e in eps]))
        return out

    def train_batch(self, batch_size, rng):
        eps = []
        for t in rng.choice(self.tasks, batch_size):
            pairs = rng.permutation(self.train_pairs[int(t)])
            eps.append(self.episode(int(t), pairs[1:], pairs[0], int(rng.integers(self.nv))))
        return self.collate(eps)

    def eval_batches(self, batch_size):
        """Every official test pair, version 0, demonstrations = the task's train pairs in order."""
        items = [(t, p) for t in range(self.n_tasks) for p in self.test_pairs[t]]
        for i in range(0, len(items), batch_size):
            yield self.collate([self.episode(t, self.train_pairs[t], p, 0) for t, p in items[i:i + batch_size]])
