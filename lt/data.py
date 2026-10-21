"""Sudoku-Extreme 1k data: checked loading, the fixed augmentation pool and the URM batch stream.

The NPZ holds raw digits (train_inputs/test_inputs 0..n with 0 = blank, labels 1..n); tokens are digit + 1. The board
size n (a square: 4 or 9) is read from the arrays; 9 is the Sudoku-Extreme data, smaller boards are toy sets.
Training puzzle g has 1 + num_aug versions: the original and num_aug augmentations (digit permutation, transpose,
band/stack and row/column-within-band permutations) drawn from a Philox stream keyed by (seed, g, index), so the pool
is fixed by the seed. Every iterator covers eval_interval epochs; each epoch visits every puzzle once in a random
order and draws one of its versions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

def _box(n):
    b = int(round(n ** 0.5))
    if b * b != n:
        raise ValueError(f'board size {n} is not a square')
    return b


def _draw_aug_params(rng, n=9):
    """Draw order (digits, transpose, bands, rows in each band, stacks, columns in each stack) is the historical 9x9 one."""
    b = _box(n)
    digits = np.pad(rng.permutation(np.arange(1, n + 1)), (1, 0))
    trans = rng.random() < 0.5
    bands = rng.permutation(b)
    rows = np.concatenate([i * b + rng.permutation(b) for i in bands])
    stacks = rng.permutation(b)
    cols = np.concatenate([s * b + rng.permutation(b) for s in stacks])
    return digits, trans, rows, cols


def _apply_aug(x, digits, trans, rows, cols):
    n = x.shape[-1]
    cells = np.arange(n * n)
    mapping = rows[cells // n] * n + cols[cells % n]
    if trans:
        x = x.T
    return digits[x.flatten()[mapping].reshape(n, n)]


class SudokuTrainDataset(IterableDataset):
    """Yields (iterator index, batch) for this rank's share of each global batch. skip_batches resumes mid-iterator."""
    def __init__(self, inputs, labels, *, seed, num_aug, global_batch_size, rank, world_size,
                 epochs_per_iter, start_iter, total_iters, skip_batches=0, ids=None):
        super().__init__()
        self.inputs, self.labels = inputs, labels
        self.ids = ids                                   # task identifier per puzzle (None: every puzzle has 0)
        self.seed, self.num_aug = seed, num_aug
        self.gbs, self.rank, self.world_size = global_batch_size, rank, world_size
        self.local_bs = global_batch_size // world_size
        self.epochs_per_iter, self.start_iter = epochs_per_iter, start_iter
        self.total_iters, self.skip_batches = total_iters, skip_batches
        self.n_groups, self.gsize = len(inputs), 1 + num_aug
        self._aug_key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def _augmented(self, gid, aug_idx):
        if aug_idx == 0:
            return self.inputs[gid], self.labels[gid]
        key = np.array([self._aug_key0, np.uint64(gid * self.gsize + aug_idx)], dtype=np.uint64)
        prm = _draw_aug_params(np.random.Generator(np.random.Philox(key=key)), self.inputs.shape[-1])
        return _apply_aug(self.inputs[gid], *prm), _apply_aug(self.labels[gid], *prm)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        if wi is not None and wi.num_workers != 1:
            raise RuntimeError('dataloader_workers must be 0 or 1.')
        skip = self.skip_batches
        for it in range(self.start_iter, self.total_iters):
            rng = np.random.Generator(np.random.Philox(seed=self.seed + it + 1))
            groups = np.concatenate([rng.permutation(self.n_groups) for _ in range(self.epochs_per_iter)])
            start = 0
            while start < len(groups):
                gsel, asel = [], []
                while start < len(groups) and len(gsel) < self.gbs:
                    g = int(groups[start])
                    pid = int(rng.integers(g * self.gsize, (g + 1) * self.gsize))
                    start += 1
                    rng.choice(1, 1, replace=False)          # keeps the historical RNG stream
                    gsel.append(g)
                    asel.append(pid - g * self.gsize)
                if len(gsel) < self.gbs:
                    break
                if skip:
                    skip -= 1
                    continue
                off = self.rank * self.local_bs
                inp = np.empty((self.local_bs, self.inputs[0].size), dtype=np.int32)
                lab = np.empty_like(inp)
                for j in range(self.local_bs):
                    x, y = self._augmented(gsel[off + j], asel[off + j])
                    inp[j], lab[j] = x.reshape(-1) + 1, y.reshape(-1) + 1
                pid = (torch.zeros(self.local_bs, dtype=torch.int32) if self.ids is None else
                       torch.from_numpy(self.ids[np.asarray(gsel[off:off + self.local_bs])].astype(np.int32)))
                yield it, dict(inputs=torch.from_numpy(inp), labels=torch.from_numpy(lab), puzzle_identifiers=pid)


ARC_N = 30


def _arc_canvas(grid, h, w, r, c):
    """Raw canvas (see data/prep_arc1.py): grid values (color + 1) at offset (r, c), EOS 11 on the row and column
    just outside, 0 elsewhere."""
    out = np.zeros((ARC_N, ARC_N), dtype=np.uint8)
    out[r:r + h, c:c + w] = grid
    if r + h < ARC_N:
        out[r + h, c:min(c + w + 1, ARC_N)] = 11
    if c + w < ARC_N:
        out[r:min(r + h + 1, ARC_N), c + w] = 11
    return out


class ArcTrainDataset(IterableDataset):
    """URM/HRM-style ARC stream with on-the-fly augmentation. Every epoch visits every demonstration pair once; each
    visit draws a version v of its task (0 = identity, 1..num_aug: dihedral transform and a permutation of colors 1..9,
    fixed by (seed, task, v)) and a random translation; the puzzle identifier is task * (num_aug + 1) + v."""
    def __init__(self, tin, tout, hw, task, *, seed, num_aug, global_batch_size, rank, world_size, epochs_per_iter,
                 start_iter, total_iters, skip_batches=0):
        super().__init__()
        self.tin, self.tout, self.hw, self.task = tin, tout, hw, task
        self.seed, self.nv = seed, num_aug + 1
        self.gbs, self.rank, self.local_bs = global_batch_size, rank, global_batch_size // world_size
        self.epochs_per_iter, self.start_iter, self.total_iters, self.skip_batches = (epochs_per_iter, start_iter,
                                                                                    total_iters, skip_batches)
        self._key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def _version(self, t, v):
        if v == 0:
            return 0, np.arange(11, dtype=np.uint8)
        key = np.array([self._key0, np.uint64(int(t) * self.nv + int(v))], dtype=np.uint64)
        g = np.random.Generator(np.random.Philox(key=key))
        k = int(g.integers(0, 8))
        perm = np.concatenate([[0, 1], 1 + g.permutation(np.arange(1, 10))]).astype(np.uint8)   # raw 0 and color 0 fixed
        return k, perm

    @staticmethod
    def _dihedral(g, k):
        g = np.rot90(g, k % 4)
        return g.T if k >= 4 else g

    def _example(self, i, v, rng):
        hi, wi, ho, wo = (int(x) for x in self.hw[i])
        k, perm = self._version(self.task[i], v)
        a = self._dihedral(perm[self.tin[i][:hi, :wi]], k)
        b = self._dihedral(perm[self.tout[i][:ho, :wo]], k)
        r = int(rng.integers(0, ARC_N - max(a.shape[0], b.shape[0]) + 1))
        c = int(rng.integers(0, ARC_N - max(a.shape[1], b.shape[1]) + 1))
        return _arc_canvas(a, *a.shape, r, c), _arc_canvas(b, *b.shape, r, c)

    def __iter__(self):
        skip = self.skip_batches
        n = len(self.tin)
        for it in range(self.start_iter, self.total_iters):
            rng = np.random.Generator(np.random.Philox(seed=self.seed + it + 1))
            order = np.concatenate([rng.permutation(n) for _ in range(self.epochs_per_iter)])
            for s in range(0, len(order) - self.gbs + 1, self.gbs):
                sel = order[s:s + self.gbs]
                vers = rng.integers(0, self.nv, len(sel))
                if skip:
                    skip -= 1
                    continue
                off = self.rank * self.local_bs
                inp = np.empty((self.local_bs, ARC_N * ARC_N), dtype=np.int32)
                lab = np.empty_like(inp)
                pid = np.empty(self.local_bs, dtype=np.int32)
                for j in range(self.local_bs):
                    i, v = int(sel[off + j]), int(vers[off + j])
                    x, y = self._example(i, v, rng)
                    inp[j], lab[j] = x.reshape(-1) + 1, y.reshape(-1) + 1
                    pid[j] = int(self.task[i]) * self.nv + v
                yield it, dict(inputs=torch.from_numpy(inp), labels=torch.from_numpy(lab),
                               puzzle_identifiers=torch.from_numpy(pid))


def ctx_canvas(grid, S):
    """Raw S x S canvas of one grid (raw colors 0..9) at the top-left: color + 1, EOS 11 on the row and column just
    outside the grid, 0 (pad) elsewhere."""
    g = np.asarray(grid, dtype=np.uint8)
    h, w = g.shape
    out = np.zeros((S, S), dtype=np.uint8)
    out[:h, :w] = g + 1
    if h < S:
        out[h, :min(w + 1, S)] = 11
    if w < S:
        out[:min(h + 1, S), w] = 11
    return out


def ctx_episode(demos, query, S, kmax):
    """In-context sample (data/prep_arc_ctx.py): 2 kmax + 2 canvases stacked along rows. Slots 2i, 2i + 1 hold the
    input and output of demonstration i (up to kmax; unused slots stay pad), slot 2 kmax the query input and slot
    2 kmax + 1 the query output, which is pad in the input and the output canvas in the label. Labels are raw 0
    (ignored with ignore_label0) everywhere else. demos: [(input, output)] raw grids; query: (input, output)."""
    G = 2 * kmax + 2
    inp = np.zeros((G * S, S), dtype=np.uint8)
    lab = np.zeros((G * S, S), dtype=np.uint8)
    for i, (x, y) in enumerate(demos[:kmax]):
        inp[2 * i * S:(2 * i + 1) * S] = ctx_canvas(x, S)
        inp[(2 * i + 1) * S:(2 * i + 2) * S] = ctx_canvas(y, S)
    inp[(G - 2) * S:(G - 1) * S] = ctx_canvas(query[0], S)
    lab[(G - 1) * S:] = ctx_canvas(query[1], S)
    return inp, lab


PACK_ROUND = 256                  # packed lengths are rounded up to this (few distinct shapes for torch.compile)


def ctx_pack(inputs, labels, S, G):
    """Pack dense in-context episodes (raw [n, G*S, S], ctx_episode layout) for ctx_packed: per episode the non-pad
    cells (grids and EOS) of slots 0 .. G-2 and the whole query-output slot G-1, slot-major and row-major, padded to
    the longest episode (rounded up to PACK_ROUND). Returns int64 tensors: inputs / labels (raw + 1; padding 1 =
    ignored label), pos_row (slot * S + row, the dense RoPE rows), pos_col, valid (1 real, 0 padding) and nbr
    [n, T, 9]: packed index of the cell at (row + i - 1, col + j - 1) of the same slot (3x3 kernel order), or T where
    that cell is not packed (local_conv reads 0 there)."""
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    n = len(inputs)
    keep = inputs.reshape(n, G, S, S) != 0
    keep[:, G - 1] = True
    lens = keep.reshape(n, -1).sum(1)
    T = int(-(-int(lens.max()) // PACK_ROUND) * PACK_ROUND)
    out = {k: np.zeros((n, T), np.int64) for k in ('inputs', 'labels', 'pos_row', 'pos_col', 'valid')}
    out['inputs'][:] = 1
    out['labels'][:] = 1
    out['nbr'] = np.full((n, T, 9), T, np.int64)
    for e in range(n):
        idx = np.full((G, S + 2, S + 2), T, np.int64)
        cells = np.flatnonzero(keep[e].reshape(-1))
        g, r, c = cells // (S * S), cells // S % S, cells % S
        L = len(cells)
        idx[g, r + 1, c + 1] = np.arange(L)
        out['inputs'][e, :L] = inputs[e].reshape(-1)[cells].astype(np.int64) + 1
        out['labels'][e, :L] = labels[e].reshape(-1)[cells].astype(np.int64) + 1
        out['pos_row'][e, :L] = g * S + r
        out['pos_col'][e, :L] = c
        out['valid'][e, :L] = 1
        for i in range(3):
            for j in range(3):
                out['nbr'][e, :L, 3 * i + j] = idx[g, r + i, c + j]
    return {k: torch.from_numpy(v) for k, v in out.items()}


def ctx_packed_length(inputs, S, G):
    """Packed length (before rounding) of dense raw episodes [n, G*S, S]."""
    s = np.asarray(inputs).reshape(len(inputs), G, S * S)
    return (s[:, :-1] != 0).sum((1, 2)) + S * S


def load_arc_ctx(path):
    """Pair store of data/prep_arc_ctx.py (in-context ARC), or None for other NPZ files."""
    with np.load(find_npz(path), allow_pickle=False) as z:
        if 'ctx_slot' not in z:
            return None
        trainable = (z['pair_trainable'] if 'pair_trainable' in z else z['task_split'][z['pair_task']] == 0)
        return dict(cells=z['pair_cells'], off=z['pair_off'], hw=z['pair_hw'], task=z['pair_task'],
                    split=z['task_split'], trainable=trainable, S=int(z['ctx_slot']), kmax=int(z['ctx_kmax']))


class ArcContextDataset(IterableDataset):
    """In-context ARC episodes drawn on the fly from the pair store of data/prep_arc_ctx.py (no task identifiers).
    Only trainable pairs enter (store['trainable']: every pair of a training task; under a transductive split also
    the train pairs of some evaluation tasks, never their test pairs), and every task with at least two of them is a
    training task. Every epoch visits every training task once in a random order. An episode of task t takes one of
    its trainable pairs (uniformly) as the query and up to kmax of the others, in random order, as demonstrations,
    then applies one dihedral transform and one permutation of all 10 colors to every grid of the episode. The
    episode's draws come from a Philox stream keyed by (seed, iterator, position), so resuming mid-iterator
    (skip_batches) reproduces the stream exactly.

    packed (ctx_packed): batches are lt.data.ctx_pack layouts. The episodes of bucket_window consecutive batches are
    sorted by packed length and cut into batches again (in a random batch order), so a batch's episodes have similar
    lengths and little padding; every episode still enters exactly one batch.

    num_aug (URM/HRM versions, for a puzzle embedding per task version): an episode of task t first draws a version v
    in 0..num_aug (uniformly) and applies that version's fixed transform to every grid instead of a fresh one: v = 0
    the identity, v >= 1 a dihedral transform and a permutation of colors 1..9 (color 0 fixed) fixed by (seed, t, v),
    as lt.data.ArcTrainDataset. The puzzle identifier is t * (num_aug + 1) + v (evaluation: v = 0)."""
    def __init__(self, store, *, seed, global_batch_size, rank, world_size, epochs_per_iter, start_iter, total_iters,
                 skip_batches=0, packed=False, bucket_window=16, task_ids=False, id_dropout=0.0, num_aug=None):
        super().__init__()
        self.packed, self.window = packed, bucket_window
        self.task_ids, self.id_dropout = task_ids, id_dropout
        if num_aug is not None and task_ids:
            raise ValueError('num_aug versions carry their own identifiers (task_ids off)')
        self.nv = None if num_aug is None else num_aug + 1
        self.store, self.S, self.kmax = store, store['S'], store['kmax']
        trainable = np.asarray(store.get('trainable', store['split'][store['task']] == 0), bool)
        pairs = {int(t): np.flatnonzero((store['task'] == t) & trainable) for t in np.unique(store['task'][trainable])}
        self.pairs = {t: p for t, p in pairs.items() if len(p) >= 2}
        self.tasks = np.array(sorted(self.pairs), dtype=np.int64)
        self.seed, self.gbs, self.rank = seed, global_batch_size, rank
        self.local_bs = global_batch_size // world_size
        self.epochs_per_iter, self.start_iter, self.total_iters, self.skip_batches = (epochs_per_iter, start_iter,
                                                                                    total_iters, skip_batches)
        self._key0 = np.uint64(0x9E3779B97F4A7C15 ^ int(seed))

    def grids(self, p):
        """Raw (input, output) grids of pair p."""
        hi, wi, ho, wo = (int(x) for x in self.store['hw'][p])
        o = int(self.store['off'][p])
        c = self.store['cells']
        return c[o:o + hi * wi].reshape(hi, wi), c[o + hi * wi:o + hi * wi + ho * wo].reshape(ho, wo)

    def version(self, t, v):
        """num_aug: the fixed (dihedral k, color permutation of raw colors 0..9) of version v of task t."""
        if v == 0:
            return 0, np.arange(10, dtype=np.uint8)
        key = np.array([self._key0 ^ np.uint64(0x5A), np.uint64(int(t) * self.nv + int(v))], dtype=np.uint64)
        g = np.random.Generator(np.random.Philox(key=key))
        k = int(g.integers(0, 8))
        return k, np.concatenate([[0], 1 + g.permutation(9)]).astype(np.uint8)

    def episode(self, t, rng):
        """(input, label, puzzle identifier) of one episode of task t; the identifier is 0 without num_aug."""
        pairs = self.pairs[int(t)]
        v = int(rng.integers(self.nv)) if self.nv else None
        q = int(rng.integers(len(pairs)))
        rest = rng.permutation(np.delete(pairs, q))[:self.kmax]
        if self.nv:
            k, perm = self.version(t, v)
        else:
            k = int(rng.integers(8))
            perm = rng.permutation(10).astype(np.uint8)
        f = lambda g: ArcTrainDataset._dihedral(perm[g], k)
        demos = [tuple(f(g) for g in self.grids(p)) for p in rest]
        x, y = ctx_episode(demos, tuple(f(g) for g in self.grids(pairs[q])), self.S, self.kmax)
        return x, y, int(t) * self.nv + v if self.nv else 0

    def _drawn(self, it, order, pos):
        key = np.array([self._key0, np.uint64(it * (1 << 32) + pos)], dtype=np.uint64)
        return self.episode(order[pos], np.random.Generator(np.random.Philox(key=key)))

    def _ids(self, it, order, positions, eps=None):
        """task_ids: puzzle identifier task + 1, or 0 (no task) with probability id_dropout (own Philox stream).
        num_aug: the episodes' version identifiers (eps)."""
        if self.nv:
            return torch.tensor([eps[i][2] for i in range(len(positions))], dtype=torch.int32)
        if not self.task_ids:
            return torch.zeros(len(positions), dtype=torch.int32)
        out = []
        for pos in positions:
            key = np.array([self._key0 ^ np.uint64(0x1D), np.uint64(it * (1 << 32) + pos)], dtype=np.uint64)
            drop = np.random.Generator(np.random.Philox(key=key)).random() < self.id_dropout
            out.append(0 if drop else int(order[pos]) + 1)
        return torch.tensor(out, dtype=torch.int32)

    def __iter__(self):
        skip = self.skip_batches
        G = 2 * self.kmax + 2
        for it in range(self.start_iter, self.total_iters):
            rng = np.random.Generator(np.random.Philox(seed=self.seed + it + 1))
            order = np.concatenate([rng.permutation(self.tasks) for _ in range(self.epochs_per_iter)])
            starts = list(range(0, len(order) - self.gbs + 1, self.gbs))
            if self.packed:
                for w0 in range(0, len(starts), self.window):
                    ws = starts[w0:w0 + self.window]
                    if skip >= len(ws):
                        skip -= len(ws)
                        continue
                    flat = [s + j for s in ws for j in range(self.gbs)]
                    eps = [self._drawn(it, order, p) for p in flat]
                    x = np.stack([e[0] for e in eps])
                    y = np.stack([e[1] for e in eps])
                    sorted_pos = np.argsort(ctx_packed_length(x, self.S, G), kind='stable')
                    key = np.array([self._key0 ^ np.uint64(0xB5), np.uint64(it * (1 << 32) + w0)], dtype=np.uint64)
                    perm = np.random.Generator(np.random.Philox(key=key)).permutation(len(ws))
                    for b in perm:
                        if skip:
                            skip -= 1
                            continue
                        sel = sorted_pos[b * self.gbs:(b + 1) * self.gbs][self.rank * self.local_bs:
                                                                        (self.rank + 1) * self.local_bs]
                        batch = ctx_pack(x[sel], y[sel], self.S, G)
                        batch['puzzle_identifiers'] = self._ids(it, order, [flat[i] for i in sel],
                                                                [eps[i] for i in sel])
                        yield it, batch
                continue
            for s in starts:
                if skip:
                    skip -= 1
                    continue
                off = self.rank * self.local_bs
                inp = np.empty((self.local_bs, G * self.S * self.S), dtype=np.int32)
                lab = np.empty_like(inp)
                pid = np.zeros(self.local_bs, dtype=np.int32)
                for j in range(self.local_bs):
                    x, y, pid[j] = self._drawn(it, order, s + off + j)
                    inp[j], lab[j] = x.reshape(-1) + 1, y.reshape(-1) + 1
                yield it, dict(inputs=torch.from_numpy(inp), labels=torch.from_numpy(lab),
                               puzzle_identifiers=torch.from_numpy(pid))


def eval_batches(inputs, labels, gbs, rank=0, world_size=1, ids=None, pack=None):
    """One yield per GLOBAL batch on every rank, None for an empty shard. ids: task identifier per puzzle.
    pack (S, G): in-context episodes in the ctx_pack layout (ctx_packed models)."""
    lbs = gbs // world_size
    for start in range(0, len(inputs), gbs):
        end = min(start + gbs, len(inputs))
        ls, le = start + rank * lbs, min(start + (rank + 1) * lbs, end)
        if le <= ls:
            yield None
        elif pack is not None:
            b = ctx_pack(inputs[ls:le], labels[ls:le], *pack)
            b['puzzle_identifiers'] = (torch.zeros(le - ls, dtype=torch.int32) if ids is None else
                                       torch.from_numpy(ids[ls:le].astype(np.int32)))
            yield b
        else:
            yield dict(inputs=torch.from_numpy(inputs[ls:le].reshape(le - ls, -1).astype(np.int32) + 1),
                       labels=torch.from_numpy(labels[ls:le].reshape(le - ls, -1).astype(np.int32) + 1),
                       puzzle_identifiers=(torch.zeros(le - ls, dtype=torch.int32) if ids is None else
                                           torch.from_numpy(ids[ls:le].astype(np.int32))))


def load_arc(path):
    """ARC-AGI extras of data/prep_arc1.py: (hw, train task, test task, number of tasks), or None."""
    with np.load(find_npz(path), allow_pickle=False) as z:
        if 'arc_train_in' not in z:
            return None
        return z['arc_train_hw'], z['arc_train_task'], z['arc_test_task'], int(z['arc_num_tasks'])


def load_task_ids(path=None, test_size=2048):
    """Task identifiers (train, test) of a multi-task NPZ, or (None, None)."""
    with np.load(find_npz(path), allow_pickle=False) as z:
        if 'train_task' not in z:
            return None, None
        return z['train_task'].astype(np.int64), z['test_task'][:test_size].astype(np.int64)


def find_npz(path=None):
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f'data_npz does not exist: {p}')
    candidates = set()
    for root in (Path('/kaggle/input'), Path(__file__).resolve().parents[1] / 'data', Path.cwd() / 'data', Path.cwd()):
        if root.is_dir():
            pattern = '**/sudoku_lt_1k.npz' if str(root) == '/kaggle/input' else 'sudoku_lt_1k.npz'
            candidates.update(str(p.resolve()) for p in root.glob(pattern))
    if len(candidates) == 1:
        return next(iter(candidates))
    if candidates:
        raise RuntimeError('Several NPZ files; set data_npz explicitly:\n' + '\n'.join(sorted(candidates)))
    raise FileNotFoundError('sudoku_lt_1k.npz not found; set data_npz.')


def _check_symbols(x, y, name, symbols):
    """ARC-style sets (inputs and outputs are different grids): only shapes and symbol ranges are checked. Samples may
    stack several grids along rows ([N, k n, n]); label 0 marks cells outside the loss (ignore_label0)."""
    if x.shape != y.shape or len(x) == 0 or x.min() < 0 or x.max() > symbols or y.min() < 0 or y.max() > symbols:
        raise ValueError(f'{name}: expected matching grids with inputs 0..{symbols} and labels 0..{symbols}')
    return np.ascontiguousarray(x.astype(np.uint8)), np.ascontiguousarray(y.astype(np.uint8))


def _check_boards(x, y, name, boxes=True):
    n = x.shape[-1] if x.ndim == 3 else int(round(x.shape[-1] ** 0.5))
    b = _box(n)
    x, y = x.reshape(-1, n, n), y.reshape(-1, n, n)
    if x.shape != y.shape or len(x) == 0:
        raise ValueError(f'{name}: expected matching nonempty [N,{n},{n}] arrays; got {x.shape}, {y.shape}.')
    if x.min() < 0 or x.max() > n or y.min() < 1 or y.max() > n:
        raise ValueError(f'{name}: raw inputs must be 0..{n} and solutions 1..{n}.')
    x, y = x.astype(np.uint8), y.astype(np.uint8)
    if not ((x == 0) | (x == y)).all():
        raise ValueError(f'{name}: clues do not match solutions.')
    digits = np.arange(1, n + 1)
    box_cells = y.reshape(-1, b, b, b, b).transpose(0, 1, 3, 2, 4).reshape(-1, n, n)
    box_ok = (np.sort(box_cells, axis=2) == digits).all() if boxes else True
    if not ((np.sort(y, axis=2) == digits).all() and (np.sort(y, axis=1) == digits[None, :, None]).all() and box_ok):
        raise ValueError(f'{name}: at least one solution is not a valid Sudoku.')
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def load_data(path=None, test_size=2048):
    """Returns train_x, train_y, test_x, test_y (raw [N,n,n] uint8), the resolved path and a content fingerprint."""
    path = find_npz(path)
    with np.load(path, allow_pickle=False) as z:
        if 'arc_train_in' in z:                          # ARC-AGI (data/prep_arc1.py): raw training pairs, finished tests
            tr_x, tr_y = z['arc_train_in'], z['arc_train_out']
            te_x, te_y = _check_symbols(z['test_inputs'][:test_size], z['test_labels'][:test_size], 'test',
                                        int(z['num_symbols']))
        elif 'num_symbols' in z:                         # ARC-style: outputs are not completions of the inputs
            s = int(z['num_symbols'])
            tr_x, tr_y = _check_symbols(z['train_inputs'], z['train_labels'], 'train', s)
            te_x, te_y = _check_symbols(z['test_inputs'][:test_size], z['test_labels'][:test_size], 'test', s)
        else:
            boxes = 'train_task' not in z                # multi-task sets mix in constraints without boxes (Latin squares)
            tr_x, tr_y = _check_boards(z['train_inputs'], z['train_labels'], 'train', boxes)
            te_x, te_y = _check_boards(z['test_inputs'][:test_size], z['test_labels'][:test_size], 'test', boxes)
    digest = hashlib.sha256()
    for a in (tr_x, tr_y, te_x, te_y):
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return tr_x, tr_y, te_x, te_y, path, digest.hexdigest()
