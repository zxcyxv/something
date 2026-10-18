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


def eval_batches(inputs, labels, gbs, rank=0, world_size=1, ids=None):
    """One yield per GLOBAL batch on every rank, None for an empty shard. ids: task identifier per puzzle."""
    lbs = gbs // world_size
    for start in range(0, len(inputs), gbs):
        end = min(start + gbs, len(inputs))
        ls, le = start + rank * lbs, min(start + (rank + 1) * lbs, end)
        if le <= ls:
            yield None
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
