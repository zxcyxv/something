"""Sudoku-Extreme 1k data: checked loading, the fixed augmentation pool and the URM batch stream.

The NPZ holds raw digits (train_inputs/test_inputs 0..9 with 0 = blank, labels 1..9); tokens are digit + 1.
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

_ROW_OF, _COL_OF = np.arange(81) // 9, np.arange(81) % 9


def _draw_aug_params(rng):
    digits = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))
    trans = rng.random() < 0.5
    bands = rng.permutation(3)
    rows = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    cols = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    return digits, trans, rows, cols


def _apply_aug(x, digits, trans, rows, cols):
    mapping = rows[_ROW_OF] * 9 + cols[_COL_OF]
    if trans:
        x = x.T
    return digits[x.flatten()[mapping].reshape(9, 9)]


class SudokuTrainDataset(IterableDataset):
    """Yields (iterator index, batch) for this rank's share of each global batch. skip_batches resumes mid-iterator."""
    def __init__(self, inputs, labels, *, seed, num_aug, global_batch_size, rank, world_size,
                 epochs_per_iter, start_iter, total_iters, skip_batches=0):
        super().__init__()
        self.inputs, self.labels = inputs, labels
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
        prm = _draw_aug_params(np.random.Generator(np.random.Philox(key=key)))
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
                inp = np.empty((self.local_bs, 81), dtype=np.int32)
                lab = np.empty_like(inp)
                for j in range(self.local_bs):
                    x, y = self._augmented(gsel[off + j], asel[off + j])
                    inp[j], lab[j] = x.reshape(81) + 1, y.reshape(81) + 1
                yield it, dict(inputs=torch.from_numpy(inp), labels=torch.from_numpy(lab),
                               puzzle_identifiers=torch.zeros(self.local_bs, dtype=torch.int32))


def eval_batches(inputs, labels, gbs, rank=0, world_size=1):
    """One yield per GLOBAL batch on every rank, None for an empty shard."""
    lbs = gbs // world_size
    for start in range(0, len(inputs), gbs):
        end = min(start + gbs, len(inputs))
        ls, le = start + rank * lbs, min(start + (rank + 1) * lbs, end)
        if le <= ls:
            yield None
        else:
            yield dict(inputs=torch.from_numpy(inputs[ls:le].reshape(-1, 81).astype(np.int32) + 1),
                       labels=torch.from_numpy(labels[ls:le].reshape(-1, 81).astype(np.int32) + 1),
                       puzzle_identifiers=torch.zeros(le - ls, dtype=torch.int32))


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


def _check_boards(x, y, name):
    x, y = x.reshape(-1, 9, 9), y.reshape(-1, 9, 9)
    if x.shape != y.shape or len(x) == 0:
        raise ValueError(f'{name}: expected matching nonempty [N,9,9] arrays; got {x.shape}, {y.shape}.')
    if x.min() < 0 or x.max() > 9 or y.min() < 1 or y.max() > 9:
        raise ValueError(f'{name}: raw inputs must be 0..9 and solutions 1..9.')
    x, y = x.astype(np.uint8), y.astype(np.uint8)
    if not ((x == 0) | (x == y)).all():
        raise ValueError(f'{name}: clues do not match solutions.')
    digits = np.arange(1, 10)
    boxes = y.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    if not ((np.sort(y, axis=2) == digits).all() and (np.sort(y, axis=1) == digits[None, :, None]).all()
            and (np.sort(boxes, axis=2) == digits).all()):
        raise ValueError(f'{name}: at least one solution is not a valid Sudoku.')
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def load_data(path=None, test_size=2048):
    """Returns train_x, train_y, test_x, test_y (raw [N,9,9] uint8), the resolved path and a content fingerprint."""
    path = find_npz(path)
    with np.load(path, allow_pickle=False) as z:
        tr_x, tr_y = _check_boards(z['train_inputs'], z['train_labels'], 'train')
        te_x, te_y = _check_boards(z['test_inputs'][:test_size], z['test_labels'][:test_size], 'test')
    digest = hashlib.sha256()
    for a in (tr_x, tr_y, te_x, te_y):
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return tr_x, tr_y, te_x, te_y, path, digest.hexdigest()
