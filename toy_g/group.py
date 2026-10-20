"""Shared definitions of the 4x4 torus toy (structure-code experiment).

Task (t, c) maps an input grid to its output grid:

    out = CMAPS[c][ roll(D_p(in), (a, b)) ]          p, a, b = decode_t(t)

- t in 0..127: a dihedral point operation D_p (p in 0..7: rot90^p for p < 4; rot90^(p-4) then transpose for p >= 4,
  the convention of data/prep_miniarc_ctx.py) followed by a cyclic translation (a rows, b columns).
- c in 0..23: a permutation of the colours 1..4; colour 0 (background) is fixed.
- task index = t * NUM_C + c (0..3071).

SRC[t, o] is the input cell that output cell o reads (cells flattened row-major): out.flat[o] = in.flat[SRC[t, o]].
DEST[t, i] is the output cell that input cell i is moved to. word(t) lists the generators of t in application order.
"""
import itertools

import numpy as np

N = 4
CELLS = N * N
COLORS = 4                          # colours 1..4; 0 is the background
NUM_T = 8 * CELLS                   # 128
NUM_C = 24                          # permutations of 1..4
NUM_TASKS = NUM_T * NUM_C           # 3,072


def decode_t(t):
    """t -> (p, a, b): point operation p in 0..7, then translation by a rows and b columns."""
    return t // CELLS, (t % CELLS) // N, t % N


def encode_t(p, a, b):
    return p * CELLS + (a % N) * N + (b % N)


def point_op(grid, p):
    """Dihedral operation on the last two axes."""
    g = np.rot90(grid, p % 4, axes=(-2, -1))
    return np.swapaxes(g, -2, -1) if p >= 4 else g


def apply_t(grid, t):
    p, a, b = decode_t(t)
    return np.roll(point_op(grid, p), shift=(a, b), axis=(-2, -1))


PERMS = [tuple(x) for x in itertools.permutations(range(1, COLORS + 1))]
CMAPS = np.array([[0, *perm] for perm in PERMS], dtype=np.int64)           # [24, 5]: CMAPS[c][colour]


def apply_task(grid, t, c):
    return CMAPS[c][apply_t(grid, t)]


def task_index(t, c):
    return t * NUM_C + c


def split_task(task):
    return task // NUM_C, task % NUM_C


SRC = np.stack([apply_t(np.arange(CELLS).reshape(N, N), t).reshape(-1) for t in range(NUM_T)])   # [128, 16]
DEST = np.argsort(SRC, axis=1)                                                                     # [128, 16]
IDENTITY = encode_t(0, 0, 0)
GENERATORS = dict(rot=encode_t(1, 0, 0), flip=encode_t(4, 0, 0), down=encode_t(0, 1, 0), right=encode_t(0, 0, 1))


def compose(t1, t2):
    """Index of 'apply t2, then t1'."""
    target = apply_t(apply_t(np.arange(CELLS).reshape(N, N), t2), t1).reshape(-1)
    hits = np.flatnonzero((SRC == target).all(1))
    if len(hits) != 1:
        raise AssertionError('the transformations are not closed under composition')
    return int(hits[0])


def inverse(t):
    return int(np.flatnonzero((SRC == DEST[t]).all(1))[0])


def word(t):
    """Generators of t in application order: rot (p mod 4 times), flip (if p >= 4), down (a times), right (b times)."""
    p, a, b = decode_t(t)
    return ['rot'] * (p % 4) + ['flip'] * (p >= 4) + ['down'] * a + ['right'] * b
