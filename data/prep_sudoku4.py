"""Build data/sudoku4_toy.npz: 4x4 Sudoku (2x2 boxes) for the recurrent linear-attention toy.

All 288 solution grids are enumerated and split by grid (test grids are never seen in training, so the FFN cannot
memorize the test solutions). Each puzzle is made minimal: clues are removed in a random order while the solution stays
unique, so no clue can be dropped. Arrays follow lt/data.py: inputs 0..4 (0 = blank), labels 1..4, shape [N, 4, 4].

    python data/prep_sudoku4.py [--per-grid 24] [--test-grids 58] [--seed 0]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

N, B = 4, 2
UNITS = ([[r * N + c for c in range(N)] for r in range(N)] + [[r * N + c for r in range(N)] for c in range(N)]
         + [[(br * B + r) * N + bc * B + c for r in range(B) for c in range(B)] for br in range(B) for bc in range(B)])
PEERS = [sorted({j for u in UNITS if i in u for j in u} - {i}) for i in range(N * N)]


def grids():
    out = []
    for rows in itertools.product(itertools.permutations(range(1, N + 1)), repeat=N):
        g = np.array(rows)
        if all(len({g.flat[i] for i in u}) == N for u in UNITS):
            out.append(g)
    return np.array(out)


def count_solutions(cells, limit=2):
    """Backtracking count of completions of a flat board (0 = blank), stopping at `limit`."""
    try:
        i = cells.index(0)
    except ValueError:
        return 1
    used = {cells[j] for j in PEERS[i]}
    total = 0
    for d in range(1, N + 1):
        if d not in used:
            cells[i] = d
            total += count_solutions(cells, limit - total)
            cells[i] = 0
            if total >= limit:
                break
    return total


def minimal_puzzle(grid, rng):
    cells = list(grid.flatten())
    for i in rng.permutation(N * N):
        keep = cells[i]
        cells[i] = 0
        if count_solutions(cells) != 1:
            cells[i] = keep
    return np.array(cells).reshape(N, N)


def naked_single_depth(puzzle):
    """Rounds of synchronous naked-single filling until solved (None if naked singles alone get stuck)."""
    cells, rounds = list(puzzle.flatten()), 0
    while 0 in cells:
        fill = {}
        for i, v in enumerate(cells):
            if v == 0:
                cand = set(range(1, N + 1)) - {cells[j] for j in PEERS[i]}
                if len(cand) == 1:
                    fill[i] = cand.pop()
        if not fill:
            return None
        for i, v in fill.items():
            cells[i] = v
        rounds += 1
    return rounds


def build(per_grid, n_test_grids, seed):
    rng = np.random.default_rng(seed)
    all_grids = grids()
    order = rng.permutation(len(all_grids))
    test_ids, train_ids = order[:n_test_grids], order[n_test_grids:]

    def make(ids):
        seen, xs, ys = set(), [], []
        for g in ids:
            for _ in range(per_grid * 4):                       # draw until per_grid distinct puzzles or give up
                p = minimal_puzzle(all_grids[g], rng)
                key = p.tobytes()
                if key not in seen:
                    seen.add(key)
                    xs.append(p)
                    ys.append(all_grids[g])
                if sum(1 for y in ys if (y == all_grids[g]).all()) >= per_grid:
                    break
        perm = rng.permutation(len(xs))
        return np.array(xs, dtype=np.uint8)[perm], np.array(ys, dtype=np.uint8)[perm]

    tr_x, tr_y = make(train_ids)
    te_x, te_y = make(test_ids)
    return all_grids, train_ids, test_ids, tr_x, tr_y, te_x, te_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-grid', type=int, default=24)
    ap.add_argument('--test-grids', type=int, default=58)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'sudoku4_toy.npz'))
    opt = ap.parse_args()
    all_grids, train_ids, test_ids, tr_x, tr_y, te_x, te_y = build(opt.per_grid, opt.test_grids, opt.seed)
    np.savez_compressed(opt.out, train_inputs=tr_x, train_labels=tr_y, test_inputs=te_x, test_labels=te_y)
    depth = lambda xs: dict(zip(*np.unique([str(naked_single_depth(x)) for x in xs], return_counts=True)))
    meta = dict(grids=len(all_grids), train_grids=len(train_ids), test_grids=len(test_ids),
                train_puzzles=len(tr_x), test_puzzles=len(te_x), seed=opt.seed, per_grid=opt.per_grid,
                clues_train=dict(zip(*map(lambda a: a.tolist(), np.unique((tr_x > 0).sum((1, 2)), return_counts=True)))),
                clues_test=dict(zip(*map(lambda a: a.tolist(), np.unique((te_x > 0).sum((1, 2)), return_counts=True)))),
                naked_single_rounds_train={k: int(v) for k, v in depth(tr_x).items()},
                naked_single_rounds_test={k: int(v) for k, v in depth(te_x).items()})
    Path(opt.out).with_suffix('.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    print(json.dumps(meta, indent=2, default=str))


if __name__ == '__main__':
    main()
