"""Build data/grid_tasks4.npz: three 4x4 constraint tasks that share the digits 1..4 and differ in their constraint graph.

    task 0  standard Sudoku   rows, columns, 2x2 boxes        (288 solutions)
    task 1  Latin square      rows, columns                   (576 solutions)
    task 2  diagonal Sudoku   standard + both main diagonals  ( 48 solutions)

Solution grids are split once for all tasks (a test grid is a test grid of every task it solves, so no test solution is
seen in training under any task); test grids are drawn until each task has about `test_frac` of its solutions in the
test split. Puzzles are minimal for their own task (no clue can be removed with the solution staying unique under that
task's constraints). Arrays follow lt/data.py plus train_task / test_task (int8 task identifiers).

    python data/prep_grid_tasks.py [--train-per-task 2000] [--test-per-task 400] [--test-frac 0.2] [--seed 0]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

N = 4
ROWS = [[r * N + c for c in range(N)] for r in range(N)]
COLS = [[r * N + c for r in range(N)] for c in range(N)]
BOXES = [[(br * 2 + r) * N + bc * 2 + c for r in range(2) for c in range(2)] for br in range(2) for bc in range(2)]
DIAGS = [[i * N + i for i in range(N)], [i * N + N - 1 - i for i in range(N)]]
TASKS = [('standard', ROWS + COLS + BOXES), ('latin', ROWS + COLS), ('diagonal', ROWS + COLS + BOXES + DIAGS)]


def peers_of(units):
    return [sorted({j for u in units if i in u for j in u} - {i}) for i in range(N * N)]


PEERS = [peers_of(u) for _, u in TASKS]


def valid(grid, t):
    g = grid.flatten()
    return all(g[i] != g[j] for i in range(N * N) for j in PEERS[t][i])


def count_solutions(cells, peers, limit=2):
    try:
        i = cells.index(0)
    except ValueError:
        return 1
    used = {cells[j] for j in peers[i]}
    total = 0
    for d in range(1, N + 1):
        if d not in used:
            cells[i] = d
            total += count_solutions(cells, peers, limit - total)
            cells[i] = 0
            if total >= limit:
                break
    return total


def minimal_puzzle(grid, peers, rng):
    cells = list(grid.flatten())
    for i in rng.permutation(N * N):
        keep = cells[i]
        cells[i] = 0
        if count_solutions(cells, peers) != 1:
            cells[i] = keep
    return np.array(cells).reshape(N, N)


def build(train_per_task, test_per_task, test_frac, seed):
    rng = np.random.default_rng(seed)
    latin = np.array([np.array(g) for g in itertools.product(itertools.permutations(range(1, N + 1)), repeat=N)
                      if all(len({g[r][c] for r in range(N)}) == N for c in range(N))])
    member = np.array([[valid(g, t) for t in range(len(TASKS))] for g in latin])          # [grids, tasks]
    test = np.zeros(len(latin), dtype=bool)
    for t in np.argsort(member.sum(0)):                                                    # smallest task first
        pool = rng.permutation(np.flatnonzero(member[:, t]))
        for g in pool:
            if (test & member[:, t]).sum() >= round(test_frac * member[:, t].sum()):
                break
            test[g] = True

    def make(t, split_mask, count):
        ids = np.flatnonzero(member[:, t] & split_mask)
        seen, xs, ys, tries = set(), [], [], 0
        while len(xs) < count and tries < count * 20:
            g = ids[tries % len(ids)] if tries < len(ids) else rng.choice(ids)
            tries += 1
            p = minimal_puzzle(latin[g], PEERS[t], rng)
            key = p.tobytes()
            if key not in seen:
                seen.add(key)
                xs.append(p)
                ys.append(latin[g])
        return np.array(xs, dtype=np.uint8), np.array(ys, dtype=np.uint8)

    parts = {'train': [], 'test': []}
    for t in range(len(TASKS)):
        for split, mask, count in (('train', ~test, train_per_task), ('test', test, test_per_task)):
            px, py = make(t, mask, count)
            parts[split].append((px, py, np.full(len(px), t, dtype=np.int8)))
    out = {}
    for split in ('train', 'test'):
        x = np.concatenate([p[0] for p in parts[split]])
        y = np.concatenate([p[1] for p in parts[split]])
        k = np.concatenate([p[2] for p in parts[split]])
        perm = rng.permutation(len(x))
        out[split] = (x[perm], y[perm], k[perm])
    stats = dict(grids=len(latin), test_grids=int(test.sum()),
                 tasks={name: dict(solutions=int(member[:, t].sum()), test_solutions=int((member[:, t] & test).sum()),
                                   train_puzzles=int((out['train'][2] == t).sum()),
                                   test_puzzles=int((out['test'][2] == t).sum()),
                                   mean_clues_train=float((out['train'][0][out['train'][2] == t] > 0).sum((1, 2)).mean()))
                        for t, (name, _) in enumerate(TASKS)})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-per-task', type=int, default=2000)
    ap.add_argument('--test-per-task', type=int, default=400)
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'grid_tasks4.npz'))
    opt = ap.parse_args()
    out, stats = build(opt.train_per_task, opt.test_per_task, opt.test_frac, opt.seed)
    np.savez_compressed(opt.out, train_inputs=out['train'][0], train_labels=out['train'][1], train_task=out['train'][2],
                        test_inputs=out['test'][0], test_labels=out['test'][1], test_task=out['test'][2])
    stats.update(seed=opt.seed, tasks_order=[name for name, _ in TASKS])
    Path(opt.out).with_suffix('.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
