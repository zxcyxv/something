"""6x6 sudoku (2x3 boxes) puzzles that singles cannot finish but one level of contradiction can.

Usage (repository root):  python docs/research/2026-10-14/scripts/gen_sudoku6_contradiction.py OUT.npz [--target 4000] [--workers 24]

Each puzzle: random solution grid, clues removed in random order while the solution stays unique (minimal puzzle).
Level 0 = naked + hidden singles reach the solution. Level 1 = singles stall; assuming a candidate and running singles
to a contradiction eliminates it, and repeating that (with singles) reaches the solution. Kept: level 1 only.
Also stores, per puzzle, the cells and candidates eliminated by contradiction (the 'forced' steps).
"""
import argparse
import multiprocessing as mp
import random

import numpy as np

N, BH, BW = 6, 2, 3
UNITS = ([[r * N + c for c in range(N)] for r in range(N)] + [[r * N + c for r in range(N)] for c in range(N)]
         + [[(br + i) * N + bc + j for i in range(BH) for j in range(BW)] for br in range(0, N, BH) for bc in range(0, N, BW)])
PEERS = [set() for _ in range(N * N)]
for u in UNITS:
    for a in u:
        PEERS[a] |= set(u) - {a}


def count_solutions(g, lim=2):
    g = list(g)
    n = [0]

    def rec():
        if n[0] >= lim:
            return
        best = None
        for i in range(N * N):
            if g[i] == 0:
                c = [d for d in range(1, N + 1) if all(g[q] != d for q in PEERS[i])]
                if not c:
                    return
                if best is None or len(c) < len(best[1]):
                    best = (i, c)
        if best is None:
            n[0] += 1
            return
        i, c = best
        for d in c:
            g[i] = d
            rec()
            g[i] = 0
    rec()
    return n[0]


def propagate(cand):
    changed = True
    while changed:
        changed = False
        for i in range(N * N):
            if not cand[i]:
                return False
            if len(cand[i]) == 1:
                d = next(iter(cand[i]))
                for q in PEERS[i]:
                    if d in cand[q]:
                        cand[q] = cand[q] - {d}
                        changed = True
                        if not cand[q]:
                            return False
        for u in UNITS:
            for d in range(1, N + 1):
                w = [i for i in u if d in cand[i]]
                if not w:
                    return False
                if len(w) == 1 and len(cand[w[0]]) > 1:
                    cand[w[0]] = {d}
                    changed = True
    return True


def level(p):
    cand = [{p[i]} if p[i] else set(range(1, N + 1)) for i in range(N * N)]
    if not propagate(cand):
        return None, []
    if all(len(c) == 1 for c in cand):
        return 0, []
    eliminated = []
    while True:
        prog = False
        for i in range(N * N):
            if len(cand[i]) > 1:
                for d in sorted(cand[i]):
                    t = [set(c) for c in cand]
                    t[i] = {d}
                    if not propagate(t):
                        cand[i] = cand[i] - {d}
                        eliminated.append((i, d))
                        prog = True
                        if not propagate(cand):
                            return None, []
        if all(len(c) == 1 for c in cand):
            return 1, eliminated
        if not prog:
            return 2, []


def random_grid(rng):
    g = [0] * (N * N)

    def rec(i):
        if i == N * N:
            return True
        ds = list(range(1, N + 1))
        rng.shuffle(ds)
        for d in ds:
            if all(g[q] != d for q in PEERS[i]):
                g[i] = d
                if rec(i + 1):
                    return True
                g[i] = 0
        return False
    rec(0)
    return g


def worker(seed):
    rng = random.Random(seed)
    sol = random_grid(rng)
    p = list(sol)
    idx = list(range(N * N))
    rng.shuffle(idx)
    for i in idx:
        v = p[i]
        p[i] = 0
        if count_solutions(p) != 1:
            p[i] = v
    lv, elim = level(p)
    return lv, p, sol, elim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--target', type=int, default=4000)
    ap.add_argument('--workers', type=int, default=24)
    opt = ap.parse_args()
    puzzles, sols, n_elim, seen = [], [], [], set()
    stats = {0: 0, 1: 0, 2: 0, None: 0}
    seed = 0
    with mp.Pool(opt.workers) as pool:
        while len(puzzles) < opt.target:
            for lv, p, sol, elim in pool.map(worker, range(seed, seed + 2000)):
                stats[lv] += 1
                if lv == 1 and tuple(p) not in seen:
                    seen.add(tuple(p))
                    puzzles.append(p)
                    sols.append(sol)
                    n_elim.append(len(elim))
            seed += 2000
            print(f'generated {seed}, kept {len(puzzles)}, levels {stats}', flush=True)
    np.savez_compressed(opt.out, inputs=np.array(puzzles[:opt.target], np.int8), labels=np.array(sols[:opt.target], np.int8),
                        n_contradictions=np.array(n_elim[:opt.target], np.int16),
                        level_counts=np.array([stats[0], stats[1], stats[2]], np.int64))


if __name__ == '__main__':
    main()
