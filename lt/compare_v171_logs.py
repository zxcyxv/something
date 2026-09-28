"""v1.71 Kaggle 런을 v1.1·v1.7 원시 기록과 같은 스텝에서 비교한다.

학습 로그: 10k 마일스톤마다 직전 10k 구간의 [LT] 지표 평균(40줄 = step%16 위치 8종을 5번씩)과
         그 구간 [EVAL] (EMA, seg16, 2048문제)의 마지막·최소·평균.
외삽: 마일스톤 파일의 seg1..128 완답 수로 오차 감소율 = (seg16 오차 - seg≤128 최소 오차) / seg16 오차.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TRAIN_RE = re.compile(r"\[LT\] step (\d+)\s+lm_loss ([0-9.eE+-]+)\s+(?:lr \S+\s+)?acc ([0-9.]+)\s+exact ([0-9.]+)")
EVAL_RE = re.compile(r"\[EVAL\] step (\d+)\s+acc ([0-9.]+)\s+exact (\d+)/(\d+)")
ROW_RE = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+(\d+)\s+")
MONITOR_RE = re.compile(r"목표 (\d+) 도달 .*— (step_\d+)\.pt.*\n.*오차 감소율 = ([0-9.]+)%")


def parse_log(path):
    train, evals = {}, {}
    for line in Path(path).read_text().splitlines():
        if m := TRAIN_RE.match(line):   # 재개로 중복된 스텝은 나중 실행 값
            train[int(m[1])] = (float(m[2]), float(m[3]), float(m[4]))
        elif m := EVAL_RE.match(line):
            evals[int(m[1])] = (float(m[2]), int(m[3]), int(m[4]))
    return train, evals


def reduction(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("#") and (m := ROW_RE.match(line)):
            rows[int(m[1])] = int(m[3])
    assert 16 in rows and max(rows) >= 128, path
    n = 512
    e16 = rows[16]
    best_seg = min((s for s in rows if s <= 128), key=lambda s: (-rows[s], s))
    eb = rows[best_seg]
    return dict(seg16=e16, best=eb, best_seg=best_seg, seg128=rows.get(128),
                rate=100 * (eb - e16) / (n - e16))


def window(train, evals, lo, hi):
    t = np.array([v for s, v in sorted(train.items()) if lo < s <= hi])
    e = [(s, v) for s, v in sorted(evals.items()) if lo < s <= hi]
    if len(t) < 30 or not e:
        return None
    ex = np.array([v[1] for _, v in e])
    return dict(loss=t[:, 0].mean(), acc=t[:, 1].mean(), exact=t[:, 2].mean(),
                eval_step=e[-1][0], eval_last=int(ex[-1]), eval_acc=e[-1][1][0],
                eval_min=int(ex.min()), eval_mean=float(ex.mean()),
                eval_cv=float(100 * ex.std() / ex.mean()) if ex.mean() else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v171", type=Path, default=Path("runs/v171_kaggle/lt_v171"))
    ap.add_argument("--ref", type=Path, default=Path("runs/v11_v17_ref/2026-09-09"))
    ap.add_argument("--out", type=Path, default=Path("runs/v171_kaggle/compare.json"))
    args = ap.parse_args()
    runs = {"v1.1": (args.ref / "train_v11.log", args.ref / "run_v11/milestones"),
            "v1.7": (args.ref / "train_v17.log", args.ref / "run_v17/milestones"),
            "v1.71": (args.v171 / "train.log", args.v171 / "milestones")}
    out = {"train": {}, "extrap": {}}
    for name, (log, mdir) in runs.items():
        train, evals = parse_log(log)
        out["train"][name] = {k: window(train, evals, k - 10000, k) for k in range(10000, 400001, 10000)}
        out["extrap"][name] = {int(re.search(r"(\d+)", p.stem)[1]): reduction(p)
                               for p in sorted(mdir.glob("extrap_step_*.txt"))}
    # v1.7 는 100k/200k/300k 외에 모니터가 중간 스텝을 같은 스크립트로 외삽했다 (seg16·best 는 기록 없음)
    mon = (args.ref / "results/extrap_monitor.log").read_text()
    out["v17_monitor"] = {int(m[2].split("_")[1]): float(m[3]) for m in MONITOR_RE.finditer(mon)}

    steps = range(10000, 210001, 10000)
    fmt = lambda w, key, f: "—" if w is None else format(w[key], f)
    print("## 학습 지표 (각 10k 구간: [LT] 평균, [EVAL] EMA seg16 /2048)\n")
    print("| step | 판 | train loss | train acc | train exact | EVAL 마지막 (step) | EVAL 구간 최소 | EVAL 변동계수 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for k in steps:
        for name in runs:
            w = out["train"][name].get(k)
            if w is None:
                continue
            print(f"| {k//1000}k | {name} | {w['loss']:.4f} | {100*w['acc']:.2f}% | {100*w['exact']:.1f}% | "
                  f"{w['eval_last']} ({w['eval_step']}) | {w['eval_min']} | {w['eval_cv']:.1f}% |")
    print("\n## 오차 감소율 (held-out 512, seg128, EMA)\n")
    print("| step | v1.1 | v1.7 | v1.71 | v1.71 seg16 → best (@seg) |")
    print("|---:|---:|---:|---:|---|")
    mon_by_k = {round(s, -4): (s, r) for s, r in out["v17_monitor"].items()}
    for k in steps:
        cells = []
        for name in runs:
            r = out["extrap"][name].get(k)
            if r is not None:
                cells.append(f"{r['rate']:.1f}")
            elif name == "v1.7" and k in mon_by_k:
                s, r_ = mon_by_k[k]
                cells.append(f"{r_:.1f}" + ("" if s == k else f" ({s//1000}k)"))
            else:
                cells.append("—")
        r = out["extrap"]["v1.71"].get(k)
        tail = "—" if r is None else f"{r['seg16']} → {r['best']} (@{r['best_seg']})"
        print(f"| {k//1000}k | " + " | ".join(cells) + f" | {tail} |")
    args.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
