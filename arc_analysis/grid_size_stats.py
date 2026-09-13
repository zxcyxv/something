"""ARC-AGI-1 과제별 최대 격자 변을 세어 캔버스 축소(30→S)로 어느 과제가 살아남는지 본다.

LT 의 결합 기억 w 는 [B,H,T,T] 이고 T = S² 이므로 캔버스를 S 로 줄이면 w 와 어텐션 비용이 (S/30)⁴ 배가 된다.
사용: python arc_analysis/grid_size_stats.py --prefix /tmp/URM/kaggle/combined/arc-agi
"""
import argparse, json
from collections import Counter
from pathlib import Path


def task_max_side(task, solutions):
    sides = []
    for pair in task["train"]:
        for g in (pair["input"], pair["output"]):
            sides.append(max(len(g), len(g[0])))
    for i, pair in enumerate(task["test"]):
        sides.append(max(len(pair["input"]), len(pair["input"][0])))
        out = pair.get("output") or solutions[i]
        sides.append(max(len(out), len(out[0])))
    return max(sides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/tmp/URM/kaggle/combined/arc-agi")
    ap.add_argument("--subsets", nargs="+", default=["training", "evaluation", "concept"])
    ap.add_argument("--thresholds", nargs="+", type=int, default=[10, 12, 15, 20, 30])
    args = ap.parse_args()
    print(f"{'subset':12s} {'tasks':>6s} " + " ".join(f"≤{s:>2d}" .rjust(6) for s in args.thresholds))
    grand = Counter(); total = 0
    for subset in args.subsets:
        ch = json.loads(Path(f"{args.prefix}_{subset}-challenges.json").read_text())
        sol_path = Path(f"{args.prefix}_{subset}-solutions.json")
        sol = json.loads(sol_path.read_text()) if sol_path.exists() else {}
        sides = {name: task_max_side(t, sol.get(name, [])) for name, t in ch.items()}
        n = len(sides); total += n
        row = []
        for s in args.thresholds:
            k = sum(v <= s for v in sides.values()); grand[s] += k
            row.append(f"{k:>3d}({100*k/n:3.0f}%)".rjust(6))
        print(f"{subset:12s} {n:6d} " + " ".join(row))
    print(f"{'all':12s} {total:6d} " + " ".join(f"{grand[s]:>3d}({100*grand[s]/total:3.0f}%)".rjust(6) for s in args.thresholds))
    print("\n캔버스 S 에 대한 T=S² 와 w 크기 비율 (30 기준):")
    for s in args.thresholds:
        print(f"  S={s:2d}  T={s*s:4d}  w/attn 비용 = {(s/30)**4:.4f}x  (batch2·8헤드 fp32 w = {2*8*s**4*4/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
