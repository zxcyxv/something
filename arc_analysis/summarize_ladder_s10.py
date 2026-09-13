"""S=10 사다리 결과 요약: runs/arc_s10/<model>_aug<N>/ 의 eval JSON 을 표와 그림으로.
사용: python arc_analysis/summarize_ladder_s10.py [--out runs/arc_s10/ladder]
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(run):
    evals = {}
    for p in run.glob("eval_step_*_seg*.json"):
        m = re.match(r"eval_step_(\d+)_seg(\d+)\.json", p.name)
        d = json.loads(p.read_text())
        evals[(int(m.group(1)), int(m.group(2)))] = d
    train = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines()] if (run / "train.jsonl").exists() else []
    return evals, train


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=str(ROOT / "runs/arc_s10")); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = sorted(p for p in Path(args.runs).iterdir() if re.match(r"(base|d4)_aug\d+$", p.name))
    rows = []
    curves = {}
    for run in runs:
        model, aug = run.name.split("_aug"); aug = int(aug)
        evals, train = load(run)
        if not evals:
            continue
        last_step = max(s for s, _ in evals)
        e16 = evals.get((last_step, 16), {}); e128 = evals.get((last_step, 128), {})
        rows.append((model, aug, last_step, e16.get("ARC/pass@1"), e16.get("ARC/pass@2"), e16.get("max_augmentations_per_task"),
                     e128.get("ARC/pass@1"), e128.get("ARC/pass@2"), e16.get("token_accuracy"),
                     train[-1]["loss"] if train else None))
        curves[run.name] = sorted((s, d["ARC/pass@1"], d["ARC/pass@2"]) for (s, seg), d in evals.items() if seg == 16 and d.get("max_augmentations_per_task") == 32)
    print(f"{'model':5s} {'aug':>5s} {'step':>6s} {'p@1 s16':>8s} {'p@2 s16':>8s} {'augcap':>6s} {'p@1 s128':>8s} {'p@2 s128':>8s} {'tok_acc':>7s} {'loss':>6s}")
    for r in rows:
        f = lambda v, w=8: (f"{v:{w}.3f}" if isinstance(v, float) else f"{str(v):>{w}s}")
        print(f"{r[0]:5s} {r[1]:5d} {r[2]:6d} {f(r[3])} {f(r[4])} {str(r[5]):>6s} {f(r[6])} {f(r[7])} {f(r[8],7)} {f(r[9],6)}")
    print("\n중간 평가(증강 32 상한, seg16) pass@1 곡선:")
    for name, c in curves.items():
        print(f"  {name:12s} " + " ".join(f"{s//1000}k:{p1:.3f}" for s, p1, _ in c))
    if args.out:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for model, marker in (("base", "o"), ("d4", "s")):
            pts = sorted((r[1], r[3], r[6]) for r in rows if r[0] == model and r[3] is not None)
            if pts:
                ax[0].plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, label=f"{model} seg16")
                if any(p[2] is not None for p in pts):
                    ax[0].plot([p[0] for p in pts if p[2] is not None], [p[2] for p in pts if p[2] is not None], marker=marker, ls="--", label=f"{model} seg128")
        ax[0].set_xscale("log"); ax[0].set_xlabel("augmentations per task"); ax[0].set_ylabel("ARC pass@1 (eval tasks ≤10)"); ax[0].legend(); ax[0].grid(alpha=.3)
        for name, c in curves.items():
            ax[1].plot([s for s, _, _ in c], [p for _, p, _ in c], marker=".", label=name)
        ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("pass@1 (aug cap 32)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=130)
        print(f"\n그림: {args.out}.png")


if __name__ == "__main__":
    main()
