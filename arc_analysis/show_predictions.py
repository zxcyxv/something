"""체크포인트의 예측을 격자로 찍어 본다 (CPU, 학습 런과 병행 가능).

evaluation 과제의 원본(증강 없는) 질의를 골라 입력 / 정답 / 세그먼트별 예측을 나란히 출력한다.
숫자 = 색 0..9, '.' = PAD, '#' = EOS/테두리. 예측은 crop 전 캔버스 전체를 보여 준다 (크기 예측·테두리까지 보이게).

사용: python arc_analysis/show_predictions.py --run runs/arc_s10/base_aug1000 [--tasks 4] [--segments 1 4 16] [--raw] [--png out.png]
--png 이면 ARC 색으로 그린 그림을 저장한다 (PAD 회색 빗금 없음: 진회색, 테두리(EOS) 흰색).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import train_arc                                   # noqa: E402
import train_arc_s                                 # noqa: E402
from train import ACTLossHead, IGNORE_LABEL_ID     # noqa: E402

CH = {0: ".", 1: "#"}


def render(tokens, S):
    g = np.asarray(tokens).reshape(S, S)
    return [" ".join(CH.get(int(v), str(int(v) - 2)) for v in row) for row in g]


def side_by_side(blocks, gap="   "):
    width = max(len(l) for b in blocks for l in b[1:]) if blocks else 0
    lines = [gap.join(b[0].ljust(width) for b in blocks)]
    for i in range(max(len(b) for b in blocks) - 1):
        lines.append(gap.join((b[i + 1] if i + 1 < len(b) else "").ljust(width) for b in blocks))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--segments", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--raw", action="store_true", help="EMA 대신 raw 가중치")
    ap.add_argument("--data", type=Path, help="기본: 체크포인트 run_cfg 의 data")
    ap.add_argument("--png", type=Path, help="그림 저장 경로")
    ap.add_argument("--augmented", action="store_true", help="증강 사본도 포함")
    ap.add_argument("--only-exact", action="store_true", help="마지막 세그먼트 예측이 정답과 완전히 일치하는 질의만")
    ap.add_argument("--best", type=int, default=0, help="훑은 질의 중 마지막 세그먼트 토큰 정확도 상위 K개만 (과제당 1개)")
    ap.add_argument("--exclude-exact", action="store_true", help="--best 에서 완전 일치는 제외 (틀린 것만)")
    ap.add_argument("--max-aug", type=int, default=32, help="--augmented 일 때 과제당 훑을 증강 사본 수 (평가의 32 상한과 같게)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true", help="평가와 같은 bf16 autocast (cuda)")
    ap.add_argument("--ids", nargs="+", help="이 식별자(정확히 일치)만. --augmented 포함 여부와 무관")
    ap.add_argument("--wrap", type=int, default=0, help="그림에서 한 줄에 놓을 패널 수 (0 = 전부 한 줄)")
    args = ap.parse_args()
    figs = []
    ck = torch.load(args.run / "latest.pt", map_location="cpu", weights_only=False)
    side = json.loads((args.run / train_arc_s.SIDECAR).read_text())
    train_arc_s.install(side["canvas"], side["model"], side["encoding"], side.get("stdp", True), side.get("transport", "adj"), side.get("sheaf_cond", "none"))
    S = side["canvas"]
    root = (args.data or Path(ck["run_cfg"]["data"])).resolve()
    test = train_arc.ARCSplit(root, "test")
    identifiers = json.loads((root / "identifiers.json").read_text())
    cfg = dict(ck["model_cfg"], amp=bool(args.amp and args.device.startswith("cuda")))
    base = ACTLossHead(train_arc.LT(cfg), "stablemax_cross_entropy", q_weight=0)
    base.load_state_dict(ck["raw_model_state_dict"] if args.raw else ck["model_state_dict"], strict=False)
    base.eval()
    base.to(args.device)
    lt = base.model
    print(f"{args.run.name}: step {ck['step']}, {'raw' if args.raw else 'EMA'} 가중치, model={side['model']} encoding={side['encoding']} canvas={S}")
    data = test.arrays["all"]
    shown = 0
    pool = []
    from collections import Counter
    seen = Counter()
    for puzzle, ident in enumerate(data["puzzle_identifiers"]):
        name = identifiers[int(ident)]
        task = name.split("|||")[0]
        if args.ids:
            if name not in args.ids:
                continue
        elif "|||" in name and not args.augmented:
            continue                                  # 원본(무증강) 사본만
        elif seen[task] >= args.max_aug:
            continue
        seen[task] += 1
        start, end = int(data["puzzle_indices"][puzzle]), int(data["puzzle_indices"][puzzle + 1])
        batch = {k: v.to(args.device) for k, v in test.batch(data, list(range(start, end)), [puzzle] * (end - start)).items()}
        B = batch["inputs"].shape[0]
        cfg_loops = lt.config.loops
        lt.config.loops = max(args.segments)
        with torch.inference_mode():
            with torch.device(args.device):
                carry = lt.initial_carry(batch)
            preds = {}
            for seg in range(1, max(args.segments) + 1):
                carry, out = lt(carry, batch)
                if seg in args.segments:
                    preds[seg] = out["logits"].argmax(-1)
        lt.config.loops = cfg_loops
        batch = {k: v.cpu() for k, v in batch.items()}; preds = {k: v.cpu() for k, v in preds.items()}
        last = preds[max(preds)]
        hit = False
        for i in range(B):
            labels = batch["labels"][i].clone(); labels[labels == IGNORE_LABEL_ID] = 0
            mask = batch["labels"][i] != IGNORE_LABEL_ID
            if args.only_exact and not bool(((last[i] == batch["labels"][i]) | ~mask).all()):
                continue
            hit = True
            last_acc = float(((last[i] == batch["labels"][i]) & mask).sum() / mask.sum())
            blocks = [["input"] + render(batch["inputs"][i], S), ["target"] + render(labels, S)]
            for seg, p in preds.items():
                acc = float(((p[i] == batch["labels"][i]) & mask).sum() / mask.sum())
                exact = "*" if acc == 1.0 else ""
                blocks.append([f"seg{seg} pred {acc:.2f}{exact}"] + render(p[i], S))
            entry = (f"{name} q{i + 1}", [(b[0], np.asarray(t).reshape(S, S)) for b, t in
                                          zip(blocks, [batch["inputs"][i], labels] + [p[i] for p in preds.values()])])
            if args.best:
                if args.exclude_exact and last_acc >= 1.0:
                    continue
                pool.append((last_acc, task, entry, blocks, name, i, B))
                continue
            print(f"\n[{name}] query {i + 1}/{B}")
            print(side_by_side(blocks))
            figs.append(entry)
        shown += hit
        if shown >= args.tasks and not args.best:
            break
    if args.best:
        pool.sort(key=lambda e: -e[0])
        used = set()
        for acc, task, entry, blocks, name, i, B in pool:
            if task in used:
                continue
            used.add(task)
            print(f"\n[{name}] query {i + 1}/{B}  last-seg token acc {acc:.3f}")
            print(side_by_side(blocks))
            figs.append(entry)
            if len(figs) >= args.best:
                break
        print(f"\n훑은 질의 {len(pool)}개, 정확도 분포: 1.0={sum(a == 1.0 for a, *_ in pool)}, ≥0.9={sum(a >= 0.9 for a, *_ in pool)}, ≥0.7={sum(a >= 0.7 for a, *_ in pool)}")
    if args.png and figs:
        save_png(figs, args.png, f"{args.run.name} · step {ck['step']} · {'raw' if args.raw else 'EMA'}", wrap=args.wrap)
        print(f"\n그림: {args.png}")


# ARC 공식 색: 0 검정 1 파랑 2 빨강 3 초록 4 노랑 5 회색 6 자홍 7 주황 8 하늘 9 갈색. 토큰 = 색+2. PAD(0) 진회색, EOS(1) 흰색
PALETTE = ["#2b2b2b", "#ffffff", "#000000", "#0074d9", "#ff4136", "#2ecc40", "#ffdc00", "#aaaaaa",
           "#f012be", "#ff851b", "#7fdbff", "#870c25"]


def save_png(figs, path, title, wrap=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(PALETTE)
    npan = max(len(f[1]) for f in figs)
    cols = min(wrap, npan) if wrap else npan
    per = -(-npan // cols)                                   # 예제 하나가 차지하는 줄 수
    rows = len(figs) * per
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.3 * rows), squeeze=False)
    for r0, (label, panels) in enumerate(figs):
        for k in range(per * cols):
            ax = axes[r0 * per + k // cols][k % cols]; ax.set_xticks([]); ax.set_yticks([])
            if k >= len(panels):
                ax.axis("off"); continue
            sub, grid = panels[k]
            c = k
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=11, interpolation="nearest")
            S = grid.shape[0]
            ax.set_xticks(np.arange(-.5, S, 1), minor=True); ax.set_yticks(np.arange(-.5, S, 1), minor=True)
            ax.grid(which="minor", color="#555555", linewidth=0.4); ax.tick_params(which="minor", length=0)
            ax.set_title(sub, fontsize=8)
            if k == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
