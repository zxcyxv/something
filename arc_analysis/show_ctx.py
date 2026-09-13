"""in-context 런의 체크포인트 예측 그림: 시범 K쌍 · 질의 입력 · 정답 · 세그먼트별 질의 출력 예측.
사용: python arc_analysis/show_ctx.py --run runs/arc_s10_ctx/d4_k2 --tasks 5 --segments 4 8 16 32 --png out.png [--device cuda --amp]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import train_arc_ctx, train_arc_s, ctx_data, train_arc          # noqa: E402
from train import ACTLossHead, IGNORE_LABEL_ID                   # noqa: E402
from show_predictions import save_png                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--segments", type=int, nargs="+", default=[4, 8, 16]); ap.add_argument("--png", type=Path)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", action="store_true"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--wrap", type=int, default=0)
    a = ap.parse_args()
    ck = torch.load(a.run / "latest.pt", map_location="cpu", weights_only=False)
    train_arc_ctx.SETTINGS.update(json.loads((a.run / train_arc_ctx.SIDECAR).read_text()))
    train_arc_ctx.install()
    root = Path(ck["run_cfg"]["data"]).resolve()
    test = train_arc.ARCSplit(root, "test")
    identifiers = json.loads((root / "identifiers.json").read_text()); tasks = json.loads((root / "test_puzzles.json").read_text())
    base = ACTLossHead(train_arc.LT(dict(ck["model_cfg"], amp=bool(a.amp and a.device.startswith("cuda")))), "stablemax_cross_entropy", q_weight=0)
    base.load_state_dict(ck["raw_model_state_dict"] if a.raw else ck["model_state_dict"], strict=False); base.eval(); base.to(a.device)
    lt = base.model; S = test.S; S2 = S * S; K = test.k; qi, qo = test.query_slots()
    print(f"{a.run.name}: step {ck['step']}, {'raw' if a.raw else 'EMA'}, {train_arc_ctx.SETTINGS}")
    figs = []
    originals = {n: t for n, t in tasks.items()}
    for batch in ctx_data.eval_batches(test, identifiers, originals, 1, 1):     # 과제당 원본 1개
        name = identifiers[int(batch["puzzle_identifiers"][0])]
        b = {k: v.to(a.device) for k, v in batch.items()}
        lt.config.loops = max(a.segments)
        with torch.inference_mode(), torch.device(a.device):
            carry = lt.initial_carry(b); preds = {}
            for seg in range(1, max(a.segments) + 1):
                carry, out = lt(carry, b)
                if seg in a.segments: preds[seg] = out["logits"].argmax(-1)[0, qo].cpu()
        x = batch["inputs"][0]; lab = batch["labels"][0, qo].clone(); mask = lab != IGNORE_LABEL_ID; lab[~mask] = 0
        panels = []
        for k in range(K):
            if (x[(2 * k) * S2:(2 * k + 1) * S2] > 0).any():
                panels += [(f"demo{k + 1} in", x[(2 * k) * S2:(2 * k + 1) * S2].view(S, S).numpy()), (f"demo{k + 1} out", x[(2 * k + 1) * S2:(2 * k + 2) * S2].view(S, S).numpy())]
        panels += [("query in", x[qi].view(S, S).numpy()), ("target", lab.view(S, S).numpy())]
        for seg, p in preds.items():
            acc = float(((p == batch["labels"][0, qo]) & mask).sum() / mask.sum())
            panels.append((f"seg{seg} pred {acc:.2f}{'*' if acc == 1 else ''}", p.view(S, S).numpy()))
        print(f"[{name}] " + "  ".join(t for t, _ in panels[len(panels) - len(preds):]))
        figs.append((name, panels))
        if len(figs) >= a.tasks: break
    if a.png: save_png(figs, a.png, f"{a.run.name} · step {ck['step']} · in-context", wrap=a.wrap); print("그림:", a.png)


if __name__ == "__main__":
    main()
