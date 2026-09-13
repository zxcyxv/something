"""세그먼트별 Dirichlet 에너지 vs 정답 여부 (v3 계열 체크포인트).
E(seg) 가 단조 감소하는가, 맞는 사본과 틀린 사본의 E 가 갈리는가(E 가 신뢰도가 되는가), seg 를 늘리면 완전 일치가 느는가.
사용: python arc_analysis/probe_energy.py --run runs/arc_s10/v3_aug8 --segments 64 [--device cuda --amp] [--max-aug 8]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import train_arc, train_arc_s                                     # noqa: E402
from train import ACTLossHead, IGNORE_LABEL_ID                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True); ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", action="store_true"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--max-aug", type=int, default=0); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ck = torch.load(a.run / "latest.pt", map_location="cpu", weights_only=False)
    side = json.loads((a.run / train_arc_s.SIDECAR).read_text())
    train_arc_s.install(side["canvas"], side["model"], side["encoding"], side.get("stdp", True), side.get("transport", "adj"), side.get("sheaf_cond", "none"))
    root = Path(ck["run_cfg"]["data"]).resolve()
    test = train_arc.ARCSplit(root, "test")
    identifiers = json.loads((root / "identifiers.json").read_text()); tasks = json.loads((root / "test_puzzles.json").read_text())
    base = ACTLossHead(train_arc.LT(dict(ck["model_cfg"], amp=bool(a.amp and a.device.startswith("cuda")))), "stablemax_cross_entropy", q_weight=0)
    base.load_state_dict(ck["raw_model_state_dict"] if a.raw else ck["model_state_dict"], strict=False); base.eval(); base.to(a.device)
    lt = base.model; lt.config.loops = a.segments
    E, exact, tokacc = [], [], []                                    # [N, segs]
    with torch.inference_mode():
        for batch in train_arc.eval_batches(test, identifiers, tasks, a.batch_size, a.max_aug):
            b = {k: v.to(a.device) for k, v in batch.items()}
            with torch.device(a.device):
                carry = lt.initial_carry(b)
            mask = b["labels"] != IGNORE_LABEL_ID; valid = mask.any(-1)
            e_rows, x_rows, t_rows = [], [], []
            for seg in range(a.segments):
                carry, out = lt(carry, b)
                pred = out["logits"].argmax(-1)
                corr = (pred == b["labels"]) & mask
                x_rows.append(((corr.sum(-1) == mask.sum(-1)) & valid).float().cpu())
                t_rows.append((corr.sum(-1) / mask.sum(-1).clamp_min(1)).cpu())
                e_rows.append(out["dirichlet"].float().cpu() if out.get("dirichlet") is not None else torch.zeros(pred.shape[0]))
            keep = valid.cpu()
            E.append(torch.stack(e_rows, 1)[keep]); exact.append(torch.stack(x_rows, 1)[keep]); tokacc.append(torch.stack(t_rows, 1)[keep])
    E, exact, tokacc = (torch.cat(v).numpy() for v in (E, exact, tokacc))
    N = len(E); segs = [s for s in (1, 2, 4, 8, 16, 32, 64, 128) if s <= a.segments]
    print(f"{a.run.name} step {ck['step']} · {N} 예제 · transport={side.get('transport','adj')} sheaf={side.get('sheaf_cond','none')}")
    print(f"{'seg':>4s} {'E 평균':>12s} {'E(맞음)':>12s} {'E(틀림)':>12s} {'완전일치':>8s} {'토큰정확':>8s}")
    for s in segs:
        e = E[:, s - 1]; ok = exact[:, s - 1] > 0
        print(f"{s:4d} {e.mean():12.1f} {(e[ok].mean() if ok.any() else float('nan')):12.1f} {(e[~ok].mean() if (~ok).any() else float('nan')):12.1f} {int(ok.sum()):8d} {tokacc[:, s - 1].mean():8.3f}")
    dec = np.mean(np.diff(E, axis=1) <= 0); print(f"세그먼트 간 E 비증가 비율: {dec:.3f}")
    if a.out:
        np.savez(a.out, E=E, exact=exact, tokacc=tokacc); print("저장:", a.out)


if __name__ == "__main__":
    main()
