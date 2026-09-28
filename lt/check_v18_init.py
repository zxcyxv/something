"""v1.71 체크포인트 → v1.8 (0 게이트) 변환이 같은 함수인지 실제 GPU 추론으로 확인한다.

같은 held-out 퍼즐을 v1.71(원본)과 v1.8(변환)로 seg128까지 돌려 세그먼트별 완답 수,
예측이 다른 칸 수, 로짓 최대 차이를 비교한다. 기본은 EMA 가중치·BF16 (마일스톤 평가와 같은 조건).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_npz  # noqa: E402


def load_train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lt_train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", nargs="?", default="runs/v171_kaggle/lt_v171/milestones/step_100000.pt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--fp32", action="store_true", help="autocast 끔 (함수 동일성 확인용)")
    ap.add_argument("--out", default="runs/v171_kaggle/v18_init_check.json")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = load_train_module()
    extra = dict(amp=False) if args.fp32 else {}
    old, cfg, step = ckpt_npz.load_lt(args.ckpt, mod=mod, batch_size=args.n, loops=args.segs + 1, **extra)
    c18 = dict(cfg, plastic_select=True, select_g_max=cfg.get("select_g_max", 4.0))
    new = mod.LT(c18).cuda().eval()
    sd = {k: v for k, v in old.state_dict().items()}
    new.load_state_dict(mod.convert_plastic_state(sd, c18["select_g_max"]), strict=True)
    _, _, b = ckpt_npz.load_data(n=args.n)
    y = b["labels"]
    ca, cb = old.initial_carry(b), new.initial_carry(b)
    rows = []
    with torch.no_grad():
        for si in range(args.segs):
            ca, oa = old(ca, b)
            cb, ob = new(cb, b)
            pa, pb = oa["logits"].argmax(-1), ob["logits"].argmax(-1)
            rows.append(dict(seg=si + 1, exact_v171=int((pa == y).all(-1).sum()), exact_v18=int((pb == y).all(-1).sum()),
                             cells_differ=int((pa != pb).sum()), logit_maxdiff=float((oa["logits"] - ob["logits"]).abs().max())))
    for r in rows:
        if r["seg"] in (1, 2, 4, 8, 16, 32, 64, 96, 128) or r["seg"] == args.segs:
            print(r)
    first = next((r["seg"] for r in rows if r["cells_differ"]), None)
    summary = dict(ckpt=args.ckpt, step=step, n=args.n, segs=args.segs, precision="fp32" if args.fp32 else "bf16", first_segment_with_different_cell=first,
                   max_cells_differ=max(r["cells_differ"] for r in rows),
                   best_v171=max(r["exact_v171"] for r in rows), best_v18=max(r["exact_v18"] for r in rows), rows=rows)
    print({k: v for k, v in summary.items() if k != "rows"})
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
