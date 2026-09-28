"""v1.71 두 체크포인트 사이 파라미터 군 교환 → seg128 외삽 (held-out 앞 512, EMA, BF16).

감쇠 없는 스칼라(위상·가소성·흔적)와 weight decay 대상 행렬 중 어느 쪽이
100k→200k 의 외삽 저하를 옮기는지 본다. 교환은 공동 적응을 깨는 개입이므로
효과가 가법적이라고 가정하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt_npz import _strip, load_data, load_lt  # noqa: E402

L0 = "inner.layers.0."
GROUPS = {
    "beta": ["beta"],
    "plastic": ["eta_raw", "lam_raw", "gain_raw"],
    "eta": ["eta_raw"],
    "lam_gain": ["lam_raw", "gain_raw"],
    "trace": ["mu_rho_raw", "mu_omega"],
    "read_phase": ["psi", "theta", "alpha_raw"],
    "scalars": ["beta", "eta_raw", "lam_raw", "gain_raw", "mu_rho_raw", "mu_omega", "psi", "theta", "alpha_raw"],
    "matrices": ["wc", "w_sh", "b_gate_up.weight", "b_down.weight",
                 "@inner.embed.weight", "@inner.w_cls.weight", "@inner.w_cls.bias", "@inner.puzzle_emb.weights"],
}


def ema_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {_strip(k): v for k, v in ck["model_state_dict"].items()}
    sd.update({_strip(k): v for k, v in ck["ema_shadow"].items()})
    return sd


def keys(group):
    return [k[1:] if k.startswith("@") else L0 + k for k in GROUPS[group]]


@torch.no_grad()
def run(m, batch, segs):
    y = batch["labels"]
    carry = m.initial_carry(batch)
    rows = []
    for _ in range(segs):
        carry, out = m(carry, batch)
        p = out["logits"].argmax(-1)
        rows.append((float((p == y).float().mean()), int((p == y).all(-1).sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mdir", default="runs/v171_kaggle/lt_v171/milestones")
    ap.add_argument("--a", type=int, default=100000)
    ap.add_argument("--b", type=int, default=200000)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--groups", nargs="*", default=["none", "beta", "plastic", "trace", "read_phase", "scalars", "matrices"])
    ap.add_argument("--out", default="runs/v171_kaggle/param_swap.json")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = {s: f"{args.mdir}/step_{s}.pt" for s in (args.a, args.b)}
    sds = {s: ema_sd(p) for s, p in paths.items()}
    _, _, batch = load_data(n=args.n)
    res = []
    for base, donor in ((args.b, args.a), (args.a, args.b)):
        m, cfg, _ = load_lt(paths[base], batch_size=args.n, loops=args.segs + 1)
        for g in args.groups:
            sd = dict(sds[base])
            if g != "none":
                for k in keys(g):
                    sd[k] = sds[donor][k]
            m.load_state_dict(sd, strict=False)
            rows = run(m, batch, args.segs)
            e16 = rows[15][1]; eb = max(r[1] for r in rows); sb = 1 + max(range(len(rows)), key=lambda i: (rows[i][1], -i))
            r = dict(base=base, donor=donor, group=g, seg1_acc=rows[0][0], seg16_acc=rows[15][0], seg16=e16,
                     best=eb, best_seg=sb, seg128=rows[-1][1], seg128_acc=rows[-1][0],
                     reduction=100 * (eb - e16) / (args.n - e16), exact=[x[1] for x in rows])
            res.append(r)
            print(f"base {base//1000}k  {g:10s}←{donor//1000}k  seg16 {e16:3d} ({r['seg16_acc']:.4f})  "
                  f"best {eb:3d}@{sb:<3d} seg128 {r['seg128']:3d}  감소율 {r['reduction']:5.1f}%", flush=True)
            Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
