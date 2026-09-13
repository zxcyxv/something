"""캔버스 S 와 배치에 따른 1 optimizer step(세그먼트 1개 = 8블록 fwd+bwd) 시간과 최고 메모리. 기본 v1.1 폭 832·8헤드.
사용: python arc_analysis/bench_canvas.py [--configs 30:2 20:2 15:2 15:32 10:2 10:64]
"""
import argparse, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "lt"))
from train import ACTLossHead, LT, PRESETS   # noqa: E402


def bench(S, B, steps=5, d=832, heads=8, blocks=8, ids=1000):
    cfg = dict(batch_size=B, seq_len=S * S, grid=S, vocab_size=12, num_puzzle_identifiers=ids, puzzle_emb_ndim=d,
               hidden_size=d, num_heads=heads, loops=16, blocks_per_seg=blocks, num_layers=1, amp=True,
               forward_dtype="float32", **PRESETS["v1.1"])
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    with torch.device(dev):
        base = ACTLossHead(LT(cfg), "stablemax_cross_entropy", q_weight=0)
    opt = torch.optim.Adam(base.parameters(), lr=1e-4)
    batch = {"inputs": torch.randint(0, 12, (B, S * S), device=dev),
             "labels": torch.randint(2, 12, (B, S * S), device=dev),
             "puzzle_identifiers": torch.randint(1, ids, (B,), device=dev)}
    with torch.device(dev):
        carry = base.initial_carry(batch)
    times = []
    for i in range(steps + 2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        carry, loss, *_ = base(carry=carry, batch=batch, return_keys=set())
        (loss / B).backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        if i >= 2: times.append(time.perf_counter() - t0)
    return sum(times) / len(times), torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", nargs="+", default=["30:2", "20:2", "15:2", "15:16", "10:2", "10:64"])
    args = ap.parse_args()
    print(f"{'S':>3s} {'T':>4s} {'batch':>5s} {'s/step':>8s} {'ms/example':>10s} {'peak GiB':>8s}   ({torch.cuda.get_device_name()})")
    for c in args.configs:
        S, B = map(int, c.split(":"))
        try:
            t, mem = bench(S, B)
            print(f"{S:3d} {S*S:4d} {B:5d} {t:8.3f} {1000*t/B:10.1f} {mem:8.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{S:3d} {S*S:4d} {B:5d}      OOM")
        torch.cuda.empty_cache()
