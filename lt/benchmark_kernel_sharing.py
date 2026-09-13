"""Small FP32 eager benchmark of a shared psi/beta address contraction.

This is an isolated implementation experiment, not a change to the trainer.
The two mathematical kernels and recurrent memory retain their separate roles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
from ckpt_npz import load_data, load_lt
from probe_phase_feedback import Blocks


class SharedAddressKernels:
    def __init__(self, blocks):
        self.b = blocks
        base = blocks.inner.kernel(blocks.layer, torch.zeros_like(blocks.layer.psi))
        self.decay, self.cos_pos, self.sin_pos = base[:3]
        self.rotations = [(phase.cos(), phase.sin())
                          for phase in (blocks.layer.psi, blocks.layer.beta)]

    def __call__(self, xy):
        x, y = xy
        rx = x * self.cos_pos - y * self.sin_pos
        ry = x * self.sin_pos + y * self.cos_pos
        key = torch.cat((rx, ry), -1).permute(0, 2, 1, 3)
        queries = [torch.cat((rx * c - ry * s, rx * s + ry * c), -1)
                   for c, s in self.rotations]
        query = torch.cat(queries, 1).permute(0, 2, 1, 3)
        n, heads, _, _ = query.shape
        length = x.shape[1]
        out = (query @ key.transpose(-1, -2)).reshape(n, heads, 2, length, length)
        out = out * self.decay[None, :, None]
        return out.unbind(2)


def time_functions(functions, repetitions, rounds=5):
    for fn in functions.values():
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    for r in range(rounds):
        names = list(functions)
        if r % 2:
            names.reverse()
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repetitions)
    return {name: {"median_ms": float(np.median(values)), "rounds_ms": values}
            for name, values in samples.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--repetitions", type=int, default=50)
    ap.add_argument("--out", default="runs/kernel_sharing_v11/summary.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, _ = load_lt("checkpoints/v1.1_step160000.npz", mod=train,
                          batch_size=args.n, loops=17, amp=False)
    _, _, batch = load_data(n=args.n)
    b = Blocks(model, batch)
    shared = SharedAddressKernels(b)
    h = b.inner.init_hidden.expand(args.n, 81, -1).clone()
    w = None
    for _ in range(128):
        h, w = b.block(h, w)
    q = b.prepare(h)
    xy = b.inner.addr(q, b.ab)
    original = lambda address: (b.inner.attn_xy(address, b.kc), b.inner.attn_xy(address, b.kcb))
    old_k, new_k = original(xy), shared(xy)
    errors = {name: float((old - new).abs().max())
              for name, old, new in zip(("psi", "beta"), old_k, new_k)}
    for old, new in zip(old_k, new_k):
        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-5)

    def block(kernels):
        prepared = b.prepare(h)
        address = b.inner.addr(prepared, b.ab)
        a, window = kernels(address)
        v = torch.einsum("btd,hcd->bthc", prepared, b.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + b.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
        wn = (1-b.eta)*w + b.eta*(b.gain*(window*agree))
        effective = (1-b.lam)*a + b.lam*wn
        values = torch.einsum("bhtn,bnhc->bthc", effective, v)
        update = torch.einsum("bthc,hcd->btd", values, b.layer.w_sh)
        return b.inner.phi(prepared + update), wn

    old_block, new_block = block(original), block(shared)
    reference = b.block(h, w)
    for old, ref in zip(old_block, reference):
        torch.testing.assert_close(old, ref, atol=0, rtol=0)
    for name, old, new in zip(("hidden", "memory"), old_block, new_block):
        errors[name] = float((old - new).abs().max())
        torch.testing.assert_close(old, new, atol=3e-6, rtol=2e-5)
    report = {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "precision": "FP32; eager; autocast and TF32 off; no backward or compile benchmark",
        "args": vars(args), "one_block_max_abs_error": errors,
        "kernel_pair": time_functions({"original": lambda: original(xy),
                                       "shared": lambda: shared(xy)}, args.repetitions),
        "whole_block": time_functions({"original": lambda: block(original),
                                       "shared": lambda: block(shared)}, args.repetitions),
        "note": "Fixed segment-16 snapshot; excludes per-segment QR/cache setup. "
                "Real-arithmetic equivalence does not establish equal long-rollout predictions in FP32.",
    }
    for group in ("kernel_pair", "whole_block"):
        report[group]["speedup"] = (report[group]["original"]["median_ms"] /
                                    report[group]["shared"]["median_ms"])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
