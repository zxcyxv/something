"""Frozen final-loss gradients with hidden/memory truncation separated.

This is a local gradient diagnostic, not a longer-BPTT training experiment.
All modes use the identical 128-block forward path and the same final loss.
The actual trainer additionally supervises earlier segments; this probe does
not claim that their earlier optimizer updates are absent in real training.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.checkpoint import checkpoint

from . import train as t
from .kv_stability import install
from .audit_kv_stdp_runtime import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    result = dict(blocks=args.blocks, batch=args.batch, device="cpu", precision="fp32",
                  final_loss_only=True,
                  caveat="Actual training supervises every segment and updates weights; this isolates final-loss credit paths at frozen weights, not their total optimizer effect.", runs={})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    common_batch = None
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        variant = ck["cfg"].get("research_variant", "original")
        install(variant)
        cfg = dict(ck["cfg"], batch_size=args.batch, seq_len=81, num_puzzle_identifiers=1,
                   amp=False, activation_checkpoint=False, nograd_blocks=0)
        model = t.LT(cfg)
        model.load_state_dict({k.removeprefix("model."): v for k, v in ck["raw_model_state_dict"].items()})
        model.eval()
        inner, layer = model.inner, model.inner.layers[0]
        if common_batch is None:
            common_batch = {k: v[:args.batch].clone() for k, v in ck["rank_states"][0]["carry"]["current_data"].items()}
        variants = {}
        reference = None
        # Full first so all later comparisons have the same reference.
        for mode in ("full", "cut_all_every8", "cut_hidden_every8", "cut_memory_every8"):
            started = time.monotonic()
            model.zero_grad(set_to_none=True)
            injection = inner.injection(common_batch)
            state = (inner.init_hidden[None, None, :].expand(args.batch, 81, -1), None, None, None)
            for block in range(args.blocks):
                if block and block % 8 == 0:
                    h, m, ek, ev = state
                    if mode in ("cut_all_every8", "cut_hidden_every8"):
                        h = h.detach()
                    if mode in ("cut_all_every8", "cut_memory_every8"):
                        m, ek, ev = (x.detach() if x is not None else None for x in (m, ek, ev))
                    state = h, m, ek, ev
                state = checkpoint(inner.block, layer, state[0], injection, *state[1:], None,
                                   use_reentrant=False, preserve_rng_state=False)
            logits = inner.w_cls(state[0])
            loss = t.stablemax_cross_entropy(logits, common_batch["labels"]).mean()
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            combined = torch.cat([g.flatten() for g in grads.values()])
            record = dict(loss=float(loss.detach()), seconds=time.monotonic()-started,
                          accuracy=float((logits.argmax(-1)==common_batch["labels"]).float().mean()),
                          gradient_norm=float(combined.norm()),
                          norms={n: float(g.norm()) for n, g in grads.items()})
            if reference is None:
                reference = dict(logits=logits.detach(), gradients=grads, combined=combined)
            else:
                record["logits_vs_full"] = compare(logits, reference["logits"])
                record["gradient_vs_full"] = compare(combined, reference["combined"])
                record["parameter_gradients_vs_full"] = {n: compare(g, reference["gradients"][n]) for n, g in grads.items()}
            variants[mode] = record
            result["runs"][variant] = dict(checkpoint=str(Path(path).resolve()), modes=variants)
            out.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(variant, mode, {k: record[k] for k in ("loss", "accuracy", "seconds", "gradient_norm")},
                  record.get("gradient_vs_full", {}), flush=True)
            del state, loss, logits, grads, combined
        del model, inner, layer, ck, reference


if __name__ == "__main__":
    main()
