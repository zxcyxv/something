"""Isolate the gradient change caused by retaining event-time key coordinates.

At fixed weights, original/raw-key and historical/rotated-key traces describe
the same forward state. At a truncated-gradient boundary, the stored state is
constant: only the original code differentiates its re-rotation by today's
theta. This probe compares both at identical weights, inputs and physical state.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from . import train as t
from .audit_kv_stdp_runtime import compare
from .kv_stability import install


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"], batch_size=args.batch, seq_len=81, num_puzzle_identifiers=1,
               amp=False, activation_checkpoint=True, nograd_blocks=0)
    saved = ck["rank_states"][0]["carry"]
    batch = {k: v[:args.batch].to(args.device) for k, v in saved["current_data"].items()}
    inputs = {k: v[:args.batch].to(args.device) for k, v in saved.items()
              if k != "current_data" and isinstance(v, torch.Tensor)}
    inputs["halted"] = torch.zeros(args.batch, dtype=torch.bool, device=args.device)
    inputs["steps"] = torch.full((args.batch,), cfg["loops"]-1, dtype=torch.int32, device=args.device)
    carry = t.LTCarry(**inputs, current_data=batch)
    models = {}
    for variant in ("original", "historical_key_trace"):
        install(variant)
        with torch.device(args.device):
            model = t.ACTLossHead(t.LT(cfg), q_weight=cfg["q_weight"])
        model.load_state_dict(ck["raw_model_state_dict"])
        models[variant] = model
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), device=args.device,
                  precision="fp32", batch=args.batch, cases={})
    for case in ("fresh", "detached_carry"):
        values = {}
        for variant, model in models.items():
            model.zero_grad(set_to_none=True)
            c = model.initial_carry(batch) if case == "fresh" else carry
            if case == "detached_carry" and variant == "historical_key_trace":
                with torch.no_grad():
                    c = replace(c, key_trace=model.model.inner.apply_rope(c.key_trace).detach().requires_grad_(True))
            _, loss, _, outputs, _ = model(carry=c, batch=batch, return_keys={"logits"})
            (loss/args.batch).backward()
            values[variant] = dict(loss=float(loss.detach()/args.batch), logits=outputs["logits"].cpu(),
                                   gradients={n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                                              if p.grad is not None})
            if case == "detached_carry" and variant == "historical_key_trace":
                boundary = model.model.inner.apply_rope(carry.key_trace)
                term = torch.autograd.grad((boundary*c.key_trace.grad.detach()).sum(),
                                           model.model.inner.layers[0].theta)[0]
                values[variant]["boundary_theta_term"] = term.detach().cpu()
        a, b = values["original"], values["historical_key_trace"]
        gradients = {n: compare(b["gradients"][n], g) for n, g in a["gradients"].items()}
        result["cases"][case] = dict(original_loss=a["loss"], historical_loss=b["loss"],
                                      logits=compare(b["logits"], a["logits"]), gradients=gradients)
        if case == "detached_carry":
            difference = a["gradients"]["model.inner.layers.0.theta"]-b["gradients"]["model.inner.layers.0.theta"]
            result["cases"][case]["gradient_difference_vs_boundary_chain_rule"] = compare(
                difference, b["boundary_theta_term"])
        print(case, "logits", result["cases"][case]["logits"],
              "theta_gradient", gradients["model.inner.layers.0.theta"], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
