"""Compare real-checkpoint forward/backward across the training execution paths.

Read-only with respect to checkpoints; no optimizer updates or new checkpoints.
Also measure subtraction accuracy on identical captured Q/K/V inputs and
report the learned trace decay range.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from . import train as t


def compare(actual, expected):
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    return dict(max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float(F.cosine_similarity(a, b, dim=0, eps=1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(47)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"], batch_size=args.batch, seq_len=81,
               num_puzzle_identifiers=1, nograd_blocks=0)
    saved = ck["rank_states"][0]["carry"]
    batch = {k: v[:args.batch].cuda() for k, v in saved["current_data"].items()}
    inputs = {k: v[:args.batch].cuda() for k, v in saved.items()
              if k != "current_data" and isinstance(v, torch.Tensor)}
    # Exercise non-reset carry from a real checkpoint. Termination flags do not
    # affect the inner dynamics; force continuation to test all state inputs.
    inputs["halted"] = torch.zeros(args.batch, dtype=torch.bool, device="cuda")
    inputs["steps"] = torch.full((args.batch,), cfg["loops"]-1, dtype=torch.int32, device="cuda")
    carry = t.LTCarry(**inputs, current_data=batch)
    with torch.device("cuda"):
        base = t.ACTLossHead(t.LT(cfg), q_weight=cfg["q_weight"])
    base.load_state_dict(ck["raw_model_state_dict"])
    base.train()
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck["step"],
                  batch=args.batch, state="saved carry, forced continuation, frozen parameters",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  tf32=torch.backends.cuda.matmul.allow_tf32,
                  float32_matmul_precision=torch.get_float32_matmul_precision(), modes={})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(result, indent=2, allow_nan=False))

    def run(model):
        base.zero_grad(set_to_none=True)
        if base.model.puzzle_emb is not None:
            base.model.puzzle_emb.local_weights.grad = None
        next_carry, loss, _, outputs, _ = model(carry=carry, batch=batch, return_keys={"logits"})
        (loss/args.batch).backward()
        torch.cuda.synchronize()
        grads = {n: p.grad.detach().cpu().clone() for n, p in base.named_parameters() if p.grad is not None}
        return dict(loss=float(loss.detach()/args.batch), logits=outputs["logits"].cpu(),
                    state={n: getattr(next_carry, n).detach().cpu() for n in
                           ("current_hidden", "coupling", "key_trace", "value_trace")}, grads=grads)

    modes = [("fp32_eager", False, False), ("bf16_eager", True, False),
             ("bf16_checkpoint", True, True)]
    if not args.skip_compile:
        modes.append(("bf16_compiled_checkpoint", True, True))
    reference = None
    for name, amp, checkpoint in modes:
        base.model.config.amp = amp
        base.model.config.activation_checkpoint = checkpoint
        if "compiled" in name:
            import torch._inductor.config as ic
            ic.triton.persistent_reductions = False
            model = torch.compile(base, dynamic=False)
        else:
            model = base
        started = time.monotonic()
        actual = run(model)
        record = dict(loss=actual["loss"], seconds=time.monotonic()-started,
                      absent_gradients=[n for n, p in base.named_parameters() if p.grad is None],
                      nonfinite_gradients=[n for n, g in actual["grads"].items() if not bool(g.isfinite().all())],
                      zero_gradients=[n for n, g in actual["grads"].items() if not bool(g.count_nonzero())])
        if reference is not None:
            record["reference"] = "bf16_eager" if name != "bf16_eager" else "fp32_eager"
            record["logits"] = compare(actual["logits"], reference["logits"])
            record["state"] = {n: compare(x, reference["state"][n]) for n, x in actual["state"].items()}
            record["gradients"] = {n: compare(g, reference["grads"][n])
                                   for n, g in actual["grads"].items() if n in reference["grads"]}
            record["all_gradients"] = compare(torch.cat([g.flatten() for g in actual["grads"].values()]),
                                              torch.cat([g.flatten() for g in reference["grads"].values()]))
        if name in ("fp32_eager", "bf16_eager"):
            reference = actual
        result["modes"][name] = record
        print(name, {k: record[k] for k in ("loss", "seconds", "absent_gradients", "nonfinite_gradients")},
              record.get("all_gradients", {}), flush=True)
        save()

    # Single-write cancellation on the same inputs, without changing trajectory.
    inner, layer = base.model.inner, base.model.inner.layers[0]
    inner.config.amp = True
    inner.config.activation_checkpoint = False
    original = inner.memory_step
    samples = []

    def inspected(L, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
        actual = original(L, q, k, v, memory, e_k, e_v, fresh)
        with torch.autocast("cuda", enabled=False):
            ek = torch.zeros_like(k, dtype=torch.float32) if e_k is None else e_k
            ev = torch.zeros_like(v, dtype=torch.float32) if e_v is None else e_v
            if fresh is not None:
                ek = torch.where(fresh[:, None, None, None], 0., ek)
                ev = torch.where(fresh[:, None, None, None], 0., ev)
            kr = inner.apply_rope(k.float(), L)
            er = inner.apply_rope(ek.float(), L)
            positive = v.float().transpose(-1, -2) @ er
            negative = ev.float().transpose(-1, -2) @ kr
            delta = positive-negative
            exact = v.double().transpose(-1, -2) @ er.double() - ev.double().transpose(-1, -2) @ kr.double()
            now = v.float().transpose(-1, -2) @ kr
            record = dict(block=len(samples)+1, fp32_subtraction=compare(delta, exact),
                          difference_over_current=float(delta.norm()/now.norm().clamp_min(1e-30)),
                          cancellation_ratio=float(delta.norm()/(positive.norm()+negative.norm()).clamp_min(1e-30)),
                          delta_norm=float(delta.norm()/k.shape[-2]),
                          memory_norm=float(actual[1].norm()))
            samples.append(record)
        return actual

    inner.memory_step = inspected
    with torch.no_grad():
        base.model(carry, batch)
    inner.memory_step = original
    result["same_input_subtraction"] = samples
    lam = layer.trace_decay.detach().cpu()
    result["trace_decay"] = dict(min=float(lam.min()), max=float(lam.max()),
                                 mean=float(lam.mean()), per_head_std=lam.std(-1).tolist(),
                                 effective_one_block_tau_min=float(-1/lam.min().log()),
                                 effective_one_block_tau_max=float(-1/lam.max().log()))
    save()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
