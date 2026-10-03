"""Quantify historical-key vs current-write RoPE conventions across updates.

The production trace is in unrotated coordinates. At fixed theta this is
exactly equivalent to a trace of rotated activities. Changing theta while
retaining the trace distinguishes the two interpretations; this script makes
that distinction explicit, without modifying the model or training it.
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .audit_kv_stdp_runtime import compare
from .test_kv_stdp_reference import config


def synthetic_counterexample():
    inner = t.KVSTDPInner(config()).double()
    layer = inner.layers[0]
    z = torch.zeros(2, 2, 4, 8, dtype=torch.float64)
    k, v = z.clone(), z.clone()
    k[0, 0, 1, 0] = 1
    v[0, 0, 1, 2] = 1
    with torch.no_grad():
        layer.theta.zero_()
        _, m, ek, ev = inner.memory_step(layer, z, k, z)
        old_rotated_ek = inner.apply_rope(ek, layer)
        layer.theta[0, 0, 1] = torch.pi/2
        _, actual, _, _ = inner.memory_step(layer, z, z, v, m, ek, ev)
        historical = v.transpose(-1, -2) @ old_rotated_ek/4
    return dict(actual_row=actual[0, 0, 2, :2].tolist(),
                historical_key_row=historical[0, 0, 2, :2].tolist(),
                difference=compare(actual, historical),
                interpretation="Different only when theta changes between paired events; not a fixed-weight algebra error.")


@torch.no_grad()
def checkpoint_measurement(path, batch_size):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"], batch_size=batch_size, seq_len=81,
               num_puzzle_identifiers=1, activation_checkpoint=False, nograd_blocks=0)
    with torch.device("cuda"):
        base = t.ACTLossHead(t.LT(cfg), q_weight=cfg["q_weight"])
    base.load_state_dict(ck["raw_model_state_dict"])
    optimizers, _ = t.create_optimizers(base, cfg, 1)
    for optimizer, saved in zip(optimizers, ck["optimizer_states"]):
        optimizer.load_state_dict(saved)
    layer = base.model.inner.layers[0]
    for group in optimizers[-1].param_groups:
        if any(p is layer.theta for p in group["params"]):
            state = optimizers[-1].state[layer.theta]
            b1, b2 = group["betas"]
            step = state["step"]
            correction = group["lr"]*torch.atan2(state["m"]/(1-b1**step),
                                                 (state["v"]/(1-b2**step)).sqrt())
            assert group["weight_decay"] == 0
            break
    inner = base.model.inner
    current_theta = layer.theta.clone()
    previous_theta = current_theta+correction
    saved = ck["rank_states"][0]["carry"]
    batch = {k: v[:batch_size].cuda() for k, v in saved["current_data"].items()}
    h, memory, ek, ev = (saved[n][:batch_size].cuda() for n in
                         ("current_hidden", "coupling", "key_trace", "value_trace"))
    base.eval()
    hp = h + inner.embed_scale*inner.injection(batch)
    def heads(x):
        return x.reshape(batch_size, 81, inner.H, inner.dh).transpose(1, 2)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        q, k, v = [heads(projection(hp)).float() for projection in
                   (layer.q_proj, layer.k_proj, layer.v_proj)]
    layer.theta.copy_(previous_theta)
    old_rotated_ek = inner.apply_rope(ek, layer)
    layer.theta.copy_(current_theta)
    current_rotated_ek = inner.apply_rope(ek, layer)
    kr, qr = inner.apply_rope(k, layer), inner.apply_rope(q, layer)
    delta = (v.transpose(-1, -2) @ current_rotated_ek - ev.transpose(-1, -2) @ kr)/81
    historical = (v.transpose(-1, -2) @ old_rotated_ek - ev.transpose(-1, -2) @ kr)/81
    read = qr @ (memory+delta).transpose(-1, -2)
    reference_read = qr @ (memory+historical).transpose(-1, -2)
    lam = layer.trace_decay_channels[None, :, None, :]
    head_lam = lam.mean(-1, keepdim=True)
    # At identical stationary activities, unequal channel decays create a
    # finite startup boundary term. This is a heterogeneous window, not the
    # single shared antisymmetric L(d) used in a simplified derivation.
    steady_k, steady_v = k, v
    e_k, e_v = torch.zeros_like(k), torch.zeros_like(v)
    learned_m = torch.zeros_like(memory)
    shared_m = torch.zeros_like(memory)
    shared_ek, shared_ev = e_k.clone(), e_v.clone()
    for _ in range(32):
        learned_m += (steady_v.transpose(-1, -2) @ inner.apply_rope(e_k, layer)
                      - e_v.transpose(-1, -2) @ kr)/81
        shared_m += (steady_v.transpose(-1, -2) @ inner.apply_rope(shared_ek, layer)
                     - shared_ev.transpose(-1, -2) @ kr)/81
        e_k, e_v = lam*e_k+(1-lam)*steady_k, lam*e_v+(1-lam)*steady_v
        shared_ek = head_lam*shared_ek+(1-head_lam)*steady_k
        shared_ev = head_lam*shared_ev+(1-head_lam)*steady_v
    current_kv = steady_v.transpose(-1, -2) @ kr/81
    return dict(checkpoint=str(Path(path).resolve()), step=ck["step"], batch=batch_size,
                theta_update_absmax=float(correction.abs().max()),
                theta_update_rms=float(correction.square().mean().sqrt()),
                rotated_trace=compare(current_rotated_ek, old_rotated_ek),
                write=compare(delta, historical), read=compare(read, reference_read),
                limitation="One actual optimizer theta update, applied to the saved raw trace; not a replay of every historical theta or a training intervention.",
                constant_activity_startup=dict(
                    heterogeneous_memory_over_current_kv=float(learned_m.norm()/current_kv.norm()),
                    shared_memory_over_current_kv=float(shared_m.norm()/current_kv.norm()),
                    explanation="Per-channel time constants define L_ij(d), not one common odd L(d)."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(2)
    result = dict(synthetic=synthetic_counterexample(), checkpoint=checkpoint_measurement(args.checkpoint, args.batch))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
