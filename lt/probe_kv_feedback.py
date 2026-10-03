"""Intervene on a frozen recurrent KV model after an identical warm-up.

Each branch starts from the same state. These are causal inference-time probes,
not trained-model comparisons or evidence of a successful training correction.
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .kv_stability import install


def rms(x):
    return float(x.float().square().mean().sqrt())


def mean_metrics(rows, keys):
    # The first row has no two-step predecessor. Keep the full available window
    # for other metrics and average two-step metrics over their valid rows.
    return {key: sum(row[key] for row in rows if key in row)/sum(key in row for row in rows)
            for key in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--burn", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--data-checkpoint", help="Use these saved puzzles for cross-model comparisons.")
    ap.add_argument("--variants", nargs="+", choices=("original", "freeze_memory", "read_previous",
                    "freeze_traces", "mute_read", "read_gain_0.75", "read_gain_0.5", "read_gain_0.25", "pre_ffn_phi"),
                    default=("original", "freeze_memory", "read_previous", "freeze_traces", "mute_read",
                             "read_gain_0.75", "read_gain_0.5", "read_gain_0.25", "pre_ffn_phi"))
    args = ap.parse_args()
    if args.blocks < 3:
        ap.error('--blocks must be at least three for two-step comparisons in both parities')
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    install(ck["cfg"].get("research_variant", "original"))
    cfg = dict(ck["cfg"], batch_size=args.batch, seq_len=81, num_puzzle_identifiers=1,
               activation_checkpoint=False, nograd_blocks=0)
    with torch.device("cuda"):
        model = t.LT(cfg)
    model.load_state_dict({k.removeprefix("model."): v for k, v in ck["raw_model_state_dict"].items()})
    model.eval()
    inner, layer = model.inner, model.inner.layers[0]
    data_ck = (torch.load(args.data_checkpoint, map_location="cpu", weights_only=False)
               if args.data_checkpoint else ck)
    saved = data_ck["rank_states"][0]["carry"]["current_data"]
    batch = {k: v[:args.batch].cuda() for k, v in saved.items()}
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck["step"],
                  batch=args.batch, burn=args.burn, blocks=args.blocks,
                  precision="fp32" if args.fp32 else "bf16_projections_fp32_states",
                  data_checkpoint=str(Path(args.data_checkpoint or args.checkpoint).resolve()),
                  caveat="Frozen inference intervention; branch accuracy is not a retraining result.", phases={})
    original_memory, original_update = inner.memory_step, inner.update_memory
    original_boundary = inner.boundary
    branch, current = "original", {}

    def memory_step(L, q, k, v, m=None, ek=None, ev=None, fresh=None):
        read, new_m, new_ek, new_ev = original_memory(L, q, k, v, m, ek, ev, fresh)
        if branch == "read_previous":
            with torch.autocast("cuda", enabled=False):
                read = inner.apply_rope(q.float(), L) @ m.float().transpose(-1, -2)
        elif branch == "mute_read":
            read = torch.zeros_like(read)
        elif branch.startswith("read_gain_"):
            read = read*float(branch.removeprefix("read_gain_"))
        elif branch == "freeze_traces":
            new_ek, new_ev = ek, ev
        current.update(memory_rms=rms(new_m), write_rms=rms(new_m-m), read_rms=rms(read))
        return read, new_m, new_ek, new_ev

    def boundary(L, h):
        current["ffn_input_rms"] = rms(h)
        if branch == "pre_ffn_phi":
            h = inner.phi(h)
        after = original_boundary(L, h)
        current["ffn_delta_rms"] = rms(after-h)
        return after

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32,
                                        cache_enabled=False):
        injection = inner.injection(batch)
        state = (inner.init_hidden[None, None, :].expand(args.batch, 81, -1), None, None, None)
        phases = []
        for block in range(1, args.burn+1):
            state = inner.block(layer, state[0], injection, *state[1:], None)
            if block >= args.burn-1:
                phases.append((block, state))
        inner.memory_step, inner.boundary = memory_step, boundary
        for phase, initial in phases:
            results = {}
            for branch in args.variants:
                inner.update_memory = (lambda m, write: m) if branch == "freeze_memory" else original_update
                state = tuple(x.clone() for x in initial)
                history = [state[0]]
                predictions = [inner.w_cls(state[0]).argmax(-1)]
                rows = []
                for block in range(1, args.blocks+1):
                    current.clear()
                    state = inner.block(layer, state[0], injection, *state[1:], None)
                    h = state[0]
                    logits = inner.w_cls(h).float()
                    pred = logits.argmax(-1)
                    row = dict(block=block, **current, hidden_one_step_rms=rms(h-history[-1]),
                               accuracy=float((pred==batch["labels"]).float().mean()),
                               exact=float((pred==batch["labels"]).all(-1).float().mean()),
                               loss=float(t.stablemax_cross_entropy(logits, batch["labels"]).mean()),
                               prediction_one_step_flip=float((pred!=predictions[-1]).float().mean()))
                    if len(history) > 1:
                        row["hidden_two_step_rms"] = rms(h-history[-2])
                        row["prediction_two_step_flip"] = float((pred!=predictions[-2]).float().mean())
                    rows.append(row)
                    history.append(h); predictions.append(pred)
                    history, predictions = history[-2:], predictions[-2:]
                keys = ("accuracy", "loss", "hidden_one_step_rms", "hidden_two_step_rms",
                        "prediction_one_step_flip", "prediction_two_step_flip", "memory_rms",
                        "write_rms", "read_rms", "ffn_input_rms", "ffn_delta_rms")
                tail = mean_metrics(rows[-16:], keys)
                by_parity = {}
                for parity in (0, 1):
                    selected = [row for row in rows[-16:] if (phase+row['block']) % 2 == parity]
                    by_parity[str(parity)] = mean_metrics(selected, keys)
                results[branch] = dict(last=rows[-1], last16_mean=tail,
                                      last16_by_absolute_block_parity=by_parity, rows=rows)
                print(phase, branch, {key: round(tail[key], 6) for key in
                      ("accuracy", "hidden_one_step_rms", "hidden_two_step_rms", "prediction_one_step_flip")}, flush=True)
            result["phases"][str(phase)] = results
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
