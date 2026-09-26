"""Opt-in training entry point for the untrained causal-STDP architecture draft."""

import argparse
import copy
import json
import os

if __package__:
    from . import train
    from .causal_stdp import CausalLT
else:
    import train
    from causal_stdp import CausalLT


LT = CausalLT
CFG = dict(train.CFG, **train.PRESETS["v1.1"])
CFG.update(architecture="causal_stdp_v1_2_draft",
           out_dir=os.path.join(train.REPO, "runs", "causal_stdp_v1_2"),
           activity_channels=16, timing_tau_init=[2., 8., 32., 128.],
           timing_tau_min=1., timing_tau_max=256., timing_amplitude_floor=1e-4)


def configure_harness(cfg):
    train.LT = CausalLT
    train.CFG = copy.deepcopy(cfg)
    train.NO_DECAY_KEYS = tuple(dict.fromkeys(
        (*train.NO_DECAY_KEYS, "timing_tau_raw", "timing_amplitude_logits")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON overrides")
    parser.add_argument("--out_dir")
    parser.add_argument("--resume_from", help="A checkpoint from this architecture only")
    for key in ("max_steps", "seed", "global_batch_size", "milestone_every"):
        parser.add_argument("--"+key, type=int)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()
    cfg = copy.deepcopy(CFG)
    if args.config:
        with open(args.config) as file:
            cfg.update(json.load(file))
    for key in ("out_dir", "resume_from", "max_steps", "seed", "global_batch_size", "milestone_every"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_compile:
        cfg["compile"] = False
    configure_harness(cfg)
    print(f"[Causal STDP draft] out_dir={cfg['out_dir']}", flush=True)
    train.run()


if __name__ == "__main__":
    main()
