"""Opt-in training entry point for the even phase-plasticity research candidate.

Uses the existing LT data, loss, optimizer, EMA, and evaluation harness. The
production train.py and its default architecture are unchanged. Checkpoints
written here use the even model's parameter names and packed carry format.
"""

import os

import train
from even_plasticity import EvenLT

# For explicit checkpoint loading with ckpt_npz.load_lt(..., mod=train_even).
# Importing this module alone does not change the production train module.
LT = EvenLT
CFG = train.CFG


def configure_harness():
    train.LT = EvenLT
    train.NO_DECAY_KEYS = tuple(dict.fromkeys(
        (*train.NO_DECAY_KEYS, "base_raw", "plastic_spectrum")))
    train.CFG["out_dir"] = os.path.join(train.REPO, "runs", "even_v11")
    train.CFG["architecture"] = "even_phase_plasticity"
    train.CFG["plastic_memory_format"] = "upper_triangle_including_diagonal"
    train.CFG["plastic_initialization"] = "first_write_bootstrap"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Experimental even-window LT training, v1.1 scaffold.")
    ap.add_argument("--preset", choices=["v1.1"], default="v1.1")
    ap.add_argument("--out_dir", default=os.path.join(train.REPO, "runs", "even_v11"))
    ap.add_argument("--resume_from")
    ap.add_argument("--data")
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--global_batch_size", type=int)
    ap.add_argument("--milestone_every", type=int)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()
    configure_harness()
    train.CFG.update(train.PRESETS["v1.1"])
    for key in ("out_dir", "resume_from", "max_steps", "epochs", "seed",
                "global_batch_size", "milestone_every"):
        value = getattr(args, key)
        if value is not None:
            train.CFG[key] = value
    if args.data:
        train.CFG["data_npz"] = args.data
    if args.no_compile:
        train.CFG["compile"] = False
    print(f"[Even LT] learned even plasticity window; out_dir={train.CFG['out_dir']}", flush=True)
    train.run()


if __name__ == "__main__":
    main()
