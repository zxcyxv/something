"""Exercise the actual training/loss/optimizer/checkpoint harness on CPU.

CUDA is hidden before importing torch because the original checkpoint writer
also saves CUDA RNG state when a GPU is visible. No external job is changed.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path

import torch

import train
from train_even import configure_harness


def main():
    torch.set_num_threads(2)
    torch.manual_seed(91)
    configure_harness()
    cfg = dict(train.CFG, **train.PRESETS["v1.1"])
    cfg.update(batch_size=2, seq_len=9, grid=3, hidden_size=32, num_heads=2,
               num_puzzle_identifiers=1, puzzle_emb_ndim=32, global_batch_size=2,
               blocks_per_seg=3, loops=2, amp=False, compile=False, lr_warmup_steps=0)
    base = train.ACTLossHead(train.LT(cfg), "stablemax_cross_entropy", q_weight=cfg["q_weight"])
    opts, lrs = train.create_optimizers(base, cfg, world_size=1)
    no_decay = {id(p) for group in opts[1].param_groups if group["weight_decay"] == 0
                for p in group["params"]}
    for name, p in base.named_parameters():
        if name.endswith(("base_raw", "plastic_spectrum", "eta_raw", "psi")):
            assert id(p) in no_decay, name
        if name.endswith("w_sh"):
            assert id(p) not in no_decay, name
    batch = dict(inputs=torch.randint(1, 11, (2, 9)), labels=torch.randint(1, 11, (2, 9)),
                 puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
    state = train.TrainState()
    ema = train.EMAHelper(mu=cfg["ema_rate"])
    ema.register(base)
    metrics = []
    for _ in range(6):
        row = train.train_batch(base, base, state, batch, cfg, opts, lrs,
                                total_steps=6, rank=0, world_size=1, device=torch.device("cpu"))
        metrics.append({key: float(value) for key, value in row.items()})
        ema.update(base)
    assert state.step == 6 and torch.isfinite(state.carry.current_hidden).all()
    assert state.carry.coupling.shape == (2, 2, 45)
    out = Path("runs/even_harness_cpu")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = train.save_checkpoint(str(out), state.step, base, opts, ema, 0, 6, cfg, keep_last=1)
    restored = train.ACTLossHead(train.LT(cfg), "stablemax_cross_entropy", q_weight=cfg["q_weight"])
    ropts, _ = train.create_optimizers(restored, cfg, world_size=1)
    ck = train.load_checkpoint(checkpoint, restored, ropts, torch.device("cpu"))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert ck["cfg"]["architecture"] == "even_phase_plasticity"
    base.eval(); restored.eval()
    with torch.no_grad():
        _, _, _, original_out, _ = base(carry=base.initial_carry(batch), batch=batch, return_keys={"logits"})
        _, _, _, restored_out, _ = restored(carry=restored.initial_carry(batch), batch=batch, return_keys={"logits"})
    torch.testing.assert_close(original_out["logits"], restored_out["logits"], atol=0, rtol=0)
    report = dict(train_steps=state.step, packed_carry_shape=list(state.carry.coupling.shape),
                  no_decay_policy_checked=True, checkpoint_raw_weights_exact=True,
                  checkpoint_logits_exact=True, architecture=ck["cfg"]["architecture"],
                  metrics=metrics, cuda_initialized=torch.cuda.is_initialized())
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
