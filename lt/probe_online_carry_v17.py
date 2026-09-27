"""Compare real online training with fixed-weight inference on the same puzzles.

The last online forward uses theta15, before the 16th optimizer update. Its
comparison with a fresh theta15 replay isolates inconsistent retained history;
theta16 replay separately includes the final supervised update.
"""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt.ckpt_npz import load
from lt.compare_raw_ema_v17 import digest, import_original, normalize


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value


def clone_carry(carry):
    return replace(carry, **{field.name: clone_tree(getattr(carry, field.name))
                             for field in fields(carry)})


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def check_state(module, expected):
    actual = module.state_dict()
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key].detach().cpu(), value)
               for key, value in expected.items()), "Fixed-forward weights/buffers changed"


def measure(tv, logits, labels):
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    assert torch.isfinite(logits).all()
    pred = logits.argmax(-1)
    correct = pred == labels
    return dict(exact=int(correct.all(-1).sum()), cell_accuracy=float(correct.float().mean()),
                lm_loss=float(tv.stablemax_cross_entropy(logits, labels).mean())), pred.numpy().astype(np.uint8), logits.numpy()


def compare(reference_logits, candidate_logits, labels):
    a, b = reference_logits.argmax(-1), candidate_logits.argmax(-1)
    sa, sb = (a == labels).all(-1), (b == labels).all(-1)
    delta = candidate_logits.astype(np.float64) - reference_logits.astype(np.float64)
    return dict(logits_identical=bool(np.array_equal(reference_logits, candidate_logits)),
                logits_rms_difference=float(np.sqrt(np.mean(delta * delta))),
                logits_max_difference=float(np.abs(delta).max()),
                changed_cells=int((a != b).sum()), changed_puzzles=int((a != b).any(-1).sum()),
                gained_exact=int((~sa & sb).sum()), lost_exact=int((sa & ~sb).sum()),
                both_exact=int((sa & sb).sum()), neither_exact=int((~sa & ~sb).sum()))


def state_gap(a, b):
    result = {}
    for key in ("current_hidden", "trace", "coupling"):
        x, y = getattr(a, key, None), getattr(b, key, None)
        if x is None or y is None:
            assert x is None and y is None, f"Mismatched presence of carry component {key}"
            result[key] = dict(present=False)
            continue
        x, y = x.detach().float(), y.detach().float()
        delta = y - x
        result[key] = dict(present=True, reference_l2=float(x.norm()), difference_l2=float(delta.norm()),
                           relative_l2=float(delta.norm() / x.norm().clamp_min(1e-30)),
                           difference_rms=float(delta.square().mean().sqrt()))
    return result


def install_smooth_qr(inner):
    """Use positive-diagonal R, with a fixed sign anchor for the loaded model.

    The anchor is fixed across online updates and all snapshot evaluations.
    Re-anchoring at each snapshot would silently restore the discontinuous rule.
    """
    anchors, initial = {}, {}
    with torch.no_grad():
        for layer in inner.layers:
            _, r = torch.linalg.qr(layer.wc_raw.transpose(-1, -2))
            diagonal = r.diagonal(dim1=-2, dim2=-1)
            assert torch.all(diagonal != 0), "Smooth QR requires full-column-rank initial projection"
            anchors[id(layer)] = diagonal.sign().detach().clone()
            initial[id(layer)] = torch.cat(inner.W_C(layer), dim=1).detach().clone()

    def smooth_W_C(layer):
        q, r = torch.linalg.qr(layer.wc_raw.transpose(-1, -2))
        signs = r.diagonal(dim1=-2, dim2=-1).sign() * anchors[id(layer)]
        ab = (q * signs.unsqueeze(-2)).transpose(-1, -2)
        return ab[:, :inner.p, :], ab[:, inner.p:, :]

    inner.W_C = smooth_W_C
    with torch.no_grad():
        assert all(torch.equal(initial[id(layer)], torch.cat(inner.W_C(layer), dim=1))
                   for layer in inner.layers), "Smooth QR changed the initial model"
    return [anchors[id(layer)].cpu().tolist() for layer in inner.layers]


@torch.no_grad()
def fixed_rollout(tv, base, weights, batch, retained=None):
    base.model.load_state_dict(weights, strict=True)
    base.eval()
    with torch.device("cuda"):
        carry = base.model.initial_carry(batch)
    records, predictions = [], []
    consistent15 = None
    for segment in range(1, 17):
        carry, outputs = base.model(carry, batch)
        metrics, pred, logits = measure(tv, outputs["logits"], carry.current_data["labels"])
        records.append(dict(segment=segment, **metrics))
        predictions.append(pred)
        if segment == 15 and retained is not None:
            consistent15 = clone_carry(carry)
    result = dict(per_segment=records, final=records[-1], parameters_fixed=True)
    if retained is not None:
        assert torch.all(retained.steps == 15) and not retained.halted.any()
        assert all(torch.equal(retained.current_data[key], consistent15.current_data[key])
                   for key in retained.current_data)
        result["state_gap_before_final_segment"] = state_gap(retained, consistent15)
        replay_carry, outputs = base.model(clone_carry(retained), batch)
        replay_metrics, _, replay_logits = measure(tv, outputs["logits"], replay_carry.current_data["labels"])
        result["retained_carry_replay"] = replay_metrics
        labels = batch["labels"].cpu().numpy()
        result["retained_vs_consistent"] = compare(replay_logits, logits, labels)
    else:
        replay_logits = None
    check_state(base.model, weights)
    return result, np.stack(predictions), logits, replay_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/v17_recovery/checkpoints/v17_step_380000.npz")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/v17_recovery/source/2026-09-09/analysis/train_v17.py")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw",
                        help="Checkpoint initialization; EMA is a fresh-optimizer diagnostic, not a historical raw continuation")
    parser.add_argument("--smooth-qr", action="store_true",
                        help="Fix QR column signs with a positive-R convention anchored to the initial loaded model")
    parser.add_argument("--data", type=Path, default=ROOT / "data/sudoku_lt_1k.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/v17_recovery/online_carry_380k")
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    assert args.episodes > 0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tv = import_original(args.source)
    loaded, meta = load(str(args.checkpoint), which=args.weights)
    assert loaded, f"Checkpoint has no {args.weights} weights: {args.checkpoint}"
    cfg = dict(meta["cfg"])
    cfg.update(batch_size=128, global_batch_size=128, seq_len=81,
               num_puzzle_identifiers=1, loops=16, blocks_per_seg=8,
               lr=1e-4, puzzle_emb_lr=1e-4, lr_min_ratio=1., lr_warmup_steps=0,
               grad_accum_steps=1, amp=True, compile=False)
    with torch.device("cuda"):
        base = tv.ACTLossHead(tv.LT(cfg), "stablemax_cross_entropy", q_weight=cfg["q_weight"])
    base.model.load_state_dict(normalize(tv, loaded), strict=True)
    qr_anchors = install_smooth_qr(base.model.inner) if args.smooth_qr else None
    base.train()
    optimizers, optimizer_lrs = tv.create_optimizers(base, cfg, world_size=1)
    state = tv.TrainState(step=int(meta["step"]))
    with np.load(args.data, allow_pickle=False) as data:
        ds = tv.SudokuTrainDataset(data["train_inputs"].reshape(-1, 9, 9),
                                  data["train_labels"].reshape(-1, 9, 9), seed=0,
                                  num_aug=cfg["num_aug"], global_batch_size=128,
                                  rank=0, world_size=1, epochs_per_iter=250,
                                  start_iter=0, total_iters=(args.episodes * 16 + 1952) // 1953)
    stream = iter(ds)
    write_json(args.out / "metadata.json", dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        initialization_weights=args.weights,
        initialization_semantics=("EMA-initialized diagnostic with fresh optimizer; not historical raw training continuation"
                                  if args.weights == "ema" else "Raw checkpoint with fresh optimizer; historical optimizer/carry not restored"),
        source=str(args.source), source_sha256=digest(args.source), data_sha256=digest(args.data),
        qr_rule=("Q * sign(diag(R)) * initial_sign(diag(R)); initial model exactly preserved; anchor fixed across all snapshots"
                 if args.smooth_qr else "Unmodified source QR rule"),
        qr_initial_anchors=qr_anchors,
        config=cfg, episodes=args.episodes, optimizer="fresh original AdamATan2 + sparse signSGD",
        data_stream="original seed0 augmented stream starting at iter0; consume one loader batch per optimizer step",
        carry="original online carry/reset/detach semantics; no historical optimizer or carry restored",
        conditions={"start_fixed": "theta0, fresh 16 segments, before any update on this episode",
                    "online": "original 16 train_batch calls, one supervised update after every segment",
                    "last_forward_fixed": "theta15, fresh 16 segments; same weights as online final prediction",
                    "end_fixed": "theta16, fresh 16 segments; includes the final supervised update"},
        primary_comparison="online final vs last_forward_fixed: same theta15 and puzzles, different retained history",
        q_loss="q_halt logits are constant -5; q loss changes scalar total but contributes no parameter gradient",
        limitations=["Same training puzzles; no held-out performance claim.",
                     "Short continuation with fresh optimizer, not a reconstruction of 328k collapse.",
                     "Theta16 fixed improvement can be ordinary learning on this minibatch."] ))
    started = time.monotonic()
    all_results = []
    for episode in range(1, args.episodes + 1):
        theta0 = snapshot(base.model)
        online_records, online_predictions = [], []
        captured = {}
        def capture(module, inputs, outputs):
            carry, out = outputs
            metrics, pred, logits = measure(tv, out["logits"], carry.current_data["labels"])
            captured.update(metrics=metrics, pred=pred, logits=logits)
        hook = base.model.register_forward_hook(capture)
        episode_batch = None
        for segment in range(1, 17):
            if segment == 16:
                theta15 = snapshot(base.model)
                retained15 = clone_carry(state.carry)
            _, batch = next(stream)
            metrics = tv.train_batch(base, base, state, batch, cfg, optimizers, optimizer_lrs,
                                     total_steps=int(meta["step"]) + args.episodes * 16,
                                     rank=0, world_size=1, device=torch.device("cuda"))
            assert int(state.carry.steps[0]) == segment
            if segment == 1:
                episode_batch = clone_tree(state.carry.current_data)
            assert all(torch.equal(state.carry.current_data[key], episode_batch[key])
                       for key in episode_batch)
            online_records.append(dict(segment=segment, step=state.step,
                                       original_lm_loss=float(metrics["lm_loss"]), **captured["metrics"]))
            online_predictions.append(captured["pred"])
        hook.remove()
        online_logits = captured["logits"]
        theta16 = snapshot(base.model)
        labels = episode_batch["labels"].cpu().numpy()
        record = dict(episode=episode, final_step=state.step, online=dict(per_segment=online_records, final=online_records[-1]))
        arrays = dict(inputs=episode_batch["inputs"].cpu().numpy(), labels=labels,
                      online_predictions=np.stack(online_predictions), online_final_logits=online_logits)
        try:
            for name, weights in [("start_fixed", theta0), ("last_forward_fixed", theta15), ("end_fixed", theta16)]:
                result, preds, logits, retained_logits = fixed_rollout(
                    tv, base, weights, episode_batch, retained15 if name == "last_forward_fixed" else None)
                record[name] = result
                arrays[name + "_predictions"] = preds
                arrays[name + "_final_logits"] = logits
                record[name]["online_to_fixed"] = compare(online_logits, logits, labels)
                if retained_logits is not None:
                    control = compare(online_logits, retained_logits, labels)
                    record["online_final_reproduction"] = control
                    assert control["logits_identical"], "Eval replay of same theta15/carry did not reproduce online forward"
                    arrays["retained_carry_replay_logits"] = retained_logits
                print(json.dumps(dict(episode=episode, condition=name, **result["final"]), ensure_ascii=False), flush=True)
        finally:
            base.model.load_state_dict(theta16, strict=True)
            base.train()
            check_state(base.model, theta16)
        record["elapsed_seconds"] = time.monotonic() - started
        np.savez_compressed(args.out / f"episode_{episode:02d}_outputs.npz", **arrays)
        write_json(args.out / f"episode_{episode:02d}.json", record)
        all_results.append(record)
        write_json(args.out / "progress.json", dict(episodes_completed=len(all_results), results=all_results))
        del theta0, theta15, theta16, retained15
    summary = dict(episodes_completed=len(all_results), final_step=state.step,
                   elapsed_seconds=time.monotonic() - started,
                   all_same_weight_retained_replays_exact=True,
                   results=[dict(episode=r["episode"],
                                 exact={name:r[name]["final"]["exact"] for name in ("online", "start_fixed", "last_forward_fixed", "end_fixed")},
                                 lm_loss={name:r[name]["final"]["lm_loss"] for name in ("online", "start_fixed", "last_forward_fixed", "end_fixed")},
                                 same_theta_retained_vs_consistent=r["last_forward_fixed"]["retained_vs_consistent"])
                            for r in all_results])
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
