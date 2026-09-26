"""Matched two-cell pulse order tests of a trained v1.1 checkpoint.

Two fixed hidden-pattern pulses, identical start state, fixed directed edges.
Four-arm subtraction isolates interaction beyond each pulse alone.  This
measures conditional order sensitivity, not a global STDP law or its origin
in training.  Positive interaction is not the same as positive total Delta W.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def interaction(values, offset, order):
    # Each six-arm group: Aearly, Blate, AB, Bearly, Alate, BA.
    indices = (3, 1, 2) if order == "AB" else (6, 4, 5)
    both, first, second = (offset + i for i in indices)
    return values[:, both] - values[:, first] - values[:, second] + values[:, 0]


def classify(ab, ba, coarse_ab, coarse_ba, absolute_floor=1e-6):
    # A positive pulse at source n before target t is the causal order for t<-n.
    scale = max(abs(ab), abs(ba))
    error = max(abs(ab - coarse_ab), abs(ba - coarse_ba))
    converged = error <= max(absolute_floor, 0.1 * scale)
    reliable_ab = abs(ab) > max(absolute_floor, 5 * abs(ab - coarse_ab))
    reliable_ba = abs(ba) > max(absolute_floor, 5 * abs(ba - coarse_ba))
    order_difference = abs(ab - ba) > max(
        absolute_floor, 5 * (abs(ab - coarse_ab) + abs(ba - coarse_ba)))
    if not converged:
        label = "not_converged"
    elif not (reliable_ab and reliable_ba):
        label = "weak_or_sign_uncertain"
    elif ab > 0 and ba < 0:
        label = "source_first_positive_reverse_negative"
    elif ab < 0 and ba > 0:
        label = "source_first_negative_reverse_positive"
    elif ab > 0 and ba > 0:
        label = "both_positive"
    else:
        label = "both_negative"
    return dict(classification=label, epsilon_converged=bool(converged),
                reliable_order_difference=bool(order_difference),
                max_epsilon_difference=float(error),
                source_first=float(ab), target_first=float(ba),
                even_component=float((ab + ba) / 2),
                odd_component=float((ab - ba) / 2))


def run_lag(blocks, h0, w0, start, lag, tail, epsilons, reference):
    source, target = 66, 48
    arms = 1 + 6 * len(epsilons)
    h = h0.repeat(arms, 1, 1)
    w = w0.repeat(arms, 1, 1, 1)
    # The same vectors are used for early and late injections in every arm.
    # eps=0.002 means adding 0.2% of the cell's starting hidden pattern.
    a, b = h0[0, source].clone(), h0[0, target].clone()
    records = {key: [] for key in ("G", "W", "window", "agree")}
    baseline_disagreements = 0
    for tick in range(lag + tail + 1):
        for group, epsilon in enumerate(epsilons):
            offset = 6 * group
            if tick == 0:
                h[[offset + 1, offset + 3], source] += epsilon * a
                h[[offset + 4, offset + 6], target] += epsilon * b
            if tick == lag:
                h[[offset + 2, offset + 3], target] += epsilon * b
                h[[offset + 5, offset + 6], source] += epsilon * a
        parts = blocks.parts(h)
        wn = blocks.rho * w + blocks.eta * parts["G"]
        for key in records:
            value = wn if key == "W" else parts[key]
            selected = torch.stack((value[:, :, target, source],
                                    value[:, :, source, target]), dim=-1)
            records[key].append(selected.numpy().copy())
        h = blocks.transmit_parts(parts, wn)
        w = wn
        predicted = blocks.inner.w_cls(h[:1]).argmax(-1)[0].numpy()
        baseline_disagreements += int((predicted != reference[start + tick + 1]).sum())
    assert baseline_disagreements == 0, (start, lag, baseline_disagreements)
    arrays = {key: np.stack(value) for key, value in records.items()}
    responses = {}
    max_identity_error, max_before_second = 0.0, 0.0
    for group, epsilon in enumerate(epsilons):
        offset = 6 * group
        for order in ("AB", "BA"):
            dg = interaction(arrays["G"], offset, order)
            dw = interaction(arrays["W"], offset, order)
            previous = np.concatenate((np.zeros_like(dw[:1]), dw[:-1]), axis=0)
            expected = blocks.rho.numpy().reshape(1, -1, 1) * previous
            expected += blocks.eta.numpy().reshape(1, -1, 1) * dg
            max_identity_error = max(max_identity_error, float(np.max(np.abs(dw - expected))))
            if lag:
                max_before_second = max(max_before_second, float(np.max(np.abs(dw[:lag]))),
                                         float(np.max(np.abs(dg[:lag]))))
            responses[f"eps{group}_{order}_G"] = dg / epsilon**2
            responses[f"eps{group}_{order}_W"] = dw / epsilon**2
    assert max_identity_error < 1e-12
    assert max_before_second < 1e-12
    if lag == 0:
        for group in range(len(epsilons)):
            for kind in ("G", "W"):
                np.testing.assert_allclose(responses[f"eps{group}_AB_{kind}"],
                                           responses[f"eps{group}_BA_{kind}"], atol=1e-8, rtol=1e-8)
    head_results = []
    for head in range(w0.shape[1]):
        fields = {}
        for name, kind, tick in (("G_at_second_pulse", "G", lag),
                                 ("W_at_second_pulse", "W", lag),
                                 ("W_after_tail", "W", -1)):
            edges = []
            for edge in range(2):
                ab = responses[f"eps1_AB_{kind}"][tick, head, edge]
                ba = responses[f"eps1_BA_{kind}"][tick, head, edge]
                cab = responses[f"eps0_AB_{kind}"][tick, head, edge]
                cba = responses[f"eps0_BA_{kind}"][tick, head, edge]
                # Source-first for the reverse edge is the BA chronology.
                edges.append(classify(ab, ba, cab, cba) if edge == 0
                             else classify(ba, ab, cba, cab))
            fields[name] = dict(forward_edge=edges[0], reverse_edge=edges[1])
        head_results.append(dict(head_zero_based=head, **fields))
    report = dict(snapshot_after_block=start, lag_blocks=lag, tail_blocks=tail,
                  baseline_prediction_disagreements=baseline_disagreements,
                  unscaled_pair_memory_recurrence_max_error=max_identity_error,
                  unscaled_pair_effect_before_second_pulse_max=max_before_second,
                  heads=head_results)
    arrays.update(responses)
    return report, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/order_pair_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        checkpoint="checkpoints/v1.1_step160000.npz",
        puzzle_test_index=1, source_cell_zero_based=66, target_cell_zero_based=48,
        starts=[152, 192], lags=[0, 1, 2, 4, 8, 16], tail_blocks=8,
        pulse="Before boundary MLP, add epsilon times that cell's starting hidden vector; the vectors are fixed across pulse times and orders.",
        epsilons=[0.002, 0.001], precision="CPU FP64",
        arms="baseline; Aearly, Blate, AB; Bearly, Alate, BA, for each epsilon. A=cell66, B=cell48.",
        interaction="(both - first_alone - second_alone + baseline) / epsilon^2",
        primary="Interaction in W, eight blocks after the second pulse, for each head and each directed edge.",
        secondary="Interaction in G and W at the second pulse; full temporal response retained.",
        expected_causal_sign="For each directed edge, source-first interaction positive and target-first negative; test all preselected nonzero lags and heads without choosing a favorable sign orientation.",
        classification="Convergence: epsilon difference <= max(1e-6,0.1*max magnitude). Reliable sign: magnitude > max(1e-6,5*epsilon difference). Order difference: > max(1e-6,5*sum epsilon differences).",
        limits="Conditional finite-pulse test on one pair and two states. Pair interaction is not total Delta W. Baseline drift and nonlinear recurrence remain. No untrained checkpoint comparison, so this cannot establish that training created the behavior, or prove global absence/presence of a universal STDP law.",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model, _ = load_cpu(protocol["checkpoint"])
    blocks = CPUBlocks(model, puzzle_batch(1))
    snapshots = load_snapshots()
    with np.load("runs/relation_transport_v11/normal.npz") as reference:
        predictions = reference["pred"][:, 1].copy()
    report = dict(protocol=protocol, cases=[])
    tic = time.monotonic()
    for start in protocol["starts"]:
        for lag in protocol["lags"]:
            result, arrays = run_lag(blocks, *snapshots[start], start, lag,
                                     protocol["tail_blocks"], protocol["epsilons"], predictions)
            report["cases"].append(result)
            np.savez_compressed(out / f"block_{start}_lag_{lag}.npz", **arrays)
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(start=start, lag=lag, seconds=time.monotonic()-tic,
                forward_W_classes=[h["W_after_tail"]["forward_edge"]["classification"]
                                   for h in result["heads"]])), flush=True)
    report.update(elapsed_seconds=time.monotonic()-tic,
                  cuda_initialized=torch.cuda.is_initialized(), torch_version=torch.__version__)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
