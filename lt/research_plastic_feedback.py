"""Measure the actual activity -> plastic update -> next activity derivative.

The diagnostic direction that raises the old wrong digit is defined using the
classifier for analysis only. It is never an input to the model or a new update
rule. A matched memory replay removes only the response of writes to the small
perturbation, while preserving the entire normal memory trajectory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def dot(a, b):
    return float((a*b).sum())


def analyze(blocks, snapshots, start, horizon):
    h0, w0 = snapshots[start]
    path = [(h0, w0)]
    with torch.no_grad():
        for _ in range(horizon):
            path.append(blocks.block(*path[-1]))
    target, gold, old = 48, 2, 4
    decoder = blocks.inner.w_cls.weight[old]-blocks.inner.w_cls.weight[gold]
    wrong = torch.zeros_like(h0)
    wrong[0, target] = decoder/decoder.norm()
    velocity = path[1][0]-h0
    velocity = velocity/velocity.norm()
    original = np.load("runs/relation_transport_v11/normal.npz")["pred"]
    predicted = torch.stack([blocks.inner.w_cls(h).argmax(-1)[0]
                             for h, _ in path]).numpy()
    result = dict(start_block=start, target=target, directions={},
                  baseline_cell_prediction_disagreements=int(
                      (predicted != original[start:start+horizon+1, 1]).sum()))
    for name, direction in (("toward_old_wrong_digit", wrong),
                            ("observed_next_state_velocity", velocity)):
        dh, dw, replay = direction.clone(), torch.zeros_like(w0), direction.clone()
        records = []
        identity_errors = []
        for step in range(1, horizon+1):
            h, w = path[step-1]
            wn = path[step][1]
            # dG = C dh; A differentiates transmission with updated w fixed.
            _, dG = jvp(blocks.write, (h,), (dh,))
            dwn = blocks.rho*dw+blocks.eta*dG
            _, A_dh = jvp(lambda x: blocks.transmit(x, wn), (h,), (dh,))
            _, B_dw = jvp(lambda x: blocks.transmit(h, x), (wn,), (dwn,))
            _, replay = jvp(lambda x: blocks.transmit(x, wn), (h,), (replay,))
            if step == 1:
                _, joint = jvp(blocks.block, (h, w), (dh, dw))
                error = float((joint[0]-A_dh-B_dw).abs().max())
                identity_errors.append(error)
                torch.testing.assert_close(joint[0], A_dh+B_dw, atol=3e-11, rtol=3e-10)
                torch.testing.assert_close(joint[1], dwn, atol=3e-11, rtol=3e-10)
            dh, dw = A_dh+B_dw, dwn
            if step in (1, 2, 4, 8, horizon):
                records.append(dict(blocks=step, closed_norm=float(dh.norm()),
                                    replay_norm=float(replay.norm()),
                                    closed_projection_on_initial=dot(dh, direction),
                                    replay_projection_on_initial=dot(replay, direction),
                                    closed_wrong_minus_gold_derivative=dot(dh[0, target], decoder),
                                    replay_wrong_minus_gold_derivative=dot(replay[0, target], decoder),
                                    feedback_wrong_minus_gold_derivative=dot((dh-replay)[0, target], decoder)))
        finite = []
        for eps in (1e-3, 5e-4):
            derivatives = {}
            with torch.no_grad():
                for mode in ("closed", "replay"):
                    ends = []
                    for sign in (1, -1):
                        hp, wp = h0+sign*eps*direction, w0.clone()
                        for t in range(1, horizon+1):
                            if mode == "closed":
                                hp, wp = blocks.block(hp, wp)
                            else:
                                hp = blocks.transmit(hp, path[t][1])
                        ends.append(hp)
                    derivatives[mode] = (ends[0]-ends[1])/(2*eps)
            finite.append(dict(epsilon=eps,
                               closed_relative_error=float((derivatives["closed"]-dh).norm()/dh.norm()),
                               replay_relative_error=float((derivatives["replay"]-replay).norm()/replay.norm())))
        if max(row["closed_relative_error"] for row in finite) > 1e-4:
            raise AssertionError("closed-loop tangent not reproduced by finite differences")
        if max(row["replay_relative_error"] for row in finite) > 1e-4:
            raise AssertionError("replay tangent not reproduced by finite differences")
        result["directions"][name] = dict(records=records, chain_identity_max_error=max(identity_errors),
                                            finite_difference=finite)
        print(json.dumps(dict(start=start, direction=name, last=records[-1], finite=finite)), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="runs/plastic_feedback_derivative_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, _ = load_cpu()
    blocks = CPUBlocks(model, puzzle_batch())
    snapshots = load_snapshots()
    report = dict(horizon=args.horizon, method="FP64 CPU JVP plus two central finite differences",
                  states=[analyze(blocks, snapshots, start, args.horizon) for start in (152, 192)],
                  cuda_initialized=torch.cuda.is_initialized())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
