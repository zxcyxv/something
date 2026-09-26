"""Causal contribution of phase-order-sensitive writing in trained v1.1.

Start from fresh state and alter only G before every memory write. Preserve
the original instantaneous read kernel and all model parameters. The 12
puzzles and 8192-block horizon come from the existing late-puzzle baseline.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe_late_puzzles import EventBlocks, setup, write_json


def completion(predictions, gold, ids):
    exact = (predictions == gold[None]).all(-1)
    rows = []
    for i, puzzle in enumerate(ids):
        hits = np.flatnonzero(exact[:, i])
        wrong = np.flatnonzero(~exact[:, i])
        stable = int(wrong[-1]+1) if len(wrong) else 0
        rows.append(dict(puzzle=int(puzzle), first_complete_block=int(hits[0]) if len(hits) else None,
                         final_complete=bool(exact[-1, i]),
                         final_256_all_complete=bool(exact[-256:, i].all()),
                         stable_complete_block=stable if stable < len(exact) else None,
                         final_wrong_cells=int(np.count_nonzero(predictions[-1, i] != gold[i]))))
    return dict(final_complete_count=int(exact[-1].sum()),
                final_256_stable_count=int(exact[-256:].all(0).sum()),
                puzzles=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/phase_timing_v11/write_ablation"))
    parser.add_argument("--blocks", type=int, default=8192)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load("runs/late_puzzle_probe_v11/baseline.npz") as data:
        ids, reference, gold = data["indices"], data["predictions"], data["Y"]+1
    modes = ("intrinsic_phase_even", "cell_exchange_even", "intrinsic_phase_mirrored",
             "intrinsic_phase_even_energy_matched")
    protocol = dict(
        checkpoint="checkpoints/v1.1_step160000.npz", puzzles=ids.tolist(), blocks=args.blocks,
        state="Fresh model initialization; no old full-G memory is retained",
        intrinsic_phase_even="G_new = (G(u)+G(conj(u)))/2; spatial position, beta, values and amplitudes retained",
        cell_exchange_even="G_new = (G+G.transpose(-1,-2))/2; remove cell-exchange antisymmetric writing, not intrinsic-phase odd writing",
        intrinsic_phase_mirrored="G_new=G(conj(u)); all intrinsic phase differences reversed, spatial phase fixed",
        intrinsic_phase_even_energy_matched="Follow-up magnitude control: sign(E)*sqrt(E^2+O^2), E=(G+Gmirror)/2, O=(G-Gmirror)/2. This remains phase-even and preserves the RMS magnitude of the two phase orders per edge. Exactly E=0 maps to zero.",
        read="Original instantaneous a_psi(u), original v, original lambda, original eta; only the write target is changed",
        end_point="Whole-puzzle completion at block8192 and uninterrupted completion for the last256 blocks; first completion and final errors also saved",
        scope="Inference intervention on twelve previously selected late puzzles; no retraining or unbiased accuracy estimate",
        arithmetic="Both normal and conjugate G use gain*(window*agree), exactly the same FP32 multiplication order. Phase-even and paired-energy targets are checked bitwise invariant under swapping G/Gmirror.",
        expected_interpretation="Preserved solutions show the removed component is not necessary for those solutions in this trained inference run; failure shows dependence on that component, not a proof of temporal STDP",
    )
    write_json(out/"protocol.json", protocol)
    model, batch, _, _ = setup(ids)
    b = EventBlocks(model, batch)
    h = b.inner.init_hidden.expand(len(ids), 81, -1).clone()
    w = None
    for k in range(1, 129):
        p = b.parts(h, w)
        h, _, _ = b.read(p)
        w = p["w"]
        assert np.array_equal(b.inner.w_cls(h).argmax(-1).cpu().numpy(), reference[k])
    report = dict(protocol=protocol, original=completion(reference[:args.blocks+1], gold, ids),
                  baseline_128block_prediction_match=True, modes={})
    tic = time.monotonic()
    for mode in modes:
        completion_path = out/f"{mode}_summary.json"
        if completion_path.exists():
            report["modes"][mode] = json.loads(completion_path.read_text())
            continue
        h = b.inner.init_hidden.expand(len(ids), 81, -1).clone()
        w = None
        predicted = np.empty((args.blocks+1, len(ids), 81), dtype=np.uint8)
        predicted[0] = b.inner.w_cls(h).argmax(-1).cpu().numpy()
        for k in range(1, args.blocks+1):
            p = b.parts(h, w)
            normal_g = p["target"]
            if mode == "cell_exchange_even":
                target = .5*(normal_g+normal_g.transpose(-1, -2))
            else:
                ux, uy = b.inner.addr(p["q"], b.ab)
                mirror = b.gain*(b.inner.attn_xy((ux, -uy), b.kcb)*p["agree"])
                even = .5*(normal_g+mirror)
                if mode == "intrinsic_phase_mirrored":
                    target = mirror
                elif mode == "intrinsic_phase_even_energy_matched":
                    odd = .5*(normal_g-mirror)
                    target = even.sign()*torch.sqrt(even.square()+odd.square())
                else:
                    target = even
                if k == 1 or k % 512 == 0:
                    reverse_even = .5*(mirror+normal_g)
                    assert torch.equal(even, reverse_even)
                    if mode == "intrinsic_phase_even_energy_matched":
                        reverse_odd = .5*(mirror-normal_g)
                        reverse_target = reverse_even.sign()*torch.sqrt(reverse_even.square()+reverse_odd.square())
                        assert torch.equal(target, reverse_target)
            p["target"] = target
            p["w"] = target if w is None else (1-b.eta)*w+b.eta*target
            h, _, _ = b.read(p)
            w = p["w"]
            predicted[k] = b.inner.w_cls(h).argmax(-1).cpu().numpy()
            if k % 512 == 0:
                item = dict(mode=mode, block=k,
                            complete=int((predicted[k] == gold).all(-1).sum()),
                            elapsed_seconds=time.monotonic()-tic)
                write_json(out/"progress.json", item)
                print(json.dumps(item), flush=True)
        np.savez_compressed(out/f"{mode}.npz", predictions=predicted, indices=ids, gold=gold)
        result = completion(predicted, gold, ids)
        report["modes"][mode] = result
        write_json(completion_path, result)
        write_json(out/"summary.json", report)
        print("RESULT", mode, json.dumps(result), flush=True)
    report["elapsed_seconds"] = time.monotonic()-tic
    write_json(out/"summary.json", report)


if __name__ == "__main__":
    main()
