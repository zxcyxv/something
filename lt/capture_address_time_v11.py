"""Capture native raw addresses along matched, unmodified v1.1 trajectories."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from analyze_late_puzzles import restore
from probe_late_puzzles import EventBlocks, setup, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/phase_timing_v11/time_data"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    root = Path("runs/late_puzzle_probe_v11")
    with np.load(root/"baseline.npz") as data:
        ids, reference = data["indices"], data["predictions"]
    events = json.loads((root/"events.json").read_text())["selected_events"]
    protocol = dict(
        checkpoint="checkpoints/v1.1_step160000.npz", source="unmodified cached baseline replay",
        precision="GPU FP32; autocast and TF32 disabled; original batch of 12 preserved",
        tensor="raw complex address W_C q, BEFORE normalization; shape [block,cell,head,channel]",
        train_puzzles=[38, 72, 111, 128], holdout_puzzles=[58, 209, 230],
        early_blocks_inclusive=[129, 384],
        late_blocks="Each preselected late event T-128 through T+127 inclusive",
        validation="Every puzzle prediction at every replayed block must match the saved reference",
        interventions="None; neither h nor address nor W is changed",
    )
    write_json(out/"protocol.json", protocol)
    model, batch, _, _ = setup(ids)
    b = EventBlocks(model, batch)
    windows = [("early", 129, 384, protocol["train_puzzles"]+protocol["holdout_puzzles"])]
    windows += [(f"late_{e['puzzle']}", e["stable_complete_block"]-128,
                 e["stable_complete_block"]+127, [e["puzzle"]]) for e in events]
    report = dict(protocol=protocol, windows=[])
    tic = time.monotonic()
    for name, start, stop, puzzles in windows:
        existing = out/f"{name}_complete.json"
        if existing.exists():
            saved = json.loads(existing.read_text())
            assert all((out/s["file"]).exists() for s in saved["files"])
            report["windows"].append(saved)
            continue
        selected = [int(np.flatnonzero(ids == p)[0]) for p in puzzles]
        h, w = restore(root, b, start, reference)
        arrays = []
        files = []
        for puzzle in puzzles:
            filename = f"{name}_puzzle_{puzzle}_raw.npy"
            arrays.append(np.lib.format.open_memmap(out/filename, mode="w+", dtype=np.complex64,
                          shape=(stop-start+1, 81, b.inner.H, b.inner.p)))
            files.append(dict(puzzle=puzzle, file=filename))
        disagreements = 0
        for block in range(start, stop+1):
            parts = b.parts(h, w)
            real, imag = b.inner.addr_raw(parts["q"], b.ab)
            z = torch.complex(real[selected], imag[selected]).cpu().numpy()
            for i, array in enumerate(arrays):
                array[block-start] = z[i]
            h, _, _ = b.read(parts)
            w = parts["w"]
            prediction = b.inner.w_cls(h).argmax(-1).cpu().numpy()
            mismatch = int(np.count_nonzero(prediction != reference[block]))
            disagreements += mismatch
            assert mismatch == 0, (name, block, mismatch)
            if (block-start+1) % 64 == 0:
                print(json.dumps(dict(window=name, block=block, seconds=time.monotonic()-tic)), flush=True)
        for array in arrays:
            array.flush()
        item = dict(name=name, start_block=start, stop_block=stop, files=files,
                    predictions_checked=int((stop-start+1)*len(ids)*81),
                    prediction_disagreements=disagreements)
        write_json(existing, item)
        report["windows"].append(item)
        write_json(out/"summary.json", report)
    report["elapsed_seconds"] = time.monotonic()-tic
    report["checkpoint_sha256"] = hashlib.sha256(Path(protocol["checkpoint"]).read_bytes()).hexdigest()
    write_json(out/"summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
