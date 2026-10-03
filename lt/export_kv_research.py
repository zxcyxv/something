"""Archive scalar experiment evidence without checkpoints or datasets."""
import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


RUNS = {
    "baseline": "M; supplementary 8 no-grad + 8 grad blocks",
    "baseline_ng0": "M = M_previous + G",
    "v17_no_address_norm": "v1.7, raw Q/K addresses, mixed current/persistent read",
    "v17_no_norm_memory_read": "v1.7, raw Q/K addresses, W-only read",
    "kv_interpolated_read_ng0": "(1-alpha) B + alpha M; NOT G/M interpolation",
    "kv_current_only_ng0": "B = mean(V_current K_current.T)",
    "kv_complex_current_only_ng0": "G = mean(V_current eK_past.T - eV_past K_current.T)",
    "kv_current_plus_stdp_ng0": "B + G; no persistent M read",
    "kv_historical_key_trace_ng0": "M, with event-time rotated key eligibility",
    "kv_read_gain_quarter_ng0": "0.25 M read",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def window(rows, endpoint):
    chosen = [r for r in rows if endpoint-256 < r["step"] <= endpoint]
    return dict(endpoint=endpoint, samples=len(chosen),
                loss=sum(r["lm_loss"] for r in chosen)/len(chosen),
                accuracy=sum(r["accuracy"] for r in chosen)/len(chosen),
                exact=sum(r["exact_accuracy"] for r in chosen)/len(chosen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/kv_collapse_20261003"))
    ap.add_argument("--out", type=Path, default=Path("docs/research/2026-10-03"))
    args = ap.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)
    copied, summaries, terminal_rows = [], {}, []

    def record(src, dest, transform=None):
        raw = src.read_bytes()
        data = transform(raw) if transform else raw
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        copied.append(dict(source=str(src), archive=str(dest.relative_to(out)),
                           source_bytes=len(raw), source_sha256=sha(raw),
                           archive_bytes=len(data), archive_sha256=sha(data)))

    for run, operator in RUNS.items():
        directory = root/run
        log = directory/"train.jsonl"
        if not log.exists():
            continue
        record(log, out/"logs"/(run+".jsonl.gz"),
               lambda data: gzip.compress(data, compresslevel=9, mtime=0))
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        terminal = [r for r in rows if r.get("_count_raw", 0) > 0]
        for r in terminal:
            terminal_rows.append(dict(run=run, step=r["step"], segment=r["segment"],
                loss=r["lm_loss"], accuracy=r["accuracy"], exact=r["exact_accuracy"],
                count=r["_count_raw"]))
        endpoints = sorted(set([x for x in (512,1008,1504,2000,2496,2992,4000,5008,6000)
                                if x <= terminal[-1]["step"]]+[terminal[-1]["step"]]))
        summaries[run] = dict(operator=operator, last_optimizer_step=rows[-1]["step"],
            last_terminal_step=terminal[-1]["step"], last_terminal=terminal[-1],
            checkpoints=[window(terminal, e) for e in endpoints],
            latest_window=window(terminal, terminal[-1]["step"]))
        for src in sorted(directory.glob("*.json")):
            record(src, out/"protocols"/run/src.name)

    for src in sorted(root.glob("*.json")):
        record(src, out/"comparisons"/src.name)
    for src in sorted(root.glob("*.csv")):
        record(src, out/"comparisons"/src.name)
    for name in ("audit", "theory_1h", "reference"):
        for src in sorted((root/name).iterdir()):
            if src.is_file() and src.suffix in (".json", ".csv", ".py", ".log", ".yaml"):
                record(src, out/name/src.name)

    with (out/"terminal_training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("run","step","segment","loss","accuracy","exact","count"))
        writer.writeheader()
        writer.writerows(terminal_rows)
    (out/"training_summary.json").write_text(json.dumps(dict(
        protocol="Terminal training rows only; endpoint-256 < step <= endpoint; proportions, not percentages. No EMA/test rows.",
        runs=summaries), indent=2, allow_nan=False)+"\n")
    manifest = dict(snapshot_at_utc=datetime.now(timezone.utc).isoformat(),
        status="published evidence snapshot; continued research may follow",
        baseline_commit="70a8dbae82a135251b83c91a53a8f14bd411946d",
        baseline_trainer_sha256=sha(Path("lt/train.py").read_bytes()),
        archived_runs=len(summaries), terminal_rows=len(terminal_rows),
        omissions=["Checkpoints, optimizer tensors, datasets, and large activity tensors are kept locally; scalar logs are complete."],
        files=copied)
    (out/"archive_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    for entry in copied:
        data = (out/entry["archive"]).read_bytes()
        assert sha(data) == entry["archive_sha256"]
        if entry["archive"].endswith(".jsonl.gz"):
            assert sha(gzip.decompress(data)) == entry["source_sha256"]
    print(json.dumps(dict(runs=len(summaries), terminal_rows=len(terminal_rows),
                          copied_files=len(copied), bytes=sum(x["archive_bytes"] for x in copied))))


if __name__ == "__main__":
    main()
