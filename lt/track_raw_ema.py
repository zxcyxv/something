"""Evaluate raw and EMA weights of every checkpoint a live run saves.

The trainer keeps only the latest few checkpoints, so each new step_*.pt is
hard-linked first (retention then only unlinks the run's name), evaluated with
lt.evaluate_free_phase_raw_ema in a subprocess, and the link is removed after
the result is recorded. Results are appended to <run>/raw_ema_track.jsonl.
The training process is never touched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def step_of(path):
    return int(re.fullmatch(r'step_(\d+)\.pt', path.name).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--min-step', type=int, default=0)
    ap.add_argument('--poll', type=float, default=20.)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    hold = run / 'raw_ema_hold'
    hold.mkdir(exist_ok=True)
    track = run / 'raw_ema_track.jsonl'
    done = {json.loads(line)['step'] for line in track.read_text().splitlines()} if track.exists() else set()
    while True:
        pending = sorted((p for p in run.glob('step_*.pt') if step_of(p) >= opt.min_step and step_of(p) not in done),
                         key=step_of)
        for path in pending:
            link = hold / path.name
            try:
                if not link.exists():
                    os.link(path, link)
            except FileNotFoundError:
                continue  # removed by retention before we could hold it
            step = step_of(path)
            started = time.monotonic()
            result = subprocess.run([sys.executable, '-u', '-m', 'lt.evaluate_free_phase_raw_ema',
                                     '--run', str(run), '--checkpoint', str(link)],
                                    capture_output=True, text=True)
            row = dict(step=step, returncode=result.returncode, seconds=time.monotonic() - started)
            complete = [line for line in result.stdout.splitlines() if line.startswith('COMPLETE ')]
            if result.returncode == 0 and complete:
                results = json.loads(Path(complete[-1].split(' ', 1)[1]).read_text())['results']
                for name in ('raw', 'ema'):
                    r = results[name]
                    row[name] = dict(accuracy=r['accuracy'], exact_count=r['exact_count'],
                                     exact_accuracy=r['exact_accuracy'], lm_loss=r['lm_loss'])
            else:
                row['stderr_tail'] = result.stderr[-2000:]
            with track.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print('TRACKED', json.dumps(row), flush=True)
            link.unlink(missing_ok=True)
            done.add(step)
        if (run / 'finished.json').exists() and not pending:
            status = json.loads((run / 'finished.json').read_text())
            if status.get('status') in ('finished', 'stopped', None):
                break
        time.sleep(opt.poll)


if __name__ == '__main__':
    main()
