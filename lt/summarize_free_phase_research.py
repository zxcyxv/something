"""Matched terminal-segment summaries of the controlled research runs."""
import json
from pathlib import Path
import statistics

from .research_free_phase_windows import DEST, ROOT


RUNS = {
    'historical_fixed_exp':'kv_phase_exp_current_fresh_20261005',
    'historical_ordered_warp':'kv_phase_local_warp_exp_fresh_20261005',
    'fixed_fourier4_e05':'research_free_phase_fixed_fourier4_e05_20261005',
    'free_fourier4_e05':'research_free_phase_free_fourier4_e05_20261005',
    'free_exact_exp':'research_free_phase_free_exact_exp_20261005',
}


def record_directory(directory):
    live = ROOT/'runs'/directory
    return live if (live/'train.jsonl').exists() else DEST/'training_records'/directory


def main():
    results = {}
    for name,directory in RUNS.items():
        p = record_directory(directory)/'train.jsonl'
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        selected = [r for r in rows if r['step']<=3008]
        terminal = [r for r in selected if r.get('_count_raw',0)>0]
        windows = {}
        for count in (32,64):
            part = terminal[-count:]
            if len(part)!=count:
                continue
            windows[str(count)] = dict(first_step=part[0]['step'],last_step=part[-1]['step'],
                batches=count,examples=sum(r['_count_raw'] for r in part),
                accuracy=statistics.mean(r['accuracy'] for r in part),
                exact_accuracy=statistics.mean(r['exact_accuracy'] for r in part),
                lm_loss=statistics.mean(r['lm_loss'] for r in part),
                batch_accuracy_std=statistics.stdev(r['accuracy'] for r in part))
        metadata_path = p.parent/'archive_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        results[name] = dict(run=str(ROOT/'runs'/directory),latest_step=metadata.get('latest_source_step', rows[-1]['step']),comparison_step=selected[-1]['step'],
                            complete_3008=selected[-1]['step']==3008,terminal_windows=windows,
                            median_seconds_last512=statistics.median(r['seconds'] for r in selected[-512:]),
                            finite_losses=all(abs(r['lm_loss'])<float('inf') for r in selected),
                            checkpoints=metadata.get('checkpoint_names', [p.name for p in sorted(p.parent.glob('step_*.pt'))]))
    report = dict(protocol='Same upper step limit 3008, terminal-segment batches only; single seed; train metrics, not validation. Incomplete controls must not be compared as final.',runs=results)
    (DEST/'training_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in results.items():
        w=r['terminal_windows'].get('32',{})
        print(name,r['comparison_step'],w.get('accuracy'),w.get('exact_accuracy'),r['median_seconds_last512'])


if __name__=='__main__':
    main()
