"""Evaluate raw and EMA weights from one saved free-phase checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from . import train as t
from .experiment_free_phase_windows import model_class
from .kv_stability import ORIGINAL_MODEL_ID


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    path = args.checkpoint or t.find_latest_checkpoint(str(run))
    if path is None:
        raise FileNotFoundError(f'No checkpoint in {run}')
    path = Path(path).resolve()
    # Keep the descriptor open across hashing/loading: retention may unlink it.
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        stream.seek(0)
        ck = torch.load(stream, map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    expected_id = ORIGINAL_MODEL_ID(cfg) + ':research-' + protocol['variant']
    assert ck['model_id'] == expected_id
    assert cfg['research_variant'] == protocol['variant']
    states = {'raw': ck['raw_model_state_dict'], 'ema': ck['model_state_dict']}
    shadow = ck['ema_shadow']
    assert shadow and set(states['raw']) == set(states['ema'])
    assert all(torch.equal(states['ema'][key], value) for key, value in shadow.items())
    common_keys = set(states['raw']) - set(shadow)
    assert all(torch.equal(states['raw'][key], states['ema'][key]) for key in common_keys)
    step = int(ck['step'])
    del ck, shadow
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = run / 'diagnostics' / f'raw_vs_ema_step{step}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'CHECKPOINT step={step} path={path}\nOUTPUT {output}', flush=True)

    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    _, _, test_x, test_y, data_path, fingerprint = t.load_data(cfg)
    assert fingerprint == cfg['data_fingerprint'], 'Evaluation data changed'
    assert len(test_x) == cfg['test_size']
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'],
                               protocol['modes'], protocol['epsilon'], protocol['generator'],
                               protocol.get('feature_precision', 'float32'),
                               protocol.get('window_scale_factor', 1.0))
    model_cfg = dict(cfg, batch_size=cfg['global_batch_size'],
                     seq_len=cfg['grid'] ** 2, num_puzzle_identifiers=1)
    with torch.device(device):
        base = t.ACTLossHead(t.LT(model_cfg), q_weight=cfg['q_weight'])
    base.eval()
    metadata = dict(checkpoint=str(path), checkpoint_sha256=digest.hexdigest(),
                    step=step, model_id=expected_id, config=cfg, protocol=protocol,
                    data_path=data_path, data_fingerprint=fingerprint,
                    test_size=len(test_x), batch_size=cfg['global_batch_size'],
                    segments=cfg['loops'], blocks_per_segment=cfg['blocks_per_seg'],
                    evaluation='Original lt.train.evaluate; fresh carry per batch; eager CUDA',
                    ema_matches_saved_shadow=True, identical_non_ema_tensors=sorted(common_keys),
                    training_continues_concurrently=True, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(), created_utc=stamp)
    write_json(output / 'metadata.json', metadata)
    original_batches = t.eval_batches
    results = {}
    try:
        for name, state in states.items():
            base.load_state_dict(state, strict=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()

            def progress_batches(*batch_args, **batch_kwargs):
                for index, batch in enumerate(original_batches(*batch_args, **batch_kwargs), 1):
                    yield batch
                    if index % 4 == 0:
                        print(f'PROGRESS weights={name} batches={index} '
                              f'seconds={time.monotonic() - started:.1f}', flush=True)

            t.eval_batches = progress_batches
            metrics = t.evaluate(base, test_x, test_y, cfg, 0, 1, device, step, ema=None)
            torch.cuda.synchronize()
            assert int(metrics['count']) == len(test_x), 'Partial evaluation'
            assert metrics['steps'] == cfg['loops'], 'Inference horizon changed'
            row = dict(metrics, exact_count=round(metrics['exact_accuracy'] * metrics['count']),
                       elapsed_seconds=time.monotonic() - started,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            results[name] = row
            write_json(output / 'results.json', dict(step=step, results=results))
            print('RESULT', name, json.dumps(row), flush=True)
    finally:
        t.eval_batches = original_batches
    print(f'COMPLETE {output / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
