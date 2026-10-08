"""Continue a saved free-phase experiment with its original training recipe.

Reads the model variant from protocol.json and restores the trainer's complete
checkpoint. Defaults to the full epoch schedule and retains three checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import train as t
from .experiment_free_phase_windows import model_class, exclude_phase_gain_from_decay
from .kv_stability import ORIGINAL_MODEL_ID


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def continue_run(run, *, steps=0, save_every=1000, keep_last=3):
    protocol = json.loads((run / 'protocol.json').read_text())
    checkpoint = t.find_latest_checkpoint(str(run))
    if checkpoint is None:
        raise FileNotFoundError(f'No checkpoint in {run}')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    name = protocol['variant']
    expected_id = ORIGINAL_MODEL_ID(cfg) + ':research-' + name
    if ck['model_id'] != expected_id or cfg['research_variant'] != name:
        raise ValueError('Checkpoint and saved model protocol disagree')
    if steps and steps <= ck['step']:
        raise ValueError('--steps must exceed the saved step, or be 0 for the full schedule')
    records_path = run / 'train.jsonl'
    records = records_path.read_text().splitlines() if records_path.exists() else []
    last = json.loads(records[-1]) if records else {}
    if last and last['step'] != ck['step']:
        raise ValueError('Log/checkpoint step mismatch; reconcile the interrupted tail before resuming')
    elapsed_before = last.get('elapsed', 0.)
    cfg.update(out_dir=str(run), resume_from=str(Path(checkpoint).resolve()), require_resume=True,
               init_from=None, max_steps=steps or None, max_hours=None,
               save_every_steps=save_every, keep_last=keep_last, milestone_every=0)
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    t.validate_run_cfg(cfg)
    if cfg.get('phase_gain_no_decay'):
        exclude_phase_gain_from_decay()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(protocol['precision'])
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'],
                               protocol['epsilon'], protocol['generator'], protocol.get('feature_precision', 'float32'),
                               protocol.get('window_scale_factor', 1.0),
                               protocol.get('tie_qk', False), protocol.get('tie_vo', False),
                               protocol.get('qk_l2', False), protocol.get('write_sum', False),
                               protocol.get('tau_phi', 2.0), protocol.get('phase_floor', 0.5), protocol.get('v_norm', 'none'),
                               protocol.get('tie_all', False), protocol.get('phase_kappa', 1.0), protocol.get('phase_omega', 0.0))
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']

    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = run / 'continuations' / session
    archive.mkdir(parents=True)
    for filename in ('config.json', 'protocol.json', 'finished.json', 'resume_status.json'):
        source = run / filename
        if source.exists():
            shutil.copy2(source, archive / ('previous_' + filename))
    source_hashes = {}
    for module in (Path(__file__), Path(__file__).with_name('experiment_free_phase_windows.py'),
                   Path(__file__).with_name('research_free_phase_windows.py'),
                   Path(__file__).with_name('tanhsech_phase_window.py'),
                   Path(__file__).with_name('kv_stability.py'), Path(t.__file__)):
        content = module.read_bytes()
        (archive / module.name).write_bytes(content)
        source_hashes[module.name] = hashlib.sha256(content).hexdigest()
    audit = dict(session=session, pid=os.getpid(), status='loading', checkpoint=checkpoint,
                 start_step=int(ck['step']), next_iter=int(ck['iter_id']),
                 consumed_batches=int(ck['batch_in_iter']), model_id=expected_id,
                 max_steps=cfg['max_steps'], max_hours=None, epochs=cfg['epochs'],
                 keep_last=keep_last, save_every_steps=save_every,
                 phase_protocol=protocol, source_sha256=source_hashes,
                 recipe_changes={k:[ck['cfg'].get(k), cfg[k]] for k in cfg if ck['cfg'].get(k) != cfg[k]})
    write_json(archive / 'resume.json', audit)
    write_json(run / 'resume_status.json', audit)
    original_load, original_batch = t.load_training_checkpoint, t.train_batch
    expected_step, expected_iter, expected_cursor = ck['step'], ck['iter_id'], ck['batch_in_iter']

    def load(*args, **kwargs):
        state = original_load(*args, **kwargs)
        base, optimizers, ema = args[1:4]
        assert (state.step, state.iter_id, state.batch_in_iter) == (expected_step, expected_iter, expected_cursor)
        # Verify actual restored parameters/EMA, beyond the trainer's strict key checks.
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), ck['raw_model_state_dict'][key], rtol=0, atol=0)
        if ema is not None:
            for key, value in ema.shadow.items():
                torch.testing.assert_close(value.detach().cpu(), ck['ema_shadow'][key], rtol=0, atol=0)
        assert len(optimizers) == len(ck['optimizer_states'])
        assert state.carry is not None
        audit.update(status='running', restore_verified=True,
                     restored='raw weights and EMA exact equality; trainer restored optimizer, carry, RNG and data cursor')
        write_json(archive / 'restore_verified.json', audit)
        write_json(run / 'resume_status.json', audit)
        (run / 'finished.json').unlink(missing_ok=True)
        print('RESUME_VERIFIED', json.dumps({k:audit[k] for k in
              ('session','pid','start_step','model_id','max_steps','keep_last','restored')}), flush=True)
        ck.clear()
        return state

    began = time.monotonic()
    def batch(model, base, state, *args, **kwargs):
        started = time.monotonic()
        result = original_batch(model, base, state, *args, **kwargs)
        row = dict(step=state.step, seconds=time.monotonic()-started,
                   elapsed=elapsed_before+time.monotonic()-began, resume_session=session,
                   segment=int(state.carry.steps.max()), **result)
        with records_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return result

    t.load_training_checkpoint, t.train_batch = load, batch
    try:
        print('CONTINUE', json.dumps(audit), flush=True)
        final_step = t.main(cfg)
        audit.update(status='stopped' if t._STOP_REQUESTED else 'finished', actual_step=final_step,
                     final_checkpoint=t.find_latest_checkpoint(str(run)),
                     session_seconds=time.monotonic()-began)
        write_json(run / 'finished.json', audit)
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}',
                     session_seconds=time.monotonic()-began)
        raise
    finally:
        t.load_training_checkpoint, t.train_batch = original_load, original_batch
        write_json(run / 'resume_status.json', audit)
        write_json(archive / 'result.json', audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=0, help='Absolute stopping step; 0 removes the research step cap')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--keep-last', type=int, default=3)
    args = parser.parse_args()
    if args.steps < 0 or args.save_every < 1 or args.keep_last < 1:
        parser.error('steps must be nonnegative; save-every and keep-last must be positive')
    run = args.run.resolve(strict=True)
    with (run / '.resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        continue_run(run, steps=args.steps, save_every=args.save_every, keep_last=args.keep_last)


if __name__ == '__main__':
    main()
