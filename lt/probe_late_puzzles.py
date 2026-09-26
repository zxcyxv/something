"""Capture and intervene on late puzzle-completion events.

See docs/late_puzzle_probe_plan_v11.md. Retrospective case selection is based
only on output trajectories, before examining internal message contributions.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import train
from ckpt_npz import load_lt
from probe_phase_feedback import Blocks, verify_runner


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def select_candidates(root):
    p = np.load('runs/sudoku_compare_seg1024_2048/lt_v11/predictions.npy', mmap_mode='r')
    with np.load('data/sudoku_lt_1k.npz') as z:
        y = z['test_labels'].reshape(-1, 81) + 1
    correct = (p == y[None]).all(-1)
    first = np.where(correct.any(0), correct.argmax(0) + 1, -1)
    ids = np.flatnonzero((first >= 256) & correct[-64:].all(0))
    chosen = ids[:12]
    record = {'source': 'completed BF16 batch2048 seg1024 run', 'eligible_n': len(ids),
              'selection': 'first 12 original test indices; no internal-state selection',
              'indices': chosen.tolist(), 'source_first_complete_segments': first[chosen].tolist()}
    write_json(root / 'candidates.json', record)
    return chosen


def setup(ids):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load('data/sudoku_lt_1k.npz') as z:
        x = z['test_inputs'].reshape(-1, 81)[ids].astype(np.int32)
        y = z['test_labels'].reshape(-1, 81)[ids].astype(np.int64)
    batch = {'inputs': torch.from_numpy(x + 1).cuda(), 'labels': torch.from_numpy(y + 1).cuda(),
             'puzzle_identifiers': torch.zeros(len(ids), dtype=torch.int32, device='cuda')}
    model, cfg, step = load_lt('checkpoints/v1.1_step160000.npz', mod=train,
                              batch_size=len(ids), loops=1025, amp=False)
    model.requires_grad_(False)
    return model, batch, x, y


class EventBlocks(Blocks):
    def parts(self, h, w):
        q = self.prepare(h)
        address = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(address, self.kc)
        v = torch.einsum('btd,hcd->bthc', q, self.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
        agree = torch.einsum('bthc,bnhc->bhtn', vv, vv)
        window = self.inner.attn_xy(address, self.kcb)
        target = self.gain * (window * agree)
        wn = target if w is None else (1-self.eta)*w + self.eta*target
        return dict(q=q, a=a, v=v, agree=agree, window=window, target=target, w=wn)

    def read(self, p, read_w=None):
        effective = (1-self.lam)*p['a'] + self.lam*(p['w'] if read_w is None else read_w)
        values = torch.einsum('bhtn,bnhc->bthc', effective, p['v'])
        update = torch.einsum('bthc,hcd->btd', values, self.layer.w_sh)
        pre = p['q'] + update
        return self.inner.phi(pre), pre, effective


def event_list(pred, y, ids):
    records = []
    total = len(pred) - 1
    for i, puzzle in enumerate(ids):
        ok = (pred[:, i] == y[i] + 1).all(-1)
        first = int(np.flatnonzero(ok)[0]) if ok.any() else None
        wrong = pred[:, i] != y[i] + 1
        last_wrong = np.where(wrong, np.arange(len(pred))[:, None], -1).max(0)
        t = int(last_wrong.max() + 1)
        row = {'puzzle': int(puzzle), 'batch_index': i, 'first_complete_block': first,
               'stable_complete_block': t if t <= total else None,
               'final_exact': bool(ok[-1]), 'eligible': False}
        if ok[-256:].all() and 2048 <= t <= total-256:
            cells = np.flatnonzero(last_wrong == t-1)
            fractions = wrong[max(1, t-1024):t, cells].mean(0)
            c = int(cells[np.argmax(fractions)])
            row.update(cell=c, target_rc=[c//9+1, c%9+1],
                       prior_1024_wrong_fraction=float(fractions.max()),
                       old_digit=int(pred[t-1, i, c])-1, gold_digit=int(y[i, c]),
                       eligible=bool(fractions.max() >= 0.9))
        records.append(row)
    return {'all_cases': records, 'selected_events': [r for r in records if r['eligible']][:3]}


def baseline(root):
    ids = select_candidates(root)
    model, batch, x, y = setup(ids)
    b = EventBlocks(model, batch)
    verification = verify_runner(b, model, batch)
    h = b.inner.init_hidden.expand(len(ids), 81, -1).clone()
    w = None
    rh, rw = h, None
    for _ in range(8):
        parts = b.parts(h, w)
        h, _, _ = b.read(parts)
        w = parts['w']
        rh, rw = b.block(rh, rw)
    torch.testing.assert_close(h, rh, atol=0, rtol=0)
    torch.testing.assert_close(w, rw, atol=0, rtol=0)
    verification['decomposed_8_blocks_exact'] = True
    print('verification', json.dumps(verification), flush=True)
    snapshots = root / 'snapshots'
    snapshots.mkdir(exist_ok=True)
    h = b.inner.init_hidden.expand(len(ids), 81, -1).clone()
    w = None
    pred = np.empty((8193, len(ids), 81), dtype=np.uint8)
    pred[0] = b.inner.w_cls(h).argmax(-1).cpu().numpy()
    torch.save({'h': h.cpu(), 'w': None}, snapshots / 'block_00000.pt')
    started = time.monotonic()
    for k in range(1, 8193):
        h, w = b.block(h, w)
        pred[k] = b.inner.w_cls(h).argmax(-1).cpu().numpy()
        if k % 128 == 0:
            torch.save({'h': h.cpu(), 'w': w.cpu()}, snapshots / f'block_{k:05d}.pt')
        if k % 256 == 0:
            row = {'block': k, 'segment': k//8,
                   'exact': int((pred[k] == y+1).all(-1).sum()),
                   'n': len(ids), 'elapsed_seconds': round(time.monotonic()-started, 2)}
            print(json.dumps(row), flush=True)
            write_json(root / 'progress.json', row)
    np.savez_compressed(root / 'baseline.npz', indices=ids, X=x, Y=y, predictions=pred)
    events = event_list(pred, y, ids)
    events.update(verification=verification, elapsed_seconds=time.monotonic()-started,
                  precision='FP32; autocast/TF32 off', checkpoint='v1.1_step160000.npz')
    write_json(root / 'events.json', events)
    print('EVENTS', json.dumps(events), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/late_puzzle_probe_v11')
    p.add_argument('--stage', choices=['baseline'], default='baseline')
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'baseline':
        if (root / 'baseline.npz').exists():
            raise RuntimeError('Baseline already exists; do not silently replace it.')
        baseline(root)


if __name__ == '__main__':
    main()
