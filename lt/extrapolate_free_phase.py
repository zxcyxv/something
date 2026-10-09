"""Segment extrapolation of a free-phase checkpoint beyond the 16 training segments.

Runs the trainer's own extrapolate() (carry kept across segments, loops raised to segs+1)
for EMA and raw weights, on the first --n held-out puzzles (512 matches the milestone
tables of v1.1/v1.71; 2048 is the full held-out set).

python -m lt.extrapolate_free_phase --run runs/<run> --checkpoint runs/<run>/step_100000.pt --segments 128 --n 512 2048
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .analyze_stdp_direct import make_model
from .kv_stability import ORIGINAL_MODEL_ID


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--segments', type=int, default=128)
    ap.add_argument('--n', type=int, nargs='+', default=[512, 2048])
    ap.add_argument('--weights', nargs='+', default=['ema', 'raw'], choices=('ema', 'raw'))
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x, y, _, _ = t.load_data(cfg)
    step = int(ck['step'])
    base = make_model(protocol, cfg, device)
    states = dict(ema=ck['model_state_dict'], raw=ck['raw_model_state_dict'])
    out_dir = run / 'diagnostics'
    for weights in opt.weights:
        base.load_state_dict(states[weights], strict=True)
        for n in opt.n:
            out = out_dir / f'extrap_step{step}_{weights}_n{n}_seg{opt.segments}.txt'
            t.extrapolate(base, x, y, dict(cfg, milestone_extrap_n=n), 0, 1, device, step, None, opt.segments, out)
            print(f'WROTE {out} ({weights})', flush=True)


if __name__ == '__main__':
    main()
