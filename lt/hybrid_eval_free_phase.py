"""Evaluate one free-phase checkpoint with raw/EMA weights mixed per parameter group.

EMA only affects evaluation (training always uses raw weights), so "exclude a
group from EMA" can be answered from saved checkpoints: load raw weights and
copy the EMA shadow into every parameter the group predicate does not select.
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .experiment_free_phase_windows import model_class

GROUPS = {
    'qkv': lambda n: any(s in n for s in ('q_proj', 'k_proj', 'v_proj')),
    'kv': lambda n: any(s in n for s in ('k_proj', 'v_proj')),
    'qk': lambda n: any(s in n for s in ('q_proj', 'k_proj')),
    'attn': lambda n: any(s in n for s in ('q_proj', 'k_proj', 'v_proj', 'out_proj')),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--groups', default='qkv', help='comma-separated group names to keep raw inside the EMA model')
    ap.add_argument('--puzzles', type=int, default=2048)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    raw, shadow = ck['raw_model_state_dict'], ck['ema_shadow']
    protocol = json.loads((run / 'protocol.json').read_text())
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
                               protocol['generator'], protocol.get('feature_precision', 'float32'),
                               protocol.get('window_scale_factor', 1.0), protocol.get('tie_qk', False),
                               protocol.get('tie_vo', False), protocol.get('qk_l2', False),
                               protocol.get('write_sum', False), protocol.get('tau_phi', 2.0), protocol.get('phase_floor', 0.5), protocol.get('v_norm', 'none'),
                               protocol.get('tie_all', False), protocol.get('phase_kappa', 1.0), protocol.get('phase_omega', 0.0))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    _, _, tx, ty, *_ = t.load_data(cfg)
    tx, ty = tx[:opt.puzzles], ty[:opt.puzzles]
    with torch.device(device):
        base = t.ACTLossHead(t.LT(dict(cfg, batch_size=cfg['global_batch_size'], seq_len=cfg['grid'] ** 2,
                                       num_puzzle_identifiers=1)), q_weight=cfg['q_weight'])
    base.eval()
    results = {}

    def run_case(name, use_ema):
        base.load_state_dict(raw, strict=True)
        with torch.no_grad():
            for n, p in base.named_parameters():
                if n in shadow and use_ema(n):
                    p.copy_(shadow[n])
        m = t.evaluate(base, tx, ty, dict(cfg, test_size=len(tx)), 0, 1, device, int(ck['step']), ema=None)
        row = dict(accuracy=m['accuracy'], exact=round(m['exact_accuracy'] * m['count']), count=int(m['count']),
                   lm_loss=m['lm_loss'])
        results[name] = row
        print(f"{name:28s} acc {row['accuracy']*100:.2f}%  exact {row['exact']}/{row['count']}  loss {row['lm_loss']:.3f}", flush=True)

    run_case('raw', lambda n: False)
    run_case('ema', lambda n: True)
    for g in opt.groups.split(','):
        pred = GROUPS[g]
        run_case(f'ema, {g} raw', lambda n, pred=pred: not pred(n))
        run_case(f'raw, {g} ema', pred)
    out = run / 'diagnostics' / f"hybrid_step{int(ck['step'])}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(step=int(ck['step']), checkpoint=str(opt.checkpoint), puzzles=len(tx),
                                   results=results), indent=2) + '\n')
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
