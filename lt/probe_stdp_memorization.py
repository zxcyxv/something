"""Does the trained STDP term carry memorization? Inference-time ablations on train vs held-out.

Loads one checkpoint of a pairangle dc-hebbian run and evaluates (16 segments, official
evaluate) the same weights on training puzzles and held-out puzzles while
  - scaling alpha_h by a factor (1, 0.75, 0.5, 0.25, 0),
  - removing the second harmonic (c_2 = 0, c_1 unchanged),
  - shuffling the per-cell phases across cells (same permutation for K and V phases,
    so each cell's write gets another cell's gain; the gain distribution is kept).
A term that mainly memorizes costs more on training puzzles than on held-out ones.
Runs the torch write path (feature_precision from the protocol) so phases can be patched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .experiment_free_phase_windows import model_class


def build(cfg, protocol, device):
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
                               protocol['generator'], protocol.get('feature_precision', 'float32'),
                               protocol.get('window_scale_factor', 1.0), protocol.get('tie_qk', False),
                               protocol.get('tie_vo', False), protocol.get('qk_l2', False),
                               protocol.get('write_sum', False), protocol.get('tau_phi', 2.0),
                               protocol.get('phase_floor', 0.5), protocol.get('v_norm', 'none'),
                               protocol.get('tie_all', False), protocol.get('phase_kappa', 1.0),
                               protocol.get('phase_omega', 0.0), protocol.get('phase_frame', 'rotated'),
                               protocol.get('dc_hebbian', False), protocol.get('dc_alpha_init', 0.0),
                               protocol.get('boundary_ffn', 'bilinear'), 'torch')
    with torch.device(device):
        return t.ACTLossHead(t.LT(dict(cfg, batch_size=cfg['global_batch_size'], seq_len=cfg['grid'] ** 2,
                                       num_puzzle_identifiers=1)), q_weight=cfg['q_weight'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    ap.add_argument('--out', type=Path)
    opt = ap.parse_args()
    device = torch.device('cuda')
    protocol = json.loads((opt.run / 'protocol.json').read_text())
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    torch.manual_seed(0)
    torch.set_float32_matmul_precision(protocol['precision'])
    t._resolve_precision(cfg, device)
    tr_x, tr_y, te_x, te_y, *_ = t.load_data(cfg)
    base = build(cfg, protocol, device)
    base.load_state_dict(ck['raw_model_state_dict'])
    if opt.weights == 'ema':
        base.load_state_dict(ck['model_state_dict'])
    base.eval()
    inner = base.model.inner
    layer = inner.layers[0]
    alpha0 = layer.stdp_alpha.detach().clone()
    coef0 = inner.window_coefficients.detach().clone()
    perm = torch.randperm(cfg['grid'] ** 2, generator=torch.Generator().manual_seed(1)).to(device)
    original_phases = inner.phases

    def shuffled_phases(*args, **kwargs):
        return tuple(x[..., perm, :] for x in original_phases(*args, **kwargs))

    def condition(name):
        with torch.no_grad():
            layer.stdp_alpha.copy_(alpha0)
            inner.window_coefficients.copy_(coef0)
            inner.__dict__.pop('phases', None)
            if name.startswith('alpha x'):
                layer.stdp_alpha.mul_(float(name.split('x')[1]))
            elif name == 'no 2nd harmonic':
                inner.window_coefficients[1:] = 0
            elif name == 'phases shuffled across cells':
                inner.phases = shuffled_phases

    sets = dict(train=(tr_x[:opt.n], tr_y[:opt.n]), held_out=(te_x[:opt.n], te_y[:opt.n]))
    names = ['alpha x1', 'alpha x0.75', 'alpha x0.5', 'alpha x0.25', 'alpha x0', 'no 2nd harmonic',
             'phases shuffled across cells']
    rows = []
    for name in names:
        condition(name)
        row = dict(condition=name)
        for split, (x, y) in sets.items():
            m = t.evaluate(base, x, y, cfg, 0, 1, device, int(ck['step']), None)
            row[split] = dict(acc=m['accuracy'], exact=round(m['exact_accuracy'] * m['count']), n=int(m['count']))
        rows.append(row)
        print(json.dumps(row), flush=True)
    condition('alpha x1')
    ref = rows[0]
    print(f"\nstep {ck['step']} ({opt.weights}); exact solved / {opt.n}, cell accuracy; drop relative to alpha x1")
    print(f"{'condition':32s} {'train':>16s} {'held-out':>16s}   train drop  held-out drop")
    for r in rows:
        tr, te = r['train'], r['held_out']
        dtr = 1 - tr['exact'] / max(ref['train']['exact'], 1)
        dte = 1 - te['exact'] / max(ref['held_out']['exact'], 1)
        print(f"{r['condition']:32s} {tr['exact']:5d} ({tr['acc']*100:5.1f}%) {te['exact']:5d} ({te['acc']*100:5.1f}%)"
              f"   {dtr*100:9.1f}%  {dte*100:12.1f}%")
    if opt.out:
        opt.out.write_text(json.dumps(dict(step=int(ck['step']), weights=opt.weights, n=opt.n, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
