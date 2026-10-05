"""Isolated full-batch compiled GPU preflight; never resumes or stops a run."""
import argparse
import json
from pathlib import Path
import time

import torch

from . import train as t
from .kv_stability import install


def main():
    ap = argparse.ArgumentParser()
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument('--coactivity', action='store_true', help='Test full-profile Gram attention')
    choice.add_argument('--local-warp', action='store_true', help='Test token-local monotone phase warp')
    args = ap.parse_args()
    tag = 'local_warp' if args.local_warp else 'channel_gram' if args.coactivity else 'channel'
    branch_name = 'phase_warp' if args.local_warp else 'phase_attention'
    torch.set_num_threads(2)
    cfg = dict(t.CFG)
    cfg.update(json.loads(Path(f'configs/kv_phase_{tag}_exp_full_research.json').read_text()))
    cfg.update(data_npz=str(Path('data/sudoku_lt_1k.npz').resolve()),
               research_variant=f'phase_{tag}_exp_current_only')
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    install(cfg['research_variant'])
    torch.manual_seed(cfg['seed'])
    x, y, _, _, _, _ = t.load_data(cfg)
    batch = next(t.eval_batches(x[:128], y[:128], 128, 0, 1))
    mcfg = dict(cfg, batch_size=128, seq_len=81, num_puzzle_identifiers=1)
    with torch.device(device):
        base = t.ACTLossHead(t.LT(mcfg), q_weight=cfg['q_weight'])
    base.train()
    opts, lrs = t.create_optimizers(base, cfg, 1)
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    compiled = torch.compile(base, dynamic=False)
    state = t.TrainState()
    original_check = t._check_finite_gradients
    gradient_records = []
    def check(model, loss, device, ws):
        original_check(model, loss, device, ws)
        gradient_records.append({n:float(p.grad.norm()) for n,p in model.named_parameters()
                                 if branch_name in n and p.grad is not None})
    t._check_finite_gradients = check
    report = dict(kind='isolated preflight; timings include concurrent existing training',
                  variant=cfg['research_variant'],
                  batch=128, hidden=832, heads=8, blocks=8, amp=cfg['amp_dtype'],
                  compile=True, activation_checkpoint=True, steps=[])
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3):
            started = time.monotonic()
            metrics = t.train_batch(compiled, base, state, batch, cfg, opts, lrs,
                                    390625, 0, 1, device)
            torch.cuda.synchronize()
            report['steps'].append(dict(step=state.step, elapsed=time.monotonic()-started,
                                        **metrics))
            print('preflight', report['steps'][-1], flush=True)
    finally:
        t._check_finite_gradients = original_check
    expected = [n for n,p in base.named_parameters() if branch_name in n]
    assert set(gradient_records[-1]) == set(expected)
    assert all(value > 0 for value in gradient_records[-1].values())
    report.update(phase_gradients=gradient_records,
                  parameters=sum(p.numel() for p in base.parameters()),
                  phase_branch_parameters=sum(p.numel() for n,p in base.named_parameters()
                                              if branch_name in n),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    branch = getattr(base.model.inner.layers[0], branch_name)
    projection = branch.projection if args.local_warp else branch.readout
    report['projection_weight_norm' if args.local_warp else 'readout_weight_norm'] = float(
        projection.weight.norm().detach())
    path = Path(f'docs/research/2026-10-05/{tag}_phase_gpu_preflight.json')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('PASS', path, flush=True)


if __name__ == '__main__':
    main()
