"""Run real v1.7 updates with the original harness and per-step diagnostics."""
import json
import math
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt.ckpt_npz import load
from lt.compare_raw_ema_v17 import import_original, normalize, digest


def scalarize(x):
    if torch.is_tensor(x):
        return x.detach().item()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: scalarize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scalarize(v) for v in x]
    return x


def tensor_stats(x):
    x = x.detach().float()
    return dict(norm=x.norm(), max_abs=x.abs().max(), finite=torch.isfinite(x).all())


def assert_finite_tree(x):
    if isinstance(x, dict):
        for value in x.values():
            assert_finite_tree(value)
    elif isinstance(x, (list, tuple)):
        for value in x:
            assert_finite_tree(value)
    elif isinstance(x, (float, int)):
        assert math.isfinite(x), 'Nonfinite logged measurement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/v17_recovery/train_probe_380k_lr1e4')
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT / 'runs/v17_recovery/source/2026-09-09/analysis/train_v17.py'
    checkpoint = ROOT / 'runs/v17_recovery/checkpoints/v17_step_380000.npz'
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tv = import_original(source)
    raw, meta = load(str(checkpoint), which='raw')
    cfg = dict(meta['cfg'])
    cfg.update(batch_size=128, global_batch_size=128, seq_len=81,
               num_puzzle_identifiers=1, loops=16, blocks_per_seg=8,
               lr=1e-4, puzzle_emb_lr=1e-4, lr_min_ratio=1.,
               lr_warmup_steps=0, grad_accum_steps=1, amp=True, compile=False)
    with torch.device('cuda'):
        base = tv.ACTLossHead(tv.LT(cfg), 'stablemax_cross_entropy', q_weight=cfg['q_weight'])
    base.model.load_state_dict(normalize(tv, raw), strict=True)
    base.train()
    optimizers, optimizer_lrs = tv.create_optimizers(base, cfg, world_size=1)
    state = tv.TrainState(step=meta['step'])
    with np.load(ROOT / 'data/sudoku_lt_1k.npz', allow_pickle=False) as data:
        ds = tv.SudokuTrainDataset(data['train_inputs'].reshape(-1, 9, 9),
                                  data['train_labels'].reshape(-1, 9, 9), seed=0,
                                  num_aug=cfg['num_aug'], global_batch_size=128,
                                  rank=0, world_size=1, epochs_per_iter=250,
                                  start_iter=0, total_iters=1)
    batches = iter(ds)
    current = {}
    dense = dict(base.named_parameters())
    emb = base.model.puzzle_emb

    # Observe the actual optimizer calls made inside the original train_batch.
    for index, opt in enumerate(optimizers):
        old_step = opt.step
        def measured_step(old_step=old_step, index=index):
            tensors = {'puzzle_emb.weights': emb.weights} if index == 0 else dense
            before = {name: value.detach().clone() for name, value in tensors.items()}
            grads = {}
            if index == 0:
                grad = emb.local_weights.grad
                assert grad is not None and torch.all(emb.local_ids == 0)
                grads['puzzle_emb.local_weights'] = tensor_stats(grad)
                grads['puzzle_emb.shared_bias'] = tensor_stats(grad.sum(0))
            else:
                grads = {name: tensor_stats(value.grad) for name, value in dense.items()
                         if value.grad is not None}
                if args.details:
                    for name, value in dense.items():
                        if value.grad is not None and '.layers.0.' in name and value.shape[0] == 8:
                            gg = value.grad.detach().float().reshape(8, -1)
                            grads[name]['per_head_norm'] = gg.norm(dim=-1).tolist()
                            grads[name]['per_head_rms'] = gg.square().mean(-1).sqrt().tolist()
            current['gradients'].update(grads)
            assert all(bool(v['finite']) for v in grads.values()), 'Nonfinite gradient'
            old_step()
            for name, value in tensors.items():
                diff = value.detach() - before[name]
                stats = tensor_stats(diff)
                stats['relative_norm'] = diff.norm() / before[name].norm().clamp_min(1e-30)
                current['updates'][name] = stats
                assert bool(torch.isfinite(value).all()), 'Nonfinite updated weight'
        opt.step = measured_step

    inner = base.model.inner
    old_step, old_phi = inner.step, inner.phi
    old_boundary, old_trace = inner.boundary, inner.trace_step
    def record_gradient(tensor, record, key):
        if tensor.requires_grad:
            def hook(grad):
                record[key] = tensor_stats(grad)
            tensor.register_hook(hook)

    def observed_step(*args, **kwargs):
        q = args[1]
        record = {'block': len(current['blocks']) + 1,
                  'q_norm_mean': q.detach().float().norm(dim=-1).mean()}
        current['blocks'].append(record)
        record_gradient(q, record, 'q_gradient')
        return old_step(*args, **kwargs)

    def observed_phi(r):
        h = old_phi(r)
        record = current['blocks'][-1]
        record.update(pre_phi_norm_mean=r.detach().float().norm(dim=-1).mean(),
                      post_phi_norm_mean=h.detach().float().norm(dim=-1).mean())
        record_gradient(h, record, 'post_phi_gradient')
        if args.details:
            record_gradient(r, record, 'pre_phi_gradient')
            with torch.no_grad():
                ss = (1 + r.float().square().sum(-1) / inner.d).sqrt()
                record['phi_tangent_scale_mean'] = ss.reciprocal().mean()
                record['phi_tangent_scale_max'] = ss.reciprocal().max()
        return h
    inner.step, inner.phi = observed_step, observed_phi

    def observed_boundary(L, y, gate=None):
        r = old_boundary(L, y, gate)
        record = current['blocks'][-1]
        with torch.no_grad():
            yn = y.float().norm(dim=-1)
            bn = (r.float()-y.float()).norm(dim=-1)
            rn = r.float().norm(dim=-1)
            record.update(mlp_input_norm_mean=yn.mean(), mlp_input_norm_max=yn.max(),
                          mlp_output_delta_norm_mean=bn.mean(),
                          mlp_delta_over_input_mean=(bn/yn.clamp_min(1e-12)).mean(),
                          mlp_residual_over_input_mean=(rn/yn.clamp_min(1e-12)).mean(),
                          mlp_input_above_sqrt_d_fraction=(yn > math.sqrt(inner.d)).float().mean())
        record_gradient(y, record, 'mlp_input_gradient')
        return r

    def observed_trace(L, ux, uy, ztr, fresh):
        zx, zy, zz = old_trace(L, ux, uy, ztr, fresh)
        with torch.no_grad():
            zn = (zx.float().square()+zy.float().square()).sum(-1).sqrt()
            un = (ux.float().square()+uy.float().square()).sum(-1).sqrt()
            ratio = zn/un.clamp_min(1e-12)
            current['blocks'][-1].update(trace_norm_min=zn.min(), trace_norm_mean=zn.mean(),
                trace_over_address_mean=ratio.mean(), trace_over_address_max=ratio.max(),
                trace_over_address_mean_per_head=ratio.mean((0,1)).tolist())
        return zx, zy, zz
    if args.details:
        inner.boundary, inner.trace_step = observed_boundary, observed_trace

    def predictions_hook(module, args, result):
        carry, outputs = result
        correct = outputs['logits'].detach().argmax(-1) == carry.current_data['labels']
        current.update(segment=int(carry.steps[0]), current_cell_accuracy=correct.float().mean(),
                       current_exact=correct.all(-1).sum(),
                       logits_finite=torch.isfinite(outputs['logits']).all())
    hook = base.model.register_forward_hook(predictions_hook)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    source=str(source), source_sha256=digest(source), config=cfg,
                    optimizer='fresh original AdamATan2 + sparse signSGD; no restored moments',
                    initial_carry='fresh; carry preserved and detached as in original model',
                    data_stream='original augmented dataset, seed0, start_iter0; not historical stream resume',
                    steps=64, precision='original bfloat16 autocast, float32 weights',
                    additional_mlp_and_trace_observation=args.details,
                    gradient_clipping=False, compile=False,
                    block_gradient_meaning='actual loss adjoints at injected q and post-Phi h; not Jacobian norms',
                    ema='not used for gradients; no EMA evaluation in this short training run')
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    rows = []
    start = time.monotonic()
    with (out / 'steps.jsonl').open('w', buffering=1) as log:
        for local_step in range(1, 65):
            current.clear()
            current.update(local_step=local_step, gradients={}, updates={}, blocks=[])
            _, batch = next(batches)
            metrics = tv.train_batch(base, base, state, batch, cfg, optimizers, optimizer_lrs,
                                     total_steps=meta['step']+64, rank=0, world_size=1,
                                     device=torch.device('cuda'))
            current.update(step=state.step, metrics=metrics, elapsed_seconds=time.monotonic()-start)
            assert len(current['blocks']) == 8 and bool(current['logits_finite'])
            row = scalarize(current)
            row['dense_gradient_norm'] = sum(v['norm']**2 for k,v in row['gradients'].items()
                                             if not k.startswith('puzzle_emb.'))**0.5
            row['dense_update_norm'] = sum(v['norm']**2 for k,v in row['updates'].items()
                                           if k != 'puzzle_emb.weights')**0.5
            assert_finite_tree(row)
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False)+'\n')
            print(json.dumps({k:row[k] for k in ['step','segment','current_exact','dense_gradient_norm',
                                                'dense_update_norm','elapsed_seconds']} |
                             {'lm_loss':row['metrics']['lm_loss']}, ensure_ascii=False), flush=True)
    hook.remove()
    torch.save(dict(raw_model_state_dict=base.state_dict(),
                    optimizer_states=[o.state_dict() for o in optimizers],
                    carry=state.carry, step=state.step, cfg=cfg), out / 'step_380064.pt')
    result = dict(steps_completed=len(rows), final_step=state.step,
                  elapsed_seconds=time.monotonic()-start,
                  max_dense_gradient_norm=max(r['dense_gradient_norm'] for r in rows),
                  min_dense_gradient_norm=min(r['dense_gradient_norm'] for r in rows),
                  max_dense_update_norm=max(r['dense_update_norm'] for r in rows),
                  halted=[dict(step=r['step'], exact=r['current_exact'],
                               cell_accuracy=r['current_cell_accuracy'], lm_loss=r['metrics']['lm_loss'])
                          for r in rows if r['segment']==16], all_logged_values_finite=True)
    (out / 'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
