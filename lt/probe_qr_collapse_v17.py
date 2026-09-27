"""Causal QR-continuity comparison using actual original v1.7 training updates.

The smooth branch equals the saved model at initialization, then removes only
QR column-sign discontinuities with a positive-diagonal R convention anchored
to that initial model. Raw and EMA evaluation have separate initial anchors.
This is a diagnostic intervention, not a production architecture change.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt.ckpt_npz import load
from lt.compare_raw_ema_v17 import import_original, normalize, digest, predict, make_batch
from lt.probe_training_v17 import scalarize


def qr_parts(w):
    q, r = torch.linalg.qr(w.transpose(-1, -2))
    return q, r.diagonal(dim1=-2, dim2=-1)


@torch.no_grad()
def transition(before, after):
    qa, ra = qr_parts(before)
    qb, rb = qr_parts(after)
    dots = (qa * qb).sum(-2)
    smooth_a = qa * ra.sign().unsqueeze(-2)
    smooth_b = qb * rb.sign().unsqueeze(-2)
    return dict(raw_relative_change=float((after-before).norm()/before.norm()),
                q_relative_change=float((qb-qa).norm()/qa.norm()),
                positive_r_q_relative_change=float((smooth_b-smooth_a).norm()/qa.norm()),
                negative_column_dots_per_head=(dots < 0).sum(-1).tolist(),
                r_diagonal_sign_changes_per_head=(ra.sign() != rb.sign()).sum(-1).tolist(),
                min_column_dot_per_head=dots.min(-1).values.tolist(),
                minimum_abs_r_diagonal=float(rb.abs().min()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['original', 'smooth'], required=True)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--eval-every', type=int, default=16)
    ap.add_argument('--eval-n', type=int, default=128)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT / 'runs/v17_recovery/source/2026-09-09/analysis/train_v17.py'
    ckpt = ROOT / 'runs/v17_recovery/checkpoints/v17_step_380000.npz'
    tv = import_original(source)
    raw, meta = load(str(ckpt), which='raw')
    ema_sd, _ = load(str(ckpt), which='ema')
    raw, ema_sd = normalize(tv, raw), normalize(tv, ema_sd)
    cfg = dict(meta['cfg'])
    cfg.update(batch_size=128, global_batch_size=128, seq_len=81,
               num_puzzle_identifiers=1, loops=16, blocks_per_seg=8,
               lr=1e-4, puzzle_emb_lr=1e-4, lr_min_ratio=1.,
               lr_warmup_steps=0, grad_accum_steps=1, amp=True, compile=False)
    with torch.device('cuda'):
        base = tv.ACTLossHead(tv.LT(cfg), 'stablemax_cross_entropy', q_weight=cfg['q_weight'])
    base.model.load_state_dict(raw, strict=True)
    optimizers, rates = tv.create_optimizers(base, cfg, world_size=1)
    state = tv.TrainState(step=meta['step'])
    ema = tv.EMAHelper(mu=cfg['ema_rate'])
    ema.shadow = {name: ema_sd[name.removeprefix('model.')].cuda().clone()
                  for name, p in base.named_parameters() if p.requires_grad}
    inner = base.model.inner
    wc_name = 'model.inner.layers.0.wc_raw'
    wc = inner.layers[0].wc_raw
    with torch.no_grad():
        raw_anchor = qr_parts(wc)[1].sign()
        ema_anchor = qr_parts(ema.shadow[wc_name])[1].sign()
    active_anchor = raw_anchor
    initial_q = torch.cat(inner.W_C(inner.layers[0]), dim=1).detach()
    if args.mode == 'smooth':
        def smooth_W_C(L):
            q, rd = qr_parts(L.wc_raw)
            ab = (q * (rd.sign()*active_anchor).unsqueeze(-2)).transpose(-1, -2)
            return ab[:, :inner.p, :], ab[:, inner.p:, :]
        inner.W_C = smooth_W_C
    assert torch.equal(initial_q, torch.cat(inner.W_C(inner.layers[0]), dim=1))

    with np.load(ROOT / 'data/sudoku_lt_1k.npz', allow_pickle=False) as z:
        dataset = tv.SudokuTrainDataset(z['train_inputs'].reshape(-1,9,9),
                    z['train_labels'].reshape(-1,9,9), seed=0, num_aug=cfg['num_aug'],
                    global_batch_size=128, rank=0, world_size=1, epochs_per_iter=250,
                    start_iter=0, total_iters=1)
        test_x = z['test_inputs'].reshape(-1,81)[:args.eval_n].astype(np.int32)+1
        test_y = z['test_labels'].reshape(-1,81)[:args.eval_n].astype(np.int32)+1
    batches = iter(dataset)
    test = make_batch(test_x, test_y)
    current = {}
    training = True
    def forward_hook(module, inputs, result):
        if training:
            carry, outputs = result
            correct = outputs['logits'].detach().argmax(-1) == carry.current_data['labels']
            current.update(train_exact=int(correct.all(-1).sum()),
                           train_cell_accuracy=float(correct.float().mean()),
                           segment=int(carry.steps[0]))
    hook = base.model.register_forward_hook(forward_hook)
    old_step = optimizers[1].step
    def observed_optimizer():
        current['dense_gradient_norm'] = float(torch.stack(
            [p.grad.detach().float().square().sum() for p in base.parameters() if p.grad is not None]).sum().sqrt())
        assert np.isfinite(current['dense_gradient_norm'])
        return old_step()
    optimizers[1].step = observed_optimizer

    metadata = dict(mode=args.mode, checkpoint_sha256=digest(ckpt), source_sha256=digest(source),
                    config=cfg, steps=args.steps, eval_n=args.eval_n, eval_every=args.eval_every,
                    initial_projection_identical=True,
                    smooth_rule='Q * sign(diag(R)) * initial_sign(diag(R)); separate raw and EMA initial anchors',
                    optimizer='fresh original AdamATan2 + sparseSignSGD',
                    ema='restore saved EMA then original update; raw common embedding buffer',
                    hypothesis='QR gauge discontinuities cause loss/accuracy jumps without large raw updates',
                    criterion='Original jump coincides with QR sign event; continuous chart removes jump and matched projection-only replay restores outputs',
                    limitations='Fresh optimizer/data stream; this does not reconstruct historical 328k weights')
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    started = time.monotonic()
    eval_log = (args.out/'eval.jsonl').open('w', buffering=1)
    def evaluate(local_step):
        nonlocal training, active_anchor
        training = False
        base.eval()
        raw_before = {n:p.detach().clone() for n,p in base.named_parameters()}
        for kind in ['raw','ema']:
            active_anchor = raw_anchor if kind == 'raw' else ema_anchor
            with tv._EMASwap(base, None if kind=='raw' else ema):
                preds = predict(base.model, test, 16)
            matches = preds == test_y[None]
            row = dict(local_step=local_step, step=state.step, weights=kind,
                       exact=int(matches[-1].all(-1).sum()), cell_accuracy=float(matches[-1].mean()),
                       exact_per_segment=matches.all(-1).sum(-1).tolist(),
                       elapsed_seconds=time.monotonic()-started)
            np.save(args.out/f'{kind}_pred_{local_step:04d}.npy', preds)
            eval_log.write(json.dumps(row)+'\n')
            print(json.dumps(dict(event='eval',mode=args.mode,**row)),flush=True)
        assert all(torch.equal(p,raw_before[n]) for n,p in base.named_parameters())
        active_anchor = raw_anchor
        base.train()
        training = True

    evaluate(0)
    with (args.out/'steps.jsonl').open('w',buffering=1) as log:
        for local_step in range(1,args.steps+1):
            current.clear()
            raw_prev = wc.detach().clone()
            ema_prev = ema.shadow[wc_name].clone()
            _, batch = next(batches)
            metrics = tv.train_batch(base,base,state,batch,cfg,optimizers,rates,
                                    total_steps=meta['step']+args.steps,
                                    rank=0,world_size=1,device=torch.device('cuda'))
            ema.update(base)
            current.update(local_step=local_step,step=state.step,metrics=scalarize(metrics),
                           qr=transition(raw_prev,wc), ema_qr=transition(ema_prev,ema.shadow[wc_name]),
                           elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(current)+'\n')
            if local_step%args.eval_every==0 or local_step==args.steps:
                torch.save(dict(raw_model_state_dict=base.state_dict(),ema_shadow=ema.shadow,
                                optimizer_states=[o.state_dict() for o in optimizers],carry=state.carry,
                                step=state.step,cfg=cfg,raw_anchor=raw_anchor,ema_anchor=ema_anchor),
                           args.out/f'step_{state.step}.pt')
                print(json.dumps(dict(event='train',mode=args.mode,**current)),flush=True)
                evaluate(local_step)
    hook.remove()
    eval_log.close()
    (args.out/'complete.json').write_text(json.dumps(dict(steps=args.steps,elapsed_seconds=time.monotonic()-started))+'\n')


if __name__=='__main__':
    main()
