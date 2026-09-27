"""Isolate QR sign jumps in saved, actually trained v1.7 checkpoints."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lt.ckpt_npz import load
from lt.compare_raw_ema_v17 import import_original, normalize, make_batch, predict
from lt.probe_qr_collapse_v17 import qr_parts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,nargs='+',default=[16,64,256])
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--out',type=Path,default=ROOT/'runs/v17_recovery/qr_collapse/counterfactual')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    tv=import_original(ROOT/'runs/v17_recovery/source/2026-09-09/analysis/train_v17.py')
    raw,meta=load(str(ROOT/'runs/v17_recovery/checkpoints/v17_step_380000.npz'),which='raw')
    initial=normalize(tv,raw)
    wc_key='inner.layers.0.wc_raw'
    qref,rref=qr_parts(initial[wc_key].cuda())
    with np.load(ROOT/'data/sudoku_lt_1k.npz') as z:
        x=z['test_inputs'].reshape(-1,81)[:args.n].astype(np.int32)+1
        y=z['test_labels'].reshape(-1,81)[:args.n].astype(np.int32)+1
    batch=make_batch(x,y)
    result=[]
    for step in args.steps:
        path=ROOT/f'runs/v17_recovery/qr_collapse/original/step_{380000+step}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        trained=normalize(tv,state['raw_model_state_dict'])
        cfg=dict(state['cfg']);cfg.update(batch_size=args.n)
        model=tv.LT(cfg).cuda().eval()
        model.load_state_dict(trained,strict=True)
        inner=model.inner
        original=inner.W_C
        qcur,rcur=qr_parts(inner.layers[0].wc_raw)
        flips=(rcur.sign()!=rref.sign()).sum(-1).tolist()
        for variant in ['original','restore_sign_all','restore_sign_head0','restore_sign_other_heads',
                        'restore_wc_all','only_wc_update']:
            model.load_state_dict(initial if variant=='only_wc_update' else trained,strict=True)
            inner.W_C=original
            with torch.no_grad():
                if variant=='restore_wc_all':
                    inner.layers[0].wc_raw.copy_(initial[wc_key])
                elif variant=='only_wc_update':
                    inner.layers[0].wc_raw.copy_(trained[wc_key])
            if variant.startswith('restore_sign'):
                def aligned(L,variant=variant):
                    q,r=qr_parts(L.wc_raw)
                    signs=r.sign()*rref.sign()
                    if variant=='restore_sign_head0': signs[1:]=1
                    if variant=='restore_sign_other_heads': signs[0]=1
                    ab=(q*signs.unsqueeze(-2)).transpose(-1,-2)
                    return ab[:,:inner.p],ab[:,inner.p:]
                inner.W_C=aligned
            preds=predict(model,batch,16)
            if variant=='original' and args.n==128:
                baseline=np.load(ROOT/f'runs/v17_recovery/qr_collapse/original/raw_pred_{step:04d}.npy')
                assert np.array_equal(preds,baseline)
            matches=preds==y[None]
            row=dict(step=380000+step,variant=variant,n=args.n,
                     exact=int(matches[-1].all(-1).sum()),cell_accuracy=float(matches[-1].mean()),
                     exact_per_segment=matches.all(-1).sum(-1).tolist(),
                     sign_difference_from_initial_per_head=flips)
            np.save(args.out/f'{step:04d}_{variant}.npy',preds)
            result.append(row)
            (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        del model,state,trained


if __name__=='__main__':
    main()
