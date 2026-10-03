"""One isolated diagnostic episode with actual within-episode optimizer steps.

Starts from a completed checkpoint and its exact saved batch. It writes JSON
only, never checkpoints. This is not a new long training run. Fixed-parameter
replays at the initial and final weights bracket the online trajectory.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from . import train as t
from .kv_stability import install
from .probe_kv_frozen_episode import metrics


@torch.no_grad()
def frozen(inner, batch, segments, block_count):
    training = inner.training
    inner.eval()
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            injection = inner.injection(batch)
            state = inner.init_hidden[None,None,:].expand(len(batch['labels']),81,-1), None, None, None
            rows = []
            for segment in range(1, segments+1):
                for _ in range(block_count):
                    for layer in inner.layers:
                        state = inner.block(layer, state[0], injection, *state[1:], None)
                rows.append({'segment':segment, **metrics(inner.w_cls(state[0]).float(),batch['labels'])})
        return rows
    finally:
        inner.train(training)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'method':'one original-harness 16-segment training episode per checkpoint, same saved 128 puzzles, fresh carry; fixed-parameter replays at starting and ending weights',
              'precision':'same BF16/FP32 as training; no compile',
              'caveat':'Local diagnostic at step6000. Earlier training history and collapse are not replayed. Updating on a previously seen saved batch can be beneficial, so a harmful effect is not assumed.',
              'runs':{}}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    for path in args.checkpoints:
        started = time.monotonic()
        ck = torch.load(path,map_location='cpu',weights_only=False)
        variant = ck['cfg'].get('research_variant','original')
        install(variant)
        data = ck['rank_states'][0]['carry']['current_data']
        n = len(data['labels'])
        cfg = dict(ck['cfg'], batch_size=n, global_batch_size=n, seq_len=81,
                   num_puzzle_identifiers=1, amp=True, activation_checkpoint=True,
                   nograd_blocks=0, compile=False)
        with torch.device('cuda'):
            base = t.ACTLossHead(t.LT(cfg),q_weight=cfg['q_weight'])
        base.load_state_dict(ck['raw_model_state_dict'])
        opts,rates = t.create_optimizers(base,cfg,1)
        for opt,sd in zip(opts,ck['optimizer_states']):
            opt.load_state_dict(sd)
        batch = {k:v.cuda() for k,v in data.items()}
        segments,blocks = cfg['loops'],cfg['blocks_per_seg']
        start_rows = frozen(base.model.inner,batch,segments,blocks)
        base.train()
        ts = t.TrainState(step=ck['step'])
        capture = {}
        def save_logits(module,inputs,output):
            capture['logits'] = output.detach().float()
        hook = base.model.inner.w_cls.register_forward_hook(save_logits)
        online=[]
        for segment in range(1,segments+1):
            logged=t.train_batch(base,base,ts,batch,cfg,opts,rates,ck['step']+segments,0,1,torch.device('cuda'))
            with torch.no_grad():
                online.append({'segment':segment,**metrics(capture['logits'],batch['labels']),
                               'logged_lm_loss':logged['lm_loss'],'lr':logged['lr']})
        hook.remove()
        finish_rows=frozen(base.model.inner,batch,segments,blocks)
        record={'checkpoint':str(Path(path).resolve()),'puzzles':n,'optimizer_steps':segments,
                'elapsed_seconds':time.monotonic()-started,
                'initial_frozen':start_rows,'online':online,'final_frozen':finish_rows}
        result['runs'][variant]=record
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(variant,json.dumps({k:record[k][-1] for k in ('initial_frozen','online','final_frozen')}),flush=True)
        del ck,base,opts,ts
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
