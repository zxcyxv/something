"""Historical v1.7 initialization/equations with the current aligned logging harness."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from . import train as t
from .audit_training_harness import load_old


NATIVE_INNER=t.LT_Inner


def install_historical_initialization(old):
    class HistoricalInitInner(NATIVE_INNER):
        def __init__(self,config):
            # Native construction consumes no lasting random state. The original
            # initializer determines both initial weights and the post-init RNG.
            with torch.random.fork_rng(devices=[]):
                super().__init__(config)
            reference=old.LT_Inner(old.LTConfig.from_dict(dict(old.CFG, **vars(config))))
            self.load_state_dict(reference.state_dict(),strict=True)
    t.LT_Inner=HistoricalInitInner
    t.model_id_of=lambda cfg:'v17-historical-init-current-harness-aligned-v1'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=6000)
    args=ap.parse_args()
    source=Path('docs/research/2026-10-03/reference/train_v17.py')
    old=load_old(source)
    cfg=dict(t.CFG,**old.CFG);cfg.update(t.PRESETS['v1.7'])
    cfg.update(json.loads(Path(args.config).read_text()))
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg.update(out_dir=str(out),max_steps=args.steps,max_hours=None,log_every=16,
               run_selftests=False,num_processes=1,milestone_every=0)
    assert cfg['loops']==16 and cfg['blocks_per_seg']==8 and cfg['num_layers']==1
    assert cfg['memory_type']=='address' and cfg['address_projection']=='qr'
    if any(cfg.get(k,0) for k in ('nograd_fixed','nograd_every','late_sup_prob')):
        raise ValueError('Matched run requires no extra no-grad or unsupervised segments.')
    protocol=dict(model='historical v1.7',source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initialization='exact historical initializer, strict state load into equivalent native model',
        loops=16,iterations_per_segment=8,layers=1,act_enabled=False,
        optimizer_update_per_segment=True,detach_between_segments=True,
        architecture='original QR shared complex address, eligibility trace, STDP and historical phi; no new RMSNorm',
        logging='All steps in train.jsonl; console every 16 steps at terminal segment. Loss/accuracy at same step.',
        criterion='Compare aligned loss/accuracy and post-warmup window means; historical sparse mixed-segment log is not a matched baseline.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'launch_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for src in [source,Path(t.__file__),Path(__file__)]:
        shutil.copyfile(src,out/(src.stem+'_snapshot.py'))
    install_historical_initialization(old)
    original=t.train_batch
    start=time.monotonic()
    def train_batch(*a,**kw):
        result=original(*a,**kw)
        if result is not None:
            carry=a[2].carry
            record=dict(result,step=a[2].step,segment=int(carry.steps[0]),elapsed=time.monotonic()-start)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return result
    t.train_batch=train_batch
    print('[v1.7] Historical model/initialization; current harness; loops=16; aligned terminal logging',flush=True)
    t.main(cfg)
    (out/'finished.json').write_text(json.dumps(dict(elapsed=time.monotonic()-start))+'\n')


if __name__=='__main__':main()
