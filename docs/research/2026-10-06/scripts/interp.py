import torch, sys
from lt import train as t
from lt.experiment_free_phase_windows import model_class
import json
def go(R,step,tie):
    ck=torch.load(f'{R}/step_{step}.pt' if 'tieqkvo' not in R else f'{R}/raw_ema_hold/interp_{step}.pt',map_location='cpu',weights_only=False); cfg=dict(ck['cfg'])
    raw,sh=ck['raw_model_state_dict'],ck['ema_shadow']
    t.KVSTDPInner=model_class('tanhsech',True,8,.35,'diagonal','float32',2.,tie,tie)
    dev=torch.device('cuda'); t._resolve_precision(cfg,dev); cfg['data_npz']='data/sudoku_lt_1k.npz'
    _,_,tx,ty,*_=t.load_data(cfg)
    with torch.device(dev): base=t.ACTLossHead(t.LT(dict(cfg,batch_size=cfg['global_batch_size'],seq_len=81,num_puzzle_identifiers=1)),q_weight=cfg['q_weight'])
    base.eval(); base.load_state_dict(raw,strict=True)
    params={n:p for n,p in base.named_parameters() if n in sh}
    rawp={n:p.detach().clone() for n,p in params.items()}
    print('==',R.split('/')[-1],step,flush=True)
    for a in (0,.25,.5,.75,1):
        with torch.no_grad():
            for n,p in params.items(): p.copy_((1-a)*rawp[n]+a*sh[n].to(p.device))
        m=t.evaluate(base,tx[:1024],ty[:1024],dict(cfg,test_size=1024),0,1,dev,0,ema=None)
        print(f'  alpha(EMA)={a:.2f}  acc {m["accuracy"]*100:.2f}%  exact {round(m["exact_accuracy"]*m["count"])}/1024  loss {m["lm_loss"]:.3f}',flush=True)
torch.set_float32_matmul_precision('highest')
go('runs/free_tanhsech_x2_diag_20261006',16000,False)
go('runs/free_tanhsech_x2_diag_20261006',15624,False)
go(sys.argv[1],int(sys.argv[2]),True)
