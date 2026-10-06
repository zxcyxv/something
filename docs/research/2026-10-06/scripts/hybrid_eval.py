import torch, json
from lt import train as t
from lt.experiment_free_phase_windows import model_class
R='runs/free_tanhsech_x2_tieqkvo_diag_20261006'
ck=torch.load(R+'/step_11000.pt',map_location='cpu',weights_only=False); cfg=dict(ck['cfg'])
raw,sh=ck['raw_model_state_dict'],ck['ema_shadow']
t.KVSTDPInner=model_class('tanhsech',True,8,.35,'diagonal','float32',2.,True,True)
torch.set_float32_matmul_precision('highest'); dev=torch.device('cuda'); t._resolve_precision(cfg,dev)
cfg['data_npz']='data/sudoku_lt_1k.npz'; _,_,tx,ty,*_=t.load_data(cfg)
with torch.device(dev): base=t.ACTLossHead(t.LT(dict(cfg,batch_size=cfg['global_batch_size'],seq_len=81,num_puzzle_identifiers=1)),q_weight=cfg['q_weight'])
base.eval()
def run(name,use_ema):
    base.load_state_dict(raw,strict=True)
    with torch.no_grad():
        for n,p in base.named_parameters():
            if use_ema(n): p.copy_(sh[n])
    m=t.evaluate(base,tx[:1024],ty[:1024],dict(cfg,test_size=1024),0,1,dev,0,ema=None)
    print(f'{name:38s} acc {m["accuracy"]*100:.2f}%  exact {round(m["exact_accuracy"]*m["count"])}/1024  loss {m["lm_loss"]:.3f}',flush=True)
attn=lambda n:('q_proj' in n or 'v_proj' in n)
run('raw',lambda n:False)
run('ema all',lambda n:True)
run('ema except shared QK/VO (raw)',lambda n:not attn(n))
run('raw except shared QK/VO (ema)',attn)
run('ema except Q=K only',lambda n:'q_proj' not in n)
run('ema except V=O only',lambda n:'v_proj' not in n)
