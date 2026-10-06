import torch, json, glob, math
from lt import train as t
from lt.experiment_free_phase_windows import model_class
from lt.kv_stability import ORIGINAL_MODEL_ID
R='runs/free_tanhsech_x2_diag_20261006'
path=sorted(glob.glob(R+'/step_*.pt'),key=lambda p:int(p.split('_')[-1][:-3]))[-1]
ck=torch.load(path,map_location='cpu',weights_only=False); cfg=dict(ck['cfg']); print('checkpoint',path)
proto=json.load(open(R+'/protocol.json'))
t.KVSTDPInner=model_class(proto['window'],True,proto['modes'],proto['epsilon'],proto['generator'],'float32',proto['window_scale_factor'])
torch.manual_seed(0)
base=t.ACTLossHead(t.LT(dict(cfg,batch_size=8,seq_len=81,num_puzzle_identifiers=1)),q_weight=cfg['q_weight'])
base.load_state_dict(ck['model_state_dict']); base.eval()
inner=base.model.inner if hasattr(base.model,'inner') else base.model
L=inner.layers[0]
g=L.phase_local_gain.detach(); th=torch.cat([L.theta_k_raw,L.theta_v_raw],-1).detach()
print('gain |a|: mean %.3f  median %.3f  max %.3f  frac>0.1 %.2f'%(g.abs().mean(),g.abs().median(),g.abs().max(),(g.abs()>.1).float().mean()))
print('theta_raw |θ|: mean %.3f'%th.abs().mean())
rec=[]
orig=inner.phases
def phases(layer,rk=None,v=None):
    pk,pv=orig(layer,rk,v)
    act=torch.cat([rk,v],-1)
    rec.append(dict(pk=pk.detach(),pv=pv.detach(),act_rms=float(act.square().mean().sqrt())))
    return pk,pv
inner.phases=phases
cfg['data_npz']='data/sudoku_lt_1k.npz'
_,_,tx,ty,*_=t.load_data(cfg)
batch=next(t.eval_batches(tx[:8],ty[:8],8,0,1))
with torch.no_grad():
    carry=base.initial_carry(batch)
    for _ in range(4): carry,*_=base(return_keys=(),carry=carry,batch=batch)
for name,idx in (('first block',0),('last block',-1)):
    r=rec[idx]
    for role in ('pk','pv'):
        x=r[role]                      # [B,H,T,D]
        tok_std=x.std(dim=2).mean()    # variation across tokens for same channel
        puz_std=x.mean(2).std(0).mean()
        print(f'{name} {role}: token std {tok_std:.3f} rad, puzzle std {puz_std:.3f} rad, |phase| mean {x.abs().mean():.3f}, activity rms {r["act_rms"]:.2f}')
# fraction of pairs whose order differs from theta order
r=rec[-1]; d=r['pv'][...,:,None]-r['pk'][...,None,:]
base_k=(math.pi/2)*torch.tanh(L.theta_k_raw.detach()); base_v=(math.pi/2)*torch.tanh(L.theta_v_raw.detach())
d0=base_v[:,:,None]-base_k[:,None,:]
flip=((d.sign()!=d0[None,:,None].sign())).float().mean()
print('last block: pair/token order reversed vs theta-only: %.2f%%'%(flip*100))
