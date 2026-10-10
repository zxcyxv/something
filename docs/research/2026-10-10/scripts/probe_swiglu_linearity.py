"""How far is URM's SwiGLU from its bilinear limit on real FFN inputs?

SiLU(g) = g/2 + g^2/4 - g^4/96 + ...  so  SwiGLU(x) = 1/2 D(g*u) + 1/4 D(g^2*u) + ...
Reports gate pre-activation stats and the relative error of the degree-2 (bilinear)
and degree-3 truncations of the actual FFN output, over all 8 x 16 recurrent calls.
"""
import json, sys
import torch
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.urm_full_bptt import install

run, ckpt, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256
cfg = dict(t.CFG); cfg.update(json.load(open(f'{run}/config.json')))
cfg.update(compile=False, activation_checkpoint=False, test_size=n, batch_size=cfg['global_batch_size'], seq_len=cfg['grid']**2, num_puzzle_identifiers=1)
install(cfg.get('urm_ffn', 'swiglu'), cfg['num_layers'])
dev = torch.device('cuda')
base = t.ACTLossHead(t.LT(cfg), q_weight=cfg['q_weight']).to(dev)
ck = {'step': 0, 'raw_model_state_dict': None} if ckpt == 'init' else torch.load(ckpt, map_location=dev, weights_only=False)
swap = len(sys.argv) > 5 and sys.argv[5] == 'bilinear'
which = sys.argv[4] if len(sys.argv) > 4 else 'ema'
sd = None
if which == 'ema':
    for key in ('ema_shadow', 'ema_state_dict', 'ema'):
        if key in ck:
            sd = ck[key]; break
    if sd is None:
        print('no EMA in checkpoint; keys:', list(ck)); which = 'raw'
if sd is None and ckpt != 'init':
    sd = ck.get('raw_model_state_dict') or ck['model_state_dict']
if sd is not None:
    sd = t.strip_prefix(sd)
if ckpt == 'init': which = 'init'
elif which == 'ema':  # EMA shadow lacks buffers/sparse embedding: start from raw, overlay EMA
    base.load_state_dict(t.strip_prefix(ck['raw_model_state_dict']))
if sd is not None:
    missing, unexpected = base.load_state_dict(sd, strict=False)
print(f'[{which}] step {ck.get("step")} swap_to_bilinear={swap}')

stats = dict(res=0., rerr=0., calls=0, g_abs=[], g_gt1=0., g_gt2=0., numel=0, e2=0., e3=0., norm=0., lin_share=0.)
def hook(mod, inp, out):
    x = inp[0].float()
    gate, up = mod.gate_up_proj(inp[0]).float().chunk(2, dim=-1)
    W = mod.down_proj.weight.float()
    full = out.float()
    bil = (0.5 * gate * up) @ W.T
    cub = bil + (0.25 * gate.square() * up) @ W.T
    stats['calls'] += 1
    stats['numel'] += gate.numel()
    stats['g_gt1'] += (gate.abs() > 1).sum().item()
    stats['g_gt2'] += (gate.abs() > 2).sum().item()
    stats['g_abs'].append(gate.abs().mean().item())
    stats['e2'] += (full - bil).square().sum().item()
    stats['e3'] += (full - cub).square().sum().item()
    stats['norm'] += full.square().sum().item()
    stats['res'] += (x + full).square().sum().item()
    stats['rerr'] += (full - bil).square().sum().item()
import types
def bilinear_forward(self, x):
    gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
    return self.down_proj(0.5 * gate * up)
for layer in base.model.inner.layers:
    if swap: layer.mlp.forward = types.MethodType(bilinear_forward, layer.mlp)
    else: layer.mlp.register_forward_hook(hook)

data = t.load_data(cfg)
te_x, te_y = data[2], data[3]
with torch.no_grad():
    t.evaluate(base, te_x[:n], te_y[:n], cfg, 0, 1, dev, ck.get('step', 0), None)
if swap: sys.exit()
N = stats['numel']
print(f'calls {stats["calls"]}  mean|g| {sum(stats["g_abs"])/len(stats["g_abs"]):.3f}  '
      f'P(|g|>1) {stats["g_gt1"]/N:.3f}  P(|g|>2) {stats["g_gt2"]/N:.3f}')
print(f'rel err bilinear (1/2 g*u): {(stats["e2"]/stats["norm"])**.5:.3f}   '
      f'with cubic term: {(stats["e3"]/stats["norm"])**.5:.3f}')
print(f'bilinear error relative to residual sum ||x + FFN(x)||: {(stats["rerr"]/stats["res"])**.5:.3f}   '
      f'||FFN|| / ||x||: {(stats["norm"]/max(stats["res"]-stats["norm"],1e-9))**.5:.3f} (approx)')
