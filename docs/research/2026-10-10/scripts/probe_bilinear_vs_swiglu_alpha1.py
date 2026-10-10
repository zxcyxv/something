"""alpha=1 general-window model at initialization: would SwiGLU differ from its bilinear boundary?

Bilinear boundary (current): h + D(1/2 g*u). Counterfactual with the same weights:
h + D(silu(g)*u). Hooks every boundary call over held-out puzzles (16 segments x 8 blocks).
"""
import json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.experiment_free_phase_windows import model_class

logs = Path('/workspace/something/docs/research/2026-10-09/logs/pairangle_unrotated_dca1_tau2_r2_qkl2_sum_20261009')
n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
cfg = json.loads((logs / 'config.json').read_text())
protocol = json.loads((logs / 'protocol.json').read_text())
cfg.update(compile=False, activation_checkpoint=False, data_npz='/workspace/something/data/sudoku_lt_1k.npz')
torch.manual_seed(cfg['seed'])
torch.set_float32_matmul_precision(protocol['precision'])
dev = torch.device('cuda')
t._resolve_precision(cfg, dev)
t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
    protocol['generator'], protocol.get('feature_precision', 'float32'), protocol.get('window_scale_factor', 1.0),
    protocol.get('tie_qk', False), protocol.get('tie_vo', False), protocol.get('qk_l2', False),
    protocol.get('write_sum', False), protocol.get('tau_phi', 2.0), protocol.get('phase_floor', 0.5),
    protocol.get('v_norm', 'none'), protocol.get('tie_all', False), protocol.get('phase_kappa', 1.0),
    protocol.get('phase_omega', 0.0), protocol.get('phase_frame', 'rotated'), protocol.get('dc_hebbian', False),
    protocol.get('dc_alpha_init', 0.0))
mcfg = dict(cfg, batch_size=cfg['global_batch_size'], seq_len=cfg['grid'] ** 2, num_puzzle_identifiers=1)
with torch.device(dev):
    base = t.ACTLossHead(t.LT(mcfg), q_weight=cfg['q_weight'])
inner = base.model.inner
print('inner', type(inner).__mro__[1].__name__, 'alpha', [round(x, 3) for x in inner.layers[0].stdp_alpha.tolist()])

S = dict(zi=0., zb=0., wnorm=0., calls=0, numel=0, g1=0, g2=0, gsq=0., xstd=0., xnorm=0., e=0., nb=0., ns=0., res=0.)
def hook(mod, inp, out):
    x = inp[0].float(); g, u = out.float().chunk(2, dim=-1)
    W = layer.b_down.weight.float()
    bil, swi = F.linear(0.5 * g * u, W), F.linear(F.silu(g) * u, W)
    S['calls'] += 1; S['numel'] += g.numel()
    S['g1'] += (g.abs() > 1).sum().item(); S['g2'] += (g.abs() > 2).sum().item()
    S['gsq'] += g.square().sum().item()
    S['xstd'] += x.square().mean().item() ** .5; S['xnorm'] += x.norm(dim=-1).mean().item()
    S['e'] += (swi - bil).square().sum().item()
    S['nb'] += bil.square().sum().item(); S['ns'] += swi.square().sum().item()
    S['res'] += (x + bil).square().sum().item()
    zb, zs = 0.5 * g * u, F.silu(g) * u
    S['zi'] += (zs - zb).square().sum().item(); S['zb'] += zb.square().sum().item()
    S['wnorm'] = W.norm().item()
for layer in inner.layers:
    layer.b_gate_up.register_forward_hook(hook)

_, _, te_x, te_y, *_ = t.load_data(cfg)
with torch.no_grad():
    t.evaluate(base, te_x[:n], te_y[:n], cfg, 0, 1, dev, 0, None)
N, c = S['numel'], S['calls']
print(f'boundary calls {c}; FFN input: element RMS {S["xstd"]/c:.3f}, token norm {S["xnorm"]/c:.1f} (sqrt d = {cfg["hidden_size"]**.5:.1f})')
print(f'gate g: std {(S["gsq"]/N)**.5:.3f}  P(|g|>1) {S["g1"]/N:.3f}  P(|g|>2) {S["g2"]/N:.3f}')
print(f'b_down weight norm {S["wnorm"]:.3g}')
print(f'intermediate ||silu(g)u - g u/2|| / ||g u/2||: {(S["zi"]/S["zb"])**.5:.3f}')
if S['nb'] == 0: sys.exit()
print(f'||SwiGLU - bilinear|| / ||bilinear||: {(S["e"]/S["nb"])**.5:.3f}   / ||SwiGLU||: {(S["e"]/S["ns"])**.5:.3f}')
print(f'difference relative to residual sum ||x + bilinear||: {(S["e"]/S["res"])**.5:.3f}')
