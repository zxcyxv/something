import json, sys, torch
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.urm_full_bptt import install
run = '/workspace/something/runs/urm_swiglu_1layer_loops16_20261010'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
cfg = dict(t.CFG); cfg.update(json.load(open(f'{run}/config.json')))
cfg.update(compile=False, activation_checkpoint=False, batch_size=cfg['global_batch_size'], seq_len=81, num_puzzle_identifiers=1)
install('swiglu', 1)
dev = torch.device('cuda'); t._resolve_precision(cfg, dev)
tr_x, tr_y, te_x, te_y, *_ = t.load_data(cfg)
base = t.ACTLossHead(t.LT(cfg), q_weight=cfg['q_weight']).to(dev)
ck = torch.load(f'{run}/step_{sys.argv[1]}.pt', map_location=dev, weights_only=False)
base.load_state_dict(t.strip_prefix(ck['raw_model_state_dict']))
base.load_state_dict(t.strip_prefix(ck['ema_shadow']), strict=False)
base.eval()
for split, (x, y) in dict(train=(tr_x[:n], tr_y[:n]), held_out=(te_x[:n], te_y[:n])).items():
    m = t.evaluate(base, x, y, cfg, 0, 1, dev, ck['step'], None)
    print(f'URM step {ck["step"]} EMA {split}: exact {round(m["exact_accuracy"]*m["count"])}/{int(m["count"])} acc {m["accuracy"]*100:.1f}%', flush=True)
