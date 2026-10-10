"""FP32 vs BF16-GEMM pairangle path on the same trained weights (alpha=1 + SwiGLU, step 3475).

1. one training step (eager, fresh carry): loss and parameter-gradient agreement
2. held-out eval (EMA, 16 segments x 8 blocks): accuracy / exact, per-puzzle agreement
3. compiled training-step time on a fixed batch
"""
import json, sys, time, statistics
from pathlib import Path
import torch
sys.path.insert(0, '/workspace/something')
from lt import train as t
from lt.experiment_free_phase_windows import model_class

run = Path('/workspace/something/runs/pairangle_unrotated_dca1_swiglu_tau2_r2_qkl2_sum_20261010')
ck = torch.load(run / 'step_3475.pt', map_location='cpu', weights_only=False)
protocol = json.loads((run / 'protocol.json').read_text())
cfg = dict(ck['cfg'])
n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = torch.device('cuda')
torch.set_float32_matmul_precision(protocol['precision'])
t._resolve_precision(cfg, dev)
tr_x, tr_y, te_x, te_y, *_ = t.load_data(cfg)
train_batch = next(t.eval_batches(tr_x[:128], tr_y[:128], 128, 0, 1))

def build(precision, kernel='torch'):
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
        protocol['generator'], precision, protocol.get('window_scale_factor', 1.0), False, False, True, True,
        protocol.get('tau_phi', 2.0), 0.5, 'none', False, 1.0, 0.0, 'unrotated', True, 1.0, 'swiglu', kernel)
    with torch.device(dev):
        base = t.ACTLossHead(t.LT(dict(cfg, batch_size=128, seq_len=81, num_puzzle_identifiers=1)), q_weight=cfg['q_weight'])
    return base

def one_step(precision, kernel='torch'):
    torch.manual_seed(0)
    base = build(precision, kernel); base.load_state_dict(ck['raw_model_state_dict']); base.train()
    opts, lrs = t.create_optimizers(base, dict(cfg, lr=0.0, puzzle_emb_lr=0.0), 1)
    grads, old = {}, t._check_finite_gradients
    def capture(model, loss, device, ws):
        old(model, loss, device, ws)
        if not grads:
            grads.update({n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None})
    t._check_finite_gradients = capture
    try:
        m = t.train_batch(base, base, t.TrainState(), train_batch, cfg, opts, lrs, 390625, 0, 1, dev)
    finally:
        t._check_finite_gradients = old
    return m, grads

def evaluate(precision, kernel='torch'):
    base = build(precision, kernel); base.load_state_dict(ck['model_state_dict']); base.eval()
    preds = []
    orig = base.forward
    m = t.evaluate(base, te_x[:n_eval], te_y[:n_eval], cfg, 0, 1, dev, ck['step'], None)
    return m

def timing(precision, kernel='torch', steps=14, ckpt=True):
    torch.manual_seed(0)
    cfg['activation_checkpoint'] = ckpt
    base = build(precision, kernel); base.load_state_dict(ck['raw_model_state_dict']); base.train()
    opts, lrs = t.create_optimizers(base, cfg, 1)
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    compiled = torch.compile(base, dynamic=False)
    state, sec = t.TrainState(), []
    for _ in range(steps):
        torch.cuda.synchronize(); s = time.monotonic()
        t.train_batch(compiled, base, state, train_batch, cfg, opts, lrs, 390625, 0, 1, dev)
        torch.cuda.synchronize(); sec.append(time.monotonic() - s)
    torch._dynamo.reset()
    return statistics.median(sec[4:])


def compare(pa, pb):
    (ma, ga), (mb, gb) = one_step(*pa), one_step(*pb)
    num = sum((gb[k] - ga[k]).square().sum() for k in ga); den = sum(ga[k].square().sum() for k in ga)
    worst = max((((gb[k]-ga[k]).norm()/ga[k].norm().clamp_min(1e-30)).item(), k) for k in ga)
    print(f'[1] {pa} vs {pb}: loss {ma["lm_loss"]:.6f} / {mb["lm_loss"]:.6f}  grad rel L2 {(num/den).sqrt().item():.2e}  worst {worst[0]:.2e} {worst[1]}')
print('amp', cfg['amp'], cfg.get('amp_dtype'))
for p, k, ck_ in (('float32','torch',True), ('bfloat16','torch',True), ('bfloat16','triton',True), ('float32','torch',False), ('bfloat16','triton',False)):
    torch.cuda.reset_peak_memory_stats()
    print(f'[3] compiled step {p}/{k} ckpt={ck_}: {timing(p, k, ckpt=ck_)*1000:.1f} ms  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
