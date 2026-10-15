"""Does the training gradient of the EMA rate eta see the cross-segment value of a long memory?
Partial (as training computes it): dL_seg/d eta with the carried state detached at segment boundaries.
Total (finite differences): change of the summed 16-segment loss when eta is changed for the whole run.
Raw (training) weights, 32 training puzzles without augmentation, 16 segments.  usage: etagrad.py <ckpt>"""
import sys, torch
from lt.analysis.common import load_model
from lt.train import segment_loss
from lt.data import load_data, eval_batches
dev = torch.device('cuda'); torch.set_float32_matmul_precision('highest')
model, cfg = load_model(sys.argv[1], dev, weights='raw')
x, y, _, _, _, _ = load_data(None, 2048)
batch = next(eval_batches(x[:32], y[:32], 32)); b = {k: v.to(dev) for k, v in batch.items()}
eta0 = float(cfg['outer_eta'])

def run(eta_value, grad=False):
    eta = torch.tensor(eta_value, device=dev, dtype=torch.float32, requires_grad=grad)
    model.config.outer_eta = eta
    carry = model.initial_carry(b); losses, grads, exact = [], [], []
    for seg in range(16):
        with torch.set_grad_enabled(grad):
            carry, logits = model(carry, b)
            loss, _ = segment_loss(logits, b['labels'], carry.halted)
            if grad:
                eta.grad = None; loss.backward(); grads.append(float(eta.grad))
        losses.append(float(loss)); exact.append(int((logits.argmax(-1) == b['labels']).all(-1).sum()))
    model.config.outer_eta = eta0
    return losses, grads, exact

lp, gp, ep = run(eta0, grad=True)
print(f'ckpt {sys.argv[1]} eta {eta0}: per-segment partial dL/deta (training gradient, carry detached):')
print('  segs 1-4  :', ' '.join(f'{g:+.3f}' for g in gp[:4]))
print('  segs 5-8  :', ' '.join(f'{g:+.3f}' for g in gp[4:8]))
print('  segs 9-16 :', ' '.join(f'{g:+.3f}' for g in gp[8:]))
print(f'  mean {sum(gp)/16:+.4f}  -> gradient descent moves eta {"UP (shorter memory)" if sum(gp) < 0 else "DOWN (longer memory)"}')
res = {}
for e in (eta0 * 0.5, eta0 * 0.8, eta0, eta0 * 1.25, eta0 * 2):
    l, _, ex = run(e); res[e] = (l, ex)
    print(f'eta {e:.4f} (window {1/e:5.1f}): total loss {sum(l):8.3f} | loss segs 1-4 {sum(l[:4]):7.3f}, 5-8 {sum(l[4:8]):7.3f}, 9-16 {sum(l[8:]):7.3f} | exact seg16 {ex[-1]}')
l_lo, l_hi = res[eta0 * 0.8][0], res[eta0 * 1.25][0]
fd = (sum(l_hi) - sum(l_lo)) / (eta0 * 1.25 - eta0 * 0.8)
print(f'finite-difference total dL/deta (0.8x..1.25x): {fd:+.3f} -> a longer memory {"LOWERS" if fd > 0 else "RAISES"} the 16-segment loss')
