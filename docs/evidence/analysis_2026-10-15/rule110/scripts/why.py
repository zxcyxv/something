"""Why does the Rule 110 model collapse past block 8?
(1) do late predictions move toward the row after 8 / 12 steps (truth table applied to its own output)?
(2) W keeps growing (EMA from zero reaches 1 - (7/8)^8 = 66% at block 8): freeze W after block B0, or rescale it."""
import sys, torch
sys.path.insert(0, sys.argv[1]); from r110lib import *
torch.set_grad_enabled(False)
model, cfg = build(f'{sys.argv[1]}/model.pt'); eta0 = cfg.outer_eta
x, y = sample(8192, torch.Generator(device=dev).manual_seed(7))
st = evolve_all(x, 16); SHOW = [8, 12, 16, 24, 32, 64, 128]
def go(mode, B0=8):
    acc, near, wn = {}, {}, {}
    box = {}
    def each(b, h, logits, _, state):
        p = logits.argmax(-1) - 1
        if b in SHOW:
            acc[b] = float((p == y).all(-1).float().mean())
            near[b] = {k: float((p == st[k]).float().mean()) for k in (0, 4, 8, 12, 16)}
            wn[b] = float(state[3].norm(dim=(-1, -2)).mean())
    # schedule eta per block through the config (the model reads cfg.outer_eta each block)
    orig = model.block_step
    cnt = [0]
    def step(h, inj, state):
        cnt[0] += 1; b = cnt[0]
        if mode == 'freeze' and b > B0: cfg.outer_eta = 1e-9
        elif mode == 'bias' :            # bias-corrected EMA: W_n = EMA_n / (1 - (1-eta)^n), the scale the model saw at block B0 kept
            pass
        h, state = orig(h, inj, state); cfg.outer_eta = eta0
        if mode == 'clamp' and b > B0:   # keep ||W|| at its block-B0 value (direction free)
            w = state[3]; s = box.setdefault('n', None)
            if b == B0 + 1 and s is None: pass
            state = (*state[:3], w * (box['ref'] / w.norm(dim=(-1, -2), keepdim=True).clamp_min(1e-12)))
        if mode == 'clamp' and b == B0:
            box['ref'] = state[3].norm(dim=(-1, -2), keepdim=True)
        return h, state
    model.block_step = step
    try: run(model, x, 128, each)
    finally: model.block_step = orig; cfg.outer_eta = eta0
    return acc, near, wn
for mode in ('base', 'freeze', 'clamp'):
    acc, near, wn = go(mode)
    print(f'\n[{mode}] exact rows / |W|')
    print('  block ' + ' '.join(f'{b:>7}' for b in SHOW))
    print('  exact ' + ' '.join(f'{acc[b]*100:6.1f}%' for b in SHOW))
    print('  |W|   ' + ' '.join(f'{wn[b]:7.3f}' for b in SHOW))
    if mode == 'base':
        for k in (0, 4, 8, 12, 16):
            print(f'  cell agreement with row after {k:2d} steps ' + ' '.join(f'{near[b][k]*100:6.1f}%' for b in SHOW))
