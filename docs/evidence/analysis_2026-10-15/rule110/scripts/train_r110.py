"""Current model (lt/model.py, B in EMA form: W <- (1 - eta) W + eta Im S, eta 0.125) at hidden 32 on Rule 110.
Protocol of the archived 1x8 toy (docs/research/2026-10-14 rule110_leaky_train.py, adamatan2 lr 1e-3 wd 0.1, 10k):
36 cells, target = the row after 4 Rule 110 steps (zero boundary), tokens bit + 1, one head, 1D positions
(row index = cell, col = 0), one segment of 8 blocks, cross-entropy after every block, full backprop, grad clip 1.0,
fresh 512 random rows per step. Evaluation: 4,096 held-out rows (seed 7) iterated to 128 blocks.
"""
import json, sys, time
import torch, torch.nn.functional as F
from lt.model import LT, ModelConfig
from lt.train import AdamATan2

T, K_STEPS, BLOCKS, STEPS, ITER = 36, 4, 8, int(sys.argv[2]) if len(sys.argv) > 2 else 10000, 128
OUT = sys.argv[1]
dev = torch.device('cuda'); torch.set_float32_matmul_precision('highest')
RULE = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0], device=dev)

def evolve(x, steps):
    for _ in range(steps):
        l, r = F.pad(x, (1, 0))[:, :-1], F.pad(x, (0, 1))[:, 1:]
        x = RULE[4 * l + 2 * x + r]
    return x

def sample(n, gen):
    x = torch.randint(0, 2, (n, T), generator=gen, device=dev)
    return x, evolve(x, K_STEPS)

torch.manual_seed(0)
cfg = ModelConfig(batch_size=512, seq_len=T, grid=6, vocab_size=3, num_puzzle_identifiers=1, puzzle_emb_ndim=0,
                  hidden_size=32, num_heads=1, blocks_per_seg=BLOCKS, loops=1, outer_eta=0.125, amp=False, kernel='torch')
model = LT(cfg).to(dev)
model.pos_row = torch.arange(T, dtype=torch.float32, device=dev); model.pos_col = torch.zeros(T, device=dev)

def forward(x, blocks, stats=False):
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=dev))
    h = model.init_hidden.float().expand(x.shape[0], T, cfg.hidden_size).clone()
    state = model.initial_state(x.shape[0], dev); inj = model.injection(batch)
    logits, wn, drift = [], [], []
    for _ in range(blocks):
        h_new, state = model.block_step(h, inj, state)
        if stats:
            wn.append(float(state[3].norm(dim=(-1, -2)).mean())); drift.append(float((h_new - h).norm(dim=-1).mean()))
        h = h_new
        logits.append(model.classifier(h))
    return torch.stack(logits), wn, drift

named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
decay = lambda n, p: p.ndim >= 2 and 'rope_theta' not in n
optim = AdamATan2([dict(params=[p for n, p in named if not decay(n, p)], weight_decay=0.0),
                   dict(params=[p for n, p in named if decay(n, p)], weight_decay=0.1)], lr=1e-3)
gen = torch.Generator(device=dev).manual_seed(1000)
xt, yt = sample(4096, torch.Generator(device=dev).manual_seed(7))
print(f'params {sum(p.numel() for p in model.parameters())}', flush=True)
SHOW = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
def evaluate(step, loss, began):
    model.eval()
    with torch.no_grad():
        lg, wn, drift = forward(xt, ITER, stats=True)
    pred = lg.argmax(-1) - 1
    exact = (pred == yt).all(-1).float().mean(-1).tolist(); cell = (pred == yt).float().mean((-1, -2)).tolist()
    row = dict(step=step, loss=loss, seconds=round(time.monotonic() - began), alpha=float(model.block.alpha),
               exact=[round(v, 4) for v in exact], cell=[round(v, 4) for v in cell],
               w_norm=[round(v, 3) for v in wn], drift=[round(v, 4) for v in drift])
    print(f'[EVAL] step {step} loss {loss:.4f} alpha {row["alpha"]:.3f}  exact@' +
          ' '.join(f'{b}:{exact[b-1]:.4f}' for b in SHOW), flush=True)
    model.train()
    return row

log, began = [], time.monotonic()
for step in range(1, STEPS + 1):
    x, y = sample(512, gen)
    lg, _, _ = forward(x, BLOCKS)
    loss = F.cross_entropy(lg.reshape(-1, 3), (y + 1).repeat(BLOCKS, 1).reshape(-1))
    optim.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
    if step % 2000 == 0 or step == STEPS:
        log.append(evaluate(step, float(loss), began))
        json.dump(log, open(f'{OUT}/log.json', 'w'))
torch.save(model.state_dict(), f'{OUT}/model.pt')
r = log[-1]
print('\nblock   ' + ' '.join(f'{b:>7}' for b in SHOW))
for k in ('exact', 'cell', 'w_norm', 'drift'):
    print(f'{k:7s} ' + ' '.join(f'{r[k][b-1]:7.4f}' for b in SHOW))
