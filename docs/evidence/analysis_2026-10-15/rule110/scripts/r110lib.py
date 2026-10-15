"""Shared pieces for the Rule 110 interpretability analysis (same model/protocol as train_r110.py)."""
import torch, torch.nn.functional as F
from lt.model import LT, ModelConfig
T, K_STEPS = 36, 4
dev = torch.device('cuda')
RULE = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0], device=dev)

def step110(x):
    l, r = F.pad(x, (1, 0))[:, :-1], F.pad(x, (0, 1))[:, 1:]
    return RULE[4 * l + 2 * x + r]

def evolve_all(x, steps=K_STEPS):
    out = [x]
    for _ in range(steps):
        out.append(step110(out[-1]))
    return out

def sample(n, gen):
    x = torch.randint(0, 2, (n, T), generator=gen, device=dev)
    return x, evolve_all(x)[-1]

def build(path):
    cfg = ModelConfig(batch_size=512, seq_len=T, grid=6, vocab_size=3, num_puzzle_identifiers=1, puzzle_emb_ndim=0,
                      hidden_size=32, num_heads=1, blocks_per_seg=8, loops=1, outer_eta=0.125, amp=False, kernel='torch')
    torch.manual_seed(0)
    model = LT(cfg).to(dev)
    model.pos_row = torch.arange(T, dtype=torch.float32, device=dev); model.pos_col = torch.zeros(T, device=dev)
    model.load_state_dict(torch.load(path, map_location=dev)); model.eval()
    return model, cfg

def run(model, x, blocks, each):
    """each(b, h, logits, box, state); box has 'ffn' [B,T,256] and 'qr','kr' [B,1,T,D] of block b."""
    cfg = model.config
    box = {}
    hk = model.block.ffn_down.register_forward_hook(lambda m, i, o: box.__setitem__('ffn', i[0]))
    model.probe = lambda rec: box.update(qr=rec.qr, kr=rec.kr) or None
    try:
        batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=dev))
        h = model.init_hidden.float().expand(x.shape[0], T, cfg.hidden_size).clone()
        state = model.initial_state(x.shape[0], dev); inj = model.injection(batch)
        for b in range(1, blocks + 1):
            h, state = model.block_step(h, inj, state)
            each(b, h, model.classifier(h), box, state)
    finally:
        hk.remove(); model.probe = None

def windows(x, r, lo=8, hi=28):
    """Window index (bit k = offset k - r) of radius r around cells lo..hi-1 -> [B, hi-lo] int64."""
    u = x.unfold(1, 2 * r + 1, 1)[:, lo - r:hi - r]                      # [B, C, 2r+1]
    return (u * (2 ** torch.arange(2 * r + 1, device=x.device))).sum(-1)

def truth9():
    """f(w) for every radius-4 window w (bit k = offset k - 4): the centre after 4 steps."""
    w = torch.arange(512, device=dev)
    bits = (w[:, None] >> torch.arange(9, device=dev)) & 1
    return evolve_all(bits)[-1][:, 4]

def hadamard(n=9):
    w = torch.arange(2 ** n, device=dev)
    pc = lambda a: sum(((a >> k) & 1) for k in range(n))
    return (1 - 2 * (pc(w[:, None] & w[None, :]) % 2)).float()      # [S, w]
