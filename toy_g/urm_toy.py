"""Quick URM baseline on the torus toy (our reproduction: lt/urm_vendor at URM c14e55f, ConvSwiGLU -> SwiGLU).

Softmax attention + 1D RoPE + two post-residual RMSNorms, input re-injection every block, 8 blocks x 2 segments
(detached between segments, both supervised), puzzle identifier as a prefix token (plain embedding trained by Adam,
or with --signsgd the harness's sparse SignSGD embedding). Quick and dirty: own loop, CE loss, EMA only with --ema.
NOT the URM training protocol (each step: a fresh batch, 2 segments, the two losses summed into one update).
--lt-config trains LT (memory mode and widths from a harness config, vocabulary 6) in this same loop; --struct D adds
the structure head and task code of toy_g.lt_struct (code dimension D) to it.

    python -m toy_g.urm_toy --data data/gtorus_id_dense50.npz --out runs/urm_dense50 [--steps 8000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lt.model import LT, ModelConfig, PuzzleEmbedding
from lt.train import EMA, SignSGDEmbedding
from lt.urm_vendor.layers import SwiGLU
from lt.urm_vendor.urm import URMConfig, URM_Inner
from .lt_rule_bands import LTRuleBands
from .lt_struct import LTStruct


class PuzzleEmb(nn.Module):
    def __init__(self, num, dim, dtype):
        super().__init__()
        self.emb, self.dtype = nn.Embedding(num, dim), dtype
        nn.init.zeros_(self.emb.weight)

    def forward(self, ids):
        return self.emb(ids.long()).to(self.dtype)


def build(seq_len, use_id, hidden=64, heads=2, blocks=8, num_ids=3072, signsgd=False, batch_size=128):
    cfg = URMConfig(batch_size=1, seq_len=seq_len, puzzle_emb_ndim=hidden if use_id else 0, num_puzzle_identifiers=num_ids,
                    vocab_size=6, num_layers=1, hidden_size=hidden, expansion=4.0, num_heads=heads, pos_encodings='rope',
                    loops=2, L_cycles=blocks, H_cycles=1, forward_dtype='bfloat16')
    m = URM_Inner(cfg)
    for layer in m.layers:                                   # the reproduction's swap: no convolution
        plain = SwiGLU(hidden, 4.0)
        plain.gate_up_proj, plain.down_proj = layer.mlp.gate_up_proj, layer.mlp.down_proj
        layer.mlp = plain
    if use_id:                                               # the harness's sparse embedding (SignSGD on used rows)
        m.puzzle_emb = (PuzzleEmbedding(num_ids, hidden, batch_size, m.forward_dtype) if signsgd
                        else PuzzleEmb(num_ids, hidden, m.forward_dtype))
    return m


def segments(m, x, ids, nseg=2):
    inj = m._input_embeddings(x, ids)
    cos_sin = m.rotary_emb()
    h = m.init_hidden.expand(x.shape[0], inj.shape[1], -1).to(inj.dtype)
    outs = []
    for _ in range(nseg):
        for _ in range(m.config.L_cycles):
            h = h + inj
            for layer in m.layers:
                h = layer(cos_sin=cos_sin, hidden_states=h)
        outs.append(m.lm_head(h)[:, m.puzzle_emb_len:].float())
        h = h.detach()
    return outs


def build_lt(path, seq_len, batch_size, struct=0, rule_bands=False):
    """LT from a harness config (memory mode, widths, RoPE); vocabulary 6 as the URM model here."""
    cfg = ModelConfig.from_dict(dict(json.loads(Path(path).read_text()), batch_size=batch_size, seq_len=seq_len, vocab_size=6))
    m = LTStruct(cfg, d_g=struct) if struct else LTRuleBands(cfg) if rule_bands else LT(cfg)
    if m.config.loops != 2:
        raise ValueError('this loop runs 2 segments per step; the LT config must have loops = 2')
    return m


def segments_lt(m, x, ids, nseg=2):
    """The same schedule through LT.forward: a fresh carry, nseg segments (LT.forward detaches the carry)."""
    batch = dict(inputs=x, puzzle_identifiers=torch.zeros(len(x), dtype=torch.long, device=x.device) if ids is None else ids)
    carry = m.initial_carry(batch)
    outs = []
    for _ in range(nseg):
        carry, logits = m(carry, batch)
        outs.append(logits.float())
    return outs


@torch.no_grad()
def evaluate(m, x, y, ids, group, bs=512, seg=segments):
    m.eval()
    preds = []
    for s in range(0, len(x), bs):
        preds.append(seg(m, x[s:s + bs], None if ids is None else ids[s:s + bs])[-1].argmax(-1))
    m.train()
    pred = torch.cat(preds)
    keep = y > 0
    exact = ((pred == y) | ~keep).all(1).cpu().numpy()
    cell = (((pred == y) & keep).sum(1).float() / keep.sum(1)).cpu().numpy()
    return {g: dict(n=int((group == g).sum()), exact=float(exact[group == g].mean()), cell=float(cell[group == g].mean()))
            for g in range(3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--signsgd', action='store_true', help='puzzle embedding as in the harness: SignSGD on the used rows')
    ap.add_argument('--emb-lr', type=float, default=1e-3)
    ap.add_argument('--ema', type=float, default=0.0, help='EMA rate of the body (and the SignSGD table) used at evaluation')
    ap.add_argument('--lt-config', default=None, help='train LT (harness config) in this loop instead of URM')
    ap.add_argument('--struct', type=int, default=0, help='with --lt-config: add the structure head (code dimension)')
    ap.add_argument('--rule-bands', action='store_true', help='with --lt-config: LT-bands with role/cell codes and a causal window')
    opt = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(0)
    z = np.load(opt.data)
    use_id = 'train_task' in z.files
    tx = torch.as_tensor(z['train_inputs'].reshape(len(z['train_inputs']), -1).astype(np.int64), device=dev)
    ty = torch.as_tensor(z['train_labels'].reshape(len(z['train_labels']), -1).astype(np.int64), device=dev)
    ex = torch.as_tensor(z['test_inputs'].reshape(len(z['test_inputs']), -1).astype(np.int64), device=dev)
    ey = torch.as_tensor(z['test_labels'].reshape(len(z['test_labels']), -1).astype(np.int64), device=dev)
    tid = torch.as_tensor(z['train_task'].astype(np.int64), device=dev) if use_id else None
    eid = torch.as_tensor(z['test_task'].astype(np.int64), device=dev) if use_id else None
    group = z['test_group']
    with torch.device(dev):                                   # built in place: .to() would make the local rows non-leaf
        m = build_lt(opt.lt_config, tx.shape[1], opt.bs, opt.struct, opt.rule_bands) if opt.lt_config else build(tx.shape[1], use_id, signsgd=opt.signsgd, batch_size=opt.bs)
    seg = segments_lt if opt.lt_config else segments
    params = m.parameters()                                   # URM: every trainable parameter is a matrix
    if opt.lt_config:
        if use_id and not opt.signsgd:
            raise ValueError("LT's puzzle embedding is the sparse table: pass --signsgd")
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        decay = lambda n, p: (p.ndim >= 2 and 'rope_theta' not in n and 'event_theta_raw' not in n and 'band_' not in n
                              and 'task_film' not in n and 'task_u' not in n and 'task_v' not in n   # lt.train's rule
                              and 'struct.' not in n)                                 # + no decay on the generators
        params = [{'params': [p for n, p in named if not decay(n, p)], 'weight_decay': 0.0},
                  {'params': [p for n, p in named if decay(n, p)], 'weight_decay': 0.1}]
    o = torch.optim.AdamW(params, lr=opt.lr, weight_decay=0.1, betas=(0.9, 0.95))   # buffers excluded
    emb_opt = SignSGDEmbedding(m.puzzle_emb, 1, lr=opt.emb_lr, weight_decay=0.1) if (use_id and opt.signsgd) else None
    ema = EMA(m, opt.ema, with_puzzle_emb=emb_opt is not None) if opt.ema else None
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'a')
    print(f'{"LT " + opt.lt_config if opt.lt_config else "URM"}{f" + struct {opt.struct}" if opt.struct else ""} toy: {opt.data} train {len(tx)} test {len(ex)} id {use_id} signsgd {emb_opt is not None} ema {opt.ema} '
          f'params {sum(p.numel() for p in m.parameters()):,}', flush=True)
    t0 = time.time()
    for step in range(1, opt.steps + 1):
        warm = min(1.0, step / 500)
        for g in o.param_groups:
            g['lr'] = opt.lr * warm
        i = torch.randint(0, len(tx), (opt.bs,), device=dev)
        outs = seg(m, tx[i], None if tid is None else tid[i])
        loss = sum(F.cross_entropy(lo.reshape(-1, 6), ty[i].reshape(-1), ignore_index=0) for lo in outs)
        o.zero_grad(set_to_none=True)
        if emb_opt is not None:
            emb_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        if emb_opt is not None:
            emb_opt.param_groups[0]['lr'] = opt.emb_lr * warm
            emb_opt.step()
        if opt.lt_config:
            m.block.project_eta_(m.config)                    # as lt.train after every step
        if ema is not None:
            ema.update(m)
        if step % 250 == 0:
            with torch.no_grad():
                keep = ty[i] > 0
                tr_exact = float(((outs[-1].argmax(-1) == ty[i]) | ~keep).all(1).float().mean())
            print(f'step {step} loss {loss.item() / 2:.4f} train-batch exact {tr_exact:.3f} {time.time() - t0:.0f}s', flush=True)
        if step % opt.eval_every == 0 or step == opt.steps:
            if ema is None:
                r = evaluate(m, ex, ey, eid, group, seg=seg)
            else:
                with ema.swap(m):
                    r = evaluate(m, ex, ey, eid, group, seg=seg)
            log.write(json.dumps(dict(step=step, eval=r)) + '\n'); log.flush()
            print(f'[EVAL] step {step} ' + ' | '.join(f"g{g} exact {100 * v['exact']:.1f}% cell {100 * v['cell']:.1f}%" for g, v in r.items()), flush=True)


if __name__ == '__main__':
    main()
