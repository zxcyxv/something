"""LT (linear attention) + a structure head, the toy's structure code inserted as one extra attention path per block.

Structure head (toy_g.model, learned mode: codes G_(r,c) = W_down^r W_right^c g0, operators U_t = products of the four
learned orthogonal generators along the grammar's word for t):
    A_t[o, x] = cos(U_t G_o, G_x)          output cell o reads source cell x (U_t G_o ~ G_SRC[t, o])
    A = sum_t pi_t A_t                     pi = the task's distribution over the 128 transformations
    read_o = sum_x A[o, x] v_x,  v = W_sv u   (linear, no softmax; the prefix token has no cell code: zero row/column)
    h' = RMSNorm(u + W_O read_heads + W_so read)      (W_so starts at zero)
Task code in place of the free puzzle embedding: per task, logits over the 128 transformations and the 24 colour
permutations (one sparse table, SignSGD like the puzzle embedding); the prefix token is sum_c pi_c E_c (E: 24 learned
colour-permutation vectors shared by all tasks).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from lt.model import LT, PuzzleEmbedding, rms_norm
from . import group as G
from .model import StructureModel


def structure_scores(struct):
    """[128, 16(o), 16(x)] cosines between the address U_t G_o and every cell code G_x."""
    W = struct.gen_matrices()
    codes = struct.codes(W)
    U = struct.operators(W)
    addr = F.normalize(torch.einsum('tde,oe->tod', U, codes), dim=-1)
    return torch.einsum('tod,xd->tox', addr, F.normalize(codes, dim=-1))


class LTStruct(LT):
    def __init__(self, cfg, d_g=32):
        super().__init__(cfg)
        if not cfg.linear_attn or cfg.puzzle_emb_mode != 'prefix' or self.prefix_len != 1 or cfg.seq_len != G.CELLS:
            raise ValueError('LTStruct: linear attention, a one-token prefix and the 16-cell torus')
        d = cfg.hidden_size
        with torch.device('cpu'):                     # StructureModel draws its init from a CPU generator
            struct = StructureModel(mode='learned', d=d_g, generators='orthogonal')
        self.struct = struct.to(self.init_hidden.device)
        self.ncmap = len(G.CMAPS)
        self.puzzle_emb = PuzzleEmbedding(cfg.num_puzzle_identifiers, G.NUM_T + self.ncmap, cfg.batch_size, torch.float32)
        self.colour_emb = nn.Parameter(torch.randn(self.ncmap, d) / math.sqrt(d))
        self.s_v = nn.Linear(d, d, bias=False)
        self.s_out = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.s_out.weight.zero_()

    def task_code(self, ids):
        logits = self.puzzle_emb(ids).float()
        return logits[:, :G.NUM_T].softmax(-1), logits[:, G.NUM_T:].softmax(-1)

    def injection(self, batch):
        inj = self.embed(batch['inputs'].long())
        pi_t, pi_c = self.task_code(batch['puzzle_identifiers'])
        with torch.autocast(device_type=inj.device.type, enabled=False):
            A = torch.einsum('bt,tox->box', pi_t, structure_scores(self.struct))
        self.struct_A = F.pad(A, (self.prefix_len, 0, self.prefix_len, 0))         # [B, T, T]
        pe = (pi_c @ self.colour_emb).to(inj.dtype)
        return torch.cat([pe[:, None], inj], 1)

    def block_step(self, h, inj, state):
        """LT.block_step (linear-attention path) plus the structure head's read."""
        L = self.block
        u = h + self.embed_scale * inj
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.dh).transpose(1, 2)
        q, k, v = heads(L.q_proj(u)), heads(L.k_proj(u)), heads(L.v_proj(u))
        extended = len(state) > 5
        state, fast_w = state[:5], (state[5] if extended else None)
        read, state = self.memory_step(q, k, v, state)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(u.dtype)
        sv = self.s_v(u)
        sread = torch.einsum('box,bxd->bod', self.struct_A.to(sv.dtype), sv)
        h = rms_norm(u + L.out_proj(read) + self.s_out(sread))
        g, a = L.ffn_gate_up(h).chunk(2, dim=-1)
        out = L.ffn_down(F.silu(g) * a)
        return rms_norm(h + out), ((*state, fast_w) if extended else state)
