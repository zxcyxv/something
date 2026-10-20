"""LT-bands with the two architectural conditions of the rule-synapse theory, for the in-context toy (7 grids:
demo input/output x 3, then the query input, 16 cells each; no task identifier).

- Grid-role and cell codes: learned embeddings of which grid (7) and which cell (16) a token is, added to the
  re-injected input (inj = E[x] + E_role + E_cell), so that addresses exist apart from contents.
- Causal window: the bands coefficients of value-before-key pairs (b_r) are fixed at zero (an odd STDP window erases
  self-inverse rules; theory doc proposition 4). The same-block weight l(0) stays learned.
Everything else (which tokens write, the block schedule, the read) is left to training.

    python -m toy_g.make_ctx_fresh                 # data/gtorus_ctx_1m.npz (fresh determined samples, k = 3)
    python -m toy_g.urm_toy --data data/gtorus_ctx_1m.npz --lt-config configs/gtorus_ctx_bands_h64.json --rule-bands ...
"""
import math

import torch
from torch import nn

from lt.model import LT, trunc_normal_


class LTRuleBands(LT):
    def __init__(self, cfg, grid_cells=16):
        super().__init__(cfg)
        if not cfg.bands or self.prefix_len or cfg.seq_len % grid_cells:
            raise ValueError('LTRuleBands: bands memory, no prefix token, whole grids')
        d = cfg.hidden_size
        self.grid_cells, self.n_grids = grid_cells, cfg.seq_len // grid_cells
        self.role_emb = nn.Parameter(trunc_normal_(torch.empty(self.n_grids, d), std=1.0 / math.sqrt(d)))
        self.cell_emb = nn.Parameter(trunc_normal_(torch.empty(grid_cells, d), std=1.0 / math.sqrt(d)))
        with torch.no_grad():
            self.block.band_coef[2:].zero_()                                     # (Re b, Im b) = 0
        self.block.band_coef.register_hook(lambda g: torch.cat([g[:2], torch.zeros_like(g[2:])]))

    def injection(self, batch):
        inj = super().injection(batch)
        pos = torch.arange(inj.shape[1], device=inj.device)
        code = self.role_emb[pos // self.grid_cells] + self.cell_emb[pos % self.grid_cells]
        return inj + code.to(inj.dtype)[None]
