"""URM blocks (lt/urm_vendor, URM c14e55f) trained by the unmodified harness lt.train.

The harness is URM's own protocol (pretrain.py): one optimizer step per segment, the hidden state carried to the next
step, a lane takes a new puzzle after `loops` segments, stablemax CE, AdamATan2, SignSGD puzzle embedding, EMA. Only the
model is swapped: lt.train.LT is replaced by URMForHarness before the harness runs, so the training and evaluation
code is line for line the one the LT runs use.

One segment is the original URM_Inner.forward (H_cycles 1: blocks_per_seg cycles of h = h + inj, then the block), with
lane replacement as in URM.forward. Changes from the original model, all as in the LT runs: ConvSwiGLU -> SwiGLU (same
width as LT's FFN, checked against mlp_inter), the harness's PuzzleEmbedding in place of CastedSparseEmbedding (the same
sparse SignSGD table, zero init), no ACT. LT-only keys of the config (linear_attn, rope, tag_bound, ...) are unused.

    python -m toy_g.urm_harness --config configs/gtorus_id_dense50_urm_h64.json --out runs/gtorus_id_dense50_urm_h64
"""
import sys

import torch
from torch import nn

from lt.model import Carry, ModelConfig, PuzzleEmbedding
from lt.urm_vendor.layers import SwiGLU
from lt.urm_vendor.urm import URMCarry, URMConfig, URM_Inner


class _NoBlock(nn.Module):
    """The harness calls model.block.project_eta_ after every step (LT's outer-eta clamp); nothing to do here."""
    def project_eta_(self, cfg):
        pass


class URMForHarness(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.act:
            raise ValueError('URMForHarness runs a fixed number of segments (act off), as the LT runs')
        if cfg.grad_blocks and cfg.blocks_per_seg % cfg.grad_blocks:
            raise ValueError('blocks_per_seg must be a multiple of grad_blocks')
        self.config = cfg
        dtype = 'bfloat16' if cfg.amp and cfg.forward_dtype == 'float32' else cfg.forward_dtype
        self.urm = URM_Inner(URMConfig(
            batch_size=cfg.batch_size, seq_len=cfg.seq_len, puzzle_emb_ndim=cfg.puzzle_emb_ndim,
            num_puzzle_identifiers=cfg.num_puzzle_identifiers, vocab_size=cfg.vocab_size, num_layers=1,
            hidden_size=cfg.hidden_size, expansion=cfg.mlp_expansion, num_heads=cfg.num_heads, pos_encodings='rope',
            rope_theta=cfg.rope_base, loops=cfg.loops, forward_dtype=dtype,
            # grad_blocks g > 0: URM's truncated backprop, blocks_per_seg / g H-cycles of which only the last (g blocks)
            # runs with gradients; 0: one H-cycle of blocks_per_seg blocks, all with gradients
            L_cycles=cfg.grad_blocks or cfg.blocks_per_seg,
            H_cycles=cfg.blocks_per_seg // cfg.grad_blocks if cfg.grad_blocks else 1))
        for layer in self.urm.layers:                       # SwiGLU on the original projections (no convolution)
            plain = SwiGLU(cfg.hidden_size, cfg.mlp_expansion)
            plain.gate_up_proj, plain.down_proj = layer.mlp.gate_up_proj, layer.mlp.down_proj
            layer.mlp = plain
        inter = self.urm.layers[0].mlp.down_proj.weight.shape[1]
        if cfg.mlp_inter and cfg.mlp_inter != inter:
            raise ValueError(f'URM FFN width is {inter} but mlp_inter = {cfg.mlp_inter}')
        if cfg.puzzle_emb_ndim:
            self.urm.puzzle_emb = PuzzleEmbedding(cfg.num_puzzle_identifiers, cfg.puzzle_emb_ndim, cfg.batch_size,
                                                  self.urm.forward_dtype)
        self.block = _NoBlock()
        self.block_fn = None

    @property
    def puzzle_emb(self):
        return getattr(self.urm, 'puzzle_emb', None)

    def _amp(self, device):
        return self.urm.forward_dtype == torch.bfloat16 and device.type == 'cuda'

    def use_kernel(self, device, dtype):
        return False

    def use_band_kernel(self, device, dtype):
        return False

    def initial_carry(self, batch):
        B, device = batch['inputs'].shape[0], self.urm.init_hidden.device
        T = self.config.seq_len + self.urm.puzzle_emb_len
        return Carry(hidden=torch.empty(B, T, self.config.hidden_size, dtype=self.urm.forward_dtype, device=device),
                     steps=torch.zeros(B, dtype=torch.int32, device=device),
                     halted=torch.ones(B, dtype=torch.bool, device=device),
                     data={k: torch.empty_like(v, device=device) for k, v in batch.items()})

    def forward(self, carry: Carry, batch: dict):
        """One segment. Halted lanes take the puzzle in `batch` and start from init_hidden (URM.forward)."""
        fresh = carry.halted
        data = {k: torch.where(fresh.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                for k, v in carry.data.items()}
        inner = self.urm.reset_carry(fresh, URMCarry(current_hidden=carry.hidden))
        inner, logits, _ = self.urm(inner, data)
        steps = torch.where(fresh, 0, carry.steps) + 1
        halted = steps >= self.config.loops
        return Carry(hidden=inner.current_hidden, steps=steps, halted=halted, data=data), logits.float()


def main(argv=None):
    import lt.train as harness
    harness.LT = URMForHarness
    return harness.cli(argv)


if __name__ == '__main__':
    sys.exit(0 if main() is not None else 1)
