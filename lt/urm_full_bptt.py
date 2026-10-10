"""URM blocks in the shared Sudoku harness, one full recurrent graph per segment."""
from dataclasses import replace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from . import train as t
from .urm_vendor.urm import URMConfig, URM_Inner
from .urm_vendor.layers import SwiGLU


class URMFullBPTTInner(t.LT_Inner):
    def __init__(self, config):
        nn.Module.__init__(self)
        if config.loops < 1 or config.nograd_blocks:
            raise ValueError('Within-segment full BPTT requires positive loops and no no-grad blocks.')
        self.config=config
        upstream=URMConfig(batch_size=config.batch_size,seq_len=config.seq_len,
            puzzle_emb_ndim=config.puzzle_emb_ndim,
            num_puzzle_identifiers=config.num_puzzle_identifiers,vocab_size=config.vocab_size,
            num_layers=config.num_layers,hidden_size=config.hidden_size,
            expansion=config.mlp_expansion,num_heads=config.num_heads,
            pos_encodings='rope',loops=config.loops,L_cycles=config.blocks_per_seg,H_cycles=1,
            forward_dtype='bfloat16' if config.amp else 'float32')
        self.urm=URM_Inner(upstream)
        # Use the harness's identical sparse embedding implementation/optimizer.
        if config.puzzle_emb_ndim:
            self.urm.puzzle_emb=t.CastedSparseEmbedding(config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,batch_size=config.batch_size,init_std=0,
                cast_to=self.urm.forward_dtype)
        self.forward_dtype=self.urm.forward_dtype
        self.d=config.hidden_size

    @property
    def init_hidden(self):return self.urm.init_hidden

    @property
    def puzzle_emb(self):return getattr(self.urm,'puzzle_emb',None)

    @property
    def layers(self):return self.urm.layers

    def empty_carry(self,batch_size,device=None):
        return t.LTCarry(current_hidden=torch.empty(batch_size,
            self.config.seq_len+self.urm.puzzle_emb_len,self.d,
            device=self.init_hidden.device if device is None else device,
            dtype=self.forward_dtype))

    def _forward(self,carry,batch):
        if self.config.nograd_blocks:
            raise ValueError('No-grad recurrence is disabled for this baseline.')
        inj=self.urm._input_embeddings(batch['inputs'],batch['puzzle_identifiers'])
        cos_sin=self.urm.rotary_emb()
        h=carry.current_hidden.to(self.forward_dtype)
        for _ in range(self.config.blocks_per_seg):
            h=h+inj
            for layer in self.layers:
                if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                    h=checkpoint(layer,cos_sin,h,use_reentrant=False,preserve_rng_state=False)
                else:h=layer(cos_sin,h)
        logits=self.urm.lm_head(h)[:,self.urm.puzzle_emb_len:]
        # No detach within the eight internal iterations. Detach at segment
        # boundaries, preserving the shared harness update/sample-retention protocol.
        return replace(carry,current_hidden=h.detach(),coupling=None,trace=None,
                       key_trace=None,value_trace=None,fresh=None),logits.float()


class URMSwiGLUFullBPTTInner(URMFullBPTTInner):
    """Remove convolution and its extra SiLU; preserve projection initialization."""
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            original = layer.mlp
            # Preserve the original random stream and matched projection weights.
            with torch.random.fork_rng(devices=[]):
                plain = SwiGLU(config.hidden_size, config.mlp_expansion, mlp_dropout=0.0)
            plain.gate_up_proj = original.gate_up_proj
            plain.down_proj = original.down_proj
            layer.mlp = plain


def install(ffn="convswiglu", layers=2):
    if ffn not in ("convswiglu", "swiglu"):
        raise ValueError(f"Unknown URM FFN: {ffn}")
    t.LT_Inner = URMSwiGLUFullBPTTInner if ffn == "swiglu" else URMFullBPTTInner
    suffix = "-swiglu" if ffn == "swiglu" else ""
    t.model_id_of=lambda cfg:(f'urm-c14e55f-segment-bptt-{layers}layers-8iterations-loops'
                              + str(cfg.get('loops',16)) + '-v2' + suffix)
