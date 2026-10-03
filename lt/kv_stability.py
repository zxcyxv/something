"""Isolated experimental controls for the October 2 KV model.

The real current-only control removes memory and traces. The complex current-only
control preserves the original complex activity/trace product but removes matrix
accumulation. The other controls retain accumulated pair-STDP writes.
Select explicitly in the research runner; the original trainer stays unchanged.
"""
import math
import torch

from . import train as t

ORIGINAL_INNER = t.KVSTDPInner
ORIGINAL_MODEL_ID = t.model_id_of
VARIANTS = ("original", "pre_ffn_phi", "read_operator_bound", "unit_qk_activity",
            "interpolated_read", "current_only", "complex_current_only",
            "current_plus_stdp", "historical_key_trace", "read_gain_quarter")


class PreFFNPhiInner(ORIGINAL_INNER):
    def boundary(self, layer, hidden):
        # The bilinear input has ||h|| < sqrt(d), even when M's read is large.
        return super().boundary(layer, self.phi(hidden))


class BoundedReadInner(ORIGINAL_INNER):
    def memory_step(self, *args, **kwargs):
        read, memory, ek, ev = super().memory_step(*args, **kwargs)
        # Bound only the operator used to read. Stored M still equals sum G.
        cap_squared = 16 * self.dh
        scale = torch.sqrt(1 + memory.square().sum((-1, -2), keepdim=True) / cap_squared)
        return read / scale, memory, ek, ev


class QuarterReadInner(ORIGINAL_INNER):
    """Diagnostic gain control: preserve STDP and multiply only its read by 0.25."""
    def memory_step(self, *args, **kwargs):
        read, memory, ek, ev = super().memory_step(*args, **kwargs)
        return 0.25*read, memory, ek, ev


class UnitQKActivityInner(ORIGINAL_INNER):
    """Use bounded address activities before forming their temporal trace.

K traces contain past unit K activities, not normalized traces of raw K.
This retains the exponential pair-STDP window in the chosen activity space.
The first controlled training run retains the original token MEAN, so the only
change is Q/K activity normalization. This also changes the initial signal gain;
an additional gain control would be needed to distinguish gain from adaptive
normalization. V, the temporal subtraction, and additive M remain intact.
"""
    def __init__(self,config):
        if config.kv_qk_rmsnorm or config.kv_qk_l2norm:
            raise ValueError('Activity normalization cannot be combined with write-copy/trace normalization.')
        super().__init__(config)

    def memory_step(self, layer, q, k, v, *args, **kwargs):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k = q.to(dtype), k.to(dtype)
            q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + self.config.eps)
            k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.config.eps)
        return super().memory_step(layer, q, k, v, *args, **kwargs)


class InterpolatedReadInner(ORIGINAL_INNER):
    """Mix current KV and accumulated STDP reads with one scalar per head.

The fresh KV product is used only for the current read, never stored in M.
Both paths retain the original token reduction, raw Q/K/V and spatial rotation.
All original random draws finish before constant-initialized gates are added.
"""
    def __init__(self,config):
        if config.kv_qk_rmsnorm or config.kv_qk_l2norm:
            raise ValueError('This read-only control requires raw Q/K activities.')
        super().__init__(config)
        for layer in self.layers:
            layer.read_lam_raw=torch.nn.Parameter(torch.full(
                (self.H,1,1),math.log(.25/.75)))

    def memory_step(self,layer,q,k,v,*args,**kwargs):
        accumulated,memory,ek,ev=super().memory_step(layer,q,k,v,*args,**kwargs)
        with torch.autocast(device_type=q.device.type,enabled=False):
            dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            q,k,v=(x.to(dtype) for x in (q,k,v))
            tables=self.rope_tables(layer)
            qr,kr=(self.apply_rope(x,layer,tables) for x in (q,k))
            current=v.transpose(-1,-2)@kr
            if self.config.kv_write_reduction=='mean':
                current=current/k.shape[-2]
            instantaneous=qr@current.transpose(-1,-2)
            lam=torch.sigmoid(layer.read_lam_raw).to(dtype)[None]
            read=(1-lam)*instantaneous+lam*accumulated
        return read,memory,ek,ev


class CurrentReadInner(ORIGINAL_INNER):
    """Recompute signed linear attention at every recurrent block.

No previous matrix or eligibility trace enters the read or gets updated.
The returned matrix is the current KV product, retained only for compatibility
with the original carry/diagnostics interface and ignored by the next block.
Unused trace parameters remain allocated to preserve original initialization.
"""
    def __init__(self,config):
        if config.kv_qk_rmsnorm or config.kv_qk_l2norm:
            raise ValueError('This current-only control requires raw Q/K activities.')
        super().__init__(config)
        self.stdp=self.use_trace=False

    def memory_step(self,layer,q,k,v,memory=None,e_k=None,e_v=None,fresh=None):
        with torch.autocast(device_type=q.device.type,enabled=False):
            dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            q,k,v=(x.to(dtype) for x in (q,k,v))
            tables=self.rope_tables(layer)
            qr,kr=(self.apply_rope(x,layer,tables) for x in (q,k))
            current=v.transpose(-1,-2)@kr
            if self.config.kv_write_reduction=='mean':
                current=current/k.shape[-2]
            read=qr@current.transpose(-1,-2)
        return read,current,None,None


class ComplexCurrentReadInner(ORIGINAL_INNER):
    """Read the current complex KV operator with the original traces intact.

K_c = RoPE(e_K) + i RoPE(K), V_c = e_V + i V.
The original memory_step computes G = Im(V_c.T @ conj(K_c)) / T,
reads RoPE(Q) @ G.T, then updates the original eligibility traces.
Only M_new = M_old + G is replaced by M_new = G; no interpolation is added.
The carry matrix records G for diagnostics and never contributes to the next G.
"""
    def update_memory(self,memory,write):
        return write


class CurrentPlusSTDPInner(ComplexCurrentReadInner):
    """Read B + G, keeping original traces and discarding matrix accumulation.

B = V.T @ RoPE(K) / T is the current real KV term.
G = (V.T @ RoPE(e_K_previous) - e_V_previous.T @ RoPE(K)) / T.
Both coefficients are exactly one. There is no read gate or past-past term.
"""
    def __init__(self,config):
        if config.kv_qk_rmsnorm or config.kv_qk_l2norm:
            raise ValueError('This B + G control requires raw Q/K activities.')
        super().__init__(config)

    def memory_step(self,layer,q,k,v,memory=None,e_k=None,e_v=None,fresh=None):
        _,difference,ek,ev=super().memory_step(layer,q,k,v,memory,e_k,e_v,fresh)
        with torch.autocast(device_type=q.device.type,enabled=False):
            dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            q,k,v=(x.to(dtype) for x in (q,k,v))
            tables=self.rope_tables(layer)
            qr,kr=(self.apply_rope(x,layer,tables) for x in (q,k))
            current=v.transpose(-1,-2)@kr
            if self.config.kv_write_reduction=='mean':
                current=current/k.shape[-2]
            operator=current+difference
            read=qr@operator.transpose(-1,-2)
        return read,operator,ek,ev


class HistoricalKeyTraceInner(ORIGINAL_INNER):
    """Store K after its event-time RoPE, retaining the original pair-STDP rule.

    eK_new = lambda*eK_old + (1-lambda)*RoPE_theta_now(K_now).
    G = (V_now.T @ eK_old - eV_old.T @ RoPE_theta_now(K_now)) / T.
    Original initialization, writes, reads and V traces are otherwise retained.
    At fixed theta, this is a change of trace coordinates; optimizer updates and
    truncation boundaries distinguish the historical from the current-theta
    interpretation. Old raw-key carry checkpoints cannot be resumed as-is.
    """
    def __init__(self, config):
        if config.kv_qk_rmsnorm or config.kv_qk_l2norm or config.kv_trace_activity_detach:
            raise ValueError('Historical-key control requires raw differentiable activities.')
        if config.num_layers != 1:
            raise ValueError('Historical-key control requires one projection/rotation basis.')
        super().__init__(config)

    def memory_step(self, layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            e_k = torch.zeros_like(k) if e_k is None else e_k.to(dtype)
            e_v = torch.zeros_like(v) if e_v is None else e_v.to(dtype)
            memory = (v.new_zeros(v.shape[0], self.H, v.shape[-1], k.shape[-1])
                      if memory is None else memory.to(dtype))
            if fresh is not None:
                mask = fresh[:, None, None, None]
                memory = torch.where(mask, torch.zeros_like(memory), memory)
                e_k = torch.where(mask, torch.zeros_like(e_k), e_k)
                e_v = torch.where(mask, torch.zeros_like(e_v), e_v)
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            write = v.transpose(-1, -2) @ e_k - e_v.transpose(-1, -2) @ kr
            if self.config.kv_write_reduction == 'mean':
                write = write/k.shape[-2]
            memory = self.update_memory(memory, write)
            read = qr @ memory.transpose(-1, -2)
            lam = layer.trace_decay_channels.to(dtype)[None, :, None, :]
            e_k = lam*e_k + (1-lam)*kr
            e_v = lam*e_v + (1-lam)*v
        return read, memory, e_k, e_v


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown research variant {variant}")
    t.KVSTDPInner = {"original": ORIGINAL_INNER,
                    "pre_ffn_phi": PreFFNPhiInner,
                    "read_operator_bound": BoundedReadInner,
                    "unit_qk_activity": UnitQKActivityInner,
                    "interpolated_read": InterpolatedReadInner,
                    "current_only": CurrentReadInner,
                    "complex_current_only": ComplexCurrentReadInner,
                    "current_plus_stdp": CurrentPlusSTDPInner,
                    "historical_key_trace": HistoricalKeyTraceInner,
                    "read_gain_quarter": QuarterReadInner}[variant]
    t.model_id_of = lambda cfg: ORIGINAL_MODEL_ID(cfg) + (
        f":research-{variant}" if variant != "original" else "")
