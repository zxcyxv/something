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
            "interpolated_read", "current_only", "current_only_urm_norm", "phase_current_only", "phase_exp_current_only", "phase_puzzle_exp_current_only", "phase_channel_exp_current_only", "phase_channel_gram_exp_current_only", "phase_local_warp_exp_current_only", "phase_unit_gaussian_current_only", "complex_current_only",
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
Optional Q/K L2 normalization uses the v1.7 denominator norm + eps, per token
and head, before spatial rotation. V, output projection and token reduction stay
unchanged. This is selected explicitly with kv_qk_l2norm, which is resume-checked.
"""
    def __init__(self,config):
        if config.kv_qk_rmsnorm:
            raise ValueError('This current-only control supports raw or v1.7 L2 Q/K, not RMS normalization.')
        super().__init__(config)
        self.stdp=self.use_trace=False

    def memory_step(self,layer,q,k,v,memory=None,e_k=None,e_v=None,fresh=None):
        with torch.autocast(device_type=q.device.type,enabled=False):
            dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            q,k,v=(x.to(dtype) for x in (q,k,v))
            if self.config.kv_qk_l2norm:
                q=q/(torch.linalg.vector_norm(q,dim=-1,keepdim=True)+self.config.eps)
                k=k/(torch.linalg.vector_norm(k,dim=-1,keepdim=True)+self.config.eps)
            tables=self.rope_tables(layer)
            qr,kr=(self.apply_rope(x,layer,tables) for x in (q,k))
            current=v.transpose(-1,-2)@kr
            if self.config.kv_write_reduction=='mean':
                current=current/k.shape[-2]
            read=qr@current.transpose(-1,-2)
        return read,current,None,None


class URMNormCurrentReadInner(CurrentReadInner):
    """Raw current KV with URM's two affine-free post-residual RMSNorms.

    Replace phi entirely; normalize the attention residual before the existing
    bilinear FFN, then normalize its residual. URM epsilon is 1e-5, independent
    of the historical address epsilon. Projections and FFN stay unchanged.
    """
    norm_eps = 1e-5

    def __init__(self, config):
        if config.kv_qk_l2norm or config.kv_qk_rmsnorm:
            raise ValueError('URM block normalization control requires raw Q/K.')
        super().__init__(config)

    def phi(self, hidden):
        dtype = hidden.dtype
        hidden = hidden.float()
        return (hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True)
                                     + self.norm_eps)).to(dtype)

    def boundary(self, layer, hidden):
        return super().boundary(layer, self.phi(hidden))


class PhaseCurrentReadInner(URMNormCurrentReadInner):
    """Current Im(V_complex K_complex^H), read with real Q; no history.

    Phases belong to channels after spatial RoPE, are shared over tokens and
    recurrence, and are learned independently for K and V. A common carrier
    cancels algebraically. Signed activities remain ordinary real projections.
    The sine window is not an exponential STDP window.
    """
    phase_limit = math.pi / 2
    phase_init_limit = math.pi / 4

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            for name in ("theta_k_raw", "theta_v_raw"):
                angle = torch.empty(self.H, self.dh).uniform_(
                    -self.phase_init_limit, self.phase_init_limit)
                layer.register_parameter(name, torch.nn.Parameter(
                    torch.atanh(angle / self.phase_limit)))

    def phases(self, layer):
        return tuple(self.phase_limit * x.tanh()
                     for x in (layer.theta_k_raw, layer.theta_v_raw))

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            pk, pv = (x.to(dtype)[None, :, None, :]
                      for x in self.phases(layer))
            current = ((v * pv.sin()).transpose(-1, -2) @ (kr * pk.cos())
                       - (v * pv.cos()).transpose(-1, -2) @ (kr * pk.sin()))
            if self.config.kv_write_reduction == "mean":
                current = current / k.shape[-2]
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None


class ExponentialPhaseCurrentReadInner(PhaseCurrentReadInner):
    """Exact signed exponential window on fixed channel phase offsets.

    tau is measured in radians (omega*tau_time), not recurrent steps.
    Zero lag contributes zero. sign has zero derivative; ordinary gradients
    train the exponential envelope away from zero, with no surrogate gradient.
    """
    phase_tau = 1.0

    def phase_window(self, layer, dtype):
        pk, pv = (x.to(dtype) for x in self.phases(layer))
        delta = pv[:, :, None] - pk[:, None, :]
        return delta.sign() * torch.exp(-delta.abs() / self.phase_tau)

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            current = v.transpose(-1, -2) @ kr
            if self.config.kv_write_reduction == "mean":
                current = current / k.shape[-2]
            current = current * self.phase_window(layer, dtype)[None]
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None


class PuzzleStateExponentialPhaseCurrentReadInner(ExponentialPhaseCurrentReadInner):
    """Exact exponential STDP with puzzle-conditioned, token-shared phases.

    c = RMSNorm(mean_tokens(hidden + input injection))
    phi_role = (pi/2) * tanh(theta_role + W_role c), shape [batch,heads,dim].
    The two W projections start at zero, reproducing the fixed-phase forward
    initially while receiving nonzero ordinary gradients through the envelope.
    Every block recomputes the summary; no phase accumulator or history is read.
    Token sharing allows the exact window to follow the single KV GEMM.
    """
    def __init__(self, config):
        if config.kv_write_reduction != "mean":
            raise ValueError('Puzzle-state exponential STDP uses the token mean.')
        super().__init__(config)
        # Add only after all baseline parameters and offsets are initialized.
        for layer in self.layers:
            layer.phase_k_proj = torch.nn.Linear(self.d, self.d, bias=False)
            layer.phase_v_proj = torch.nn.Linear(self.d, self.d, bias=False)
            torch.nn.init.zeros_(layer.phase_k_proj.weight)
            torch.nn.init.zeros_(layer.phase_v_proj.weight)

    def phases(self, layer, hidden):
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            summary = hidden.to(dtype).mean(dim=1)
            # This summary norm is separate from the two residual norms.
            summary = summary * torch.rsqrt(
                summary.square().mean(-1, keepdim=True) + self.norm_eps)
            return tuple(self.phase_limit * (
                projection(summary).reshape(hidden.shape[0], self.H, self.dh)
                + offset[None]).tanh()
                for projection, offset in ((layer.phase_k_proj, layer.theta_k_raw),
                                           (layer.phase_v_proj, layer.theta_v_raw)))

    def phase_window(self, layer, dtype, *, phases):
        pk, pv = (x.to(dtype) for x in phases)
        delta = pv[..., :, None] - pk[..., None, :]
        return delta.sign() * torch.exp(-delta.abs() / self.phase_tau)

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None, *, phases):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            current = (v.transpose(-1, -2) @ kr) / k.shape[-2]
            current = current * self.phase_window(layer, dtype, phases=phases)
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None

    def block(self, layer, hidden, inj, memory, e_k, e_v, fresh):
        hidden = hidden + self.embed_scale * inj
        b, n, _ = hidden.shape
        def heads(x):
            return x.reshape(b, n, self.H, self.dh).transpose(1, 2)
        q = heads(layer.q_proj(hidden))
        if self.config.kv_projection_fp32:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                projection_h = hidden if hidden.dtype == torch.float64 else hidden.float()
                k, v = (heads(p(projection_h)) for p in (layer.k_proj, layer.v_proj))
        else:
            k, v = (heads(p(hidden)) for p in (layer.k_proj, layer.v_proj))
        read, current, _, _ = self.memory_step(
            layer, q, k, v, phases=self.phases(layer, hidden))
        read = read.transpose(1, 2).reshape(b, n, self.d).to(hidden.dtype)
        hidden = hidden + layer.out_proj(read)
        # Inherited boundary applies RMSNorm before FFN; phi after it applies
        # the second RMSNorm to the FFN residual, both FP32 and affine-free.
        hidden = self.phi(self.boundary(layer, hidden))
        return hidden, current, None, None


class ChannelPhaseAttention(torch.nn.Module):
    """Joint K/V channel attention on position-sensitive activity profiles.

    Each of the 2*head_dim channel rows contains its ordered token activities.
    A shared token-to-latent projection keeps spatial distinctions before the
    channels attend to each other. Heads/puzzles never mix in this branch.
    Both residual norms accumulate in FP32, including in FP64 audit callers.
    """
    latent_dim = 16
    ffn_expansion = 4
    norm_eps = 1e-5

    def __init__(self, seq_len, heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        r = self.latent_dim
        self.profile_proj = torch.nn.Linear(seq_len, r, bias=False)
        self.channel_embedding = torch.nn.Parameter(
            torch.empty(heads, 2 * head_dim, r))
        torch.nn.init.normal_(self.channel_embedding, std=.02)
        self.qkv = torch.nn.Linear(r, 3 * r, bias=False)
        self.out_proj = torch.nn.Linear(r, r, bias=False)
        self.ffn_up = torch.nn.Linear(r, self.ffn_expansion * r, bias=False)
        self.ffn_down = torch.nn.Linear(self.ffn_expansion * r, r, bias=False)
        self.readout = torch.nn.Linear(r, 1, bias=False)
        # Only this last projection is zero: it receives gradients immediately,
        # and opens the other gradients after its first nonzero-LR update.
        torch.nn.init.zeros_(self.readout.weight)

    def norm(self, hidden):
        dtype = hidden.dtype
        hidden = hidden.float()
        return (hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True)
                                     + self.norm_eps)).to(dtype)

    def forward(self, rotated_key, value):
        dtype = torch.float64 if rotated_key.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=rotated_key.device.type, enabled=False):
            profiles = torch.cat((rotated_key.to(dtype).transpose(-1, -2),
                                  value.to(dtype).transpose(-1, -2)), dim=-2)
            z = self.profile_proj(profiles) + self.channel_embedding
            q, k, v = self.qkv(z).chunk(3, dim=-1)
            weights = ((q @ k.transpose(-1, -2)) / math.sqrt(self.latent_dim)).softmax(-1)
            u = self.norm(z + self.out_proj(weights @ v))
            z = self.norm(u + self.ffn_down(torch.nn.functional.gelu(self.ffn_up(u))))
            return self.readout(z).squeeze(-1).split(self.head_dim, dim=-1)


class CoactivityChannelPhaseAttention(torch.nn.Module):
    """Channel attention using the full token-axis Gram matrix as its logits.

    Z=[RoPE(K).T; V.T], scores=Z Z.T / T; no learned token-axis projection.
    Values and residuals use channel/role identity embeddings, so no absolute
    token-slot weights re-enter through the value path or scalar readout.
    The 16-wide messages are mixed only AFTER comparing full activity profiles.
    Common permutations of already-encoded K/V token records preserve phases;
    this does not assert invariance when changing the spatial RoPE assignment.
    """
    latent_dim = 16
    ffn_expansion = 4
    norm_eps = 1e-5
    norm = ChannelPhaseAttention.norm

    def __init__(self, seq_len, heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        r = self.latent_dim
        self.channel_embedding = torch.nn.Parameter(
            torch.empty(heads, 2 * head_dim, r))
        torch.nn.init.normal_(self.channel_embedding, std=.02)
        self.value_proj = torch.nn.Linear(r, r, bias=False)
        self.out_proj = torch.nn.Linear(r, r, bias=False)
        self.ffn_up = torch.nn.Linear(r, self.ffn_expansion * r, bias=False)
        self.ffn_down = torch.nn.Linear(self.ffn_expansion * r, r, bias=False)
        self.readout = torch.nn.Linear(r, 1, bias=False)
        torch.nn.init.zeros_(self.readout.weight)

    def forward(self, rotated_key, value):
        dtype = torch.float64 if rotated_key.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=rotated_key.device.type, enabled=False):
            profiles = torch.cat((rotated_key.to(dtype).transpose(-1, -2),
                                  value.to(dtype).transpose(-1, -2)), dim=-2)
            scores = (profiles @ profiles.transpose(-1, -2)) / profiles.shape[-1]
            weights = scores.softmax(-1)
            embedding = self.channel_embedding
            messages = weights @ self.value_proj(embedding)
            u = self.norm(embedding + self.out_proj(messages))
            y = self.norm(u + self.ffn_down(torch.nn.functional.gelu(self.ffn_up(u))))
            return self.readout(y).squeeze(-1).split(self.head_dim, dim=-1)


class ChannelAttentionExponentialPhaseCurrentReadInner(ExponentialPhaseCurrentReadInner):
    """Per-channel phases from joint attention over K/V activity profiles.

    phi_role=(pi/2)*tanh(theta_role+phase_attention(RoPE(K),V)_role).
    Phases remain token-shared, so the exact signed exponential multiplies
    the single KV GEMM. The unwindowed K/V profiles avoid a circular forward.
    """
    phase_attention_cls = ChannelPhaseAttention

    def __init__(self, config):
        if config.kv_write_reduction != "mean":
            raise ValueError('Channel-attention exponential STDP uses the token mean.')
        super().__init__(config)
        # Preserve every baseline parameter and initialization draw first.
        for layer in self.layers:
            layer.phase_attention = self.phase_attention_cls(config.seq_len, self.H, self.dh)

    def phases(self, layer, rotated_key, value):
        with torch.autocast(device_type=rotated_key.device.type, enabled=False):
            dk, dv = layer.phase_attention(rotated_key, value)
            return tuple(self.phase_limit * (offset[None] + correction).tanh()
                         for offset, correction in ((layer.theta_k_raw, dk),
                                                    (layer.theta_v_raw, dv)))

    def phase_window(self, layer, dtype, *, phases):
        pk, pv = (x.to(dtype) for x in phases)
        delta = pv[..., :, None] - pk[..., None, :]
        return delta.sign() * torch.exp(-delta.abs() / self.phase_tau)

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            phases = self.phases(layer, kr, v)
            current = (v.transpose(-1, -2) @ kr) / k.shape[-2]
            current = current * self.phase_window(layer, dtype, phases=phases)
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None


class CoactivityChannelExponentialPhaseCurrentReadInner(ChannelAttentionExponentialPhaseCurrentReadInner):
    """Exact exponential write with phases generated from full channel coactivity."""
    phase_attention_cls = CoactivityChannelPhaseAttention


class LocalMonotonePhaseWarp(torch.nn.Module):
    """Token-local, common K/V warp of the learned baseline phase axis.

    A shared Linear([RoPE(K_n),V_n],8) sets positive normalized interval lengths
    on [-pi/2,pi/2]. Piecewise-linear interpolation preserves channel order in
    real arithmetic. Zero logits give the identity, using a displacement form
    to also recover the baseline phases exactly at floating-point initialization.
    No token pooling, phase attention, recurrent phase state, or extra FFN.
    """
    intervals = 8
    phase_limit = math.pi / 2

    def __init__(self, head_dim):
        super().__init__()
        self.projection = torch.nn.Linear(2 * head_dim, self.intervals, bias=False)
        torch.nn.init.zeros_(self.projection.weight)

    def warp(self, base_phase, logits):
        probabilities = logits.softmax(-1)
        # Endpoint displacement is exactly zero. The interior displacements
        # vanish exactly when the zero-initialized head returns uniform lengths.
        displacement = (2 * self.phase_limit *
                        (probabilities - 1 / self.intervals).cumsum(-1)[..., :-1])
        zero = torch.zeros_like(logits[..., :1])
        displacement = torch.cat((zero, displacement, zero), -1)
        coordinate = (base_phase + self.phase_limit) * self.intervals / (2 * self.phase_limit)
        index = coordinate.floor().long().clamp(0, self.intervals - 1)
        fraction = coordinate - index
        index = index[None, :, None].expand(*logits.shape[:-1], base_phase.shape[-1])
        lo, hi = (displacement.gather(-1, index + offset) for offset in (0, 1))
        return base_phase[None, :, None] + (lo + fraction[None, :, None] * (hi - lo))

    def forward(self, rotated_key, value, base_key_phase, base_value_phase):
        dtype = torch.float64 if rotated_key.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=rotated_key.device.type, enabled=False):
            activity = torch.cat((rotated_key.to(dtype), value.to(dtype)), -1)
            logits = self.projection(activity)
            base = torch.cat((base_key_phase.to(dtype), base_value_phase.to(dtype)), -1)
            return self.warp(base, logits).split(rotated_key.shape[-1], -1)


class LocalWarpExponentialPhaseCurrentReadInner(ExponentialPhaseCurrentReadInner):
    """Exact signed exponential with token-local, order-preserving phases.

    Fixed K/V ordering separates the two exponential halves into two GEMMs.
    Compute signs from the bounded baseline phases, rather than inferring them
    again from rounded, possibly very close warped phases. Equal baseline phases
    contribute zero. Gradients through the exponential and warp remain ordinary
    autodiff, including through the learned baseline offsets.
    """
    def __init__(self, config):
        if config.kv_write_reduction != "mean":
            raise ValueError('Local phase warp uses the token mean.')
        super().__init__(config)
        for layer in self.layers:
            layer.phase_warp = LocalMonotonePhaseWarp(self.dh)

    def base_phases(self, layer):
        return super().phases(layer)

    def phases(self, layer, rotated_key, value, *, base_phases=None):
        if base_phases is None:
            base_phases = self.base_phases(layer)
        return layer.phase_warp(rotated_key, value, *base_phases)

    def phase_window(self, layer, dtype, *, phases):
        """Dense diagnostic/reference only; training uses the two GEMMs below."""
        pk, pv = (x.to(dtype) for x in phases)
        base_k, base_v = (x.to(dtype) for x in self.base_phases(layer))
        order = (base_v[..., :, None] - base_k[..., None, :]).sign()
        delta = pv[..., :, None] - pk[..., None, :]
        return order[None, :, None] * (-delta.abs() / self.phase_tau).exp()

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            base_k, base_v = (x.to(dtype) for x in self.base_phases(layer))
            pk, pv = self.phases(layer, kr, v, base_phases=(base_k, base_v))
            plus = ((v * (-pv / self.phase_tau).exp()).transpose(-1, -2)
                    @ (kr * (pk / self.phase_tau).exp()))
            minus = ((v * (pv / self.phase_tau).exp()).transpose(-1, -2)
                     @ (kr * (-pk / self.phase_tau).exp()))
            delta0 = base_v[..., :, None] - base_k[..., None, :]
            current = ((delta0 > 0) * plus - (delta0 < 0) * minus) / k.shape[-2]
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None


class DynamicGaussianPhaseCurrentReadInner(PhaseCurrentReadInner):
    """State-dependent channel phase STDP with an instantaneous shared G.

    Real signed activities and phase projections are independent. The latter
    produce bounded angles, not unconstrained Cartesian real/imaginary parts:
    phi=(pi/2-1e-4)*tanh(W_phi h+theta_raw). A shared carrier cancels.
    The margin keeps even saturated FP32 angles away from the alias boundary.
    L=sign(sin(phiV-phiK))*exp(-(phiV-phiK)^2/tau), with exact zero lag = 0.
    No unit normalization, history, phase clock, or surrogate sign gradient.
    """
    phase_limit = math.pi / 2 - 1e-4
    phase_tau = 1.0
    phase_gain = 1.0

    def __init__(self, config):
        if config.kv_write_reduction != "mean":
            raise ValueError('The fused Gaussian write uses the token mean.')
        super().__init__(config)
        # Added after the original model/offset initialization, preserving
        # the old Q/K/V, FFN, embedding and spatial-RoPE random draws.
        for layer in self.layers:
            layer.phase_k_proj = torch.nn.Linear(self.d, self.d, bias=False)
            layer.phase_v_proj = torch.nn.Linear(self.d, self.d, bias=False)

    def phases(self, layer, hidden):
        b, n, _ = hidden.shape
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        # Phase direction is sensitive to small differences: unlike the
        # existing activity projections, these projections remain FP32 in AMP.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(dtype)
            result = []
            for projection, offset in ((layer.phase_k_proj, layer.theta_k_raw),
                                       (layer.phase_v_proj, layer.theta_v_raw)):
                raw = projection(hidden).reshape(b, n, self.H, self.dh).transpose(1, 2)
                result.append(self.phase_limit * (raw + offset[None, :, None, :]).tanh())
        return tuple(result)

    def memory_step(self, layer, q, k, v, memory=None, e_k=None,
                    e_v=None, fresh=None, *, phases):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k, v = (x.to(dtype) for x in (q, k, v))
            pk, pv = (x.to(dtype) for x in phases)
            tables = self.rope_tables(layer)
            qr, kr = (self.apply_rope(x, layer, tables) for x in (q, k))
            if k.is_cuda and dtype == torch.float32:
                from .unit_phase_stdp import unit_phase_gaussian_write
                current = unit_phase_gaussian_write(kr, v, pk, pv,
                                                    self.phase_tau, self.phase_gain)
            else:
                # Mathematical reference for small CPU/FP64 checks only.
                delta = pv[..., :, None] - pk[..., None, :]
                window = self.phase_gain * delta.sin().sign() * torch.exp(
                    -delta.square() / self.phase_tau)
                current = (v[..., :, None] * kr[..., None, :] * window).mean(-3)
            read = qr @ current.transpose(-1, -2)
        return read, current, None, None

    def block(self, layer, hidden, inj, memory, e_k, e_v, fresh):
        hidden = hidden + self.embed_scale * inj
        b, n, _ = hidden.shape
        def heads(x):
            return x.reshape(b, n, self.H, self.dh).transpose(1, 2)
        q = heads(layer.q_proj(hidden))
        if self.config.kv_projection_fp32:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                projection_h = hidden if hidden.dtype == torch.float64 else hidden.float()
                k, v = (heads(p(projection_h)) for p in (layer.k_proj, layer.v_proj))
        else:
            k, v = (heads(p(hidden)) for p in (layer.k_proj, layer.v_proj))
        read, current, _, _ = self.memory_step(
            layer, q, k, v, phases=self.phases(layer, hidden))
        read = read.transpose(1, 2).reshape(b, n, self.d).to(hidden.dtype)
        hidden = hidden + layer.out_proj(read)
        # URMNormCurrentReadInner.boundary normalizes this attention residual
        # before FFN; phi below normalizes the FFN residual (two RMSNorms).
        hidden = self.phi(self.boundary(layer, hidden))
        return hidden, current, None, None


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
                    "current_only_urm_norm": URMNormCurrentReadInner,
                    "phase_current_only": PhaseCurrentReadInner,
                    "phase_exp_current_only": ExponentialPhaseCurrentReadInner,
                    "phase_puzzle_exp_current_only": PuzzleStateExponentialPhaseCurrentReadInner,
                    "phase_channel_exp_current_only": ChannelAttentionExponentialPhaseCurrentReadInner,
                    "phase_channel_gram_exp_current_only": CoactivityChannelExponentialPhaseCurrentReadInner,
                    "phase_local_warp_exp_current_only": LocalWarpExponentialPhaseCurrentReadInner,
                    "phase_unit_gaussian_current_only": DynamicGaussianPhaseCurrentReadInner,
                    "complex_current_only": ComplexCurrentReadInner,
                    "current_plus_stdp": CurrentPlusSTDPInner,
                    "historical_key_trace": HistoricalKeyTraceInner,
                    "read_gain_quarter": QuarterReadInner}[variant]
    t.model_id_of = lambda cfg: ORIGINAL_MODEL_ID(cfg) + (
        f":research-{variant}" if variant != "original" else "")
