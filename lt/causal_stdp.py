"""Draft v1.2: v1.1 read dynamics with an explicit, signed pair-STDP write.

The write's activity is a bounded nonnegative projection of the current state.
Positive and negative branches compare current activity to *previous* traces.
They implement a known causal/anticausal exponential-mixture timing window;
its Fourier phase is derived from that window, not an unconstrained address.
This file supplies an untrained research architecture, not a checkpoint conversion.
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import train
else:
    import train


@dataclass
class CausalConfig(train.LTConfig):
    activity_channels: int = 16
    timing_tau_init: tuple = (2.0, 8.0, 32.0, 128.0)
    timing_tau_min: float = 1.0
    timing_tau_max: float = 256.0
    timing_amplitude_floor: float = 1e-4


class PairSTDP(nn.Module):
    """History is [B,T,H,2,M,C], with + and - time constants in axis 3.

    K(d>0) = sum_m a_plus[m]*(1-rho_plus[m])*rho_plus[m]**(d-1)
    K(d<0) = -sum_m a_minus[m]*(1-rho_minus[m])*rho_minus[m]**(-d-1)
    K(0) = 0. All amplitudes are positive and their joint sum is one.
    """

    def __init__(self, heads, hidden_size, channels=16, tau_init=(2, 8, 32, 128),
                 tau_min=1.0, tau_max=256.0, amplitude_floor=1e-4):
        super().__init__()
        if channels < 1 or not tau_init or not 0 < tau_min < tau_max:
            raise ValueError("invalid activity channels or timing range")
        if not all(tau_min < v < tau_max for v in tau_init):
            raise ValueError("initial time constants must be strictly inside the timing range")
        self.heads, self.channels, self.components = heads, channels, len(tau_init)
        self.tau_min, self.tau_max = tau_min, tau_max
        self.amplitude_floor = amplitude_floor
        if not 0 < amplitude_floor < 1/(2*self.components):
            raise ValueError("amplitude floor must lie between zero and 1/(2*M)")
        self.activity_weight = nn.Parameter(torch.randn(heads, channels, hidden_size)
                                           / math.sqrt(hidden_size))
        fraction = (torch.tensor(tau_init)-tau_min)/(tau_max-tau_min)
        self.timing_tau_raw = nn.Parameter(torch.logit(fraction).expand(heads, 2, -1).clone())
        self.timing_amplitude_logits = nn.Parameter(torch.zeros(heads, 2, self.components))

    def coefficients(self):
        tau = self.tau_min + (self.tau_max-self.tau_min)*self.timing_tau_raw.sigmoid()
        rho = torch.exp(-1/tau)
        a = self.timing_amplitude_logits.flatten(1).softmax(-1).view_as(rho)
        a = self.amplitude_floor + (1-2*self.components*self.amplitude_floor)*a
        return rho, a

    def activities(self, q):
        # This is the actual drive, not a retrospectively defined residual.
        # Squaring uses the existing model's bilinear kind of nonlinearity.
        # Accumulate the write in FP32 (or FP64 during algebraic audits).
        dtype = torch.float64 if self.activity_weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            y = torch.einsum("btd,hcd->bthc", q.to(dtype), self.activity_weight)
            energy = y.square()
            return energy/(1+energy.sum(-1, keepdim=True))

    def reset_history(self, x, history=None, fresh=None):
        if history is None:
            history = x.new_zeros(*x.shape[:-1], 2, self.components, self.channels)
        elif fresh is not None:
            history = torch.where(fresh.view(-1, 1, 1, 1, 1, 1),
                                  torch.zeros_like(history), history)
        return history

    def pair_terms(self, x, history):
        _, a = self.coefficients()
        plus_history = (history[:, :, :, 0]*a[:, 0, :, None]).sum(-2)
        minus_history = (history[:, :, :, 1]*a[:, 1, :, None]).sum(-2)
        # Matrix rows receive (post), columns send (pre).
        plus = torch.einsum("bthc,bnhc->bhtn", x, plus_history)
        minus = torch.einsum("bthc,bnhc->bhtn", minus_history, x)
        return plus, minus

    def advance_history(self, x, history):
        rho, _ = self.coefficients()
        return rho[..., None]*history + (1-rho[..., None])*x[..., None, None, :]

    def write(self, x, history=None, fresh=None):
        """Write from past-only history, THEN append the present activity."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            history = self.reset_history(x, history, fresh)
            plus, minus = self.pair_terms(x, history)
            new_history = self.advance_history(x, history)
        return plus-minus, new_history

    def timing_window(self, lags):
        """Exact isolated-pair window [H, number_of_lags], before gain/distance."""
        rho, a = self.coefficients()
        d = torch.as_tensor(lags, device=rho.device, dtype=rho.dtype)
        magnitude = (a[..., None]*(1-rho[..., None]) *
                     rho[..., None].pow((d.abs()-1).clamp_min(0))).sum(-2)
        return torch.where(d > 0, magnitude[:, 0],
                           torch.where(d < 0, -magnitude[:, 1], torch.zeros_like(magnitude[:, 0])))

    def timing_spectrum(self, omega):
        """DTFT of the infinite discrete pair window; no finite-bank approximation."""
        rho, a = self.coefficients()
        omega = torch.as_tensor(omega, device=rho.device, dtype=rho.dtype)
        e = torch.exp(-1j*omega)
        pos = (a[:, 0, :, None]*(1-rho[:, 0, :, None])*e /
               (1-rho[:, 0, :, None]*e)).sum(-2)
        neg = (a[:, 1, :, None]*(1-rho[:, 1, :, None])*e.conj() /
               (1-rho[:, 1, :, None]*e.conj())).sum(-2)
        return pos-neg


class CausalInner(train.LT_Inner):
    def __init__(self, config):
        if (not config.stdp or config.use_trace or config.legacy_gauge or
                config.block_order != "pre" or config.num_layers != 1):
            raise ValueError("draft requires the one-layer v1.1 scaffold, without the old address trace")
        super().__init__(config)
        layer = self.layers[0]
        del layer.beta
        layer.temporal_write = PairSTDP(
            self.H, self.d, config.activity_channels, config.timing_tau_init,
            config.timing_tau_min, config.timing_tau_max, config.timing_amplitude_floor)

    def step(self, L, h, AB, kc, w=None, fresh=None, kcb=None, ztr=None, apply_phi=True):
        if not apply_phi:
            raise ValueError("only pre-order blocks are supported in this draft")
        a = self.attn_xy(self.addr(h, AB), kc)
        v = torch.einsum("btd,hcd->bthc", h, L.w_sh)
        x = L.temporal_write.activities(h)
        signal, history = L.temporal_write.write(x, ztr, fresh)
        with torch.autocast(device_type=h.device.type, enabled=False):
            gain = F.softplus(L.gain_raw) if self.config.stdp_gain_fixed < 0 else float(self.config.stdp_gain_fixed)
            eta = L.eta_raw.sigmoid()
            target = gain*(kc[0].to(signal.dtype).unsqueeze(0)*signal)
            if w is None:
                w = torch.zeros_like(target)
            elif fresh is not None:
                w = torch.where(fresh.view(-1, 1, 1, 1), torch.zeros_like(w), w)
            w = (1-eta)*w + eta*target
            lam = (L.lam_raw.sigmoid() if self.config.stdp_lam_fixed < 0
                   else torch.full_like(L.lam_raw, self.config.stdp_lam_fixed))
            coupling = (1-lam)*a + lam*w
        message = torch.einsum("bhtn,bnhc->bthc", coupling, v)
        message = torch.einsum("bthc,hcd->btd", message, L.w_sh)
        return self.phi(h+message), w, history

    def _forward(self, carry, batch):
        h, inj = carry.current_hidden, self.injection(batch)
        L = self.layers[0]
        AB, kc = self.W_C(L), self.kernel(L)
        w, history, fresh = carry.coupling, carry.trace, carry.fresh
        for _ in range(self.config.blocks_per_seg):
            q = self.boundary(L, h) + self.embed_scale*inj
            h, w, history = self.step(L, q, AB, kc, w, fresh, ztr=history)
            fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=w.detach(),
                       trace=history.detach(), fresh=None), self.w_cls(h)


class CausalLT(train.LT):
    def __init__(self, config_dict):
        nn.Module.__init__(self)
        self.config = CausalConfig.from_dict(config_dict)
        self.inner = CausalInner(self.config)

    def initial_carry(self, batch):
        size, device = batch["inputs"].shape[0], batch["inputs"].device
        return train.LTCarry(
            current_hidden=self.inner.init_hidden.new_empty(size, self.config.seq_len, self.config.hidden_size),
            steps=torch.zeros(size, dtype=torch.int32, device=device),
            halted=torch.ones(size, dtype=torch.bool, device=device),
            current_data={key: torch.empty_like(value) for key, value in batch.items()})
