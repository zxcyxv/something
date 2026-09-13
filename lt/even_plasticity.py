"""Experimental LT with a learned even phase-plasticity window.

Fast transmission retains psi and both address/content projections. The plastic
window has real signed spectral coefficients. Its symmetric persistent state is
stored once per unordered cell pair, including the diagonal. This is a research
candidate, not a replacement for the production LT default.
"""

from dataclasses import asdict, replace

import torch
from torch import nn

import train


class EvenInner(train.LT_Inner):
    def __init__(self, config, bootstrap=True):
        if not config.stdp or config.use_trace or config.block_order != "pre":
            raise ValueError("even-window candidate requires STDP, pre order, and no address trace")
        if config.stdp_gain_fixed >= 0 or config.stdp_lam_fixed >= 0:
            raise ValueError("fixed gain/mix settings are not supported by this research candidate")
        super().__init__(config)
        self.bootstrap = bootstrap
        pair_index = torch.triu_indices(config.seq_len, config.seq_len)
        self.register_buffer("pair_row", pair_index[0], persistent=False)
        self.register_buffer("pair_col", pair_index[1], persistent=False)
        for layer in self.layers:
            lam = layer.lam_raw.detach().sigmoid()
            gain = torch.nn.functional.softplus(layer.gain_raw.detach())
            layer.base_raw = nn.Parameter(-layer.lam_raw.detach().clone())
            layer.plastic_spectrum = nn.Parameter(
                (lam*gain).squeeze(-1)*layer.beta.detach().cos())
            del layer.lam_raw, layer.gain_raw, layer.beta

    def pack(self, dense):
        return dense[..., self.pair_row, self.pair_col]

    def unpack(self, packed):
        n = self.config.seq_len
        dense = packed.new_zeros(*packed.shape[:-1], n, n)
        dense[..., self.pair_row, self.pair_col] = packed
        dense[..., self.pair_col, self.pair_row] = packed
        return dense

    def write_target(self, layer, unit_address, unit_values, distance):
        # Code convention: z_i = u_i exp(i theta.position_i).
        x, y = unit_address
        pos = (layer.theta[..., 0, None]*self.pos_u+
               layer.theta[..., 1, None]*self.pos_w).permute(2, 0, 1)
        zx = x*pos.cos()-y*pos.sin()
        zy = x*pos.sin()+y*pos.cos()
        c = layer.plastic_spectrum
        window = (torch.einsum("bthj,bnhj->bhtn", zx*c, zx)+
                  torch.einsum("bthj,bnhj->bhtn", zy*c, zy))
        agree = torch.einsum("bthc,bnhc->bhtn", unit_values, unit_values)
        return distance.unsqueeze(0)*window*agree

    def step(self, layer, h, AB, kc, w=None, fresh=None, kcb=None,
             ztr=None, apply_phi=True):
        if ztr is not None or not apply_phi:
            raise ValueError("only the v1.1 pre-order block is implemented")
        u = self.addr(h, AB)
        a = self.attn_xy(u, kc)
        v = torch.einsum("btd,hcd->bthc", h, layer.w_sh)
        vn = v/(v.norm(dim=-1, keepdim=True)+self.config.eps)
        target = self.pack(self.write_target(layer, u, vn, kc[0]))
        eta = layer.eta_raw.sigmoid().squeeze(-1)
        initial = target if self.bootstrap else eta*target
        wn = initial if w is None else (1-eta)*w+eta*target
        if fresh is not None:
            wn = torch.where(fresh.view(-1, 1, 1), initial, wn)
        coupling = layer.base_raw.sigmoid()*a+self.unpack(wn)
        msg = torch.einsum("bhtn,bnhc->bthc", coupling, v)
        msg = torch.einsum("bthc,hcd->btd", msg, layer.w_sh)
        return self.phi(h+msg), wn, None

    def _forward(self, carry, batch):
        h = carry.current_hidden
        inj = self.injection(batch)
        addresses = [self.W_C(layer) for layer in self.layers]
        kernels = [self.kernel(layer) for layer in self.layers]
        memory, fresh = carry.coupling, carry.fresh
        for _ in range(self.config.blocks_per_seg):
            for layer, AB, kc in zip(self.layers, addresses, kernels):
                q = self.boundary(layer, h)+self.embed_scale*inj
                h, memory, _ = self.step(layer, q, AB, kc, memory, fresh)
                fresh = None
        return replace(carry, current_hidden=h.detach(), coupling=memory.detach(),
                       trace=None, fresh=None), self.w_cls(h)


class EvenLT(train.LT):
    def __init__(self, config_dict, bootstrap=True):
        nn.Module.__init__(self)
        self.config = train.LTConfig.from_dict(config_dict)
        self.inner = EvenInner(self.config, bootstrap=bootstrap)


def from_v11(model, bootstrap=True):
    """Convert parameters; initial trajectories match symmetric-plastic read.

This is NOT an exact conversion of the original asymmetric model. A saved
original carry must separately be converted with convert_memory().
"""
    ref = next(model.parameters())
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        converted = EvenLT(asdict(model.config), bootstrap=bootstrap)
    converted.to(device=ref.device, dtype=ref.dtype)
    state = dict(model.state_dict())
    for index in range(model.config.num_layers):
        prefix = f"inner.layers.{index}."
        lam_raw = state.pop(prefix+"lam_raw")
        gain = torch.nn.functional.softplus(state.pop(prefix+"gain_raw"))
        beta = state.pop(prefix+"beta")
        state[prefix+"base_raw"] = -lam_raw
        state[prefix+"plastic_spectrum"] = (lam_raw.sigmoid()*gain).squeeze(-1)*beta.cos()
    converted.load_state_dict(state, strict=True)
    converted.train(model.training)
    return converted


def convert_memory(original_model, converted_model, memory, layer=0):
    lam = original_model.inner.layers[layer].lam_raw.sigmoid()
    return converted_model.inner.pack(lam*(memory+memory.transpose(-1, -2))/2)
