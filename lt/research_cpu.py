"""CPU-only v1.1 loading and differentiable blocks for isolated theory probes.

The production loader intentionally creates CUDA models. These probes instead
load the same NPZ weights on CPU and leave the training code and other jobs alone.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train
from ckpt_npz import _remap_legacy, load


def load_cpu(checkpoint="checkpoints/v1.1_step160000.npz", dtype=torch.float64):
    state, metadata = load(str(checkpoint))
    cfg = dict(metadata["cfg"])
    cfg.update(seq_len=81, num_puzzle_identifiers=1, batch_size=1,
               forward_dtype="float32", amp=False)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(0)
        model = train.LT(cfg).eval()
    missing, extra = model.load_state_dict(
        {_remap_legacy(k): v for k, v in state.items()}, strict=False)
    missing = [k for k in missing if "puzzle_emb" not in k]
    if missing or extra:
        raise ValueError(f"checkpoint mismatch: missing={missing}, extra={extra}")
    model.to(dtype=dtype)
    model.requires_grad_(False)
    assert not model.config.use_trace and model.config.block_order == "pre"
    assert model.config.num_layers == 1 and not model.config.legacy_gauge
    assert all(p.device.type == "cpu" for p in model.parameters())
    return model, metadata


def puzzle_batch(index=1, data="data/sudoku_lt_1k.npz"):
    with np.load(data, allow_pickle=False) as z:
        x = z["test_inputs"].reshape(-1, 81)[index:index+1].copy()
        y = z["test_labels"].reshape(-1, 81)[index:index+1].copy()
    return dict(inputs=torch.from_numpy(x+1).long(),
                labels=torch.from_numpy(y+1).long(),
                puzzle_identifiers=torch.zeros(1, dtype=torch.int32))


class CPUBlocks:
    def __init__(self, model, batch):
        self.model, self.inner, self.batch = model, model.inner, batch
        self.layer = self.inner.layers[0]
        self.ab = self.inner.W_C(self.layer)
        self.kc = self.inner.kernel(self.layer)
        self.kcb = self.inner.kernel(self.layer, self.layer.beta)
        self.eta = self.layer.eta_raw.sigmoid()
        self.rho = 1-self.eta
        self.lam = self.layer.lam_raw.sigmoid()
        self.gain = F.softplus(self.layer.gain_raw)
        self.inj = self.inner.embed_scale*self.inner.injection(batch)

    def prepare(self, h):
        return self.inner.boundary(self.layer, h)+self.inj

    def parts(self, h):
        q = self.prepare(h)
        u = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(u, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        vn = v/(v.norm(dim=-1, keepdim=True)+self.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vn, vn)
        window = self.inner.attn_xy(u, self.kcb)
        G = self.gain*(window*agree)
        return dict(q=q, u=u, a=a, v=v, vn=vn, agree=agree, window=window, G=G)

    def write(self, h):
        return self.parts(h)["G"]

    def transmit_parts(self, parts, memory):
        J = (1-self.lam)*parts["a"]+self.lam*memory
        msg = torch.einsum("bhtn,bnhc->bthc", J, parts["v"])
        msg = torch.einsum("bthc,hcd->btd", msg, self.layer.w_sh)
        return self.inner.phi(parts["q"]+msg)

    def transmit(self, h, memory):
        return self.transmit_parts(self.parts(h), memory)

    def block(self, h, memory):
        parts = self.parts(h)
        wn = parts["G"] if memory is None else self.rho*memory+self.eta*parts["G"]
        return self.transmit_parts(parts, wn), wn


def load_snapshots(path="runs/settling_v11/snapshots.pt", dtype=torch.float64):
    return {k: tuple(t.to(dtype=dtype) for t in value)
            for k, value in torch.load(Path(path), map_location="cpu", weights_only=True).items()}
