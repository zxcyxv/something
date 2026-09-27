"""Convert v1.7 QR address weights to v1.71 direct linear projections.

Usage: python -m lt.convert_v171 SOURCE.npz DESTINATION.pt [--device cpu]

Raw, evaluation, and EMA-shadow projections are converted independently. This
is a weights conversion, not an exact training continuation: optimizer, random
state, data cursor, and recurrent carry start fresh. No training is launched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch


_RAW_PROJECTION = re.compile(r"(?:^|\.)layers\.\d+\.wc_raw$")
_LINEAR_PROJECTION = re.compile(r"(?:^|\.)layers\.\d+\.wc$")
_V17 = re.compile(r"(?:^|[^a-z0-9])v1[._]?7(?![0-9])", re.I)
_BUFFER_SUFFIXES = (".init_hidden", ".puzzle_emb.weights")


def _clean_key(key: str) -> str:
    # Compiling the full wrapper or an inner submodule can place this component
    # at different levels. Detect collisions instead of silently losing a key.
    return ".".join(part for part in key.split(".") if part != "_orig_mod")


def _harness_state(state: dict) -> dict:
    result = {}
    for old_key, value in state.items():
        key = _clean_key(old_key)
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("inner."):
            key = "model." + key
        if not key.startswith("model.inner."):
            raise ValueError(f"Unsupported checkpoint wrapper key: {old_key}")
        if key in result:
            raise ValueError(f"Prefix normalization collision: {old_key} -> {key}")
        if not torch.is_tensor(value):
            raise TypeError(f"Model state is not a tensor: {old_key}")
        result[key] = value
    return result


@torch.no_grad()
def convert_state(state: dict, device: str = "cpu") -> dict:
    """Return a CPU state dict, replacing each layers.N.wc_raw by its QR Q.T.

    Arbitrary model prefixes are supported; compile prefixes are removed.
    All other tensors are copied unchanged. No positive-R sign convention is
    introduced: the QR function's original Q is the projection being preserved.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    result, count = {}, 0
    for old_key, value in state.items():
        key = _clean_key(old_key)
        if _LINEAR_PROJECTION.search(key):
            raise ValueError(f"Already a linear-projection checkpoint: {key}")
        if not torch.is_tensor(value):
            raise TypeError(f"Model state is not a tensor: {old_key}")
        if _RAW_PROJECTION.search(key):
            if value.ndim != 3 or value.shape[-2] > value.shape[-1]:
                raise ValueError(f"Expected wc_raw[heads, address_dim, hidden_dim]: {key} {tuple(value.shape)}")
            dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            work = value.detach().to(device=device, dtype=dtype)
            if not torch.isfinite(work).all():
                raise ValueError(f"Non-finite projection: {key}")
            q, r = torch.linalg.qr(work.transpose(-1, -2), mode="reduced")
            if torch.any(r.diagonal(dim1=-2, dim2=-1) == 0):
                raise ValueError(f"Rank-deficient projection: {key}")
            value = q.transpose(-1, -2).contiguous()
            key = key[:-len("wc_raw")] + "wc"
            count += 1
        if key in result:
            raise ValueError(f"Converted state key collision: {key}")
        result[key] = value.detach().cpu().clone()
    if not count:
        raise ValueError("No layers.N.wc_raw projections found")
    return result


def _parameter_subset(state: dict, keys: list | None = None) -> dict:
    if keys is None:
        # Original v1.7's only persistent buffers. Nonpersistent sparse local
        # workspaces and position tensors are absent from its state_dict.
        return {k: v for k, v in state.items() if not k.endswith(_BUFFER_SUFFIXES)}
    normalized = list(_harness_state({k: torch.empty(0) for k in keys}))
    missing = set(normalized) - set(state)
    if missing:
        raise ValueError(f"EMA parameter keys absent from evaluation state: {sorted(missing)}")
    return {k: state[k] for k in normalized}


def read_source(path: Path) -> tuple[dict, dict, dict | None, dict, dict]:
    """Return raw, evaluation, shadow, source metadata, and provenance details."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["__meta__"]))
            def section(prefix):
                result = {}
                for key in archive.files:
                    if key.startswith(prefix + "/"):
                        tensor = torch.from_numpy(archive[key].copy())
                        # Match lt.ckpt_npz.load, including historical fp16 files.
                        result[key.split("/", 1)[1]] = tensor.float() if tensor.is_floating_point() else tensor
                return _harness_state(result)
            raw, evaluation = section("raw"), section("ema")
        raw_origin = "raw" if raw else "ema (raw absent)"
        raw = raw or evaluation
        evaluation = evaluation or raw
        ema_keys = meta.get("ema_keys")
        shadow = _parameter_subset(evaluation, ema_keys) if ema_keys != [] and meta.get("cfg", {}).get("ema", True) else None
        shadow_origin = "npz EMA parameter subset" if shadow else "absent"
    elif path.suffix.lower() in (".pt", ".pth"):
        meta = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(meta, dict):
            raise ValueError("Expected a training-checkpoint dictionary")
        raw_origin = "raw_model_state_dict" if meta.get("raw_model_state_dict") else "model_state_dict (raw absent)"
        raw = _harness_state(meta.get("raw_model_state_dict") or meta.get("model_state_dict") or {})
        evaluation = _harness_state(meta.get("model_state_dict") or raw)
        source_shadow = meta.get("ema_shadow")
        if source_shadow:
            shadow = _harness_state(source_shadow)
            shadow_origin = "ema_shadow"
        elif meta.get("cfg", {}).get("ema", True):
            shadow = _parameter_subset(evaluation)
            shadow_origin = "reconstructed from model_state_dict parameter subset"
        else:
            shadow, shadow_origin = None, "absent"
    else:
        raise ValueError("Source must be .npz, .pt, or .pth")
    if not raw or not evaluation:
        raise ValueError("Source has no usable model weights")
    if set(raw) != set(evaluation):
        raise ValueError("Raw and evaluation state keys differ")
    if shadow is not None and set(shadow) != set(_parameter_subset(raw)):
        raise ValueError("EMA shadow does not cover exactly the v1.7 learned parameters")
    return raw, evaluation, shadow, meta, dict(raw_origin=raw_origin, ema_shadow_origin=shadow_origin)


def validate_v17(cfg: dict, state: dict, meta: dict, source: Path) -> None:
    if cfg.get("address_projection", "qr") != "qr" or any(_LINEAR_PROJECTION.search(k) for k in state):
        raise ValueError("Source is already linear or uses an unsupported address projection")
    if cfg.get("legacy_gauge") is not False or cfg.get("block_order") != "post":
        raise ValueError("Expected v1.7: fixed gauge and post block order")
    if not cfg.get("use_trace", "trace_rho_init" in cfg) or not cfg.get("stdp", False):
        raise ValueError("Expected v1.7 with address traces and STDP enabled")
    for key, expected in (("stdp_target", "faithful"), ("stdp_window", "beta"),
                          ("stdp_read", "add"), ("stdp_diag", "keep"), ("boundary", "bilinear")):
        if cfg.get(key, expected) != expected:
            raise ValueError(f"Unsupported v1.7 variant: {key}={cfg[key]!r}")
    if cfg.get("gate", False) or cfg.get("addr_dim", 0) != 0:
        raise ValueError("Gated/split-address variants are not the supported v1.7 architecture")
    for key in ("preset", "architecture"):
        if key in cfg and cfg[key] not in ("v1.7", "v17"):
            raise ValueError(f"Source explicitly identifies another architecture: {key}={cfg[key]!r}")
    provenance = " ".join(str(x) for x in (source, meta.get("source", ""), meta.get("note", ""),
                                              cfg.get("out_dir", ""), cfg.get("preset", ""), cfg.get("architecture", "")))
    if not _V17.search(provenance):
        raise ValueError("No explicit v1.7 provenance found; trace/post flags alone also describe v1.6")
    projections = [key for key in state if _RAW_PROJECTION.search(key)]
    if len(projections) != int(cfg.get("num_layers", 1)):
        raise ValueError("Projection count does not match num_layers")
    for key in projections:
        prefix = key[:-len("wc_raw")]
        if not all(prefix + name in state for name in ("mu_rho_raw", "mu_omega", "beta")):
            raise ValueError(f"Missing v1.7 trace/write parameters beside {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint(source: Path, destination: Path, device: str = "cpu", overwrite: bool = False) -> dict:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source checkpoint is immutable; choose a different destination")
    if destination.suffix.lower() != ".pt":
        raise ValueError("Destination must have a .pt suffix")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    raw, evaluation, shadow, meta, origins = read_source(source)
    cfg = dict(meta.get("cfg") or {})
    validate_v17(cfg, raw, meta, source)
    source_hash = _sha256(source)
    converted_raw = convert_state(raw, device)
    converted_eval = convert_state(evaluation, device)
    converted_shadow = convert_state(shadow, device) if shadow is not None else None
    cfg.update(address_projection="linear", use_trace=True, block_order="post", legacy_gauge=False,
               preset="v1.71", architecture="v1.71", resume_from=None, out_dir=None)
    conversion = dict(
        format_version=1, source=str(source.resolve()), source_sha256=source_hash,
        source_step=int(meta.get("step", 0)), source_note=meta.get("note", ""),
        source_config=meta.get("cfg", {}), source_architecture="v1.7", target_architecture="v1.71",
        created_utc=datetime.now(timezone.utc).isoformat(), qr_device=device,
        projection_rule="wc = torch.linalg.qr(wc_raw.transpose(-1,-2), mode='reduced').Q.transpose(-1,-2); no sign canonicalization",
        independent_states=["raw_model_state_dict", "model_state_dict"] + (["ema_shadow"] if shadow is not None else []),
        semantics="Weights conversion; fresh optimizer, seeded RNG/data stream, and recurrent carry. Source step retained for the LR schedule.",
        reset=["optimizer_states", "rng_state", "cuda_rng_state", "recurrent_carry", "iter_id", "batch_in_iter"],
        numerical_scope="QR is evaluated on the selected backend; cross-backend bit identity is not promised.",
        **origins)
    output = dict(step=int(meta.get("step", 0)), iter_id=0, batch_in_iter=0,
                  raw_model_state_dict=converted_raw, model_state_dict=converted_eval,
                  ema_shadow=converted_shadow, cfg=cfg, architecture="v1.71",
                  note="v1.7 to v1.71 direct-projection weights conversion; training states reset", conversion=conversion)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(output, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # A same-filesystem hard link publishes the complete file atomically
            # and fails if a competing writer created the destination meanwhile.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination; source is always immutable")
    args = parser.parse_args()
    result = convert_checkpoint(args.source, args.destination, args.device, args.overwrite)
    print(json.dumps(dict(destination=str(args.destination), conversion=result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
