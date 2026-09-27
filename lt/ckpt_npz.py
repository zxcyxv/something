"""체크포인트 ↔ npz (git 보관용).

저장:  python lt/ckpt_npz.py pack  <in.pt> <out.npz> [--raw] [--fp16] [--note "..."]
       ema_shadow (+ ema 에 없는 raw 키) 를 저장. --raw 면 raw_model_state_dict 도 저장.
       기본 float32 (25MB). --fp16 을 주면 numel ≥ 1e5 텐서를 float16 으로 (12MB) — 완답 수가 ±1% 흔들리므로 기록 재현용엔 쓰지 말 것.
       (2026-09-10 이전 npz 는 전부 fp16 이다.)
로드:  from lt.ckpt_npz import load ;  sd, meta = load("x.npz", which="ema")   # sd: {name: float32 tensor}
       model.load_state_dict(sd, strict=False) 로 평가. meta = {step, cfg, note, source}
"""
import json, sys, numpy as np, torch

BIG = 100_000

def _pack_sd(sd, prefix, out, fp16=False):
    for k, t in sd.items():
        if not torch.is_tensor(t): continue
        a = t.detach().cpu()
        if fp16 and a.is_floating_point() and a.numel() >= BIG: a = a.to(torch.float16)
        out[f"{prefix}/{k}"] = a.numpy()

def pack(src, dst, with_raw=False, note="", fp16=False):
    c = torch.load(src, map_location="cpu", weights_only=False)
    out = {}
    ema = c.get("ema_shadow") or {}
    raw = c.get("raw_model_state_dict") or c.get("model_state_dict") or {}
    _pack_sd(ema, "ema", out, fp16)
    _pack_sd({k: v for k, v in raw.items() if k not in ema}, "ema", out, fp16)      # ema 에 없는 키(퍼즐 임베딩 등)는 raw 로 채움
    if with_raw: _pack_sd(raw, "raw", out, fp16)
    meta = dict(step=int(c.get("step", -1)), cfg=c.get("cfg", {}), note=note or c.get("note", ""), source=src,
                ema_keys=list(ema.keys()), raw_filled_into_ema=[k for k in raw if k not in ema])
    out["__meta__"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(dst, **out)

def load(path, which="ema"):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    sd = {k.split("/", 1)[1]: torch.from_numpy(z[k]).to(torch.float32) if z[k].dtype.kind == "f" else torch.from_numpy(z[k])
          for k in z.files if k.startswith(which + "/")}
    return sd, meta

if __name__ == "__main__":
    if sys.argv[1] == "pack":
        a = sys.argv[2:]; note = ""
        if "--note" in a: i = a.index("--note"); note = a[i + 1]; del a[i:i + 2]
        raw = "--raw" in a; fp16 = "--fp16" in a; a = [x for x in a if x not in ("--raw", "--fp16")]
        pack(a[0], a[1], raw, note, fp16)
    else:
        sd, meta = load(sys.argv[2]); print(meta["step"], len(sd), sum(v.numel() for v in sd.values()))


# ─────────────────────────────────────────────────────────────────────────────
# 공용 로더: .pt(학습 체크포인트) 또는 .npz(git 보관본) → lt/train.py 의 LT 모델 (EMA 가중치)
# ─────────────────────────────────────────────────────────────────────────────
_PER = {"wc_raw", "wc", "psi", "theta", "alpha_raw", "w_sh", "eta_raw", "lam_raw", "gain_raw", "beta", "b_gate_up", "b_down",
        "mu_rho_raw", "mu_omega"}

def _strip(k):
    k = k.replace("_orig_mod.", "")
    return k[len("model."):] if k.startswith("model.") else k

def _remap_legacy(k):
    """9/1 원본 npz 는 레이어 접두(layers.0)가 없고 inj_gate 라는 옛 이름을 쓴다."""
    k = _strip(k)
    if not k.startswith("inner."): return k
    t = k[len("inner."):]
    if t == "inj_gate": return "inner.embed_scale"
    return f"inner.layers.0.{t}" if t.split(".")[0] in _PER else k

def load_lt(path, mod=None, weights="ema", **cfg_over):
    """반환 (model(eval, cuda), cfg, step).  cfg_over 로 batch_size/loops 등을 덮는다.
    mod = lt.train 모듈 (None 이면 import). 판(legacy_gauge·block_order·use_trace)은 체크포인트 cfg 에서 읽는다."""
    import importlib.util, os as _os
    if mod is None:
        spec = importlib.util.spec_from_file_location("lt_train", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "train.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if path.endswith(".npz"):
        sd, meta = load(path, which=weights); cfg = dict(meta.get("cfg") or {}); step = meta.get("step", -1)
        if not cfg: cfg = dict(mod.CFG)                       # 9/1 원본 npz 는 cfg 가 비어 있다 → v1 기본값
        sd = {_remap_legacy(k): v for k, v in sd.items()}
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False); cfg = dict(ck["cfg"]); step = int(ck.get("step", -1))
        sd = {_strip(k): v for k, v in ck["model_state_dict"].items()}
        if weights == "ema" and ck.get("ema_shadow"):
            sd.update({_strip(k): v for k, v in ck["ema_shadow"].items()})
    cfg.update(seq_len=81, num_puzzle_identifiers=1); cfg.update(cfg_over)
    m = mod.LT(cfg).cuda().eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    miss = [k for k in miss if "puzzle_emb" not in k]
    assert not miss and not unexp, f"state_dict 불일치 missing={miss[:3]} unexpected={list(unexp)[:3]}"
    return m, cfg, step

def load_data(root=None, n=512):
    """held-out 앞 n 퍼즐 → (X, Y, batch(cuda))."""
    import numpy as _np, os as _os
    root = root or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    z = _np.load(_os.path.join(root, "data/sudoku_lt_1k.npz"))
    X = z["test_inputs"].reshape(-1, 81).astype(_np.int32)[:n]; Y = z["test_labels"].reshape(-1, 81).astype(_np.int32)[:n]
    batch = dict(inputs=torch.from_numpy(X + 1).cuda(), labels=torch.from_numpy(Y + 1).cuda().long(),
                 puzzle_identifiers=torch.zeros(n, dtype=torch.int32, device="cuda"))
    return X, Y, batch
