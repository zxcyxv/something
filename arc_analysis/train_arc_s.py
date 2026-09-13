"""`lt/train_arc.py` 를 캔버스 S×S 데이터(`prep_arc_canvas.py` 출력)로, 기존 LT 또는 D4 등변 LT 로 돌린다.

train_arc.py 는 900 토큰·30 캔버스를 세 곳에 상수로 갖는다 — `ARCSplit` 의 형식 검사, `crop_grid` 의 reshape, `main` 의 모델 cfg.
이 스크립트는 그 세 곳만 데이터의 `seq_len` 에서 읽은 S 로 바꿔 끼우고 나머지(샘플러·손실·옵티마이저·평가·재개)는
train_arc.py 의 코드를 그대로 실행한다. lt/ 는 수정하지 않는다.

추가 인자 (나머지는 train_arc.py 와 동일):
  --model base|d4|color  base = lt/train.py 의 LT,  d4 = lt_d4.py 의 D4 등변 LT (헤드 8 필수),
                         color = lt_color.py 의 색 슬롯 LT (--hidden-size 768 = 12 슬롯 × 64, 헤드 8)
                         v3 = lt_v3.py (풀버전 train.py 의 v3 preset: sym_equiv dk=128, swiglu, p=192, write_addr).
                              --hidden-size 1536 --puzzle-emb-dim 140 --num-heads 8 필수
  --no-stdp              결합 기억 w 제거 (LTConfig stdp=False: a_eff = a). STDP ablation
  --transport adj|laplacian   (v3) 수송에 차수 항 추가 → Dirichlet 에너지 경사 하강 형태
  --sheaf-cond none|scale     (v3) 퍼즐별 제한사상 채널 스케일 F_h(e)=diag(1+s_h(e))W_sh,h. ID 폭이 dk+K+H·c 로 자동 확장
  --encoding urm|frame   urm   = 원본 (EOS 가 출력의 오른쪽·아래에만)
                         frame = EOS 토큰을 입력·출력 격자의 4변 테두리에 (D4 대칭 부호화). 테두리는 감독, 나머지 PAD 는 무시
재개 시 두 값은 out-dir 의 arc_s.json 에서 읽으며 CLI 와 다르면 거부한다.

사용:
  python arc_analysis/train_arc_s.py --data data/arc_s10_aug250 --out-dir runs/arc_s10/base --model base --encoding frame --batch-size 128 ...
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lt"))
import train_arc                                   # noqa: E402
from train import LT as _LT                        # noqa: E402
from lt_d4 import D4LT                             # noqa: E402
from lt_color import ColorLT                       # noqa: E402
import lt_v3                                       # noqa: E402

CANVAS = 30
MODEL = "base"
ENCODING = "urm"
STDP = True
TRANSPORT, SHEAF = "adj", "none"
SIDECAR = "arc_s.json"


def canvas_of(data_root: Path) -> int:
    meta = json.loads((data_root / "train" / "dataset.json").read_text())
    S = math.isqrt(meta["seq_len"])
    if S * S != meta["seq_len"] or not 1 <= S <= 30:
        raise ValueError(f"seq_len={meta['seq_len']} is not a square canvas in 1..30")
    return S


class ARCSplitS(train_arc.ARCSplit):
    """train_arc.ARCSplit 와 같되 토큰 수 900 대신 CANVAS² 를 요구한다."""

    def __init__(self, root: Path, split: str):
        T = CANVAS * CANVAS
        self.root, self.split = root, split
        self.metadata = json.loads((root / split / "dataset.json").read_text())
        m = self.metadata
        if (m["seq_len"], m["vocab_size"], m["pad_id"], m["ignore_label_id"],
                m["blank_identifier_id"]) != (T, 12, 0, 0, 0):
            raise ValueError(f"Expected URM ARC encoding on canvas {CANVAS}: {T} tokens, vocab 12, PAD/ignore/blank ID 0")
        self.arrays = {}
        for name in m["sets"]:
            if Path(name).name != name:
                raise ValueError(f"Invalid set name: {name}")
            data = {field: np.load(root / split / f"{name}__{field}.npy", mmap_mode="r")
                    for field in train_arc.ARRAY_FIELDS}
            self._validate(data)
            self.arrays[name] = data

    def batch(self, data, example_indices, puzzle_indices, size=None):
        if ENCODING == "urm":
            return super().batch(data, example_indices, puzzle_indices, size)
        inputs = frame_encode(np.array(data["inputs"][example_indices], dtype=np.int64, copy=True))
        labels = frame_encode(np.array(data["labels"][example_indices], dtype=np.int64, copy=True))
        identifiers = np.array(data["puzzle_identifiers"][puzzle_indices], dtype=np.int64, copy=True)
        labels[labels == 0] = train_arc.IGNORE_LABEL_ID
        if size is not None and len(identifiers) < size:
            pad = size - len(identifiers)
            inputs = np.pad(inputs, ((0, pad), (0, 0)))
            labels = np.pad(labels, ((0, pad), (0, 0)), constant_values=train_arc.IGNORE_LABEL_ID)
            identifiers = np.pad(identifiers, (0, pad))
        import torch
        return {"inputs": torch.from_numpy(inputs), "labels": torch.from_numpy(labels),
                "puzzle_identifiers": torch.from_numpy(identifiers)}

    def _validate(self, data):
        inputs, labels = data["inputs"], data["labels"]
        pi, gi, ids = (data[k] for k in ("puzzle_indices", "group_indices", "puzzle_identifiers"))
        if inputs.ndim != 2 or inputs.shape != labels.shape or inputs.shape[1] != CANVAS * CANVAS:
            raise ValueError(f"ARC inputs/labels must both have shape [examples, {CANVAS * CANVAS}]")
        if pi.ndim != 1 or gi.ndim != 1 or ids.ndim != 1 or not len(ids):
            raise ValueError("Invalid or empty puzzle/group index arrays")
        if len(pi) != len(ids) + 1 or pi[0] != 0 or pi[-1] != len(inputs) or np.any(np.diff(pi) <= 0):
            raise ValueError("Puzzle indices must delimit nonempty contiguous example ranges")
        if len(gi) < 2 or gi[0] != 0 or gi[-1] != len(ids) or np.any(np.diff(gi) <= 0):
            raise ValueError("Group indices must delimit nonempty contiguous puzzle ranges")
        if ids.min() <= 0 or ids.max() >= self.metadata["num_puzzle_identifiers"]:
            raise ValueError("Puzzle identifier out of metadata range (0 is reserved for padding)")


def frame_encode(tokens):
    """URM 부호화(EOS 오른쪽·아래) → 4변 테두리 부호화. [B,T] int, 색 토큰 2..11 의 경계상자 둘레 한 칸을 EOS(1)로, 나머지 PAD.

    격자는 색 토큰으로 꽉 찬 직사각형이므로 경계상자 = 격자. 캔버스 밖으로 나가는 테두리는 없다.
    """
    S = CANVAS
    grid = tokens.reshape(-1, S, S)
    content = grid >= 2
    if np.any((tokens < 0) | (tokens >= 12)):
        raise ValueError("Invalid ARC token")
    out = np.where(content, grid, 0)
    for b in range(grid.shape[0]):
        rows = np.flatnonzero(content[b].any(1)); cols = np.flatnonzero(content[b].any(0))
        if not len(rows):
            continue
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        if not content[b, r0:r1 + 1, c0:c1 + 1].all():
            raise ValueError("Color tokens do not form a full rectangle")
        R0, R1, C0, C1 = max(r0 - 1, 0), min(r1 + 1, S - 1), max(c0 - 1, 0), min(c1 + 1, S - 1)
        ring = np.zeros((S, S), dtype=bool); ring[R0:R1 + 1, C0:C1 + 1] = True; ring[r0:r1 + 1, c0:c1 + 1] = False
        out[b][ring] = 1
    return out.reshape(tokens.shape)


def crop_grid(tokens):
    """train_arc.crop_grid 의 캔버스 S 판: (0,0) 에 붙은 가장 큰 유효 색 직사각형."""
    S = CANVAS
    grid = np.asarray(tokens).reshape(S, S)
    width, best_area, best = S, 0, (0, 0)
    for height in range(1, S + 1):
        for col in range(width):
            if grid[height - 1, col] < 2 or grid[height - 1, col] > 11:
                width = col
                break
        if height * width > best_area:
            best_area, best = height * width, (height, width)
    return (grid[:best[0], :best[1]] - 2).astype(np.uint8)


class LTS(_LT):
    """main() 이 seq_len=900, grid=30 으로 만든 cfg 를 캔버스 S 로 바꿔 받는다 (재개 cfg 는 이미 S·stdp)."""

    def __init__(self, config_dict: dict):
        super().__init__(dict(config_dict, seq_len=CANVAS * CANVAS, grid=CANVAS, stdp=config_dict.get("stdp", True) and STDP))


class D4LTS(D4LT):
    def __init__(self, config_dict: dict):
        super().__init__(dict(config_dict, seq_len=CANVAS * CANVAS, grid=CANVAS, stdp=config_dict.get("stdp", True) and STDP))


class V3LTS(lt_v3.LT):
    def __init__(self, config_dict: dict):
        dk, K, H = 128, 12, 8
        n_scale = H * (dk // H) if SHEAF == "scale" else 0
        super().__init__(dict(config_dict, seq_len=CANVAS * CANVAS, grid=CANVAS, **lt_v3.PRESETS_V3["v3"],
                              stdp=config_dict.get("stdp", True) and STDP, transport=TRANSPORT, sheaf_cond=SHEAF,
                              puzzle_emb_ndim=dk + K + n_scale))


class ColorLTS(ColorLT):
    def __init__(self, config_dict: dict):
        super().__init__(dict(config_dict, seq_len=CANVAS * CANVAS, grid=CANVAS,
                              slot_dim=config_dict.get("slot_dim", config_dict["hidden_size"] // 12)))


def install(S: int, model: str = "base", encoding: str = "urm", stdp: bool = True, transport: str = "adj", sheaf: str = "none"):
    global CANVAS, MODEL, ENCODING, STDP, TRANSPORT, SHEAF
    TRANSPORT, SHEAF = transport, sheaf
    if (transport != "adj" or sheaf != "none") and model != "v3":
        raise ValueError("--transport/--sheaf-cond are implemented for --model v3 only")
    if model not in ("base", "d4", "color", "v3") or encoding not in ("urm", "frame"):
        raise ValueError("model must be base|d4|color|v3 and encoding urm|frame")
    if model == "color" and not stdp:
        raise ValueError("color model has no --no-stdp variant")
    CANVAS, MODEL, ENCODING, STDP = S, model, encoding, stdp
    train_arc.ARCSplit = ARCSplitS
    train_arc.crop_grid = crop_grid
    train_arc.LT = {"base": LTS, "d4": D4LTS, "color": ColorLTS, "v3": V3LTS}[model]
    if not getattr(train_arc.evaluate, "_arc_s_wrapped", False):
        install_exact_recorder()
        train_arc.evaluate._arc_s_wrapped = True


class ExactRecorder:
    """evaluate() 안의 ARCVotes.update 에 끼어들어 정답과 완전히 일치한 예제를 모은다.
    결과는 metrics["exact_hits"] 로 넣어 run_eval 이 쓰는 eval_step_N_segS.json 에 함께 저장된다 (최대 MAX 개, 수는 exact_hit_count)."""
    hits = []
    MAX = 200

    @classmethod
    def wrap(cls, votes_cls):
        orig = votes_cls.update

        def update(self, batch, predictions):
            if "labels" not in batch:
                return orig(self, batch, predictions)
            mask = batch["labels"] != train_arc.IGNORE_LABEL_ID
            exact = ((predictions == batch["labels"]) | ~mask).all(-1) & mask.any(-1)
            for i in torch.nonzero(exact).flatten().tolist():
                ident = int(batch["puzzle_identifiers"][i])
                if ident and len(cls.hits) < cls.MAX:
                    cls.hits.append(dict(identifier=self.identifiers[ident], input=batch["inputs"][i].cpu().tolist(),
                                         prediction=predictions[i].cpu().tolist()))
            cls.count += int(exact.sum())
            return orig(self, batch, predictions)
        votes_cls.update = update


def install_exact_recorder():
    ExactRecorder.wrap(train_arc.ARCVotes)
    orig_eval = train_arc.evaluate

    def evaluate(*a, **k):
        ExactRecorder.hits, ExactRecorder.count = [], 0
        metrics, submission = orig_eval(*a, **k)
        metrics["exact_hit_count"] = ExactRecorder.count
        metrics["exact_hits"] = ExactRecorder.hits
        return metrics, submission
    evaluate._arc_s_wrapped = True
    train_arc.evaluate = evaluate


def split_args(argv):
    import argparse
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--model", choices=["base", "d4", "color", "v3"])
    extra.add_argument("--encoding", choices=["urm", "frame"])
    extra.add_argument("--no-stdp", action="store_true")
    extra.add_argument("--transport", choices=["adj", "laplacian"])
    extra.add_argument("--sheaf-cond", choices=["none", "scale"])
    own, rest = extra.parse_known_args(argv)
    return own, rest


def main(argv=None):
    own, rest = split_args(sys.argv[1:] if argv is None else argv)
    args = train_arc.parse_args(rest)
    S = canvas_of(args.data.resolve())
    sidecar = args.out_dir / SIDECAR
    if args.resume is not None:
        saved = json.loads((Path(args.resume).parent / SIDECAR).read_text())
        for key in ("model", "encoding"):
            if getattr(own, key) is not None and getattr(own, key) != saved[key]:
                raise ValueError(f"--{key} {getattr(own, key)} conflicts with checkpoint's {saved[key]}")
        if saved["canvas"] != S:
            raise ValueError(f"checkpoint canvas {saved['canvas']} != data canvas {S}")
        for key in ("transport", "sheaf_cond"):
            if getattr(own, key) is not None and getattr(own, key) != saved.get(key, "adj" if key == "transport" else "none"):
                raise ValueError(f"--{key} conflicts with checkpoint")
        model, encoding, stdp = saved["model"], saved["encoding"], saved.get("stdp", True)
        transport, sheaf = saved.get("transport", "adj"), saved.get("sheaf_cond", "none")
    else:
        model, encoding, stdp = own.model or "base", own.encoding or "urm", not own.no_stdp
        transport, sheaf = own.transport or "adj", own.sheaf_cond or "none"
    if args.resume is None and model == "d4" and args.num_heads != 8:
        raise ValueError("--model d4 requires --num-heads 8")
    if args.resume is None and model == "v3" and (args.num_heads != 8 or args.hidden_size != 12 * 128):
        raise ValueError("--model v3 requires --num-heads 8 --hidden-size 1536 (puzzle-emb-dim is set automatically: 140, +128 with --sheaf-cond scale)")
    if args.resume is None and model == "color" and (args.num_heads != 8 or args.hidden_size % 12 or (args.hidden_size // 12) % 8):
        raise ValueError("--model color requires --num-heads 8 and --hidden-size = 12 × slot_dim with slot_dim divisible by 8 (e.g. 768)")
    install(S, model, encoding, stdp, transport, sheaf)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None or not sidecar.exists():
        sidecar.write_text(json.dumps(dict(canvas=S, model=model, encoding=encoding, stdp=stdp, transport=transport, sheaf_cond=sheaf)) + "\n")
    print(f"[ARC-S] canvas {CANVAS}×{CANVAS} = {CANVAS * CANVAS} tokens; model={MODEL} encoding={ENCODING} stdp={STDP}; coupling carry "
          f"{args.batch_size * args.num_heads * CANVAS ** 4 * 4 / 2 ** 20:.1f} MiB (train_arc's own line below assumes 900)", flush=True)
    train_arc.main(rest)


if __name__ == "__main__":
    main()
