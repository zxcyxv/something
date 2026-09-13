"""In-context 프로토콜 학습 CLI. 설계: protocol_context.md.  모델: lt_ctx.py, 데이터: ctx_data.py.

`lt/train_arc.py` 의 main(학습 루프·EMA·AdamATan2·체크포인트·재개·평가 스케줄)을 그대로 실행하되
데이터 클래스·샘플러·모델·옵티마이저·평가 함수를 in-context 판으로 바꿔 끼운다. lt/ 는 수정하지 않는다.

추가 인자 (나머지는 train_arc.py 와 동일; --max-steps 는 항상 주는 것을 권장):
  --model base|d4        기존 LT 커널 / D4 등변
  --k-demos K            시범 쌍 수 (슬롯 = 2(K+1)). 기본 2
  --mask-prob p          학습 시 확률 p 로 시범 출력 슬롯 하나를 가리고 함께 감독. 기본 0
  --transductive (기본) | --inductive     evaluation 과제의 시범을 학습에 포함(URM 과 같음, ID 행 학습) / 제외
  --no-puzzle-id         퍼즐 ID 임베딩 제거 (시범만으로 규칙을 읽어야 함). 기본은 ID 유지 (모든 슬롯에 더함)
슬롯 위상 σ 는 ψ 와 같이 [−π, π] 균등 초기화 (첫 런의 0 초기화는 20k 뒤에도 움직이지 않았음).
재개 시 위 값은 out-dir 의 arc_ctx.json 에서 읽고 CLI 와 다르면 거부.

사용:
  python arc_analysis/train_arc_ctx.py --data data/arc_s10_aug8 --out-dir runs/arc_s10_ctx/d4_k2 --model d4 --k-demos 2 \\
      --batch-size 16 --max-steps 20000 --eval-every 2000 --eval-max-augmentations 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "lt"))
import train_arc                                                     # noqa: E402
import train_arc_s                                                   # noqa: E402
import ctx_data                                                      # noqa: E402
from lt_ctx import LTCtx, D4LTCtx                                    # noqa: E402
from train import _EMASwap, IGNORE_LABEL_ID, AdamATan2, _is_no_decay   # noqa: E402
_orig_create_optimizers = train_arc.create_optimizers

SIDECAR = "arc_ctx.json"
DEFAULTS = dict(canvas=10, model="base", k_demos=2, mask_prob=0.0, inductive=False, puzzle_id=True)
SETTINGS = dict(DEFAULTS)


def make_lt_class():
    S, n = SETTINGS["canvas"], 2 * (SETTINGS["k_demos"] + 1)
    inner = {"base": LTCtx, "d4": D4LTCtx}[SETTINGS["model"]]

    class LTCtxS(inner):
        def __init__(self, config_dict: dict):
            super().__init__(dict(config_dict, seq_len=n * S * S, grid=S, n_slots=n, use_puzzle_id=SETTINGS["puzzle_id"]))
    return LTCtxS


def create_optimizers(model, cfg, world_size):
    """ID 임베딩이 없으면 sparse SignSGD 없이 AdamATan2 하나 (wd 그룹 분리는 원본 규칙)."""
    if SETTINGS["puzzle_id"]:
        return _orig_create_optimizers(model, cfg, world_size)
    opt = AdamATan2(
        [{"params": [p for n, p in model.named_parameters() if _is_no_decay(n, p)], "weight_decay": 0.0},
         {"params": [p for n, p in model.named_parameters() if not _is_no_decay(n, p)], "weight_decay": cfg["weight_decay"]}],
        lr=0, weight_decay=cfg["weight_decay"], betas=(cfg["beta1"], cfg["beta2"]))
    return [opt], [cfg["lr"]]


@torch.inference_mode()
def evaluate(base, ema, dataset, identifiers, tasks, args, device):
    selected = {name: tasks[name] for name in sorted(tasks)[:args.eval_max_tasks or None]}
    if not selected:
        raise ValueError("No evaluation tasks")
    lt = base.model
    loops = lt.config.loops
    was_training = base.training
    lt.config.loops = args.eval_segments or loops
    votes = train_arc.ARCVotes(identifiers, selected)
    rec = train_arc_s.ExactRecorder
    rec.hits, rec.count = [], 0
    qi, qo = dataset.query_slots()
    totals = Counter()
    started = time.monotonic()
    with _EMASwap(base, ema):
        base.eval()
        try:
            for index, batch in enumerate(ctx_data.eval_batches(dataset, identifiers, selected, args.batch_size,
                                                                args.eval_max_augmentations)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.device(device):
                    carry = lt.initial_carry(batch)
                for _ in range(lt.config.loops):
                    carry, out = lt(carry, batch)
                pred = out["logits"].argmax(-1)[:, qo]
                labels = batch["labels"][:, qo]
                mask = labels != IGNORE_LABEL_ID
                valid = mask.any(-1)
                correct = (pred == labels) & mask
                totals["examples"] += int(valid.sum())
                totals["accuracy"] += float((correct.sum(-1) / mask.sum(-1).clamp_min(1))[valid].sum())
                totals["exact"] += int(((correct.sum(-1) == mask.sum(-1)) & valid).sum())
                votes.update({"puzzle_identifiers": batch["puzzle_identifiers"], "inputs": batch["inputs"][:, qi],
                              "labels": labels}, pred)
                if (index + 1) % 100 == 0:
                    print(f"[ARC eval] {index + 1} batches, {time.monotonic() - started:.1f}s", flush=True)
        finally:
            lt.config.loops = loops
            base.train(was_training)
    metrics, submission = votes.result()
    metrics.update(token_accuracy=totals["accuracy"] / max(totals["examples"], 1),
                   sequence_exact=totals["exact"] / max(totals["examples"], 1),
                   evaluated_augmented_examples=totals["examples"], eval_segments=args.eval_segments or loops,
                   max_augmentations_per_task=args.eval_max_augmentations, total_available_tasks=len(tasks),
                   seconds=time.monotonic() - started, protocol="in-context",
                   k_demos=SETTINGS["k_demos"], inductive=SETTINGS["inductive"], puzzle_id=SETTINGS["puzzle_id"],
                   exact_hit_count=rec.count, exact_hits=rec.hits)
    return metrics, submission


def install():
    train_arc_s.install(SETTINGS["canvas"], "base", "frame")            # crop_grid·frame_encode·ExactRecorder (S)
    train_arc.ARCSplit = lambda root, split: ctx_data.CtxSplit(root, split, SETTINGS["k_demos"], SETTINGS["inductive"])
    train_arc.ARCTrainStream = lambda ds, bs, seed, epi: ctx_data.CtxStream(ds, bs, seed, epi, SETTINGS["mask_prob"])
    train_arc.LT = make_lt_class()
    train_arc.create_optimizers = create_optimizers
    train_arc.evaluate = evaluate


def split_args(argv):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--model", choices=["base", "d4"])
    extra.add_argument("--k-demos", type=int)
    extra.add_argument("--mask-prob", type=float)
    grp = extra.add_mutually_exclusive_group()
    grp.add_argument("--inductive", dest="inductive", action="store_true", default=None)
    grp.add_argument("--transductive", dest="inductive", action="store_false")
    extra.add_argument("--no-puzzle-id", dest="puzzle_id", action="store_false", default=None)
    own, rest = extra.parse_known_args(argv)
    return own, rest


def main(argv=None):
    own, rest = split_args(sys.argv[1:] if argv is None else argv)
    args = train_arc.parse_args(rest)
    S = train_arc_s.canvas_of(args.data.resolve())
    given = dict(canvas=S, model=own.model, k_demos=own.k_demos, mask_prob=own.mask_prob, inductive=own.inductive, puzzle_id=own.puzzle_id)
    if args.resume is not None:
        saved = json.loads((Path(args.resume).parent / SIDECAR).read_text())
        for key, value in given.items():
            if value is not None and value != saved[key]:
                raise ValueError(f"--{key} {value} conflicts with checkpoint's {saved[key]}")
        SETTINGS.clear(); SETTINGS.update(DEFAULTS); SETTINGS.update(saved)
    else:
        SETTINGS.clear(); SETTINGS.update(DEFAULTS)
        SETTINGS.update(canvas=S, **{k: v for k, v in given.items() if v is not None and k != "canvas"})
    if SETTINGS["model"] == "d4" and args.num_heads != 8:
        raise ValueError("--model d4 requires --num-heads 8")
    if SETTINGS["k_demos"] < 1 or not 0 <= SETTINGS["mask_prob"] <= 1:
        raise ValueError("--k-demos must be >= 1 and --mask-prob in [0,1]")
    install()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = args.out_dir / SIDECAR
    if args.resume is None or not sidecar.exists():
        sidecar.write_text(json.dumps(SETTINGS) + "\n")
    n = 2 * (SETTINGS["k_demos"] + 1)
    print(f"[ARC-ctx] canvas {S} × {n} slots = {n * S * S} tokens; {SETTINGS}; coupling carry "
          f"{args.batch_size * args.num_heads * (n * S * S) ** 2 * 4 / 2 ** 20:.1f} MiB (train_arc's own lines below assume 900)", flush=True)
    train_arc.main(rest)


if __name__ == "__main__":
    main()
