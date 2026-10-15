"""Kaggle Dataset 업로드용 최소 데이터 생성 (로컬에서 1회 실행).

무엇을 만드는가
    `sudoku-extreme` 의 train.csv 에서 **1,000개를 서브샘플**하고, test.csv 의 **앞 2,048개**를 잘라
    하나의 `sudoku_lt_1k.npz` 로 저장한다. 증강 1,000배는 업로드하지 않는다 —
    `lt/data.py` 가 학습 중에 같은 규칙으로 즉석 생성한다 (원본 1.3GB → 여기서는 ~130KB).

무엇을 그대로 따랐는가 (`refs/URM/data/build_sudoku_dataset.py`)
    - 시드: `np.random.seed(int(os.environ.get("LT_SEED", "0")))` 를 **train 변환 직전에 1회**.
      원본도 `preprocess_data` 진입 시 1회 seed 후 train 을 먼저 처리하므로,
      `np.random.choice(total, 1000, replace=False)` 가 소비하는 난수 스트림이 동일하다.
      → 뽑히는 1,000개가 원본 빌드와 **완전히 같음을 실측 확인**(2026-09-04, `all__inputs.npy` 의
        group 시작 행 1,000개와 bit-exact 일치).
    - CSV 파싱: `q` 의 '.' → '0', `np.frombuffer(...) - ord('0')` → 9×9 uint8 (0=빈칸, 1..9=숫자).
    - test 는 서브샘플도 증강도 하지 않는다 (`num_augments = num_aug if set_name=="train" else 0`).
      앞 2,048개 절단은 `sudoku/truncate_test.py` 와 동일한 슬라이스(그룹=퍼즐=예제 1:1:1 구조라 앞에서 자르면 그만).
    - **+1 오프셋은 여기서 하지 않는다.** 원본 `_seq_to_numpy` 의 `arr + 1` 은 학습 코드에서 적용한다
      (증강은 0=빈칸 좌표계에서 돌아야 자릿수 치환 `digit_map` 이 원본과 같은 식이 된다).

사용
    python data/prep_dataset.py                      # 기본: HF 캐시/로컬 CSV 자동 탐색
    python data/prep_dataset.py --out /tmp/upload    # 업로드 디렉터리 지정
    LT_SEED=0 python data/prep_dataset.py            # 시드는 원본과 같은 env 이름
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np


def find_csv(name: str, explicit: str | None) -> str:
    """CSV 탐색 순서: 명시 경로 → 저장소 로컬(sudoku_csv) → HF 허브 캐시 → HF 다운로드."""
    if explicit:
        return explicit
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local = os.path.join(root, "refs", "URM", "data", "sudoku_csv", f"{name}.csv")
    if os.path.exists(local):
        return local
    for cache in (os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface"),
                  "/workspace/.cache/huggingface"):
        if not cache:
            continue
        hits = glob.glob(os.path.join(cache, "**", f"datasets--sapientinc--sudoku-extreme",
                                      "snapshots", "*", f"{name}.csv"), recursive=True)
        if hits:
            return sorted(hits)[-1]
    from huggingface_hub import hf_hub_download   # 마지막 수단 (인터넷 필요)
    return hf_hub_download("sapientinc/sudoku-extreme", f"{name}.csv", repo_type="dataset")


def read_csv(path: str, min_difficulty: int | None = None):
    """원본 `convert_subset` 의 CSV 파싱을 그대로 옮긴 것."""
    inputs, labels = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)   # 헤더
        for source, q, a, rating in reader:
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
    return inputs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload"))
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--subsample", type=int, default=1000)
    ap.add_argument("--test-size", type=int, default=2048)
    ap.add_argument("--min-difficulty", type=int, default=None)
    args = ap.parse_args()

    seed = int(os.environ.get("LT_SEED", "0"))

    train_csv = find_csv("train", args.train_csv)
    test_csv = find_csv("test", args.test_csv)
    print(f"[data] train: {train_csv}", flush=True)
    print(f"[data] test : {test_csv}", flush=True)

    # ---- train: seed 후 곧바로 서브샘플 (원본의 난수 소비 순서와 동일)
    tr_in, tr_lb = read_csv(train_csv, args.min_difficulty)
    print(f"[data] train 전체 {len(tr_in)}행", flush=True)
    np.random.seed(seed)
    total = len(tr_in)
    if args.subsample < total:
        idx = np.random.choice(total, size=args.subsample, replace=False)
    else:
        idx = np.arange(total)
    train_inputs = np.stack([tr_in[i] for i in idx]).astype(np.uint8)     # [1000,9,9] 0..9
    train_labels = np.stack([tr_lb[i] for i in idx]).astype(np.uint8)

    # ---- test: 절단만
    te_in, te_lb = read_csv(test_csv, args.min_difficulty)
    print(f"[data] test 전체 {len(te_in)}행 → 앞 {args.test_size}개", flush=True)
    test_inputs = np.stack(te_in[: args.test_size]).astype(np.uint8)
    test_labels = np.stack(te_lb[: args.test_size]).astype(np.uint8)

    # ---- 무결성: 값 범위, 입력 단서가 정답과 일치, 정답판이 유효한 스도쿠
    for a in (train_inputs, train_labels, test_inputs, test_labels):
        assert a.min() >= 0 and a.max() <= 9
    for inp, lab in ((train_inputs, train_labels), (test_inputs, test_labels)):
        assert np.all((inp == 0) | (inp == lab)), "입력 단서가 정답과 불일치"
        assert np.all(np.sort(lab.reshape(-1, 9, 9), axis=2) == np.arange(1, 10)), "정답 행이 1..9 순열 아님"

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "sudoku_lt_1k.npz")
    np.savez_compressed(npz, train_inputs=train_inputs, train_labels=train_labels,
                        test_inputs=test_inputs, test_labels=test_labels)

    meta = {
        "source_repo": "sapientinc/sudoku-extreme",
        "lt_seed": seed,
        "subsample_size": int(args.subsample),
        "test_size": int(args.test_size),
        "train_total_rows": int(total),
        "shapes": {k: list(v.shape) for k, v in
                   dict(train_inputs=train_inputs, train_labels=train_labels,
                        test_inputs=test_inputs, test_labels=test_labels).items()},
        "encoding": "uint8 9x9, 0=빈칸, 1..9=숫자 (학습 코드가 +1 하여 vocab 0..10 으로 씀)",
        "sha256_npz": hashlib.sha256(open(npz, "rb").read()).hexdigest(),
        "note": "증강(9!·전치·밴드/스택)은 train_kaggle.py 가 학습 중 생성. test 는 증강하지 않음.",
    }
    with open(os.path.join(args.out, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[out] {npz}  {os.path.getsize(npz)/1024:.1f} KB")
    print(f"[out] {os.path.join(args.out, 'dataset_meta.json')}")
    print("\n업로드:  kaggle datasets create -p %s   (dataset-metadata.json 필요, 예외사항.md §1 참조)" % args.out)


if __name__ == "__main__":
    main()
