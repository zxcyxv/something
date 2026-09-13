"""In-context 프로토콜의 데이터: builder 출력(URM 포맷, 캔버스 S)에서 (K 시범 쌍 + 질의) 시퀀스를 만든다.

학습 샘플 = 증강 퍼즐 하나에서 예제 K+1 개 (leave-one-out: 하나가 질의). 예제가 K+1 보다 적으면 빈 시범 슬롯은 PAD.
평가 샘플 = test split 의 질의 예제 + 같은 식별자의 train split 예제(시범)를 앞 K 개.
격자는 슬롯마다 frame 부호화(4변 테두리). 라벨은 질의 출력 슬롯만 감독 (선택: 가린 시범 출력 슬롯도).
inductive 면 evaluation 과제(test_puzzles.json 의 이름)의 퍼즐을 학습에서 뺀다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_arc                                   # noqa: E402
import train_arc_s                                 # noqa: E402
from lt_ctx import ROLE_DEMO_OUT, slot_roles       # noqa: E402

IGN = train_arc.IGNORE_LABEL_ID


class CtxSplit:
    """train_arc_s.ARCSplitS 를 감싸, 슬롯 시퀀스를 만드는 데 필요한 색인을 붙인다."""

    def __init__(self, root: Path, split: str, k_demos: int, inductive: bool):
        self.root, self.split, self.k = root, split, k_demos
        self.n_slots = 2 * (k_demos + 1)
        self.base = train_arc_s.ARCSplitS(root, split)
        self.metadata, self.arrays = self.base.metadata, self.base.arrays
        self.S = train_arc_s.CANVAS
        self.T = self.n_slots * self.S * self.S
        self.identifiers = json.loads((root / "identifiers.json").read_text())
        eval_tasks = set(json.loads((root / "test_puzzles.json").read_text()))
        # 학습에 쓸 그룹: inductive 면 evaluation 과제 그룹 제외
        self.allowed_groups = {}
        for name, data in self.arrays.items():
            gi, ids = data["group_indices"], data["puzzle_identifiers"]
            groups = []
            for g in range(len(gi) - 1):
                task = self.identifiers[int(ids[gi[g]])].split("|||")[0]
                if not (inductive and split == "train" and task in eval_tasks):
                    groups.append(g)
            self.allowed_groups[name] = np.array(groups, dtype=np.int64)
        if split == "test":                          # 시범은 train split 에서 같은 식별자로
            train = train_arc_s.ARCSplitS(root, "train")
            self.demo_index = {}
            for data in train.arrays.values():
                pi, ids = data["puzzle_indices"], data["puzzle_identifiers"]
                for p, ident in enumerate(ids):
                    self.demo_index[int(ident)] = (data, int(pi[p]), int(pi[p + 1]))

    # ---- 시퀀스 조립
    def assemble(self, demo_pairs, query_in, query_out=None, masked_demo=None):
        """demo_pairs: [(in,out)] 캔버스 토큰 (URM 부호화). 반환 inputs, labels [T]."""
        S2 = self.S * self.S
        inputs = np.zeros(self.T, dtype=np.int64)
        labels = np.full(self.T, IGN, dtype=np.int64)
        fe = lambda g: train_arc_s.frame_encode(np.asarray(g, dtype=np.int64)[None])[0]
        for k, (x, y) in enumerate(demo_pairs[: self.k]):
            inputs[(2 * k) * S2:(2 * k + 1) * S2] = fe(x)
            out = fe(y)
            if masked_demo == k:
                lab = out.copy(); lab[lab == 0] = IGN
                labels[(2 * k + 1) * S2:(2 * k + 2) * S2] = lab
            else:
                inputs[(2 * k + 1) * S2:(2 * k + 2) * S2] = out
        q = 2 * self.k
        inputs[q * S2:(q + 1) * S2] = fe(query_in)
        if query_out is not None:
            lab = fe(query_out); lab[lab == 0] = IGN
            labels[(q + 1) * S2:(q + 2) * S2] = lab
        return inputs, labels

    def query_slots(self):
        S2 = self.S * self.S
        q = 2 * self.k
        return slice(q * S2, (q + 1) * S2), slice((q + 1) * S2, (q + 2) * S2)

    def to_batch(self, seqs, ids, size=None):
        inputs = np.stack([s[0] for s in seqs]); labels = np.stack([s[1] for s in seqs])
        ids = np.asarray(ids, dtype=np.int64)
        if size is not None and len(ids) < size:
            pad = size - len(ids)
            inputs = np.pad(inputs, ((0, pad), (0, 0))); labels = np.pad(labels, ((0, pad), (0, 0)), constant_values=IGN)
            ids = np.pad(ids, (0, pad))
        return {"inputs": torch.from_numpy(inputs), "labels": torch.from_numpy(labels), "puzzle_identifiers": torch.from_numpy(ids)}


class CtxStream(train_arc.ARCTrainStream):
    """그룹 → 증강 퍼즐 → 예제 K+1 개(하나가 질의) 샘플러. 상태 저장/복원은 부모와 같다."""

    def __init__(self, dataset: CtxSplit, batch_size, seed, epochs_per_iter, mask_prob=0.0):
        self.dataset, self.batch_size = dataset, batch_size
        self.seed, self.epochs_per_iter, self.mask_prob = seed, epochs_per_iter, mask_prob
        self.names = list(dataset.arrays)
        self.iteration, self.set_index, self.position = 0, -1, 0
        self.order = np.empty(0, dtype=np.int64)
        self.rng = np.random.Generator(np.random.Philox(seed))
        if min(len(g) for g in dataset.allowed_groups.values()) * epochs_per_iter < batch_size:
            raise ValueError("Too few groups for one full batch; increase --epochs-per-iter")

    def _begin_set(self):
        self.iteration += 1
        self.set_index = (self.set_index + 1) % len(self.names)
        groups = self.dataset.allowed_groups[self.names[self.set_index]]
        self.rng = np.random.Generator(np.random.Philox(self.seed + self.iteration))
        self.order = np.concatenate([groups[self.rng.permutation(len(groups))] for _ in range(self.epochs_per_iter)])
        self.position = 0

    def __next__(self):
        for _ in range(len(self.names) + 2):
            if self.position >= len(self.order):
                self._begin_set()
            data = self.dataset.arrays[self.names[self.set_index]]
            pi, gi = data["puzzle_indices"], data["group_indices"]
            seqs, ids = [], []
            while self.position < len(self.order) and len(seqs) < self.batch_size:
                group = self.order[self.position]; self.position += 1
                puzzle = int(self.rng.integers(gi[group], gi[group + 1]))
                start, end = int(pi[puzzle]), int(pi[puzzle + 1])
                if end - start < 2:
                    continue
                chosen = start + self.rng.permutation(end - start)[: self.dataset.k + 1]
                q, demos = int(chosen[0]), chosen[1:]
                pairs = [(data["inputs"][e], data["labels"][e]) for e in demos]
                masked = int(self.rng.integers(len(pairs))) if pairs and self.rng.random() < self.mask_prob else None
                seqs.append(self.dataset.assemble(pairs, data["inputs"][q], data["labels"][q], masked))
                ids.append(int(data["puzzle_identifiers"][puzzle]))
            if len(seqs) == self.batch_size:
                return self.dataset.to_batch(seqs, ids)
        raise ValueError("No full training batch")


def eval_batches(dataset: CtxSplit, identifiers, tasks, batch_size, max_augmentations):
    """train_arc.eval_batches 와 같은 순서·상한. 질의마다 시퀀스 하나."""
    from collections import Counter
    for data in dataset.arrays.values():
        seqs, ids = [], []
        counts = Counter()
        for puzzle, identifier in enumerate(data["puzzle_identifiers"]):
            name = identifiers[int(identifier)].split("|||")[0]
            if name not in tasks or (max_augmentations and counts[name] >= max_augmentations):
                continue
            counts[name] += 1
            demo = dataset.demo_index.get(int(identifier))
            pairs = [(demo[0]["inputs"][e], demo[0]["labels"][e]) for e in range(demo[1], demo[2])] if demo else []
            start, end = data["puzzle_indices"][puzzle:puzzle + 2]
            for example in range(int(start), int(end)):
                seqs.append(dataset.assemble(pairs, data["inputs"][example], data["labels"][example]))
                ids.append(int(identifier))
                if len(seqs) == batch_size:
                    yield dataset.to_batch(seqs, ids, batch_size); seqs, ids = [], []
        if seqs:
            yield dataset.to_batch(seqs, ids, batch_size)
