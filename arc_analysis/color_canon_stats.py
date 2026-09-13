"""색 정규화(canonicalization) 가능성: 과제 시범 예제의 색 빈도 순위로 색을 재명명하면 색 치환 증강을 대체할 수 있는가.

과제별로 (시범 입력+출력) 색 빈도를 세어 순위를 매긴다. 검정(0)은 URM 과 같이 고정.
  · 동률: 빈도가 같은 색 쌍이 있으면 순위가 모호 → 그 과제는 tie-break 규칙(첫 등장 위치 등)이나 소수의 증강이 필요
  · 질의 미등장 색: 시범에 없던 색이 테스트 입력에 나타나면 정규화 사상이 정의되지 않음 → 남은 순위를 등장 순으로 부여
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/tmp/URM/kaggle/combined/arc-agi")
    ap.add_argument("--subsets", nargs="+", default=["training", "evaluation", "concept"])
    args = ap.parse_args()
    print(f"{'subset':12s} {'tasks':>5s} {'동률 있음':>8s} {'동률(상위3 색 안)':>14s} {'질의에 새 색':>10s} {'시범 색 수 평균':>10s}")
    for subset in args.subsets:
        ch = json.loads(Path(f"{args.prefix}_{subset}-challenges.json").read_text())
        n = len(ch); ties = top_ties = unseen = 0; ncolors = []
        for name, task in ch.items():
            cnt = Counter()
            for pair in task["train"]:
                for g in (pair["input"], pair["output"]):
                    cnt.update(np.asarray(g).ravel().tolist())
            cnt.pop(0, None)
            ranked = sorted(cnt.items(), key=lambda kv: -kv[1])
            ncolors.append(len(ranked))
            freqs = [f for _, f in ranked]
            if len(freqs) != len(set(freqs)):
                ties += 1
                if len(set(freqs[:3])) != min(3, len(freqs)):
                    top_ties += 1
            seen = set(cnt)
            if any(set(np.asarray(pair["input"]).ravel().tolist()) - seen - {0} for pair in task["test"]):
                unseen += 1
        print(f"{subset:12s} {n:5d} {ties:4d}({100*ties/n:3.0f}%) {top_ties:8d}({100*top_ties/n:3.0f}%) {unseen:7d}({100*unseen/n:3.0f}%) {np.mean(ncolors):10.2f}")


if __name__ == "__main__":
    main()
