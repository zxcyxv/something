"""평가 JSON 에 기록된 완전 일치 예제(exact_hits)를 그림으로. 예측 = 정답이므로 입력 / 예측 두 칸.
사용: python arc_analysis/show_exact_hits.py runs/arc_s10/d4_aug1000/eval_step_8000_seg16.json [--max 12] [--png out.png]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from show_predictions import save_png, render, side_by_side   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("eval", type=Path); ap.add_argument("--max", type=int, default=12); ap.add_argument("--png", type=Path)
    a = ap.parse_args()
    j = json.loads(a.eval.read_text())
    hits = j.get("exact_hits", []); S = int(round(len(hits[0]["input"]) ** 0.5)) if hits else 0
    print(f"{a.eval}: step {j.get('step')}, exact {j.get('exact_hit_count', '?')} / {j.get('evaluated_augmented_examples')} 예제, "
          f"기록 {len(hits)}개; 과제 종류 {len({h['identifier'].split('|||')[0] for h in hits})}")
    figs = []
    seen = set()
    for h in hits:
        task = h["identifier"].split("|||")[0]
        if task in seen:
            continue                                   # 과제당 1개
        seen.add(task)
        print(f"\n[{h['identifier']}]"); print(side_by_side([["input"] + render(h["input"], S), ["prediction = target"] + render(h["prediction"], S)]))
        figs.append((task, [("input", np.array(h["input"]).reshape(S, S)), ("prediction = target", np.array(h["prediction"]).reshape(S, S))]))
        if len(figs) >= a.max:
            break
    if a.png and figs:
        save_png(figs, a.png, f"{a.eval.parent.name} · step {j.get('step')} · exact hits"); print(f"\n그림: {a.png}")


if __name__ == "__main__":
    main()
