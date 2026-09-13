"""LT 의 대칭성 수치 검사 (무작위 초기화, CPU fp32, 2 세그먼트).

  1. 기존 LT   : 평행이동 — 격자를 캔버스 안에서 옮겼을 때 격자 칸의 로짓이 얼마나 변하는가 (캔버스 경계 효과만 남아야 함)
  2. 기존 LT   : D4       — 캔버스 전체를 회전·반사했을 때 로짓이 따라오는가 (θ 가 방향을 가지므로 안 따라온다)
  3. D4LT 시제품: D4      — 입력·퍼즐 임베딩을 함께 변환하면 로짓이 정확히 따라오는가 (fp32 오차 수준이어야 함)
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lt")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LT_Inner, LTConfig, LTCarry, PRESETS          # noqa: E402
import d4                                                        # noqa: E402
from lt_d4 import D4Inner                                        # noqa: E402

torch.manual_seed(0); np.random.seed(0)
S, D, H, SEGS, BLOCKS = 12, 128, 8, 2, 3
VOCAB = 12


def make_cfg(**kw):
    base = dict(batch_size=1, seq_len=S * S, grid=S, vocab_size=VOCAB, num_puzzle_identifiers=4,
                puzzle_emb_ndim=D, hidden_size=D, num_heads=H, loops=SEGS, blocks_per_seg=BLOCKS,
                amp=False, forward_dtype="float32", **PRESETS["v1.1"])
    base.update(kw)
    return LTConfig.from_dict(base)


def run(model, tokens, pid=1):
    batch = {"inputs": tokens.view(1, -1), "puzzle_identifiers": torch.tensor([pid])}
    carry = LTCarry(current_hidden=model.init_hidden.expand(1, S * S, -1).clone())
    with torch.no_grad():
        for _ in range(SEGS):
            carry, logits = model(carry, batch)
    return logits[0], carry.coupling[0]


def random_canvas(h, w, r=0, c=0, colors=None):
    """h×w 격자(색 2..11)를 (r,c) 에 놓고 나머지는 PAD=0. EOS 없음 (대칭 부호화)."""
    canvas = torch.zeros(S, S, dtype=torch.long)
    grid = torch.randint(2, VOCAB, (h, w)) if colors is None else colors
    canvas[r:r + h, c:c + w] = grid
    return canvas, grid


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def randomize_(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "down" in n:
                p.normal_(0, 0.05)
        for n, b in model.named_buffers():
            if n.endswith("puzzle_emb.weights"):
                b.normal_(0, 0.05)


print(f"canvas {S}×{S}, d={D}, heads={H}, {SEGS} seg × {BLOCKS} blocks, v1.1 preset, fp32 CPU\n")

# ── 1·2. 기존 LT
base = LT_Inner(make_cfg()); randomize_(base); base.eval()
gh, gw = 5, 4
canvas0, grid = random_canvas(gh, gw, 0, 0)
lo0, _ = run(base, canvas0)
print("[1] 기존 LT 평행이동 (격자 5×4, 격자 칸 로짓의 상대 오차; 0 이면 정확한 등변)")
for r, c in [(0, 1), (1, 0), (2, 3), (S - gh, S - gw)]:
    canvas1, _ = random_canvas(gh, gw, r, c, grid)
    lo1, _ = run(base, canvas1)
    a = lo0.view(S, S, -1)[:gh, :gw]; b = lo1.view(S, S, -1)[r:r + gh, c:c + gw]
    print(f"   offset ({r},{c}):  rel err = {rel(b, a):.3e}   argmax 일치율 = {(a.argmax(-1) == b.argmax(-1)).float().mean():.2f}")
print("   (참고) 거리 감쇠 α 를 크게 하면 경계 효과가 줄어드는가:")
with torch.no_grad():
    base.layers[0].alpha_raw.fill_(2.0)
lo0, _ = run(base, canvas0)
canvas1, _ = random_canvas(gh, gw, 2, 3, grid); lo1, _ = run(base, canvas1)
a = lo0.view(S, S, -1)[:gh, :gw]; b = lo1.view(S, S, -1)[2:2 + gh, 3:3 + gw]
print(f"   α=softplus(2.0)≈2.1, offset (2,3):  rel err = {rel(b, a):.3e}")
with torch.no_grad():
    base.layers[0].alpha_raw.fill_(float(np.log(np.expm1(0.1))))

print("\n[2] 기존 LT D4 (캔버스 전체 변환, 전체 로짓 상대 오차)")
canvasF, _ = random_canvas(7, 6, 2, 3)
loF, wF = run(base, canvasF)
for g in range(1, d4.ORDER):
    lo_g, _ = run(base, d4.act_field(canvasF.view(1, -1), S, g).view(-1))
    expect = d4.act_field(loF.unsqueeze(0), S, g)[0]
    print(f"   g={g} M={d4.MATS[g].tolist()}:  rel err = {rel(lo_g, expect):.3e}")

# ── 3. D4LT
print("\n[3] D4LT 시제품 (입력 캔버스 + 퍼즐 임베딩 정칙표현을 함께 변환)")
eq = D4Inner(make_cfg()); randomize_(eq); eq.eval()
loE, wE = run(eq, canvasF, pid=1)
for g in range(1, d4.ORDER):
    with torch.no_grad():                    # ID 2 ← ID 1 행의 슬라이스 순열 (g·e)[k] = e[g⁻¹k]
        eq.puzzle_emb.weights[2] = d4.act_slices(eq.puzzle_emb.weights[1], g, eq.dh)
    lo_g, w_g = run(eq, d4.act_field(canvasF.view(1, -1), S, g).view(-1), pid=2)
    expect = d4.act_field(loE.unsqueeze(0), S, g)[0]
    # 결합 기억 w[m,t,n] 은 헤드가 m → g·m 으로 재명명되고 위치는 (g·t, g·n)
    wperm = torch.tensor([d4.MUL[g, m] for m in range(d4.ORDER)])
    w_exp = d4.act_field(d4.act_field(wE.index_select(0, torch.argsort(wperm)).unsqueeze(0), S, g, axis=2), S, g, axis=3)[0]
    print(f"   g={g}:  logits rel err = {rel(lo_g, expect):.3e}   coupling w rel err = {rel(w_g, w_exp):.3e}")
print("\n   (대조) D4LT 에 입력만 변환하고 퍼즐 임베딩 방향을 고정하면 — 규칙 방향이 달라져야 하므로 달라야 정상:")
lo_g, _ = run(eq, d4.act_field(canvasF.view(1, -1), S, 1).view(-1), pid=1)
print(f"   g=1, 임베딩 고정:  rel err = {rel(lo_g, d4.act_field(loE.unsqueeze(0), S, 1)[0]):.3e}")
print("\n   파라미터 수:  기존 LT = %d,  D4LT = %d  (퍼즐 임베딩 제외)" % (
    sum(p.numel() for p in base.parameters()), sum(p.numel() for p in eq.parameters())))
