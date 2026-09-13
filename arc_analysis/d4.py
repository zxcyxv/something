"""D4 (정사각 격자의 회전·반사 8원소) 도구.

원소는 (u,w) 오프셋에 작용하는 부호 있는 순열 행렬 M_g (2×2 정수) 로 표현한다.
캔버스 S×S 의 칸 x=(u,w) 는 중심 c=(S−1)/2 기준으로 g·x = M_g (x−c) + c 로 보낸다 (정확한 칸 순열).
"""
import numpy as np
import torch

_R = np.array([[0, -1], [1, 0]])     # 90° 회전
_F = np.array([[1, 0], [0, -1]])     # w 축 반사


def _closure():
    mats = [np.eye(2, dtype=int)]
    frontier = [np.eye(2, dtype=int)]
    while frontier:
        nxt = []
        for m in frontier:
            for gen in (_R, _F):
                c = gen @ m
                if not any((c == e).all() for e in mats):
                    mats.append(c); nxt.append(c)
        frontier = nxt
    return mats


MATS = _closure()                     # 8개, MATS[0] = I
ORDER = len(MATS)
assert ORDER == 8


def index_of(m):
    for i, e in enumerate(MATS):
        if (e == m).all():
            return i
    raise ValueError(m)


MUL = np.array([[index_of(MATS[a] @ MATS[b]) for b in range(ORDER)] for a in range(ORDER)])   # MUL[a,b] = idx(g_a g_b)
INV = np.array([index_of(MATS[a].T) for a in range(ORDER)])                                   # 직교 → 역 = 전치
assert all(MUL[a, INV[a]] == 0 for a in range(ORDER))


def conv_index():
    """G-합성곱 블록 색인: block(k,k') = base[k⁻¹k']  → [8,8] 정수."""
    return torch.tensor([[MUL[INV[k], kp] for kp in range(ORDER)] for k in range(ORDER)], dtype=torch.long)


def pos_perm(S, g):
    """gpos[x] = g·x 의 평탄 색인 (캔버스 중심 기준 정확한 순열)."""
    c = (S - 1) / 2
    idx = np.arange(S * S)
    u, w = idx // S, idx % S
    xy = np.stack([u - c, w - c])                 # [2,T]
    new = MATS[g] @ xy
    nu, nw = np.rint(new[0] + c).astype(int), np.rint(new[1] + c).astype(int)
    assert nu.min() >= 0 and nu.max() < S and nw.min() >= 0 and nw.max() < S
    out = nu * S + nw
    assert len(set(out.tolist())) == S * S
    return out


def act_field(field, S, g, axis=1):
    """스칼라 장 field[..., x, ...] 를 (g·field)[g·x] = field[x] 로 옮긴다."""
    gpos = torch.as_tensor(pos_perm(S, g), device=field.device)
    inv = torch.empty_like(gpos); inv[gpos] = torch.arange(S * S, device=field.device)   # inv[y] = g⁻¹y
    return field.index_select(axis, inv)


def act_slices(h, g, dh, axis=-1):
    """정칙표현: (g·h)[k] = h[g⁻¹k]  (마지막 축이 8·dh 로 슬라이스 k 를 담는다)."""
    ginv = INV[g]
    perm = torch.tensor([MUL[ginv, k] for k in range(ORDER)], dtype=torch.long, device=h.device)   # new slice k ← old slice g⁻¹k
    hs = h.reshape(*h.shape[:-1], ORDER, dh)
    return hs.index_select(-2, perm).reshape(h.shape)
