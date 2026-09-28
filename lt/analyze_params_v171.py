"""v1.71 마일스톤의 파라미터 분포 추적 (EMA 가중치; 퍼즐 공통 편향만 raw).

이론상 중요한 양을 스칼라 요약으로 뽑아 10k~200k 궤적과 v1.1 160k·v1.7 200k 참조값을 나란히 둔다.
행렬 쪽은 weight decay 대상이므로 노름·스펙트럼·주소 스케일 불변성을 함께 본다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

P = "model.inner."
L0 = P + "layers.0."
D = 832


def load_pt(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v.float() for k, v in c["ema_shadow"].items()}
    raw = {k.replace("_orig_mod.", ""): v.float() for k, v in c["raw_model_state_dict"].items()}
    for k, v in raw.items():
        sd.setdefault(k, v)
    return sd, raw, c


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    sd = {k.split("/", 1)[1]: torch.from_numpy(z[k]).float() for k in z.files if k.startswith("ema/")}
    if L0 + "wc_raw" in sd:     # QR 판: 실효 투영으로 변환해 같은 좌표로 비교
        Q, _ = torch.linalg.qr(sd[L0 + "wc_raw"].transpose(-1, -2))
        sd[L0 + "wc"] = Q.transpose(-1, -2)
    return sd


def eff_rank(s):
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def summarize(sd):
    r = {}
    sig = torch.sigmoid
    eta = sig(sd[L0 + "eta_raw"]).flatten()
    lam = sig(sd[L0 + "lam_raw"]).flatten()
    gain = torch.nn.functional.softplus(sd[L0 + "gain_raw"]).flatten()
    alpha = torch.nn.functional.softplus(sd[L0 + "alpha_raw"]).flatten()
    r["eta"] = eta.tolist(); r["tau_w"] = (1 / eta).tolist()
    r["lam"] = lam.tolist(); r["gain"] = gain.tolist(); r["alpha"] = alpha.tolist()
    r["lam_gain"] = (lam * gain).tolist()                       # 기억 경로 세기
    beta = wrap(sd[L0 + "beta"])
    r["beta_cos2"] = float((torch.cos(beta) ** 2).mean())       # 대칭(짝) 쓰기 비중
    r["beta_abs_med_deg"] = float(beta.abs().median() * 180 / math.pi)
    r["beta_head_cos2"] = (torch.cos(beta) ** 2).mean(1).tolist()
    r["beta_head_mean_deg"] = (torch.atan2(torch.sin(beta).mean(1), torch.cos(beta).mean(1)) * 180 / math.pi).tolist()
    psi = wrap(sd[L0 + "psi"])
    r["psi_cos2"] = float((torch.cos(psi) ** 2).mean())
    th = sd[L0 + "theta"]
    r["theta_norm_med"] = float(th.norm(dim=-1).median())
    if L0 + "mu_rho_raw" in sd:
        rho = sig(sd[L0 + "mu_rho_raw"])
        tau = -1 / torch.log(rho)
        om = wrap(sd[L0 + "mu_omega"])
        r["rho_mean"] = float(rho.mean()); r["rho_max"] = float(rho.max())
        r["tau_z_med"] = float(tau.median()); r["tau_z_max"] = float(tau.max())
        r["tau_z_head_mean"] = tau.mean(1).tolist()
        r["omega_abs_med"] = float(om.abs().median())
    # 주소 투영: 헤드별 [104,832]. linear 판은 스케일이 자유 (정규화로 함수에서는 대부분 사라짐)
    wc = sd[L0 + "wc"]
    s = torch.linalg.svdvals(wc)                                # [H,104]
    r["wc_fro"] = wc.norm(dim=(1, 2)).tolist()
    r["wc_effrank"] = [eff_rank(x) for x in s]
    r["wc_cond"] = (s[:, 0] / s[:, -1]).tolist()
    A, B = wc[:, :52], wc[:, 52:]
    r["wc_imag_over_real"] = (B.norm(dim=(1, 2)) / A.norm(dim=(1, 2))).tolist()
    G = wc @ wc.transpose(1, 2)                                 # 행 그람: I 에서 얼마나 벗어났나 (스케일 제거)
    Gn = G / G.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
    r["wc_gram_offdiag_rms"] = (Gn - torch.diag_embed(Gn.diagonal(dim1=1, dim2=2))).pow(2).mean((1, 2)).sqrt().tolist()
    # 값 사영 w_sh: 전달 f = W_shᵀ a W_sh h 는 w_sh 에 2차
    wsh = sd[L0 + "w_sh"]
    r["wsh_fro"] = wsh.norm(dim=(1, 2)).tolist()
    blk = torch.stack([wsh[m, :, m * 104:(m + 1) * 104] for m in range(8)])
    r["wsh_diag_block_frac"] = (blk.norm(dim=(1, 2)) ** 2 / wsh.norm(dim=(1, 2)) ** 2).tolist()
    r["wsh_top_sv"] = torch.linalg.svdvals(wsh)[:, 0].tolist()
    # 경계 MLP (유일한 비선형 혼합)
    gu, dn = sd[L0 + "b_gate_up.weight"], sd[L0 + "b_down.weight"]
    r["mlp_gate_up_fro"] = float(gu.norm()); r["mlp_down_fro"] = float(dn.norm())
    r["mlp_gate_up_sv1"] = float(torch.linalg.matrix_norm(gu, 2))
    r["mlp_down_sv1"] = float(torch.linalg.matrix_norm(dn, 2))
    # 입력 주입: √d·E(x) (+ 공통 퍼즐 편향)
    E = sd[P + "embed.weight"]
    r["embed_norm"] = E.norm(dim=1).tolist()
    digits = E[2:11]
    En = digits / digits.norm(dim=1, keepdim=True)
    C = En @ En.T
    r["embed_digit_offdiag_cos_mean"] = float((C.sum() - 9) / 72)
    r["inj_norm_digit_mean"] = float(math.sqrt(D) * digits.norm(dim=1).mean())
    r["inj_norm_blank"] = float(math.sqrt(D) * E[1].norm())
    pe = sd.get(P + "puzzle_emb.weights")
    if pe is not None:
        r["puzzle_bias_norm"] = float(pe.norm()); r["inj_bias_scaled"] = float(math.sqrt(D) * pe.norm())
    W = sd[P + "w_cls.weight"]
    r["wcls_fro"] = float(W.norm()); r["wcls_digit_norm_mean"] = float(W[2:11].norm(dim=1).mean())
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mdir", type=Path, default=Path("runs/v171_kaggle/lt_v171/milestones"))
    ap.add_argument("--out", type=Path, default=Path("runs/v171_kaggle/param_trajectory.json"))
    args = ap.parse_args()
    out = {"v1.71": {}, "ref": {}, "drift": {}}
    prev = None
    for step in range(10000, 200001, 10000):
        sd, raw, c = load_pt(args.mdir / f"step_{step}.pt")
        out["v1.71"][step] = summarize(sd)
        # raw 와 EMA 의 거리 (EMA 가 raw 를 얼마나 평활하는지), 10k 간 각도 변화 (유효 학습률)
        d = {}
        for name in ("wc", "w_sh", "b_gate_up.weight", "b_down.weight"):
            e, rw = sd[L0 + name], raw[L0 + name]
            d[f"{name}_raw_ema_rel"] = float((rw - e).norm() / e.norm())
            if prev is not None:
                p = prev[L0 + name]
                d[f"{name}_angle_deg_10k"] = float(torch.rad2deg(torch.arccos(torch.clamp(
                    (p * e).sum() / (p.norm() * e.norm()), -1, 1))))
        for name in ("embed.weight", "w_cls.weight"):
            e = sd[P + name]
            if prev is not None:
                p = prev[P + name]
                d[f"{name}_angle_deg_10k"] = float(torch.rad2deg(torch.arccos(torch.clamp(
                    (p * e).sum() / (p.norm() * e.norm()), -1, 1))))
        out["drift"][step] = d
        prev = sd
        print("loaded", step, flush=True)
    out["ref"]["v1.1_160k"] = summarize(load_npz("checkpoints/v1.1_step160000.npz"))
    out["ref"]["v1.7_200k"] = summarize(load_npz("checkpoints/v1.7_step200000.npz"))
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
