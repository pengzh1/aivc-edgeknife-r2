"""wsV4：相似度加权菌株调制迁移（机制：把已采纳的 CRD 迁移推广到任意未见菌株）。

动机：CRD←CGD β=0.35 是最终管线里的已采纳机制（val BAI 代理无害验证），
但它只用单一供体+固定 β。val_strain 划分（BAI，未见菌株×已见化合物）在
composite 里有双倍杠杆（FC 均值 1/4 + drug_resid 20%）。本模块把调制源
推广为：BAI 的响应调制 ≈ Σ_s softmax(−d(BAI,s)/τ)·m_{s,c}，
d = 1011 SNP 距离（outputs/wsK/genomes，外部登记数据），m_{s,c} = 严格
train Δ 的 (菌株,化合物) 对均值 − 化合物均值（wsN8 同式）。

训练内调定（零 val）：4 折 LOSO——留出菌株 s*，用其余 3 株为供体、以
s* 自己的 train 行为目标，网格 (γ, τ) 最大化 train 侧 FC/resid 代理；
取全局最优 (γ*, τ*) 后一次性应用于 val_strain（BAI 行）。
预注册采纳门槛：val composite ≥ 0.5541 + 0.0005（真增益）且 strain 划分
FC/resid 不双降；只作用 strain 划分行（both 划分化合物未见，m 不可得）。

合规：调制与相似度均为 train/公开登记数据；LOSO 调定全 train 内；
val 单次看；Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsV4_straintransfer
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Harness
from .wsN8_crdtransfer import strain_modulation
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsV4")
# 预注册网格（LOSO 全局共享，非逐折调参防过拟合）
GAMMAS = [0.2, 0.35, 0.5, 0.7]
TAUS = [0.001, 0.002, 0.005, 0.01]      # SNP 距离 softmax 温度（距离~0.004-0.02）


def load_distances(strains):
    p = "outputs/wsK/genomes/1011DistanceMatrixBasedOnSNPs.tab.gz"
    df = pd.read_csv(p, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.loc[strains, strains].to_numpy(np.float64)


def apply_modulation(pred, rows, m_map, donors, w, chem_of, gamma):
    """pred (全 trainval) 原地加 γ·Σ w_s m_{s,c} 到 rows 行。"""
    out = pred.copy()
    for i in rows:
        c = chem_of[i]
        m = np.zeros(pred.shape[1], np.float64)
        for s, ws in zip(donors, w):
            mv = m_map.get((s, c))
            if mv is not None:
                m += ws * mv
        out[i] = pred[i] + gamma * m
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
    ctrl_hat = np.load("outputs/wsT0/cache/control_hat.npy")
    mod = strain_modulation(h)            # {(s,c): 调制向量}，strict train Δ
    m = h.m_tr
    chem_of = m["perturbation_no_concentration"].to_numpy()
    # DHY210 为 S288c 背景实验室株，不在 1011 集合编码内——供体/LOSO 仅取
    # 矩阵中存在的训练菌株（BAH/CEK/CGD）
    _all = sorted(set(m.loc[m["split_final"] == "train", "Strains"]))
    _Dfull = pd.read_csv(
        "outputs/wsK/genomes/1011DistanceMatrixBasedOnSNPs.tab.gz",
        sep="\t", index_col=0)
    _Dfull.index = _Dfull.index.astype(str)
    _Dfull.columns = _Dfull.columns.astype(str)
    strains_all = [s for s in _all if s in _Dfull.index]
    skipped = [s for s in _all if s not in strains_all]
    if skipped:
        print(f"[note] 跳过无 1011 距离的训练菌株: {skipped}", flush=True)
    D = _Dfull.loc[strains_all + ["BAI"], strains_all + ["BAI"]].to_numpy(
        np.float64)
    smap = {s: i for i, s in enumerate(strains_all + ["BAI"])}

    def proxy_score(pred, rows):
        """train 行代理：Δ̂ vs Δ_true 的逐样本 PCC（FC 代理）。"""
        dt = h.delta_tr_all[rows].astype(np.float64)
        dp = (pred[rows] - ctrl_hat[rows]).astype(np.float64)
        pcc = []
        for i in range(len(rows)):
            mk = ~np.isnan(dt[i])
            if mk.sum() > 10 and np.std(dp[i][mk]) > 1e-6:
                pcc.append(np.corrcoef(dt[i][mk], dp[i][mk])[0, 1])
        return float(np.mean(pcc))

    # ---- LOSO 调定 (γ, τ) ----
    tr_rows = np.where((m["split_final"] == "train").to_numpy())[0]
    tr_treat = tr_rows[h.is_treat_tr[tr_rows]]
    strain_of = m["Strains"].to_numpy()
    best = None
    for tau in TAUS:
        for gamma in GAMMAS:
            scores = []
            for s_star in strains_all:
                held = np.array([i for i in tr_treat
                                 if strain_of[i] == s_star])
                donors = [s for s in strains_all if s != s_star]
                dd = np.array([D[smap[s_star], smap[s]] for s in donors])
                w = np.exp(-dd / tau)
                w = w / w.sum()
                p2 = apply_modulation(routed, held, mod, donors, w,
                                      chem_of, gamma)
                scores.append(proxy_score(p2, held))
            mean_s = float(np.mean(scores))
            print(f"  [LOSO] τ={tau} γ={gamma}: proxy FC={mean_s:.4f}",
                  flush=True)
            if np.isfinite(mean_s) and (best is None or mean_s > best[0]):
                best = (mean_s, tau, gamma)
    if best is None:
        print("[wsV4] 全部网格 LOSO 无有效值 → 机制关闭（不加任何迁移）",
              flush=True)
        return
    proxy_fc, tau_star, gamma_star = best
    base_proxy = proxy_score(routed, tr_treat)
    print(f"[tune] base proxy FC={base_proxy:.4f} → LOSO 最优 "
          f"τ*={tau_star} γ*={gamma_star} ({proxy_fc:.4f})", flush=True)

    # ---- 应用于 val_strain（BAI 行）----
    vs_rows = h._fast["val_strain_only"]["rows"]
    vs_treat = vs_rows[h.is_treat_tr[vs_rows]]
    dd = np.array([D[smap["BAI"], smap[s]] for s in strains_all])
    w = np.exp(-dd / tau_star)
    w = w / w.sum()
    print(f"[apply] BAI 供体权重: "
          f"{ {s: round(float(wi), 3) for s, wi in zip(strains_all, w)} }",
          flush=True)
    pred = apply_modulation(routed, vs_treat, mod, strains_all, w,
                            chem_of, gamma_star)
    np.save(OUT / "pred_trainval_strainxfer.npy", pred)
    res = h.score_val(pred, verbose=False)
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    s_only = res["per_split"]["val_strain_only"]
    base_res = h.score_val(routed, verbose=False)
    b_only = base_res["per_split"]["val_strain_only"]
    print(f"[wsV4] composite={res['composite']:.4f} F1={f1:.4f} "
          f"（门槛 0.5546）", flush=True)
    print(f"  strain 划分: FC {b_only['FC_PCC']:.4f}→{s_only['FC_PCC']:.4f} "
          f"resid {b_only['resid_PCC']:.4f}→{s_only['resid_PCC']:.4f}", flush=True)
    (OUT / "strainxfer.json").write_text(json.dumps(
        {"tau": tau_star, "gamma": gamma_star, "proxy_fc": proxy_fc,
         "base_proxy_fc": base_proxy, "weights": dict(
             zip(strains_all, w.tolist())), "res": res}, indent=1,
        default=float))
    print(f"[done] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
