"""CGD 响应谱 + 菌株间响应差异代理分析（Y_tr only，Y_te 零接触）。

发现：CRD 不在 train_val 元数据中（仅存在于 test split），因此
"同化合物同上下文 CGD vs CRD 的 Δ_true 差异"无法在不触 Y_te 的前提下计算。
本脚本退而提供两层合法代理：
  A. CGD 响应活跃基因谱：每蛋白在 CGD 各化合物下的 Δ_true 汇总（响应强度/频率）。
  B. 菌株背景敏感性：同化合物同上下文下 CGD vs 其余 4 株（BAH/BAI/CEK/DHY210）
     的 Δ 中位差——即"对菌株背景敏感"的基因。这类基因是 CRD 迁移调制的天然候选。

Δ_true 口径: log2(处理) − 七键组内 DMSO/Water 对照均值（复刻 src/data.py::build_control_map）。

输出:
  analysis/crd_variants/cgd_response_profile.csv      — 每蛋白 CGD 响应谱 + 菌株敏感性
  analysis/crd_variants/strain_sensitivity_pairs.csv  — 逐 (化合物,上下文,蛋白) 的 CGD-vs-others 差
  analysis/crd_variants/response_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("analysis/crd_variants")
CONTROLS = {"DMSO", "Water"}
QC = "Quality Control"
MATCH_KEYS = ["data_source", "Strains", "Medium", "Temperature",
              "pert_time", "instrument", "Yeast_cell_plate"]
CTX_KEYS = ["Strains", "Medium", "Temperature", "pert_time"]
CHEM = "perturbation_no_concentration"

RESP_ABS = 0.5  # |Δ|>=0.5 log2 视为"有响应"（叙述性阈值）


def main():
    m = pd.read_csv("input/WAYB_WAYC_metadata_train_val(1).csv")
    d = np.load("cache/proteome_log2.npz", allow_pickle=True)
    Y = d["Y_tr"].astype(np.float64)  # (8958, 5243)
    proteins = d["proteins"].astype(str)
    assert len(m) == Y.shape[0]

    # ---- 对照均值（7 键） ----
    is_ctrl = m[CHEM].isin(CONTROLS).to_numpy()
    ctrl = m[is_ctrl]
    Yc = Y[is_ctrl]
    ctrl_groups = ctrl.groupby(MATCH_KEYS).indices

    ctrl_mean = {}
    for key, idx in ctrl_groups.items():
        idx = np.asarray(sorted(idx))
        ctrl_mean[key] = np.nanmean(Yc[idx], axis=0)

    # ---- 每处理行 Δ_true ----
    is_treat = ~m[CHEM].isin(CONTROLS | {QC}).to_numpy()
    key_all = list(map(tuple, m[MATCH_KEYS].itertuples(index=False, name=None)))
    delta = np.full_like(Y, np.nan)
    n_nomatch = 0
    for i in np.where(is_treat)[0]:
        cm = ctrl_mean.get(key_all[i])
        if cm is None:
            n_nomatch += 1
            continue
        delta[i] = Y[i] - cm
    print(f"treat rows={is_treat.sum()} no-control-match={n_nomatch}")

    strains = m["Strains"].to_numpy()
    chem = m[CHEM].to_numpy()
    ctx = list(map(tuple, m[["Medium", "Temperature", "pert_time"]].itertuples(index=False, name=None)))

    # ---- A. CGD 响应谱 ----
    cgd_mask = (strains == "CGD") & is_treat
    dc = delta[cgd_mask]
    cc = chem[cgd_mask]
    print("CGD treat rows:", dc.shape[0], "compounds:", len(set(cc)))

    n_ctx = np.sum(~np.isnan(dc), axis=0)
    mean_abs = np.nanmean(np.abs(dc), axis=0)
    max_abs = np.nanmax(np.abs(dc), axis=0)
    frac_resp = np.nanmean(np.abs(dc) >= RESP_ABS, axis=0)
    mean_d = np.nanmean(dc, axis=0)

    # ---- B. 同化合物同上下文 CGD vs 其余株 ----
    # 对每 (chem, ctx) 组：CGD 中位 Δ − 其余株合并中位 Δ（逐蛋白）
    from collections import defaultdict
    groups = defaultdict(list)
    for i in np.where(is_treat)[0]:
        if np.all(np.isnan(delta[i])):
            continue
        groups[(chem[i], ctx[i])].append(i)

    pair_records = []  # (chem, ctx, protein_idx, diff, n_cgd, n_oth)
    for (c, x), idx in groups.items():
        idx = np.asarray(idx)
        s_cgd = idx[strains[idx] == "CGD"]
        s_oth = idx[strains[idx] != "CGD"]
        if len(s_cgd) == 0 or len(s_oth) == 0:
            continue
        med_cgd = np.nanmedian(delta[s_cgd], axis=0)
        med_oth = np.nanmedian(delta[s_oth], axis=0)
        diff = med_cgd - med_oth
        valid = ~np.isnan(diff)
        for j in np.where(valid)[0]:
            pair_records.append((c, x, j, diff[j], len(s_cgd), len(s_oth)))
    pr = pd.DataFrame(pair_records,
                      columns=["chem", "ctx", "pidx", "diff", "n_cgd", "n_oth"])
    print("pair records:", len(pr), "groups with both strains:",
          pr[["chem", "ctx"]].drop_duplicates().shape[0])
    pr["protein"] = proteins[pr["pidx"].to_numpy()]
    pr.to_csv(OUT / "strain_sensitivity_pairs.csv", index=False)

    # 每蛋白汇总：跨 (chem,ctx) 的差值中位数、|差|中位数、可用对数、响应频率差
    g = pr.groupby("pidx")["diff"]
    sens_med = g.median()
    sens_absmed = g.apply(lambda s: s.abs().median())
    sens_n = g.size()
    sens_q90 = g.apply(lambda s: s.abs().quantile(0.9))

    prof = pd.DataFrame({
        "protein": proteins,
        "cgd_n_obs": n_ctx,
        "cgd_mean_abs_delta": mean_abs,
        "cgd_max_abs_delta": max_abs,
        "cgd_frac_responsive": frac_resp,
        "cgd_mean_delta": mean_d,
        "sens_n_pairs": sens_n.reindex(range(len(proteins))).fillna(0).astype(int).to_numpy(),
        "sens_median_diff": sens_med.reindex(range(len(proteins))).to_numpy(),
        "sens_absmedian_diff": sens_absmed.reindex(range(len(proteins))).to_numpy(),
        "sens_absq90_diff": sens_q90.reindex(range(len(proteins))).to_numpy(),
    })
    prof.to_csv(OUT / "cgd_response_profile.csv", index=False)

    # 全局量级统计
    finite = np.isfinite(pr["diff"])
    summary = {
        "n_treat_rows": int(is_treat.sum()),
        "n_no_control_match": int(n_nomatch),
        "cgd_treat_rows": int(cgd_mask.sum()),
        "cgd_compounds": int(len(set(cc))),
        "n_pair_records": int(len(pr)),
        "strain_diff_abs_median": float(np.nanmedian(np.abs(pr["diff"][finite]))),
        "strain_diff_abs_q90": float(np.abs(pr["diff"][finite]).quantile(0.9)),
        "strain_diff_abs_q99": float(np.abs(pr["diff"][finite]).quantile(0.99)),
        "n_proteins": int(len(proteins)),
    }
    with open(OUT / "response_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
