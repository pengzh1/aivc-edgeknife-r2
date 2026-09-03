"""官方评分指标的本地复现（用于模型选择，非官方实现）。

六大模块：
1. 绝对保真度：逐样本 corr/R² + 逐蛋白 corr/R²（适用全部划分）
2. 匹配对照原始 FC：PCC(Δ_pred, Δ_true)，Δ = 处理 - 匹配对照（OOD 核心）
3. 上下文均值残差（chem_only）：PCC(Δ_pred - μ_ctx, Δ_true - μ_ctx)
4. 药物均值残差（strain_only）：PCC(Δ_pred - μ_drug, Δ_true - μ_drug)
5. 双重未知/时间外推：FC + 保真度为主
6. 高效应蛋白/DEP 检出：|Δ_true|>1 的方向准确率、高效应 PCC、F1

所有参照统计（μ_ctx、μ_drug、对照匹配）仅用 train split 冻结。
本地总分按官方权重近似合成，仅用于模型间相对比较。
"""
import numpy as np
import pandas as pd

# ---------- 基础统计 ----------


def _masked_pcc_axis1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """逐行 PCC，忽略 y_true 中的 NaN。返回每行一个值。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true)
    n = np.maximum(mask.sum(axis=1), 1)
    t = np.where(mask, y_true, 0.0)
    p = np.where(mask, np.nan_to_num(y_pred, nan=0.0), 0.0)
    mt = t.sum(1) / n
    mp = p.sum(1) / n
    tc = np.where(mask, y_true - mt[:, None], 0.0)
    pc = np.where(mask, p - mp[:, None], 0.0)
    cov = (tc * pc).sum(1)
    vt = np.sqrt((tc**2).sum(1))
    vp = np.sqrt((pc**2).sum(1))
    denom = vt * vp
    return np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 0.0)


def _masked_r2_axis1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """逐行 R²（可为负），忽略 NaN。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true)
    n = np.maximum(mask.sum(axis=1), 1)
    t = np.where(mask, y_true, 0.0)
    mt = t.sum(1) / n
    sst = (np.where(mask, y_true - mt[:, None], 0.0) ** 2).sum(1)
    sse = (np.where(mask, y_true - np.nan_to_num(y_pred, nan=0.0), 0.0) ** 2).sum(1)
    return np.where(sst > 1e-12, 1.0 - sse / np.maximum(sst, 1e-12), 0.0)


def per_protein_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐蛋白跨样本 PCC，忽略 NaN，返回蛋白间均值（自动剔除近零方差蛋白）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    vals = []
    for j in range(y_true.shape[1]):
        t = y_true[:, j]
        p = y_pred[:, j]
        m = ~np.isnan(t) & ~np.isnan(p)
        if m.sum() < 8:
            continue
        tj, pj = t[m], p[m]
        if tj.std() < 1e-3 or pj.std() < 1e-3:
            continue
        vals.append(np.corrcoef(tj, pj)[0, 1])
    return float(np.mean(vals)) if vals else 0.0


def fidelity_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    s_pcc = _masked_pcc_axis1(y_true, y_pred)
    s_r2 = _masked_r2_axis1(y_true, y_pred)
    return {
        "sample_PCC": float(np.mean(s_pcc)),
        "sample_R2": float(np.mean(np.clip(s_r2, -1, 1))),
        "protein_PCC": per_protein_pcc(y_true, y_pred),
    }


# ---------- Δ（fold change）与残差 ----------


def compute_delta(Y: np.ndarray, m: pd.DataFrame, ctrl_map: dict,
                  Y_pool: np.ndarray, idx_pool: dict) -> tuple[np.ndarray, np.ndarray]:
    """对 m 中每个处理样本计算 Δ = y_treat - mean(匹配对照)。

    Y/idx 为待计算样本所在矩阵；Y_pool/idx_pool 为对照值搜索池（可为拼接矩阵）。
    返回 (delta, is_treat_mask)；非处理样本 delta 为 NaN 行。
    """
    delta = np.full_like(Y, np.nan)
    is_treat = np.zeros(len(m), dtype=bool)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, sid in enumerate(m["sample_ID"]):
            ctrls = ctrl_map.get(sid, [])
            if not ctrls:
                continue
            rows = [idx_pool[c] for c in ctrls if c in idx_pool]
            if not rows:
                continue
            cvals = Y_pool[rows]  # (n_ctrl, n_prot)，可能含 NaN
            cmean = np.nanmean(cvals, axis=0)
            delta[i] = Y[i] - cmean
            is_treat[i] = True
    return delta, is_treat


def fc_scores(delta_true: np.ndarray, delta_pred: np.ndarray,
              rows: np.ndarray) -> dict:
    """匹配对照原始 FC：逐样本 PCC(Δ_pred, Δ_true)（Δ_true 的 NaN 忽略）。"""
    dt, dp = delta_true[rows], delta_pred[rows]
    pcc = _masked_pcc_axis1(dt, dp)
    return {"FC_PCC": float(np.nanmean(pcc))}


def residual_scores(delta_true, delta_pred, mu, rows) -> dict:
    """PCC(Δ_pred - μ, Δ_true - μ)，mu 已与 rows 对齐（len(rows) × n_prot）。"""
    dt = delta_true[rows] - mu
    dp = delta_pred[rows] - mu
    pcc = _masked_pcc_axis1(dt, dp)
    return {"resid_PCC": float(np.nanmean(pcc))}


def dep_scores(delta_true, delta_pred, rows, thr: float = 1.0) -> dict:
    """高效应蛋白与 DEP 检出：方向准确率、高效应 PCC、F1。"""
    dt = np.asarray(delta_true[rows], dtype=np.float64)
    dp = np.asarray(delta_pred[rows], dtype=np.float64)
    dir_acc, he_pcc, f1s = [], [], []
    for i in range(len(rows)):
        t, p = dt[i], dp[i]
        m = ~np.isnan(t)
        t, p = t[m], p[m]
        hi = np.abs(t) > thr
        if hi.sum() > 0:
            dir_acc.append((np.sign(t[hi]) == np.sign(p[hi])).mean())
            if t[hi].std() > 1e-3 and p[hi].std() > 1e-3:
                he_pcc.append(np.corrcoef(t[hi], p[hi])[0, 1])
        pred_hi = np.abs(p) > thr
        tp = (pred_hi & hi).sum()
        fp = (pred_hi & ~hi).sum()
        fn = (~pred_hi & hi).sum()
        if tp + fp + fn > 0:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {
        "DEP_dir_acc": float(np.mean(dir_acc)) if dir_acc else 0.0,
        "DEP_PCC": float(np.mean(he_pcc)) if he_pcc else 0.0,
        "DEP_F1": float(np.mean(f1s)) if f1s else 0.0,
    }


# ---------- 汇总 ----------

W = {"fidelity": 0.20, "FC": 0.25, "ctx_resid": 0.20, "drug_resid": 0.20,
     "both_time": 0.10, "DEP": 0.05}


def composite(per_split: dict) -> float:
    """按官方权重近似合成总分。per_split: {split: {metric: value}}。

    fidelity / FC / DEP 取全部划分的均值；ctx_resid 取 chem_only；
    drug_resid 取 strain_only；both_time 取 both 与 time 的 (FC+保真)/2 均值。
    """
    def avg(key, splits):
        v = [per_split[s][key] for s in splits if key in per_split.get(s, {})]
        return float(np.mean(v)) if v else 0.0

    all_s = list(per_split)
    fid = avg("fidelity", all_s)
    fc = avg("FC_PCC", all_s)
    dep = avg("DEP_F1", all_s)
    ctx = avg("resid_PCC", [s for s in all_s if "chem_only" in s])
    drug = avg("resid_PCC", [s for s in all_s if "strain_only" in s])
    bt = np.mean([
        0.5 * (per_split[s].get("fidelity", 0) + per_split[s].get("FC_PCC", 0))
        for s in all_s if ("both" in s or "time" in s)
    ]) if any(("both" in s or "time" in s) for s in all_s) else 0.0
    return (W["fidelity"] * fid + W["FC"] * fc + W["ctx_resid"] * ctx
            + W["drug_resid"] * drug + W["both_time"] * bt + W["DEP"] * dep)
