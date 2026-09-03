"""wsE：高效应蛋白/DEP 检出校准 —— 对预测的 Δ 分量做膨胀以对抗回归收缩。

背景：ŷ_ens = 0.8·MLP(3seed) + 0.2·Ridge（当前最佳集成，composite≈0.509）。
模型 Δ 普遍向均值收缩，|Δ_pred|>1 的召回低 → DEP F1≈0.20。
校准：ŷ_cal = control_hat + γ·(ŷ_ens − control_hat)，只作用于处理样本。
γ 只在 train split 处理样本上以 train Δ_true 的 DEP F1 为目标调参；
val 四划分仅通过 h.score_val 评分（合规）。

候选：
  a) global : 全局 γ
  b) dir    : 分方向 γ_up/γ_down（按 Δ̂_est 符号）
  c) noise  : 蛋白级噪声加权 γ_j = 1+(γ_g−1)·w_j，w_j 由 train 残差方差决定
  d) gate   : 门控膨胀 ŷ_cal = ŷ + (γ−1)·g(|Δ̂|)·Δ̂，g 为 [a,b] 线性斜坡，
              只放大接近/超过阈值的条目，保护 resid/FC/fidelity
  e) gate7  : gate + 只对 control_hat 精确 7 键命中的行生效（回退行锚点误差大）
  f) band   : 梯形门控 —— 门在 |Δ̂|=1 处达峰（值 1），[a,1] 上坡、[1,c] 下坡，
              只把阈值带内的边际条目推过 |Δ|>1，带外条目不动的单调映射

用法：
  python -m src.wsE_depcal --stage baseline   # 未校准集成基线分
  python -m src.wsE_depcal --stage tune       # train 上调 γ 网格
  python -m src.wsE_depcal --stage eval       # val 上评候选（读 cache/candidates.json）
  python -m src.wsE_depcal --stage deliver --name <候选名>  # 交付 pred + 汇总
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .evaluate import Harness

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "wsE"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

GAMMAS_FINE = np.round(np.arange(0.80, 2.001, 0.05), 3)
GAMMAS_COARSE = np.round(np.arange(0.80, 2.001, 0.10), 3)

# control_hat 分层回退键（逐级放宽）
CTRL_KEY_LEVELS = [
    D.MATCH_KEYS,  # 7 键（官方匹配键）
    ["data_source", "Strains", "Medium", "Temperature", "pert_time", "instrument"],
    ["Strains", "Medium", "Temperature", "pert_time", "instrument"],
    D.CTX_KEYS,  # Strains/Medium/Temperature/pert_time
    ["Medium", "Temperature", "pert_time"],
    ["Strains"],
]


# ---------------------------------------------------------------- 集成与对照基线

def load_ensemble(w_mlp: float = 0.8) -> np.ndarray:
    """复用缓存预测：0.8·MLP(3seed) + 0.2·Ridge。"""
    mlp = np.load(ROOT / "outputs" / "pred_mlp_3seed.npy")
    ridge = np.load(ROOT / "outputs" / "pred_ridge.npy")
    ens = (w_mlp * mlp + (1.0 - w_mlp) * ridge).astype(np.float32)
    assert ens.shape == (8958, 5243) and not np.isnan(ens).any()
    return ens


def build_control_hat(h: Harness) -> tuple[np.ndarray, np.ndarray]:
    """train split 对照样本按 7 键分组均值 + 分层回退，对全部 train_val 行取值。

    返回 (control_hat, level)；level[i] 为该行命中层级（0=7键 … 5=1键, 6=全局）。
    """
    is_ctrl = h.m_train["perturbation_no_concentration"].isin(D.CONTROLS).to_numpy()
    mc = h.m_train[is_ctrl]
    Yc = h.Y_train[is_ctrl]

    prot_mean = np.where(np.isnan(h.stats.protein_mean),
                         np.nanmean(h.Y_train), h.stats.protein_mean)
    glob = np.nanmean(Yc, axis=0)
    glob = np.where(np.isnan(glob), prot_mean, glob).astype(np.float32)

    levels = []
    for keys in CTRL_KEY_LEVELS:
        km = list(map(tuple, mc[keys].to_numpy()))
        df = pd.DataFrame({"k": km})
        gm = {}
        for k, idx in df.groupby("k").groups.items():
            idx = np.fromiter(idx, dtype=int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(Yc[idx], axis=0)
            gm[k] = np.where(np.isnan(v), glob, v)  # 蛋白级回退到全局对照均值
        levels.append((keys, gm))

    out = np.repeat(glob[None, :], len(h.m_tr), axis=0)
    hit = np.zeros(len(h.m_tr), dtype=bool)
    level = np.full(len(h.m_tr), len(levels), dtype=np.int8)  # 默认全局
    level_hit = np.zeros(len(levels) + 1, dtype=int)
    for li, (keys, gm) in enumerate(levels):
        mk = list(map(tuple, h.m_tr[keys].to_numpy()))
        for i, k in enumerate(mk):
            if not hit[i] and k in gm:
                out[i] = gm[k]
                hit[i] = True
                level[i] = li
                level_hit[li] += 1
    level_hit[-1] = int((~hit).sum())
    print("[control_hat] rows resolved per level "
          "(7k/6k/5k/4k/3k/1k/global):", level_hit.tolist())
    # 各 split 的层级分布（诊断用）
    diag = pd.DataFrame({"split": h.m_tr["split_final"].to_numpy(),
                         "level": level})
    print(pd.crosstab(diag["split"], diag["level"]))
    return out.astype(np.float32), level


# ---------------------------------------------------------------- 快速 DEP 指标（与 M.dep_scores 语义一致）

def fast_dep_f1(dt: np.ndarray, dp: np.ndarray, thr: float = 1.0) -> float:
    """向量化版 dep_scores 的 DEP_F1（逐样本 F1 后取均值，规则逐条复刻）。"""
    m = ~np.isnan(dt)
    hi = m & (np.abs(dt) > thr)
    pred_hi = m & (np.abs(dp) > thr)
    tp = (pred_hi & hi).sum(1).astype(np.float64)
    fp = (pred_hi & ~hi).sum(1).astype(np.float64)
    fn = (~pred_hi & hi).sum(1).astype(np.float64)
    valid = (tp + fp + fn) > 0
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / np.maximum(tp + fn, 1.0)
    f1 = np.where(prec + rec > 0,
                  2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1[valid].mean()) if valid.any() else 0.0


def fast_dep_dir(dt: np.ndarray, dp: np.ndarray, thr: float = 1.0) -> float:
    """向量化版 DEP_dir_acc（|Δ_true|>1 处符号一致率，逐样本均值）。"""
    m = ~np.isnan(dt)
    hi = m & (np.abs(dt) > thr)
    cnt = hi.sum(1)
    ok = cnt > 0
    if not ok.any():
        return 0.0
    match = (np.sign(dt) == np.sign(dp)) & hi
    return float((match.sum(1)[ok] / cnt[ok]).mean())


def _verify_fast_dep(dt, dp, rows_idx):
    """与 M.dep_scores 对拍一次，保证语义一致。"""
    ref = M.dep_scores(dt, dp, np.arange(len(dt)))
    f1 = fast_dep_f1(dt, dp)
    dir_acc = fast_dep_dir(dt, dp)
    assert abs(ref["DEP_F1"] - f1) < 1e-9, (ref["DEP_F1"], f1)
    assert abs(ref["DEP_dir_acc"] - dir_acc) < 1e-9, (ref["DEP_dir_acc"], dir_acc)
    print(f"[verify] fast_dep == M.dep_scores (F1={f1:.4f}, dir={dir_acc:.4f})")


# ---------------------------------------------------------------- 调参准备

def setup_tuning(h: Harness, ens: np.ndarray, ctrl_hat: np.ndarray):
    """train split 处理样本上的调参张量。

    dt     : train Δ_true（harness 提供，对照取自 train_val 池）
    d_est  : Δ̂_est = ŷ_ens − control_hat（只用 train 冻结统计）
    offset : control_hat − cmean_pool = control_hat − Y + dt，
             使 scored_Δ(γ) = γ·d_est + offset 与 h.score_val 的
             Δ_pred = ŷ_cal − 匹配对照真值 在 train 行上逐元素一致。
    另返回 train 侧副作用代理：mu_ctx/mu_drug（train 冻结参照）。
    """
    tr = h.tr_rows
    rows = tr[h.is_treat_tr[tr]]
    dt = h.delta_tr_all[rows].astype(np.float64)
    d_est = (ens[rows] - ctrl_hat[rows]).astype(np.float64)
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    # 健全性：γ=1 时 scored_Δ 应等于 ŷ_ens − cmean_pool
    dp1 = d_est + offset
    _verify_fast_dep(dt, dp1, rows)
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    print(f"[tune] train treated rows={len(rows)}")
    return rows, dt, d_est, offset, mu_ctx, mu_drug


def train_side_effects(dt, dp, mu_ctx, mu_drug) -> dict:
    """train 侧副作用代理：FC_PCC / ctx_resid / drug_resid 的逐样本 PCC。"""
    return {
        "FC": float(np.nanmean(M._masked_pcc_axis1(dt, dp))),
        "ctx": float(np.nanmean(M._masked_pcc_axis1(dt - mu_ctx, dp - mu_ctx))),
        "drug": float(np.nanmean(M._masked_pcc_axis1(dt - mu_drug, dp - mu_drug))),
    }


# ---------------------------------------------------------------- 各 stage

def cmd_baseline():
    h = Harness()
    ens = load_ensemble()
    res = h.score_val(ens)
    _dump_json(CACHE / "baseline_score.json", res)
    print("[baseline] composite =", res["composite"])


def cmd_tune():
    h = Harness()
    ens = load_ensemble()
    ctrl_hat, level = build_control_hat(h)
    np.save(CACHE / "control_hat.npy", ctrl_hat)
    np.save(CACHE / "control_level.npy", level)

    rows, dt, d_est, offset, mu_ctx, mu_drug = setup_tuning(h, ens, ctrl_hat)
    lvl_rows = level[rows]

    results = {"n_train_treat": int(len(rows))}
    base_eff = train_side_effects(dt, d_est + offset, mu_ctx, mu_drug)
    results["base_effects"] = base_eff
    print(f"[tune] train proxies @uncal: {base_eff}")

    # ---- a) 全局 γ ----
    grid_a = []
    for g in GAMMAS_FINE:
        dp = g * d_est + offset
        grid_a.append({"gamma": float(g),
                       "F1": fast_dep_f1(dt, dp),
                       "dir": fast_dep_dir(dt, dp)})
    best_a = max(grid_a, key=lambda r: r["F1"])
    results["global"] = {"grid": grid_a, "best": best_a}
    print(f"[a] global γ* = {best_a['gamma']:.2f}  F1 {best_a['F1']:.4f} "
          f"(γ=1: {next(r['F1'] for r in grid_a if r['gamma']==1.0):.4f})")

    # ---- b) 分方向 γ_up/γ_down ----
    sign_pos = d_est > 0
    grid_b = []
    for gu in GAMMAS_COARSE:
        for gd in GAMMAS_COARSE:
            dp = np.where(sign_pos, gu, gd) * d_est + offset
            grid_b.append({"gu": float(gu), "gd": float(gd),
                           "F1": fast_dep_f1(dt, dp)})
    best_b = max(grid_b, key=lambda r: r["F1"])
    # 细调：±0.1 邻域 0.05 步长
    for gu in np.round(np.arange(best_b["gu"] - 0.1, best_b["gu"] + 0.101, 0.05), 3):
        for gd in np.round(np.arange(best_b["gd"] - 0.1, best_b["gd"] + 0.101, 0.05), 3):
            if gu < 0.5 or gd < 0.5:
                continue
            dp = np.where(sign_pos, gu, gd) * d_est + offset
            grid_b.append({"gu": float(gu), "gd": float(gd),
                           "F1": fast_dep_f1(dt, dp)})
    best_b = max(grid_b, key=lambda r: r["F1"])
    results["dir"] = {"grid": grid_b, "best": best_b}
    print(f"[b] dir γ_up*={best_b['gu']:.2f} γ_down*={best_b['gd']:.2f} "
          f"F1 {best_b['F1']:.4f}")

    # ---- c) 蛋白级噪声加权 ----
    # 噪声：train 处理样本上 scored_Δ(γ=1) 对 Δ_true 的残差方差（逐蛋白）
    resid = dt - (d_est + offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v_res = np.nanvar(resid, axis=0)
        v_tot = np.nanvar(dt, axis=0)
    med_res = np.nanmedian(v_res)
    v_res = np.where(np.isnan(v_res), med_res, v_res)

    grid_c = []
    for vname, v in [("resid", v_res), ("total", np.where(
            np.isnan(v_tot), np.nanmedian(v_tot), v_tot))]:
        vref = np.median(v)
        for beta in [0.25, 0.5, 1.0]:
            w = (vref / (v + vref)) ** beta
            w = w / w.mean()  # 均值归一 → 平均膨胀水平可比
            for g in GAMMAS_COARSE:
                gj = 1.0 + (g - 1.0) * w
                gj = np.clip(gj, 0.5, 3.0)
                dp = gj[None, :] * d_est + offset
                grid_c.append({"v": vname, "beta": float(beta), "g": float(g),
                               "F1": fast_dep_f1(dt, dp)})
    best_c = max(grid_c, key=lambda r: r["F1"])
    results["noise"] = {"grid": grid_c, "best": best_c}
    print(f"[c] noise v={best_c['v']} β={best_c['beta']} γ_g={best_c['g']:.2f} "
          f"F1 {best_c['F1']:.4f}")

    # ---- d/e) 门控膨胀（可选仅 7 键可靠锚点行）----
    # ŷ_cal = ŷ + (γ−1)·g(|Δ̂|)·Δ̂ ; g 为 [a,b] 线性斜坡（a=b 即硬门控）
    ad = np.abs(d_est)
    grid_d = []
    for max_lvl in [6, 0]:  # 6=全部行, 0=仅 7 键精确命中行
        row_ok = lvl_rows <= max_lvl
        for a, b in [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8),
                     (0.4, 0.8), (0.5, 0.9), (0.6, 1.0)]:
            gate = np.clip((ad - a) / max(b - a, 1e-9), 0.0, 1.0)
            for g in [1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
                dp = d_est + (g - 1.0) * gate * d_est + offset
                if max_lvl < 6:
                    dp = np.where(row_ok[:, None], dp, d_est + offset)
                f1 = fast_dep_f1(dt, dp)
                eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
                dmg = sum(base_eff[k] - eff[k] for k in ("FC", "ctx", "drug"))
                grid_d.append({"max_lvl": max_lvl, "a": a, "b": b, "g": g,
                               "F1": f1, "eff": eff, "dmg": dmg})
    grid_d.sort(key=lambda r: -r["F1"])
    results["gate"] = {"grid": grid_d}
    f1_uncal = next(r["F1"] for r in results["global"]["grid"]
                    if r["gamma"] == 1.0)
    print("[d/e] gate top8 (max_lvl,a,b,g,F1,dmg):")
    for r in grid_d[:8]:
        print(f"   lvl{r['max_lvl']} a={r['a']} b={r['b']} g={r['g']:.1f} "
              f"F1={r['F1']:.4f} dmg={r['dmg']:.4f}")
    # 选取：train F1 增益 ≥ 全局最优的 60% 且副作用代理损失最小
    best_a_f1 = results["global"]["best"]["F1"]
    thr_f1 = f1_uncal + 0.6 * (best_a_f1 - f1_uncal)
    safe = [r for r in grid_d if r["F1"] >= thr_f1]
    best_d = min(safe, key=lambda r: r["dmg"]) if safe else grid_d[0]
    print(f"[d/e] 选取: {best_d}")

    _dump_json(CACHE / "tune_results.json", results)

    # ---- 建议进入 val 评估的候选 ----
    cands = [
        {"name": "uncal", "kind": "global", "g": 1.0},
        {"name": f"g{best_a['gamma']:.2f}", "kind": "global",
         "g": best_a["gamma"]},
        {"name": f"dir_u{best_b['gu']:.2f}_d{best_b['gd']:.2f}", "kind": "dir",
         "gu": best_b["gu"], "gd": best_b["gd"]},
        {"name": f"noise_{best_c['v']}_b{best_c['beta']}_g{best_c['g']:.2f}",
         "kind": "noise", "v": best_c["v"], "beta": best_c["beta"],
         "g": best_c["g"]},
        {"name": (f"gate_l{best_d['max_lvl']}_a{best_d['a']}_b{best_d['b']}"
                  f"_g{best_d['g']:.1f}"),
         "kind": "gate", "max_lvl": best_d["max_lvl"], "a": best_d["a"],
         "b": best_d["b"], "g": best_d["g"]},
    ]
    # 折中参考点（较小膨胀，考察副作用-收益权衡）
    for g in [1.2, 1.4]:
        if all(abs(c.get("g", 0) - g) > 1e-6 for c in cands):
            cands.append({"name": f"g{g:.2f}", "kind": "global", "g": g})
    # gate 变体参考：train F1 最高的 gate（无论副作用）+ 一个温和 gate
    top_gate = grid_d[0]
    cands.append({"name": (f"gateTop_l{top_gate['max_lvl']}_a{top_gate['a']}"
                           f"_b{top_gate['b']}_g{top_gate['g']:.1f}"),
                  "kind": "gate", "max_lvl": top_gate["max_lvl"],
                  "a": top_gate["a"], "b": top_gate["b"], "g": top_gate["g"]})
    _dump_json(CACHE / "candidates.json", cands)
    print("[tune] candidates:", [c["name"] for c in cands])


def _gate_vec(d: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip((np.abs(d) - a) / max(b - a, 1e-9), 0.0, 1.0)


def _band_vec(d: np.ndarray, a: float, c: float) -> np.ndarray:
    """梯形门：|d|<1 段 [a,1] 上坡至 1；|d|≥1 段 [1,c] 下坡至 0。c<=1 时只留上坡。"""
    ad = np.abs(d)
    up = np.clip((ad - a) / max(1.0 - a, 1e-9), 0.0, 1.0)
    if c <= 1.0:
        return np.where(ad < 1.0, up, 0.0)
    down = np.clip((c - ad) / (c - 1.0), 0.0, 1.0)
    return np.where(ad < 1.0, up, down)


def apply_calibration(ens: np.ndarray, ctrl_hat: np.ndarray,
                      is_treat: np.ndarray, cand: dict,
                      noise_w: np.ndarray | None = None,
                      level: np.ndarray | None = None) -> np.ndarray:
    """校准 ŷ，仅处理样本；其余保持 ŷ_ens。

    global/dir/noise: ŷ_cal = control_hat + γ·(ŷ_ens − control_hat)
    gate            : ŷ_cal = ŷ_ens + (γ−1)·g(|Δ̂|)·Δ̂（可选仅 7 键命中行）
    """
    out = ens.copy()
    d = (ens - ctrl_hat)[is_treat]
    kind = cand["kind"]
    if kind == "global":
        g = np.float32(cand["g"])
        out[is_treat] = ctrl_hat[is_treat] + g * d
    elif kind == "dir":
        gu, gd = np.float32(cand["gu"]), np.float32(cand["gd"])
        g = np.where(d > 0, gu, gd).astype(np.float32)
        out[is_treat] = ctrl_hat[is_treat] + g * d
    elif kind == "noise":
        assert noise_w is not None
        gj = (1.0 + (cand["g"] - 1.0) * noise_w).astype(np.float32)
        gj = np.clip(gj, 0.5, 3.0)
        out[is_treat] = ctrl_hat[is_treat] + gj[None, :] * d
    elif kind == "gate":
        gate = _gate_vec(d, cand["a"], cand["b"]).astype(np.float32)
        add = ((cand["g"] - 1.0) * gate * d).astype(np.float32)
        if cand.get("max_lvl", 6) < 6:
            assert level is not None
            ok = level[is_treat] <= cand["max_lvl"]
            add = np.where(ok[:, None], add, np.float32(0.0))
        out[is_treat] = ens[is_treat] + add
    elif kind == "band":
        gate = _band_vec(d, cand["a"], cand["c"]).astype(np.float32)
        add = ((cand["g"] - 1.0) * gate * d).astype(np.float32)
        if cand.get("max_lvl", 6) < 6:
            assert level is not None
            ok = level[is_treat] <= cand["max_lvl"]
            add = np.where(ok[:, None], add, np.float32(0.0))
        out[is_treat] = ens[is_treat] + add
    else:
        raise ValueError(kind)
    return out.astype(np.float32)


def noise_weight(h: Harness, ens, ctrl_hat, vname: str, beta: float):
    """重算 c 方案的蛋白权重 w（train 冻结）。"""
    rows = h.tr_rows[h.is_treat_tr[h.tr_rows]]
    dt = h.delta_tr_all[rows].astype(np.float64)
    d_est = (ens[rows] - ctrl_hat[rows]).astype(np.float64)
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    if vname == "resid":
        arr = dt - (d_est + offset)
    else:
        arr = dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanvar(arr, axis=0)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    vref = np.median(v)
    w = (vref / (v + vref)) ** beta
    return w / w.mean()


def cmd_tune_band():
    """f) 梯形门控网格（复用 tune 缓存），向 candidates.json 追加 band 候选。"""
    h = Harness()
    ens = load_ensemble()
    ctrl_hat = np.load(CACHE / "control_hat.npy")
    level = np.load(CACHE / "control_level.npy")
    rows, dt, d_est, offset, mu_ctx, mu_drug = setup_tuning(h, ens, ctrl_hat)
    lvl_rows = level[rows]
    row_ok = lvl_rows <= 0  # 仅 7 键精确命中行
    base_eff = train_side_effects(dt, d_est + offset, mu_ctx, mu_drug)
    res_prev = json.loads((CACHE / "tune_results.json").read_text())
    f1_uncal = next(r["F1"] for r in res_prev["global"]["grid"]
                    if r["gamma"] == 1.0)
    f1_best = res_prev["global"]["best"]["F1"]

    ad = np.abs(d_est)
    grid = []
    for a in [0.5, 0.6, 0.7]:
        for c in [1.0, 1.15, 1.3, 1.6]:
            gate = _band_vec(d_est, a, c)
            for g in [1.3, 1.5, 1.8, 2.2, 2.8]:
                dp = d_est + (g - 1.0) * gate * d_est + offset
                dp = np.where(row_ok[:, None], dp, d_est + offset)
                f1 = fast_dep_f1(dt, dp)
                eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
                dmg = sum(base_eff[k] - eff[k] for k in ("FC", "ctx", "drug"))
                grid.append({"a": a, "c": c, "g": g, "F1": f1,
                             "eff": eff, "dmg": dmg})
    grid.sort(key=lambda r: -r["F1"])
    print("[f] band top10 (a,c,g,F1,dmg):")
    for r in grid[:10]:
        print(f"   a={r['a']} c={r['c']} g={r['g']:.1f} F1={r['F1']:.4f} "
              f"dmg={r['dmg']:.4f}")
    res_prev["band"] = {"grid": grid}
    _dump_json(CACHE / "tune_results.json", res_prev)

    # 候选：F1 达到全局最优增益 70%/85% 中 dmg 最小者
    cands = json.loads((CACHE / "candidates.json").read_text())
    names = {c["name"] for c in cands}
    for frac, tag in [(0.70, "bandMild"), (0.85, "bandBest")]:
        thr = f1_uncal + frac * (f1_best - f1_uncal)
        safe = [r for r in grid if r["F1"] >= thr]
        if not safe:
            continue
        bd = min(safe, key=lambda r: r["dmg"])
        nm = f"{tag}_a{bd['a']}_c{bd['c']}_g{bd['g']:.1f}"
        if nm not in names:
            cands.append({"name": nm, "kind": "band", "max_lvl": 0,
                          "a": bd["a"], "c": bd["c"], "g": bd["g"]})
            names.add(nm)
        print(f"[f] {tag}: {bd}")
    _dump_json(CACHE / "candidates.json", cands)
    print("[f] candidates:", [c["name"] for c in cands])


def cmd_eval():
    h = Harness()
    ens = load_ensemble()
    ctrl_hat = np.load(CACHE / "control_hat.npy")
    level = np.load(CACHE / "control_level.npy")
    cands = json.loads((CACHE / "candidates.json").read_text())
    scores = {}
    if (CACHE / "eval_scores.json").exists():
        scores = json.loads((CACHE / "eval_scores.json").read_text())
    for cand in cands:
        if cand["name"] in scores:
            print(f"[eval] {cand['name']:<34} (cached)")
            continue
        t0 = time.time()
        nw = None
        if cand["kind"] == "noise":
            nw = noise_weight(h, ens, ctrl_hat, cand["v"], cand["beta"])
        pred = apply_calibration(ens, ctrl_hat, h.is_treat_tr, cand, nw, level)
        assert not np.isnan(pred).any()
        res = h.score_val(pred)
        scores[cand["name"]] = {"cand": cand, "res": res}
        print(f"[eval] {cand['name']:<34} composite={res['composite']:.4f} "
              f"({time.time()-t0:.0f}s)")
        _dump_json(CACHE / "eval_scores.json", scores)  # 增量保存


def cmd_deliver(name: str):
    h = Harness()
    ens = load_ensemble()
    ctrl_hat = np.load(CACHE / "control_hat.npy")
    level = np.load(CACHE / "control_level.npy")
    cands = {c["name"]: c for c in json.loads((CACHE / "candidates.json").read_text())}
    cand = cands[name]
    nw = None
    if cand["kind"] == "noise":
        nw = noise_weight(h, ens, ctrl_hat, cand["v"], cand["beta"])
    pred = apply_calibration(ens, ctrl_hat, h.is_treat_tr, cand, nw, level)
    # NaN 保险：train 蛋白均值填补
    if np.isnan(pred).any():
        pred = h.stats.impute(pred)
    assert pred.shape == (8958, 5243) and pred.dtype == np.float32
    assert not np.isnan(pred).any() and not np.isinf(pred).any()
    np.save(OUT / "pred_trainval.npy", pred)
    print(f"[deliver] saved {OUT/'pred_trainval.npy'}  cand={cand}")
    res = h.score_val(pred)
    _dump_json(CACHE / "final_score.json", {"cand": cand, "res": res})
    print("[deliver] composite =", res["composite"])


def _dump_json(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=conv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["baseline", "tune", "tune_band", "eval", "deliver"])
    ap.add_argument("--name", default=None, help="deliver 阶段候选名")
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "baseline":
        cmd_baseline()
    elif args.stage == "tune":
        cmd_tune()
    elif args.stage == "tune_band":
        cmd_tune_band()
    elif args.stage == "eval":
        cmd_eval()
    elif args.stage == "deliver":
        assert args.name, "deliver 需要 --name"
        cmd_deliver(args.name)
    print(f"[done] {args.stage} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
