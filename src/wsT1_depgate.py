"""wsT1：DEP 选择器（分类引导过阈推送，handoff 8.15 §T1.2+T1.3+T1.4 合体）。

思路（wsE 三轮后处理之外的第四条形态）：
  DEP F1 被召回拖死，而 dir_acc/he_PCC 只在真值 hi 条目上计算 →
  "把高置信是 DEP 的条目刚好推过 |Δ|>1" 几乎只动 F1，对其他指标扰动极小。
  wsE 带门是这个选择器的单变量（|Δ̂|）手工版；本模块用 train-only 条目级
  分类器 p(|Δ_true|>1) 替代手工门，并首次突破 7 键锚点限制（推送是 ŷ 上的
  加性操作，不依赖 control_hat 精度）→ strain/both 两划分首次可攻击。

条目特征（10 维，全部 train-only 可算）：
  absDE |Δ̂|、x_std/x_std_w 跨族分歧（wsT0 缓存）、prot_dep_rate、
  prot_absq90、absDE_rel、mu_ctx_abs、mu_drug_abs、ctrl level、pert_time。

阶段（每步产物落盘，可断点续跑）：
  build     : 条目特征/标签张量缓存（train 处理行 + val 四划分处理行）
  fit       : HistGB 分类器，样本分组 5 折 OOF（train 内 CV 即 τ 调定的
              唯一依据，防 val 过拟合红线），全量终模用于 val/test 侧
  tune      : τ 网格 → train OOF 上 F1 增益 vs 副作用代理（FC/ctx/drug/
              fidelity），按预注册预算定 τ*，产出 candidates.json（≤3 候选）
  arbitrate : 候选施加于 routed r=0.7 基线（替代/叠加 band），h.score_val
              单次裁决表（预注册候选集，一次看不迭代）
  deliver   : 保存选定 pred_trainval + 复评 + 报告

预注册候选：
  C1 选择器替代 band（p≥τ* & |Δ̂|≥0.3 → 推至 1.02，保号）
  C2 选择器叠加 band（band 后仍 <1 且 p≥τ*₂ → 推至 1.02）
  C3 C1 + 下行拉（p≤τ_low & |Δ̂|∈(1,1.25] → 拉至 0.98）
副作用预算（train 代理，预注册）：FC+ctx+drug 降幅 ≤0.010，fidelity ≤0.002。
val 裁决规则（预注册）：DEP_F1 均值最大且 composite ≥ 基线−0.001。

合规：train-only 拟合（分类器/τ/统计量）；val 仅 arbitrate 一张表；
Y_te 零接触；不改既有文件（特征缓存全部落 outputs/wsT1/）。

用法:
  python -m src.wsT1_depgate --stage build
  python -m src.wsT1_depgate --stage fit
  python -m src.wsT1_depgate --stage tune
  python -m src.wsT1_depgate --stage arbitrate
  python -m src.wsT1_depgate --stage deliver --name C2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .evaluate import Harness
from .make_submission import band_calibrate
from .wsE_depcal import fast_dep_f1, train_side_effects
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsT1")
CACHE = OUT / "cache"
T0 = Path("outputs/wsT0/cache")

FEATS = ["absDE", "x_std", "x_std_w", "prot_dep_rate", "prot_absq90",
         "absDE_rel", "mu_ctx_abs", "mu_drug_abs", "level", "pert_time"]
PUSH_TO = 1.02       # 过阈推送目标 |Δ̂|
PULL_TO = 0.98       # 下行拉目标
MIN_PUSH = 0.3       # 低于此 |Δ̂| 不推（保护 fidelity/resid）
PULL_BAND = (1.0, 1.25)
NEG_MULT = 5         # 阴性子采样倍数
DAMAGE_BUDGET = {"proxy_sum": 0.010, "fidelity": 0.002}


# ------------------------------------------------------------ 特征构建

def _pert_time_num(m: pd.DataFrame, ref_map: dict | None = None):
    v = pd.to_numeric(m["pert_time"], errors="coerce")
    if ref_map is None:
        med = float(np.nanmedian(v))
        return v.fillna(med).to_numpy(np.float32), med
    return v.fillna(ref_map).to_numpy(np.float32)


def build_entry_features(h: Harness, rows: np.ndarray, routed: np.ndarray,
                         ctrl_hat: np.ndarray, level: np.ndarray,
                         prot_stats: dict, pt_med: float):
    """对 rows（处理行）构造条目级特征；返回 dict of (n,P) f32 + avail/hi 掩膜。

    只在 Δ_true 可用（train_val 对照池可算）的条目上有标签；特征本身全行可算。
    """
    dt = h.delta_tr_all[rows].astype(np.float64)
    d_est = (routed[rows] - ctrl_hat[rows]).astype(np.float64)
    avail = ~np.isnan(dt)
    hi = avail & (np.abs(dt) > 1.0)

    x_std = np.load(T0 / "x_std.npy")[rows].astype(np.float64)
    x_std_w = np.load(T0 / "x_std_w.npy")[rows].astype(np.float64)
    pdr = np.broadcast_to(prot_stats["prot_dep_rate"], dt.shape)
    paq = np.broadcast_to(prot_stats["prot_absq90"], dt.shape)
    absDE = np.abs(d_est)
    absDE_rel = absDE / (paq + 0.1)
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    lv = np.broadcast_to(level[rows][:, None].astype(np.float64), dt.shape)
    pt = np.broadcast_to(_pert_time_num(h.m_tr.iloc[rows], pt_med)[:, None],
                         dt.shape).astype(np.float64)

    X = np.stack([absDE, x_std, x_std_w, pdr, paq, absDE_rel,
                  np.abs(mu_ctx), np.abs(mu_drug), lv, pt], axis=-1)
    return {"X": X.astype(np.float32), "avail": avail, "hi": hi,
            "dt": dt, "d_est": d_est}


def cmd_build():
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load(T0 / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0 / "control_hat.npy")
    level = np.load(T0 / "control_level.npy")

    # 蛋白级 train 统计（train 处理行，train-only）
    rows_tr = h.tr_rows[h.is_treat_tr[h.tr_rows]]
    dt_tr = h.delta_tr_all[rows_tr].astype(np.float64)
    av_tr = ~np.isnan(dt_tr)
    hi_tr = av_tr & (np.abs(dt_tr) > 1.0)
    p0 = hi_tr.sum() / av_tr.sum()
    prot_dep_rate = (hi_tr.sum(0) + 2 * p0) / (av_tr.sum(0) + 2)
    with np.errstate(all="ignore"):
        prot_absq90 = np.nanquantile(np.abs(dt_tr), 0.9, axis=0)
    g90 = np.nanmedian(prot_absq90)
    prot_absq90 = np.where(np.isnan(prot_absq90), g90, prot_absq90)
    prot_stats = {"prot_dep_rate": prot_dep_rate.astype(np.float32),
                  "prot_absq90": prot_absq90.astype(np.float32)}
    np.savez(CACHE / "prot_stats.npz", **prot_stats)
    pt_med = float(np.nanmedian(pd.to_numeric(
        h.m_tr.iloc[rows_tr]["pert_time"], errors="coerce")))
    (CACHE / "pt_med.json").write_text(json.dumps({"pt_med": pt_med}))

    # train 处理行张量
    pack = build_entry_features(h, rows_tr, routed, ctrl_hat, level,
                                prot_stats, pt_med)
    np.save(CACHE / "train_X.npy", pack["X"])
    np.save(CACHE / "train_avail.npy", pack["avail"])
    np.save(CACHE / "train_hi.npy", pack["hi"])
    np.save(CACHE / "train_dt.npy", pack["dt"].astype(np.float32))
    np.save(CACHE / "train_dest.npy", pack["d_est"].astype(np.float32))
    np.save(CACHE / "train_rows.npy", rows_tr)
    print(f"[build] train entries X={pack['X'].shape} "
          f"avail={pack['avail'].sum():,} hi={pack['hi'].sum():,}", flush=True)

    # val 四划分处理行张量（推送只需要特征；标签仅为裁决表诊断）
    for sp in SPLITS:
        rows = h._fast[sp]["rows"]
        trows = rows[h.is_treat_tr[rows]]
        pack = build_entry_features(h, trows, routed, ctrl_hat, level,
                                    prot_stats, pt_med)
        np.save(CACHE / f"val_X_{sp}.npy", pack["X"])
        np.save(CACHE / f"val_rows_{sp}.npy", trows)
        print(f"[build] {sp} entries X={pack['X'].shape}", flush=True)
    print("[build] done", flush=True)


# ------------------------------------------------------------ 分类器

def _subsample(y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.where(y)[0]
    neg = np.where(~y)[0]
    neg_sub = rng.choice(neg, size=min(len(neg), NEG_MULT * len(pos)),
                         replace=False)
    sel = np.concatenate([pos, neg_sub])
    w = np.ones(len(sel), np.float32)
    w[len(pos):] = len(neg) / max(len(neg_sub), 1)
    return sel, w


def _fit_hgb(X, y, w):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=0)
    clf.fit(X, y, sample_weight=w)
    return clf


def cmd_fit():
    from sklearn.model_selection import GroupKFold
    X = np.load(CACHE / "train_X.npy")
    avail = np.load(CACHE / "train_avail.npy")
    hi = np.load(CACHE / "train_hi.npy")
    n_rows, n_prot = avail.shape
    Xa = X.reshape(-1, X.shape[-1])[avail.ravel()]
    ya = hi.ravel()[avail.ravel()]
    groups = np.repeat(np.arange(n_rows), avail.sum(1))
    print(f"[fit] avail entries={len(ya):,} pos={ya.sum():,} "
          f"rate={ya.mean():.4f}", flush=True)

    p_oof = np.zeros(len(ya), np.float32)
    gkf = GroupKFold(n_splits=5)
    for k, (tr_i, te_i) in enumerate(gkf.split(Xa, ya, groups)):
        t0 = time.time()
        sel, w = _subsample(ya[tr_i], seed=k)
        clf = _fit_hgb(Xa[tr_i][sel], ya[tr_i][sel], w)
        p_oof[te_i] = clf.predict_proba(Xa[te_i])[:, 1]
        print(f"  fold{k} done ({time.time()-t0:.0f}s) "
                  f"pos_rate_oof={p_oof[te_i].mean():.4f}", flush=True)
    np.save(CACHE / "train_p_oof.npy", p_oof)

    t0 = time.time()
    sel, w = _subsample(ya, seed=999)
    clf_full = _fit_hgb(Xa[sel], ya[sel], w)
    import joblib
    joblib.dump(clf_full, CACHE / "hgb_full.joblib")
    print(f"[fit] full model saved ({time.time()-t0:.0f}s)", flush=True)

    # OOF 质量：整体 AUC（与 wsT0 表同口径）
    from .wsT0_varcheck import auc_ties
    print(f"[fit] OOF AUC={auc_ties(p_oof, ya.astype(np.int8)):.4f}", flush=True)


# ------------------------------------------------------------ τ 调定（train-only）

def _apply_push(dp, d_cur, flag, mode):
    """在 dp 上施加推送。目标与增量都对当前值 d_cur 计算（tune 期 d_cur 为
    scored-Δ 当前值；apply 期 d_cur 为 ŷ 当前值 − ctrl_hat），避免与 band
    叠加时双重推送。flag: 布尔条目掩膜；mode: 'push'/'pull'。"""
    out = dp.copy()
    if mode == "push":
        tgt = np.sign(d_cur) * np.maximum(np.abs(d_cur), PUSH_TO)
    else:
        tgt = np.sign(d_cur) * np.minimum(np.abs(d_cur), PULL_TO)
    delta = tgt - d_cur
    out[flag] = dp[flag] + delta[flag]
    return out


def _fidelity_proxy(h, rows, routed, d_est, delta_mat):
    """ŷ' = ŷ + δ 后 train 处理行的样本级 PCC/R2（对照 wsE 副作用口径的补全）。
    δ 在不可用条目上带 NaN（继承自 scored-Δ 空间），必须先归零——否则 yp 被
    NaN 毒害、指标退化为与 δ 无关的常数（已踩此坑，勿回退）。"""
    delta_mat = np.nan_to_num(delta_mat, nan=0.0)
    yp = routed[rows].astype(np.float64) + delta_mat
    yt = h.Y_tr[rows].astype(np.float64)
    pcc = M._masked_pcc_axis1(yt, yp)
    r2 = np.clip(M._masked_r2_axis1(yt, yp), -1, 1)
    return float(np.nanmean(pcc) + np.nanmean(r2)) / 2


def cmd_tune():
    h = Harness()
    routed = np.load(T0 / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0 / "control_hat.npy")
    rows = np.load(CACHE / "train_rows.npy")
    avail = np.load(CACHE / "train_avail.npy")
    dt = np.load(CACHE / "train_dt.npy").astype(np.float64)
    d_est = np.load(CACHE / "train_dest.npy").astype(np.float64)
    p_oof = np.load(CACHE / "train_p_oof.npy")

    # scored Δ(基线) = d_est + offset；offset = ctrl_hat − Y + dt（wsE 口径）
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    dp0 = d_est + offset
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    eff0 = train_side_effects(dt, dp0, mu_ctx, mu_drug)
    fid0 = _fidelity_proxy(h, rows, routed, d_est, np.zeros_like(d_est))
    f1_0 = fast_dep_f1(dt, dp0)
    print(f"[tune] base: F1={f1_0:.4f} eff={eff0} fid={fid0:.4f}", flush=True)

    # band γ1.3 后的 dp（C2 的地基）：与 make_submission.band_calibrate 同式
    ad = np.abs(d_est)
    gate = np.where(ad < 1.0, np.clip((ad - 0.5) / 0.5, 0.0, 1.0), 0.0)
    lv = np.load(T0 / "control_level.npy")[rows]
    add = np.where((lv <= 0)[:, None], 0.3 * gate * d_est, 0.0)
    dp_band = dp0 + add

    P = np.full(avail.shape, np.nan, np.float32)
    P[avail] = p_oof
    absDE = np.abs(d_est)

    taus = np.round(np.arange(0.05, 0.951, 0.05), 2)
    results = {"base": {"F1": f1_0, "eff": eff0, "fid": fid0},
               "C1": [], "C2": [], "C3": []}
    for tau in taus:
        # C1: 选择器替代 band（推送目标在 scored-Δ 空间对当前值计算）
        fl = (P >= tau) & (absDE >= MIN_PUSH) & avail
        dp = _apply_push(dp0, dp0, fl, "push")
        eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
        dmg = sum(eff0[k] - eff[k] for k in ("FC", "ctx", "drug"))
        fid = _fidelity_proxy(h, rows, routed, d_est, dp - dp0)
        results["C1"].append({"tau": float(tau), "F1": fast_dep_f1(dt, dp),
                              "dmg": dmg, "fid_drop": fid0 - fid,
                              "n_flag": int(fl.sum())})
        # C2: band 之上再推（band 后 |scored|≤1 的条目；目标对 dp_band 计算）
        fl2 = fl & (np.abs(dp_band) <= 1.0)
        dp2 = _apply_push(dp_band, dp_band, fl2, "push")
        eff2 = train_side_effects(dt, dp2, mu_ctx, mu_drug)
        dmg2 = sum(eff0[k] - eff2[k] for k in ("FC", "ctx", "drug"))
        fid2 = _fidelity_proxy(h, rows, routed, d_est, dp2 - dp0)
        results["C2"].append({"tau": float(tau), "F1": fast_dep_f1(dt, dp2),
                              "dmg": dmg2, "fid_drop": fid0 - fid2,
                              "n_flag": int(fl2.sum())})
        print(f"  τ={tau:.2f} C1 F1={results['C1'][-1]['F1']:.4f}/dmg={dmg:.4f} "
              f"C2 F1={results['C2'][-1]['F1']:.4f}/dmg={dmg2:.4f}", flush=True)

    # τ*：预算内 F1 最大
    def pick(lst):
        ok = [r for r in lst if r["dmg"] <= DAMAGE_BUDGET["proxy_sum"]
              and r["fid_drop"] <= DAMAGE_BUDGET["fidelity"]]
        return max(ok or lst, key=lambda r: r["F1"])
    best1 = pick(results["C1"])
    best2 = pick(results["C2"])
    print(f"[tune] C1 τ*={best1['tau']:.2f} F1={best1['F1']:.4f} | "
          f"C2 τ*={best2['tau']:.2f} F1={best2['F1']:.4f}", flush=True)

    # C3: 固定 C1 的 τ*，扫下行拉阈值 τ_low（p≤τ_low & 1<|scored|≤1.25 → 0.98）
    fl_c1 = (P >= best1["tau"]) & (absDE >= MIN_PUSH) & avail
    dp_c1 = _apply_push(dp0, dp0, fl_c1, "push")
    for tau_low in [0.05, 0.10, 0.15, 0.20, 0.30]:
        fl3 = (P <= tau_low) & (np.abs(dp_c1) > PULL_BAND[0]) \
            & (np.abs(dp_c1) <= PULL_BAND[1]) & avail
        dp3 = _apply_push(dp_c1, dp_c1, fl3, "pull")
        eff3 = train_side_effects(dt, dp3, mu_ctx, mu_drug)
        dmg3 = sum(eff0[k] - eff3[k] for k in ("FC", "ctx", "drug"))
        fid3 = _fidelity_proxy(h, rows, routed, d_est, dp3 - dp0)
        results["C3"].append({"tau": float(best1["tau"]),
                              "tau_low": float(tau_low),
                              "F1": fast_dep_f1(dt, dp3), "dmg": dmg3,
                              "fid_drop": fid0 - fid3,
                              "n_flag": int(fl_c1.sum()) + int(fl3.sum())})
        print(f"  τ_low={tau_low:.2f} C3 F1={results['C3'][-1]['F1']:.4f} "
              f"dmg={dmg3:.4f}", flush=True)
    best3 = pick(results["C3"])
    print(f"[tune] C3 τ*={best3['tau']:.2f} τ_low*={best3['tau_low']:.2f} "
          f"F1={best3['F1']:.4f}", flush=True)

    cands = [{"name": "C1", "tau": best1["tau"], "train": best1, "kind": "C1"},
             {"name": "C2", "tau": best2["tau"], "train": best2, "kind": "C2"},
             {"name": "C3", "tau": best3["tau"], "tau_low": best3["tau_low"],
              "train": best3, "kind": "C3"}]
    (CACHE / "tune_results.json").write_text(json.dumps(
        results, indent=1, default=float))
    (CACHE / "candidates.json").write_text(json.dumps(cands, indent=1))
    print(f"[tune] candidates: {[c['name'] for c in cands]}", flush=True)


# ------------------------------------------------------------ val 裁决（单次）

def _predict_val(h, clf, sp):
    X = np.load(CACHE / f"val_X_{sp}.npy")
    rows = np.load(CACHE / f"val_rows_{sp}.npy")
    p = clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1]
    return rows, p.reshape(X.shape[:2])


def _apply_candidate(h, routed, ctrl_hat, level, cand, band_first):
    """对全 trainval 施加候选推送。

    分类器特征/flag 一律用 pre-band 的 Δ̂（= routed − ctrl_hat，与拟合分布一致）；
    推送目标与增量对 `out` 的当前值计算（C2 即 band 后值），避免双重推送。
    train 行用终模（in-sample，但 train 行不参与 val 评分，无裁决后果）。
    """
    import joblib
    clf = joblib.load(CACHE / "hgb_full.joblib")
    out = band_calibrate(routed, ctrl_hat, level, h.is_treat_tr, g=1.3) \
        if band_first else routed.copy()
    tau = cand["tau"]
    tau_low = cand.get("tau_low", 0.10)
    kind = cand["kind"]

    def _push_rows(rows, P):
        nonlocal out
        d_est = (routed[rows] - ctrl_hat[rows]).astype(np.float64)
        absDE = np.abs(d_est)
        d_cur = (out[rows] - ctrl_hat[rows]).astype(np.float64)
        add = np.zeros_like(d_cur)
        fl = (P >= tau) & (absDE >= MIN_PUSH)
        if kind == "C2":
            fl &= np.abs(d_cur) <= 1.0
        tgt = np.sign(d_cur) * np.maximum(np.abs(d_cur), PUSH_TO)
        add += np.where(fl, tgt - d_cur, 0.0)
        if kind == "C3":
            fl3 = (P <= tau_low) & (np.abs(d_cur) > PULL_BAND[0]) \
                & (np.abs(d_cur) <= PULL_BAND[1])
            tgt3 = np.sign(d_cur) * np.minimum(np.abs(d_cur), PULL_TO)
            add += np.where(fl3, tgt3 - d_cur, 0.0)
        out[rows] = (out[rows].astype(np.float64) + add).astype(np.float32)

    for sp in SPLITS:
        rows, P = _predict_val(h, clf, sp)
        _push_rows(rows, P)
    rows_tr = h.tr_rows[h.is_treat_tr[h.tr_rows]]
    X = np.load(CACHE / "train_X.npy")
    P = clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(X.shape[:2])
    _push_rows(rows_tr, P)
    return out


def cmd_arbitrate():
    h = Harness()
    routed = np.load(T0 / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0 / "control_hat.npy")
    level = np.load(T0 / "control_level.npy")
    cands = json.loads((CACHE / "candidates.json").read_text())

    table = {}
    base = h.score_val(routed, verbose=False)
    table["C0_routed"] = base
    banded = band_calibrate(routed, ctrl_hat, level, h.is_treat_tr, g=1.3)
    table["C0_band13"] = h.score_val(banded, verbose=False)
    f1 = lambda r: float(np.mean([r["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    print(f"[arb] C0_routed    composite={base['composite']:.4f} F1={f1(base):.4f}",
          flush=True)
    print(f"[arb] C0_band13    composite={table['C0_band13']['composite']:.4f} "
          f"F1={f1(table['C0_band13']):.4f}", flush=True)

    for cand in cands:
        band_first = cand["kind"] in ("C2",)
        pred = _apply_candidate(h, routed, ctrl_hat, level, cand, band_first)
        res = h.score_val(pred, verbose=False)
        table[cand["name"]] = res
        np.save(CACHE / f"pred_trainval_{cand['name']}.npy", pred)
        per = {sp: round(res["per_split"][sp]["DEP_F1"], 4) for sp in SPLITS}
        print(f"[arb] {cand['name']:<12} composite={res['composite']:.4f} "
              f"F1={f1(res):.4f} per-split={per}", flush=True)
        (CACHE / "arbitrate.json").write_text(json.dumps(
            table, indent=1, default=float))
    print(f"[saved] {CACHE/'arbitrate.json'}", flush=True)


def cmd_deliver(name):
    pred = np.load(CACHE / f"pred_trainval_{name}.npy")
    np.save(OUT / "pred_trainval.npy", pred)
    print(f"[deliver] {OUT/'pred_trainval.npy'} <- {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["build", "fit", "tune", "arbitrate", "deliver"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"build": cmd_build, "fit": cmd_fit, "tune": cmd_tune,
     "arbitrate": cmd_arbitrate,
     "deliver": lambda: cmd_deliver(args.name)}[args.stage]()
    print(f"[done] {args.stage} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
