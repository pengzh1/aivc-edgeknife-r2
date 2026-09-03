"""wsT4：校准幅度推送（C5 候选）——C1 旗标不动，推送目标从固定 1.02 换成条件分位数。

动机：C1 把全部旗标条目推到恰好 |Δ̂|=1.02——F1 只看过阈（C5 与 C1 的 F1/方向
**逐位相同**，预注册要求验证），但高效应区幅度被系统性压平。若隐藏评测含
幅度感知成分（he_PCC 类），异质化目标更优。本模块拟合条件分位数回归
q50/q70（特征同 wsT1 选择器，train-only），旗标条目推送目标 =
sign·clip(max(1.02, q̂), ≤2.5)。

预注册：
- C5a = C1 旗标 + q50 目标；C5b = C1 旗标 + q70 目标。
- train 侧硬约束：F1 与 C1 完全一致（验证）；记录 he_PCC/FC/ctx/drug/fid 变化。
- val 单次裁决（2 新行并入 wsT1 裁决表）：采纳条件 = composite ≥ 0.5541−0.0005
  且 DEP_PCC 均值增益 ≥ +0.005；否则维持 C1。
- 只动旗标条目；非旗标条目与 C1 逐位相同。

合规：分位数模型 train-only；val 一次看；Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsT4_qpush
"""
import json
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .evaluate import Harness
from .wsE_depcal import fast_dep_f1, train_side_effects
from .wsN11_grandrouter import SPLITS
from .wsT1_depgate import (CACHE as T1C, MIN_PUSH, PUSH_TO, _apply_push,
                           _fidelity_proxy, _predict_val)
import joblib

OUT = Path("outputs/wsT4")
CACHE = OUT / "cache"
T0C = Path("outputs/wsT0/cache")
TAU_C1 = 0.35
QCAP = 2.5


def fast_he_pcc(dt: np.ndarray, dp: np.ndarray, thr: float = 1.0) -> float:
    """向量化 DEP_PCC（真值 hi 条目的逐样本 PCC 均值，同 M.dep_scores 口径）。"""
    m = ~np.isnan(dt)
    hi = m & (np.abs(dt) > thr)
    cnt = hi.sum(1)
    ok = cnt > 1
    if not ok.any():
        return 0.0
    vals = []
    dt64, dp64 = dt.astype(np.float64), dp.astype(np.float64)
    for i in np.where(ok)[0]:
        t, p = dt64[i][hi[i]], dp64[i][hi[i]]
        if t.std() > 1e-3 and p.std() > 1e-3:
            vals.append(np.corrcoef(t, p)[0, 1])
    return float(np.mean(vals)) if vals else 0.0


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()

    # ---- train 侧张量（wsT1 缓存）----
    rows = np.load(T1C / "train_rows.npy")
    avail = np.load(T1C / "train_avail.npy")
    dt = np.load(T1C / "train_dt.npy").astype(np.float64)
    d_est = np.load(T1C / "train_dest.npy").astype(np.float64)
    X = np.load(T1C / "train_X.npy")
    p_oof = np.load(T1C / "train_p_oof.npy")
    routed = np.load(T0C / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0C / "control_hat.npy")
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    dp0 = d_est + offset
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    eff0 = train_side_effects(dt, dp0, mu_ctx, mu_drug)
    fid0 = _fidelity_proxy(h, rows, routed, d_est, np.zeros_like(d_est))

    # ---- 旗标（C1 同式，train 用 OOF P）----
    P = np.full(avail.shape, np.nan, np.float32)
    P[avail] = p_oof
    absDE = np.abs(d_est)
    fl = (P >= TAU_C1) & (absDE >= MIN_PUSH) & avail
    print(f"[wsT4] flags={int(fl.sum()):,}", flush=True)

    # ---- 分位数回归（train-only，子采样加速）----
    Xa = X.reshape(-1, X.shape[-1])[avail.ravel()]
    ya = np.abs(dt)[avail]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ya), size=min(len(ya), 3_000_000), replace=False)
    from sklearn.ensemble import HistGradientBoostingRegressor
    qhats = {}
    for q in (0.5, 0.7):
        tq = time.time()
        qr = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=200, learning_rate=0.08,
            max_leaf_nodes=31, min_samples_leaf=200, random_state=0)
        qr.fit(Xa[sub], ya[sub])
        joblib.dump(qr, CACHE / f"qreg_{int(q*10)}.joblib")
        qhats[q] = qr
        print(f"[wsT4] q{q} fit ({time.time()-tq:.0f}s)", flush=True)

    # ---- train 侧评估：C1 vs C5a vs C5b ----
    def eval_variant(target_mat):
        dp = dp0.copy()
        delta = target_mat - dp0
        dp[fl] = dp0[fl] + delta[fl]
        eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
        return {"F1": fast_dep_f1(dt, dp),
                "hePCC": fast_he_pcc(dt, dp),
                "dmg": sum(eff0[k] - eff[k] for k in ("FC", "ctx", "drug")),
                "fid_drop": fid0 - _fidelity_proxy(h, rows, routed, d_est,
                                                   np.nan_to_num(delta, 0.0))}, dp

    tgt_c1 = np.sign(dp0) * np.maximum(np.abs(dp0), PUSH_TO)
    r_c1, dp_c1 = eval_variant(np.where(fl, tgt_c1, dp0))
    print(f"[train] C1   F1={r_c1['F1']:.4f} hePCC={r_c1['hePCC']:.4f} "
          f"dmg={r_c1['dmg']:.4f} fid_drop={r_c1['fid_drop']:.6f}", flush=True)

    report = {"C1_train": r_c1}
    preds_val = {}
    for tag, q in (("C5a", 0.5), ("C5b", 0.7)):
        qhat = qhats[q].predict(Xa).astype(np.float64)
        Q = np.full(avail.shape, np.nan)
        Q[avail] = qhat
        tgt = np.sign(dp0) * np.clip(np.maximum(np.abs(dp0), np.maximum(PUSH_TO, Q)),
                                     0, QCAP)
        r, _ = eval_variant(np.where(fl, tgt, dp0))
        report[f"{tag}_train"] = r
        print(f"[train] {tag} F1={r['F1']:.4f} hePCC={r['hePCC']:.4f} "
              f"dmg={r['dmg']:.4f} fid_drop={r['fid_drop']:.6f}", flush=True)
        assert abs(r["F1"] - r_c1["F1"]) < 1e-9, "预注册：F1 必须与 C1 一致"

        # ---- val 应用（旗标同 C1：hgb_full 预测；在 routed 上重放推送）----
        clf = joblib.load(T1C / "hgb_full.joblib")
        pred = routed.astype(np.float64)
        for sp in SPLITS:
            vrows, Pv = _predict_val(h, clf, sp)
            d_v = (routed[vrows] - ctrl_hat[vrows]).astype(np.float64)
            flv = (Pv >= TAU_C1) & (np.abs(d_v) >= MIN_PUSH)
            Xv = np.load(T1C / f"val_X_{sp}.npy")
            Qv = qhats[q].predict(Xv.reshape(-1, Xv.shape[-1])).reshape(Xv.shape[:2])
            tgtv = np.sign(d_v) * np.clip(
                np.maximum(np.abs(d_v), np.maximum(PUSH_TO, Qv)), 0, QCAP)
            add = np.where(flv, tgtv - d_v, 0.0)
            pred[vrows] = pred[vrows] + add
        # train 行（不参与评分，保持一致性）
        tgt_tr = np.sign(d_est) * np.clip(
            np.maximum(np.abs(d_est), np.maximum(PUSH_TO, Q)), 0, QCAP)
        fl_tr = (P >= TAU_C1) & (absDE >= MIN_PUSH)
        pred[rows] = pred[rows] + np.where(fl_tr, tgt_tr - d_est, 0.0)
        pred = pred.astype(np.float32)
        np.save(CACHE / f"pred_trainval_{tag}.npy", pred)
        preds_val[tag] = pred

    # ---- val 单次裁决 ----
    table = {}
    for tag, pred in preds_val.items():
        res = h.score_val(pred, verbose=False)
        table[tag] = res
        f1m = float(np.mean([res["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
        pm = float(np.mean([res["per_split"][sp]["DEP_PCC"] for sp in SPLITS]))
        print(f"[val] {tag} composite={res['composite']:.4f} F1={f1m:.4f} "
              f"DEP_PCC={pm:.4f}", flush=True)
    report["val"] = table
    (CACHE / "wsT4_report.json").write_text(json.dumps(report, indent=1,
                                                       default=float))
    print(f"[saved] {CACHE/'wsT4_report.json'} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
