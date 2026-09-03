"""wsT0：DEP 选择器前置快检（handoff 8.15 §T1.4）——跨族分歧能否定位 DEP 漏检。

做的事：
1. 重建 21 族 r=0.7 路由基线 trainval 预测并缓存（wsT 全链路的公共底座），
   叠加 band γ=1.3 后 val 评分复核 ≈0.554（基线锚定，非调参）。
2. train 处理行上构造零训练/轻量特征：
   - x_std   : 21 族预测的跨族标准差（= Δ̂ 的跨族分歧，锚点是常量）
   - x_std_w : 按 w_global 加权的跨族标准差
   - prot_dep_rate : 蛋白级 train DEP 率（|Δ_true|>1 频率，加性平滑，train-only）
3. 裁决问题（全部先在 train 上回答，val 仅最终一张表单次确认）：
   Q1 |Δ̂_ens| 单独预测 |Δ_true|>1 的 AUC（基线）
   Q2 x_std / x_std_w / prot_dep_rate 单独的同目标 AUC
   Q3 边际带内（|Δ̂|∈[0.3,1)）x_std 区分 FN(|Δ_true|>1) 的 AUC —— 选择器价值核心
   Q4 三特征 logistic（按样本 GroupKFold 5 折，子采样）是否显著超过 Q1
4. val 四划分同指标一张表（特征零拟合，单次裁决看）。
5. 判定（预注册）：GREEN = Q3 train AUC≥0.60 或 Q4≥Q1+0.02，且 val 同向
   （差≤0.03）→ 上 DEP 选择器（wsT1_depgate）；否则关闭本路线转 wsT 模型族。

合规：prot_dep_rate 等统计仅用 train split 处理行；Δ_true 口径同 wsE 调参惯例
（对照取自 train_val 池的官方匹配对照，与 γ=1.3 拟合同一约定）；val 仅本脚本
一张表做路线选择；Y_te 零接触；不改任何既有文件。

用法: python -m src.wsT0_varcheck
"""
import json
import time
from pathlib import Path

import numpy as np

from .evaluate import Harness
from . import wsH_router as H
from .make_submission import band_calibrate
from .wsN11_grandrouter import CANDIDATES, SPLITS

OUT = Path("outputs/wsT0")
CACHE = OUT / "cache"


# ------------------------------------------------------------- 工具

def auc_ties(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC，平局秩修正。x 连续/含平局，y∈{0,1}。"""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    n = len(xs)
    diff = np.concatenate(([True], xs[1:] != xs[:-1], [True]))
    idx = np.flatnonzero(diff)
    starts, ends = idx[:-1], idx[1:]
    avg_rank = (starts + 1 + ends) / 2.0  # 1-based 平均秩
    # 每段累计阳性数 → 段平均秩 × 段内阳性数 求和
    pos_cum = np.concatenate(([0], np.cumsum(ys)))
    pos_seg = pos_cum[ends] - pos_cum[starts]
    n1 = float(ys.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((avg_rank * pos_seg).sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fam_std_accum(preds: np.ndarray, w: np.ndarray):
    """跨族标准差（无额外大临时量）：返回 (unweighted, weighted) 两个 (N,P) f32。"""
    k = preds.shape[0]
    m = np.zeros(preds.shape[1:], np.float64)
    s = np.zeros_like(m)
    mw = np.zeros_like(m)
    sw = np.zeros_like(m)
    for i in range(k):
        x = preds[i].astype(np.float64)
        m += x
        s += x * x
        mw += w[i] * x
        sw += w[i] * x * x
    m /= k
    var = np.clip(s / k - m * m, 0, None)
    var_w = np.clip(sw - mw * mw, 0, None)  # w 已归一
    return np.sqrt(var).astype(np.float32), np.sqrt(var_w).astype(np.float32)


# ------------------------------------------------------------- 主流程

def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()

    gr = json.loads(Path("outputs/wsN11/grand_router.json").read_text())
    names = gr["names"]
    r = 0.7
    w_glob = np.array(gr["w_global"])
    w_glob = w_glob / w_glob.sum()
    best_w = {sp: np.array(gr["best_w_split"][sp]) for sp in SPLITS}
    w_use = {sp: r * best_w[sp] + (1 - r) * w_glob for sp in SPLITS}

    files = [(n, p) for n, p in CANDIDATES if n in names and Path(p).exists()]
    assert len(files) == len(names), f"缺族文件: {set(names)-{n for n,_ in files}}"

    # ---- 1. 路由基线（有缓存直接复用）----
    routed_path = CACHE / "routed_r07_trainval.npy"
    x_std_path, x_std_w_path = CACHE / "x_std.npy", CACHE / "x_std_w.npy"
    if routed_path.exists() and x_std_path.exists() and x_std_w_path.exists():
        routed = np.load(routed_path)
        x_std = np.load(x_std_path)
        x_std_w = np.load(x_std_w_path)
        print("[base] routed/x_std 缓存命中，跳过 21 族重载", flush=True)
    else:
        preds = H.load_preds(files, h)  # (21, 8958, 5243) f32
        routed = np.empty(preds.shape[1:], dtype=np.float32)
        routed[h.tr_rows] = H.blend_rows(preds, w_glob, h.tr_rows)
        for sp in SPLITS:
            rows = h._fast[sp]["rows"]
            routed[rows] = H.blend_rows(preds, w_use[sp], rows)
        np.save(routed_path, routed)
        print(f"[base] routed r=0.7 saved -> {routed_path}", flush=True)
        # ---- 2. 跨族标准差（pred 的 std == Δ̂ 的 std，锚点常量）----
        x_std, x_std_w = fam_std_accum(preds, w_glob)
        np.save(x_std_path, x_std)
        np.save(x_std_w_path, x_std_w)
        del preds
        print(f"[feat] x_std mean={x_std.mean():.4f} max={x_std.max():.3f}",
              flush=True)

    # ---- 3. 基线评分锚定（routed / routed+band γ1.3）----
    ch_path, lv_path = CACHE / "control_hat.npy", CACHE / "control_level.npy"
    if ch_path.exists() and lv_path.exists():
        ctrl_hat = np.load(ch_path)
        level = np.load(lv_path)
    else:
        from .wsE_depcal import build_control_hat
        ctrl_hat, level = build_control_hat(h)
        np.save(ch_path, ctrl_hat)
        np.save(lv_path, level)
        print("[feat] control_hat 重建并缓存于 wsT0/cache", flush=True)
    res_base = h.score_val(routed, verbose=False)
    routed_cal = band_calibrate(routed, ctrl_hat, level, h.is_treat_tr, g=1.3)
    np.save(CACHE / "routed_r07_band13_trainval.npy", routed_cal)
    res_cal = h.score_val(routed_cal, verbose=False)
    f1_base = np.mean([res_base["per_split"][sp]["DEP_F1"] for sp in SPLITS])
    f1_cal = np.mean([res_cal["per_split"][sp]["DEP_F1"] for sp in SPLITS])
    print(f"[anchor] routed composite={res_base['composite']:.4f} (DEP_F1 {f1_base:.4f})"
          f" | +band1.3 composite={res_cal['composite']:.4f} (DEP_F1 {f1_cal:.4f})",
          flush=True)

    # ---- 4. train 处理行条目级特征与标签 ----
    rows_tr = h.tr_rows[h.is_treat_tr[h.tr_rows]]
    dt = h.delta_tr_all[rows_tr].astype(np.float64)           # Δ_true
    d_est = (routed[rows_tr] - ctrl_hat[rows_tr]).astype(np.float64)  # Δ̂_ens
    avail = ~np.isnan(dt)
    hi = avail & (np.abs(dt) > 1.0)
    pred_hi = avail & (np.abs(d_est) > 1.0)
    base_rate = hi.sum() / avail.sum()
    print(f"[train] treated rows={len(rows_tr)} entries={avail.sum():,} "
          f"DEP base rate={base_rate:.4f} pred_hi rate={pred_hi.sum()/avail.sum():.4f}",
          flush=True)

    # 蛋白级 train DEP 率（加性平滑）
    cnt_hi = hi.sum(0).astype(np.float64)
    cnt_av = avail.sum(0).astype(np.float64)
    p0 = hi.sum() / max(avail.sum(), 1)
    prot_dep_rate = (cnt_hi + 2 * p0) / (cnt_av + 2)
    np.save(CACHE / "prot_dep_rate.npy", prot_dep_rate.astype(np.float32))

    # ---- 5. train AUC 表 ----
    av = avail.ravel()
    y = hi.ravel()[av].astype(np.int8)
    f_abs = np.abs(d_est).ravel()[av]
    f_std = x_std[rows_tr].ravel()[av].astype(np.float64)
    f_std_w = x_std_w[rows_tr].ravel()[av].astype(np.float64)
    f_prot = np.broadcast_to(prot_dep_rate, dt.shape).ravel()[av]

    q = {}
    q["Q1_absDE"] = auc_ties(f_abs, y)
    q["Q2a_std"] = auc_ties(f_std, y)
    q["Q2b_std_w"] = auc_ties(f_std_w, y)
    q["Q2c_prot_rate"] = auc_ties(f_prot, y)
    # Q3: 边际带内（|Δ̂|∈[0.3,1)）的 FN 判别
    marg = (f_abs >= 0.3) & (f_abs < 1.0)
    q["Q3_std_in_marginal"] = auc_ties(f_std[marg], y[marg])
    q["Q3_n_marginal"] = int(marg.sum())
    q["Q3_fn_rate_in_marginal"] = float(y[marg].mean())
    print("[train AUC] " + " | ".join(
        f"{k}={v:.4f}" for k, v in q.items() if isinstance(v, float)), flush=True)

    # Q4: 三特征 logistic（按样本 GroupKFold，全阳性+随机阴性子采样）
    q4 = _logit_cv(f_abs, f_std, f_prot, y, rows_tr, avail)
    q.update(q4)
    print(f"[train AUC] Q4_logit3_cv={q4['Q4_logit3_cv']:.4f} "
          f"(vs Q1 {q['Q1_absDE']:.4f})", flush=True)

    # ---- 6. val 单次确认表 ----
    val_tbl = {}
    for sp in SPLITS:
        rows = h._fast[sp]["rows"]
        trows = rows[h.is_treat_tr[rows]]
        dtv = h.delta_tr_all[trows].astype(np.float64)
        dv = (routed[trows] - ctrl_hat[trows]).astype(np.float64)
        avv = ~np.isnan(dtv)
        yv = (avv & (np.abs(dtv) > 1.0)).ravel()
        avf = avv.ravel()
        fv_abs = np.abs(dv).ravel()[avf]
        fv_std = x_std[trows].ravel()[avf].astype(np.float64)
        fv_prot = np.broadcast_to(prot_dep_rate, dtv.shape).ravel()[avf]
        yv = yv[avf].astype(np.int8)
        mv = (fv_abs >= 0.3) & (fv_abs < 1.0)
        val_tbl[sp] = {
            "AUC_absDE": auc_ties(fv_abs, yv),
            "AUC_std": auc_ties(fv_std, yv),
            "AUC_prot": auc_ties(fv_prot, yv),
            "AUC_std_in_marginal": auc_ties(fv_std[mv], yv[mv]) if mv.any() else None,
            "dep_base_rate": float(yv.mean()),
        }
        print(f"[val {sp}] " + " ".join(
            f"{k}={v:.4f}" for k, v in val_tbl[sp].items()
            if isinstance(v, float)), flush=True)

    # ---- 7. 预注册判定 ----
    green_train = (q["Q3_std_in_marginal"] >= 0.60
                   or q["Q4_logit3_cv"] >= q["Q1_absDE"] + 0.02)
    val_std = np.nanmean([v["AUC_std_in_marginal"] for v in val_tbl.values()])
    green_val = abs(val_std - q["Q3_std_in_marginal"]) <= 0.03 or val_std >= 0.58
    verdict = "GREEN" if (green_train and green_val) else (
        "YELLOW" if green_train else "RED")
    print(f"\n[verdict] {verdict}  (train Q3={q['Q3_std_in_marginal']:.4f} "
          f"val Q3 mean={val_std:.4f})", flush=True)

    report = {
        "anchor": {"routed": res_base, "routed_band13": res_cal},
        "train": {"base_rate": float(base_rate),
                  "pred_hi_rate": float(pred_hi.sum() / avail.sum()),
                  **{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k, v in q.items()}},
        "val": val_tbl,
        "verdict": verdict,
        "seconds": time.time() - t0,
    }
    (OUT / "varcheck.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"[saved] {OUT/'varcheck.json'} ({time.time()-t0:.0f}s)", flush=True)


def _logit_cv(f_abs, f_std, f_prot, y, rows_tr, avail, rng_seed=0):
    """三特征 logistic 的样本分组 5 折 CV AUC（全阳性 + 3×阴性子采样）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(rng_seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg_sub = rng.choice(neg, size=min(len(neg), 3 * len(pos)), replace=False)
    sel = np.concatenate([pos, neg_sub])
    X = np.stack([f_abs[sel], f_std[sel], f_prot[sel]], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ys = y[sel]
    # 组 = 样本行号（ravel 顺序下每个可用条目所属的处理行）
    groups = np.repeat(np.arange(len(rows_tr)), avail.sum(1))[sel]
    oof = np.zeros(len(sel))
    gkf = GroupKFold(n_splits=5)
    for tr_i, te_i in gkf.split(X, ys, groups):
        lr = LogisticRegression(max_iter=300, C=1.0)
        lr.fit(X[tr_i], ys[tr_i])
        oof[te_i] = lr.predict_proba(X[te_i])[:, 1]
    return {"Q4_logit3_cv": auc_ties(oof, ys),
            "Q4_n_sub": int(len(sel)), "Q4_pos": int(len(pos))}


if __name__ == "__main__":
    main()
