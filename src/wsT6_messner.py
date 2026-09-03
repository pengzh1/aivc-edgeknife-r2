"""wsT6：Messner 2023 酵母 KO 蛋白组先验 → DEP 选择器增强（C7 候选）。

唯一新增外部信息源（research_20260826 #5）：Messner et al. Cell 2023，
4,699 基因敲除株 × 定量蛋白组（Mendeley w8jtmnszd9 v2，CC-BY，须登记披露）。
与已证伪的"全局签名特征"不同构：那是化合物-基因适应性签名作样本级特征；
这里是**蛋白级扰动响应倾向实测统计**（同物种、同蛋白组数据类型），
直接增强选择器的蛋白先验（现仅 43 化合物 4 菌株的 train 统计）。

stage feat：解析 stat_DE（校正 p 值矩阵，行=KO 株[UniProt]，列=蛋白 locus）
  + noimpute_wide（log2 丰度宽表）→ 蛋白级特征：
  ms_sig01/ms_sig05 = p_adj<0.01/0.05 的 KO 比例（响应广度）
  ms_absmed/ms_absq90 = |log2(KO/WT 中位)| 的中位/90 分位（响应幅度）
  ms_cov = 覆盖率；对未覆盖蛋白回退全局中位。
  经 extref/hop/gene2locus.json 桥接到我方 5,243 蛋白轴（覆盖 3,634=69%）。
stage run：10+4 特征重训 HGB（同 wsT1 协议：样本组 5 折 OOF、全阳性+5×阴性
  加权、同预算调 τ*）→ **预注册闸**：train OOF AUC(14feat) ≥ AUC(10feat)+0.002
  才花 val 那一眼；过闸则 C7 单次裁决：composite ≥ 0.5541−0.0005 且
  F1 ≥ 0.2442+0.003 → 替换 C1，否则关闭。

合规：统计/拟合全 train-only + 公开外部数据（登记）；val 至多一次；Y_te 零接触。

用法: python -m src.wsT6_messner --stage feat
      python -m src.wsT6_messner --stage run
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Harness
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsT6")
CACHE = OUT / "cache"
MSDIR = Path("extref/messner2023")
T0C = Path("outputs/wsT0/cache")
T1C = Path("outputs/wsT1/cache")


def cmd_feat():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()

    prots = np.load("cache/proteome_log2.npz")["proteins"].astype(str)
    g2l = json.loads(Path("extref/hop/gene2locus.json").read_text())

    # ---- UniProt 桥：accession → {gene names, locus} ----
    up = pd.read_csv(MSDIR / "uniprot_sgd_map.tsv", sep="\t")
    up_gene = {}   # 名字（gene 或 locus）→ accession
    for _, r in up.iterrows():
        names = set()
        for col in ("Gene Names", "Gene Names (ordered locus)"):
            v = r.get(col)
            if isinstance(v, str):
                names.update(v.split())
        for nm in names:
            up_gene.setdefault(nm, r["Entry"])
    print(f"[feat] UniProt 桥 {len(up)} 条 / {len(up_gene)} 名", flush=True)

    # ---- stat_DE：行=蛋白(UniProt)，列=KO 株；逐蛋白响应广度 ----
    de = pd.read_csv(MSDIR / "yeast5k_stat_DE.csv").set_index("Protein.Group")
    P = de.to_numpy(dtype=np.float32)  # (n_prot_ms, n_ko)
    print(f"[feat] stat_DE 矩阵 {P.shape}（蛋白 × KO 株）", flush=True)
    with np.errstate(invalid="ignore"):
        meas = ~np.isnan(P)
        n_meas = np.maximum(meas.sum(1), 1)
        ms_sig01 = (meas & (P < 0.01)).sum(1) / n_meas
        ms_sig05 = (meas & (P < 0.05)).sum(1) / n_meas
        ms_cov = meas.sum(1) / P.shape[1]
    row_acc = de.index.to_numpy()
    feat_rows = {"ms_sig01": ms_sig01, "ms_sig05": ms_sig05}

    # ---- wide：行=蛋白(UniProt)，列=样本；逐蛋白响应幅度（剔除 QC 样本）----
    wide_path = MSDIR / "yeast5k_noimpute_wide.csv"
    if wide_path.exists() and wide_path.stat().st_size > 150e6:
        meta = pd.read_csv(MSDIR / "yeast5k_metadata.csv")
        qc_files = set(meta.loc[meta["sampletype"] == "qc", "Filename"])
        w = pd.read_csv(wide_path).set_index("Protein.Group")
        keep_cols = [c for c in w.columns if c not in qc_files
                     and "_qc_" not in c]
        Wm = w[keep_cols].to_numpy(dtype=np.float32)
        if np.nanmedian(Wm) > 100:  # 原始强度 → log2
            Wm = np.log2(np.where(Wm > 0, Wm, np.nan))
            print("[feat] wide 判为原始强度，已 log2", flush=True)
        med = np.nanmedian(Wm, axis=1, keepdims=True)
        D = np.abs(Wm - med)
        meas = ~np.isnan(D)
        cnt = meas.sum(1)
        ms_absmed = np.array([np.nanmedian(D[i][meas[i]]) if cnt[i] > 0
                              else np.nan for i in range(D.shape[0])],
                             np.float32)
        ms_absq90 = np.array([np.nanquantile(D[i][meas[i]], 0.9)
                              if cnt[i] > 10 else np.nan
                              for i in range(D.shape[0])], np.float32)
        amp = pd.Series(ms_absmed, index=w.index)
        q90 = pd.Series(ms_absq90, index=w.index)
        cov = pd.Series(cnt / max(Wm.shape[1], 1), index=w.index)
        feat_rows["ms_absmed"] = amp.reindex(row_acc).to_numpy(np.float32)
        feat_rows["ms_absq90"] = q90.reindex(row_acc).to_numpy(np.float32)
        feat_rows["ms_cov"] = cov.reindex(row_acc).to_numpy(np.float32)
        print(f"[feat] wide 幅度特征完成（{len(keep_cols)} 样本列）",
              flush=True)
    else:
        for k in ("ms_absmed", "ms_absq90", "ms_cov"):
            feat_rows[k] = np.full(len(row_acc), np.nan, np.float32)
        print("[feat] wide 未齐，幅度/覆盖特征置 NaN 回退", flush=True)

    # ---- 映射到我方蛋白轴（先 gene 名直查，再 locus 桥）----
    acc_idx = {a: i for i, a in enumerate(row_acc)}
    out = {}
    for k, v in feat_rows.items():
        arr = np.full(len(prots), np.nan, np.float32)
        hit = 0
        for j, p in enumerate(prots):
            acc = up_gene.get(p) or up_gene.get(g2l.get(p, ""))
            if acc in acc_idx:
                arr[j] = v[acc_idx[acc]]
                hit += 1
        fallback = np.nanmedian(arr)
        out[k] = np.where(np.isnan(arr), fallback, arr)
        if k == "ms_sig01":
            print(f"[feat] 映射命中 {hit}/{len(prots)}", flush=True)
        print(f"[feat] {k}: 回退值={fallback:.4f} "
              f"range=[{out[k].min():.4f},{out[k].max():.4f}]", flush=True)
    np.savez(CACHE / "ms_prot_feats.npz", **out)
    print(f"[feat] saved {CACHE/'ms_prot_feats.npz'} ({time.time()-t0:.0f}s)",
          flush=True)


def cmd_run():
    """14 特征重训选择器（wsT1 协议复刻）+ 预注册闸 + 过闸后的单次 val 裁决。"""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from .wsT0_varcheck import auc_ties
    from .wsE_depcal import fast_dep_f1, train_side_effects
    from .wsT1_depgate import (MIN_PUSH, PUSH_TO, _apply_push, _fidelity_proxy,
                               _predict_val)
    from . import metrics as M

    t0 = time.time()
    h = Harness()
    h.prepare_fast_eval()
    msf = np.load(CACHE / "ms_prot_feats.npz")
    ms_names = list(msf.keys())

    X10 = np.load(T1C / "train_X.npy")
    avail = np.load(T1C / "train_avail.npy")
    hi = np.load(T1C / "train_hi.npy")
    dt = np.load(T1C / "train_dt.npy").astype(np.float64)
    d_est = np.load(T1C / "train_dest.npy").astype(np.float64)
    rows = np.load(T1C / "train_rows.npy")
    ms_block = np.stack([np.broadcast_to(msf[k], avail.shape) for k in ms_names],
                        axis=-1).astype(np.float32)
    X14 = np.concatenate([X10, ms_block], axis=-1)

    n_rows = avail.shape[0]
    Xa14 = X14.reshape(-1, X14.shape[-1])[avail.ravel()]
    Xa10 = X10.reshape(-1, X10.shape[-1])[avail.ravel()]
    ya = hi.ravel()[avail.ravel()]
    groups = np.repeat(np.arange(n_rows), avail.sum(1))

    def fit_sub(X, y, seed):
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y)[0], np.where(~y)[0]
        neg_sub = rng.choice(neg, size=min(len(neg), 5 * len(pos)),
                             replace=False)
        sel = np.concatenate([pos, neg_sub])
        w = np.ones(len(sel), np.float32)
        w[len(pos):] = len(neg) / max(len(neg_sub), 1)
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, random_state=0)
        clf.fit(X[sel], y[sel], sample_weight=w)
        return clf

    # ---- 5 折 OOF（10 vs 14 特征）----
    p10 = np.zeros(len(ya), np.float32)
    p14 = np.zeros(len(ya), np.float32)
    gkf = GroupKFold(n_splits=5)
    for k, (tr_i, te_i) in enumerate(gkf.split(Xa14, ya, groups)):
        c10 = fit_sub(Xa10[tr_i], ya[tr_i], k)
        p10[te_i] = c10.predict_proba(Xa10[te_i])[:, 1]
        c14 = fit_sub(Xa14[tr_i], ya[tr_i], k)
        p14[te_i] = c14.predict_proba(Xa14[te_i])[:, 1]
        print(f"  fold{k} done", flush=True)
    a10 = auc_ties(p10, ya.astype(np.int8))
    a14 = auc_ties(p14, ya.astype(np.int8))
    print(f"[gate] OOF AUC 10feat={a10:.4f} 14feat={a14:.4f} "
          f"Δ={a14-a10:+.4f}", flush=True)

    report = {"auc10": a10, "auc14": a14}
    if a14 < a10 + 0.002:
        print("[gate] 未过闸（<+0.002）→ 关闭，不花 val 那一眼", flush=True)
        report["verdict"] = "CLOSED_AT_GATE"
        (CACHE / "wsT6_report.json").write_text(json.dumps(report, indent=1))
        return

    # ---- 过闸：全量终模 + τ 调定（wsT1 同预算）+ 保存 val 侧 P14（不评分）----
    clf14 = fit_sub(Xa14, ya, 999)
    joblib.dump(clf14, CACHE / "hgb14_full.joblib")
    routed = np.load(T0C / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0C / "control_hat.npy")
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    dp0 = d_est + offset
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    eff0 = train_side_effects(dt, dp0, mu_ctx, mu_drug)
    fid0 = _fidelity_proxy(h, rows, routed, d_est, np.zeros_like(d_est))
    P = np.full(avail.shape, np.nan, np.float32)
    P[avail] = p14
    absDE = np.abs(d_est)
    best = None
    for tau in np.round(np.arange(0.05, 0.951, 0.05), 2):
        fl = (P >= tau) & (absDE >= MIN_PUSH) & avail
        dp = _apply_push(dp0, dp0, fl, "push")
        eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
        dmg = sum(eff0[k] - eff[k] for k in ("FC", "ctx", "drug"))
        fid_drop = fid0 - _fidelity_proxy(h, rows, routed, d_est, dp - dp0)
        f1 = fast_dep_f1(dt, dp)
        ok = dmg <= 0.010 and fid_drop <= 0.002
        if ok and (best is None or f1 > best[1]):
            best = (tau, f1, dmg)
    tau_c7, f1_c7, dmg_c7 = best
    print(f"[C7] train τ*={tau_c7:.2f} F1={f1_c7:.4f} dmg={dmg_c7:.4f}",
          flush=True)

    # val 侧 P14 落盘（裁决统一在 wsT8 一次看）
    for sp in SPLITS:
        Xv10 = np.load(T1C / f"val_X_{sp}.npy")
        Xv14 = np.concatenate([Xv10, np.stack(
            [np.broadcast_to(msf[k], Xv10.shape[:2]) for k in ms_names],
            axis=-1).astype(np.float32)], axis=-1)
        Pv = clf14.predict_proba(Xv14.reshape(-1, Xv14.shape[-1]))[:, 1]
        np.save(CACHE / f"P14_{sp}.npy", Pv.reshape(Xv10.shape[:2]))
    report.update({"tau": tau_c7, "train_F1": f1_c7, "verdict": "GATE_PASSED"})
    (CACHE / "wsT6_report.json").write_text(json.dumps(report, indent=1,
                                                       default=float))
    print(f"[done] run in {time.time()-t0:.0f}s（val 裁决在 wsT8）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["feat", "run"])
    args = ap.parse_args()
    {"feat": cmd_feat, "run": cmd_run}[args.stage]()


if __name__ == "__main__":
    main()
