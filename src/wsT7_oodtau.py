"""wsT7：OOD 感知阈值规则（C8 候选）——LOSO/LOCO 模拟 tuning 修复 τ 的 val 迁移差。

机理（C6 失败的正面教训）：C1 的 τ=0.35 在"分布内 OOF p"上调定；val 的
strain/both 划分对分类器是 OOD，同一 τ 对应完全不同的工作点。本模块不碰 val
调 τ，而是**在 train 内做 OOD 模拟**：留出菌株（LOSO）/留出化合物（LOCO）/
双留出（BOTH）分组折，得到"仿佛该菌株/化合物未见"的 p_lo，在其上调 τ：
  τ_chem ← LOCO p 上调（模拟 val_chem_only）
  τ_strain ← LOSO p 上调（模拟 val_strain_only）
  τ_both ← 双留出 p 上调（模拟 val_both）
  τ_time ← 普通 OOF（时间插值为分布内）
规则按 test 行 seen/unseen 元数据套用（route_keys 同口径），**τ 全程 train-only**。

预注册：
- 副作用预算同 wsT1（proxy ≤0.010、fid ≤0.002，在各自 p_lo 子集上计）。
- 与 C7（wsT6 Messner 特征）共用**同一张 val 裁决表**（一次看）：
  候选 = {C7（若过闸）, C8, C7+C8 组合}；采纳门槛不变（composite ≥ 0.5541−0.0005
  且 F1 ≥ 0.2442+0.003）。
- 若 C8 单过而 C7 未过闸：C8 单独上看。

合规：train-only 全部拟合与调参；val 一次；Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsT7_oodtau --stage fit    # 3 套留出 p（~20min）
      python -m src.wsT7_oodtau --stage tune   # 调 τ 表（train）
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .evaluate import Harness

OUT = Path("outputs/wsT7")
CACHE = OUT / "cache"
T1C = Path("outputs/wsT1/cache")


def _fit_hgb(X, y, w, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=seed)
    clf.fit(X, y, sample_weight=w)
    return clf


def _sub(y, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    ns = rng.choice(neg, size=min(len(neg), 5 * len(pos)), replace=False)
    sel = np.concatenate([pos, ns])
    w = np.ones(len(sel), np.float32)
    w[len(pos):] = len(neg) / max(len(ns), 1)
    return sel, w


def cmd_fit():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()
    h = Harness()
    rows = np.load(T1C / "train_rows.npy")
    avail = np.load(T1C / "train_avail.npy")
    hi = np.load(T1C / "train_hi.npy")
    X = np.load(T1C / "train_X.npy")
    m_rows = h.m_tr.iloc[rows].reset_index(drop=True)
    strains = m_rows["Strains"].to_numpy()
    chems = m_rows["perturbation_no_concentration"].to_numpy()

    Xa = X.reshape(-1, X.shape[-1])[avail.ravel()]
    ya = hi.ravel()[avail.ravel()]
    row_of = np.repeat(np.arange(len(rows)), avail.sum(1))

    u_strains = np.unique(strains)
    rng = np.random.default_rng(7)
    u_chems = np.unique(chems)
    chem_grp = {c: i % 8 for i, c in enumerate(
        rng.permutation(u_chems))}  # 8 个化合物折
    cg = np.array([chem_grp[c] for c in chems])

    p_loso = np.full(len(ya), np.nan, np.float32)
    p_loco = np.full(len(ya), np.nan, np.float32)
    p_both = np.full(len(ya), np.nan, np.float32)
    for si, s in enumerate(u_strains):
        te_rows = np.where(strains == s)[0]
        te_mask = np.isin(row_of, te_rows)
        tr_mask = ~np.isin(row_of, np.where(strains == s)[0])
        # LOSO：仅按菌株留出
        sel, w = _sub(ya[tr_mask], si)
        clf = _fit_hgb(Xa[tr_mask][sel], ya[tr_mask][sel], w, si)
        p_loso[te_mask] = clf.predict_proba(Xa[te_mask])[:, 1]
        # BOTH：该菌株 + 一组化合物（1/8）双留出
        held_chems = [c for c in u_chems if chem_grp[c] == si % 8]
        both_rows = np.where((strains == s) & np.isin(chems, held_chems))[0]
        te_b = np.isin(row_of, both_rows)
        tr_b = ~np.isin(row_of, te_rows) & ~np.isin(
            row_of, np.where(np.isin(chems, held_chems))[0])
        sel, w = _sub(ya[tr_b], si + 100)
        clf = _fit_hgb(Xa[tr_b][sel], ya[tr_b][sel], w, si + 100)
        p_both[te_b] = clf.predict_proba(Xa[te_b])[:, 1]
        print(f"  [LOSO/BOTH] strain={s} done ({time.time()-t0:.0f}s)",
              flush=True)
    for gi in range(8):
        held = [c for c in u_chems if chem_grp[c] == gi]
        te_mask = np.isin(row_of, np.where(np.isin(chems, held))[0])
        tr_mask = ~te_mask
        sel, w = _sub(ya[tr_mask], gi + 200)
        clf = _fit_hgb(Xa[tr_mask][sel], ya[tr_mask][sel], w, gi + 200)
        p_loco[te_mask] = clf.predict_proba(Xa[te_mask])[:, 1]
        print(f"  [LOCO] group{gi} done ({time.time()-t0:.0f}s)", flush=True)

    np.save(CACHE / "p_loso.npy", p_loso)
    np.save(CACHE / "p_loco.npy", p_loco)
    np.save(CACHE / "p_both.npy", p_both)
    # 双留出覆盖行（train 处理行子集），供 tune 期子集评估
    both_cov_rows = np.zeros(len(rows), bool)
    for si, s in enumerate(u_strains):
        held_chems = [c for c in u_chems if chem_grp[c] == si % 8]
        both_cov_rows |= (strains == s) & np.isin(chems, held_chems)
    np.save(CACHE / "both_cov_rows.npy", both_cov_rows)
    cov = {"loso": float(np.isfinite(p_loso).mean()),
           "loco": float(np.isfinite(p_loco).mean()),
           "both": float(np.isfinite(p_both).mean()),
           "both_rows": int(both_cov_rows.sum())}
    print(f"[fit] coverage={cov} ({time.time()-t0:.0f}s)", flush=True)
    (CACHE / "fit_meta.json").write_text(json.dumps(cov, indent=1))


def cmd_tune():
    from .wsE_depcal import fast_dep_f1, train_side_effects
    from .wsT1_depgate import MIN_PUSH, _apply_push
    t0 = time.time()
    h = Harness()
    rows = np.load(T1C / "train_rows.npy")
    avail = np.load(T1C / "train_avail.npy")
    dt = np.load(T1C / "train_dt.npy").astype(np.float64)
    d_est = np.load(T1C / "train_dest.npy").astype(np.float64)
    ctrl_hat = np.load("outputs/wsT0/cache/control_hat.npy")
    offset = (ctrl_hat[rows].astype(np.float64)
              - h.Y_tr[rows].astype(np.float64) + dt)
    dp0 = d_est + offset
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float64)
    eff0 = train_side_effects(dt, dp0, mu_ctx, mu_drug)
    f1_0 = fast_dep_f1(dt, dp0)
    absDE = np.abs(d_est)

    def scan(p_flat, tag, row_sub=None):
        """row_sub: 只在这些行上调 τ/评 F1（both 双留出的覆盖子集）。"""
        sub = np.ones(len(rows), bool) if row_sub is None else row_sub
        P = np.full(avail.shape, np.nan, np.float32)
        ok = np.isfinite(p_flat)
        P[avail] = np.where(ok, p_flat, np.nan)
        best = None
        for tau in np.round(np.arange(0.05, 0.951, 0.05), 2):
            fl = (P >= tau) & (absDE >= MIN_PUSH) & avail
            fl[~sub] = False
            dp = _apply_push(dp0[sub], dp0[sub], fl[sub], "push")
            f1 = fast_dep_f1(dt[sub], dp)
            dp_full = _apply_push(dp0, dp0, fl, "push")
            eff = train_side_effects(dt, dp_full, mu_ctx, mu_drug)
            dmg = sum(eff0[k] - eff[k] for k in ("FC", "ctx", "drug"))
            if dmg <= 0.010 and (best is None or f1 > best[1]):
                best = (float(tau), f1, dmg, int(fl.sum()))
        print(f"  [τ-{tag}] τ*={best[0]:.2f} F1={best[1]:.4f} "
              f"dmg={best[2]:.4f} flags={best[3]:,}（base {f1_0:.4f}）",
              flush=True)
        return best

    res = {}
    res["tau_time"] = scan(np.load(T1C / "train_p_oof.npy"), "time(OOF)")
    res["tau_strain"] = scan(np.load(CACHE / "p_loso.npy"), "strain(LOSO)")
    res["tau_chem"] = scan(np.load(CACHE / "p_loco.npy"), "chem(LOCO)")
    pb = np.load(CACHE / "p_both.npy")
    cov_rows = np.load(CACHE / "both_cov_rows.npy")
    res["tau_both_cov_rows"] = int(cov_rows.sum())
    res["tau_both"] = scan(pb, "both(双留出)", row_sub=cov_rows)
    (CACHE / "taus.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[tune] done ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "tune"])
    args = ap.parse_args()
    {"fit": cmd_fit, "tune": cmd_tune}[args.stage]()


if __name__ == "__main__":
    main()
