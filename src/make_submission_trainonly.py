"""train-only 回退版提交生成器（双榜合并后的唯一提交）。

与 make_submission_v2 的差异（2026-08-10 组委会答复驱动）：
1. ROSTER = outputs/wsM/*_trainonly.npy（split_final=train 重训，见 wsM_trainonly）；
2. 路由 seen 集合 = train split 实见类别（BAI 与 val 化合物不再"已见"）；
3. 路由权重 = wsH v2 逻辑重拟合（wsB 换严格版 val 预测
   outputs/wsM/pred_trainval_wsB_strict.npy，其余沿用既有 train-split 版）；
4. DEP 带门校准的对照池（train 对照 751 行）与回退均值（Y_train）全部
   train-only（v2 用全量 train_val 对照 956 行与全量均值）；
5. 输出 5243 列全量版 + train 缺失率<80% 过滤的 4,422 列裁剪版
   （组委会口径 4,232 列按所述规则本地复现为 4,422，差异已另行邮件澄清；
   官方 feature contract 到达后按官方列表重裁）。

合规：h.Y_te 零接触；路由权重拟合仅用 val 预测（模型选择，规则允许）；
种子固定（wsH SEED=20260806）。

用法:
    python -m src.make_submission_trainonly                  # 路由重拟合 + 生成提交
    python -m src.make_submission_trainonly --skip-refit     # 复用已存权重文件
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluate import Harness
from .make_submission import CTRL_KEY_LEVELS, band_calibrate
from .wsH_router import (SPLITS, SplitScorer, build_candidates, blend_rows,
                         route_pred, run_set, search_global)
from . import wsH_router as H

OUT = Path("outputs/wsM")

# val 预测阵容（wsB 为严格版，其余为既有 train-split 版交付物）
VAL_CLOSED = [
    ("wsD_g2g3", "outputs/wsD/pred_trainval_g2g3.npy"),
    ("wsG", "outputs/wsG/pred_trainval.npy"),
    ("wsF", "outputs/wsF/pred_trainval.npy"),
    ("wsC_g3", "outputs/wsC/pred_trainval_g3.npy"),
    ("wsB", "outputs/wsM/pred_trainval_wsB_strict.npy"),
    ("ridge", "outputs/pred_ridge.npy"),
]
VAL_OPEN_EXTRA = ("wsA", "outputs/wsA/pred_trainval.npy")
# 与 wsH v2 相同的全局锚点（给定点优先规则）
V2_W_GLOBAL = np.array([0.765, 0.0, 0.0, 0.0, 0.21, 0.024])

# test 预测（wsM train-only 版；wsG 无 test 预测，权重并入 wsD——同 v2 约定）
TEST_ROSTER = {
    "wsD_g2g3": "outputs/wsM/pred_test_wsD_trainonly.npy",
    "wsF": "outputs/wsM/pred_test_wsF_trainonly.npy",
    "wsC_g3": "outputs/wsM/pred_test_wsC_trainonly.npy",
    "wsB": "outputs/wsM/pred_test_wsB_trainonly.npy",
    "ridge": "outputs/wsM/pred_test_ridge_trainonly.npy",
    "wsA": "outputs/wsM/pred_test_wsA_trainonly.npy",
}


def refit_router(h: Harness) -> dict:
    """wsH v2 开放阵容在 wsB-strict 上重拟合（r=0.5 收缩档）。返回记录。"""
    t0 = time.time()
    h.prepare_fast_eval()
    names = [n for n, _ in VAL_CLOSED]
    preds = H.load_preds(VAL_CLOSED, h)

    scorers = {sp: SplitScorer(h, sp) for sp in SPLITS}
    P_splits = {sp: np.ascontiguousarray(preds[:, h._fast[sp]["rows"], :])
                for sp in SPLITS}
    rng = np.random.default_rng(H.SEED)
    cand = build_candidates(len(VAL_CLOSED), rng)
    w_given = V2_W_GLOBAL / V2_W_GLOBAL.sum()
    w_srch, fast_c = search_global(scorers, P_splits, cand)
    all_rows = np.arange(len(h.m_tr))
    full_given = h.score_val(blend_rows(preds, w_given, all_rows)
                             .astype(np.float32), verbose=False)["composite"]
    full_srch = h.score_val(blend_rows(preds, w_srch, all_rows)
                            .astype(np.float32), verbose=False)["composite"]
    print(f"[router] 锚点: 给定 {full_given:.4f} vs 重搜 {full_srch:.4f}")
    if full_srch > full_given + 5e-4:
        w_anchor, which = w_srch, "重搜点"
    else:
        w_anchor, which = w_given, "给定点"

    files_o = VAL_CLOSED + [VAL_OPEN_EXTRA]
    w_anchor_o = np.concatenate([w_anchor, [0.0]])
    rec_o = run_set(h, files_o, w_anchor_o, "trainonly-open",
                    shrink_grid=(1.0, 0.7, 0.5, 0.3),
                    full_score_grid=(0.7, 0.5, 0.3))
    routed_val = rec_o.pop("routed_pred")  # r=0.5 档
    rec_o.pop("preds")
    np.save(OUT / "pred_trainval_routed_trainonly.npy",
            routed_val.astype(np.float32))
    rec = {"seed": H.SEED, "lineup": "v2-open(wsB-strict)",
           "anchor": {"which": which, "full_given": float(full_given),
                      "full_searched": float(full_srch),
                      "w_used": w_anchor.tolist()},
           "names": [n for n, _ in files_o],
           "open": rec_o,
           "routed_val_composite":
               rec_o["shrink"]["0.5"]["full"]["composite"],
           "seconds": time.time() - t0}
    (OUT / "router_weights_trainonly.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1, default=float))
    print(f"[router] r=0.5 routed val composite = "
          f"{rec['routed_val_composite']:.4f} ({time.time()-t0:.0f}s)")
    return rec


def build_control_hat_train(h: Harness, m_target: pd.DataFrame):
    """build_control_hat_for 的 train-only 版：对照池/回退均值仅 train split。"""
    m_train, Y_train = h.m_train, h.Y_train
    is_ctrl = m_train["perturbation_no_concentration"].isin(D.CONTROLS).to_numpy()
    mc = m_train[is_ctrl]
    Yc = Y_train[is_ctrl]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prot_mean = np.nanmean(Y_train, axis=0)
        scalar = np.nanmean(Y_train)  # 全缺失蛋白的最终回退
        prot_mean = np.where(np.isnan(prot_mean), scalar, prot_mean)
        glob = np.nanmean(Yc, axis=0)
        glob = np.where(np.isnan(glob), prot_mean, glob).astype(np.float32)

    out = np.repeat(glob[None, :], len(m_target), axis=0)
    hit = np.zeros(len(m_target), dtype=bool)
    level = np.full(len(m_target), len(CTRL_KEY_LEVELS), dtype=np.int8)
    for li, keys in enumerate(CTRL_KEY_LEVELS):
        km = list(map(tuple, mc[keys].to_numpy()))
        gm = {}
        df = pd.DataFrame({"k": km})
        for k, idx in df.groupby("k").groups.items():
            idx = np.fromiter(idx, dtype=int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(Yc[idx], axis=0)
            gm[k] = np.where(np.isnan(v), glob, v)
        mk = list(map(tuple, m_target[keys].to_numpy()))
        for i, k in enumerate(mk):
            if not hit[i] and k in gm:
                out[i] = gm[k]
                hit[i] = True
                level[i] = li
    return out.astype(np.float32), level


def route_keys_trainonly(h: Harness) -> np.ndarray:
    """test 角色映射：seen 集合 = train split 实见类别（BAI/val 化合物视为未见）。"""
    seen_strains = set(h.m_train["Strains"].unique())
    seen_chems = set(h.m_train["perturbation_no_concentration"].unique())
    keys = []
    for row in h.m_te.itertuples():
        s_seen = row.Strains in seen_strains
        c_seen = row.perturbation_no_concentration in seen_chems
        if s_seen and c_seen:
            keys.append("time")
        elif s_seen:
            keys.append("chem_only")
        elif c_seen:
            keys.append("strain_only")
        else:
            keys.append("both")
    return np.array(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--skip-refit", action="store_true",
                    help="复用既有 outputs/wsM/router_weights_trainonly.json")
    args = ap.parse_args()

    h = Harness()
    if args.skip_refit and (OUT / "router_weights_trainonly.json").exists():
        rec = json.loads((OUT / "router_weights_trainonly.json").read_text())
    else:
        rec = refit_router(h)

    names = rec["names"]  # [wsD_g2g3, wsG, wsF, wsC_g3, wsB, ridge, wsA]
    w05 = {sp: rec["open"]["shrink"]["0.5"][sp] for sp in SPLITS}
    # val 划分名 → test 角色名
    role_of = {"val_chem_only": "chem_only", "val_strain_only": "strain_only",
               "val_both": "both", "val_time": "time"}

    preds = {k: np.load(v) for k, v in TEST_ROSTER.items()}
    keys = route_keys_trainonly(h)
    dist = pd.Series(keys).value_counts().to_dict()
    print(f"[route] train-only 可见性 test 角色分布: {dist} "
          f"(官方 test 划分: chem 1640 / strain 1534 / both 1129 / time 151)")

    pred = np.zeros_like(next(iter(preds.values())))
    for sp, role in role_of.items():
        mask = keys == role
        if not mask.any():
            continue
        w = dict(zip(names, w05[sp]))
        w["wsD_g2g3"] = w.get("wsD_g2g3", 0.0) + w.pop("wsG", 0.0)  # wsG→wsD
        w = {k: v for k, v in w.items() if v > 0}
        s = sum(w.values())
        blend = sum(wt * preds[m][mask] for m, wt in w.items()) / s
        pred[mask] = blend.astype(np.float32)
        print(f"  {role:<12} n={int(mask.sum()):>5}  weights="
              f"{ {k: round(v, 3) for k, v in w.items()} }")

    # DEP 带门校准（train-only 对照池/回退均值；γ=1.3 不变，仅 7 键命中处理行）
    ctrl_hat_te, level_te = build_control_hat_train(h, h.m_te)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    print(f"[control_hat] level dist: "
          f"{np.bincount(level_te, minlength=7).tolist()} | 7k-hit treat rows: "
          f"{int((is_treat_te & (level_te == 0)).sum())}")
    pred_cal = band_calibrate(pred, ctrl_hat_te, level_te, is_treat_te,
                              g=args.gamma)
    assert np.isfinite(pred_cal).all()

    # ---- 输出 1：5243 列全量版 ----
    out_full = Path("outputs/prediction_trainonly.csv")
    df = pd.DataFrame(pred_cal, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    df.to_csv(out_full, index=False)
    np.save(out_full.with_suffix(".npy"), pred_cal)
    print(f"[saved] {out_full} shape={df.shape} "
          f"range=[{pred_cal.min():.2f},{pred_cal.max():.2f}]")

    # ---- 输出 2：train 缺失率<80% 过滤裁剪版（本地复现 4,422 列）----
    miss = np.isnan(h.Y_train).mean(axis=0)
    keep_mask = miss < 0.80
    keep = h.proteins[keep_mask]
    keep_df = pd.DataFrame({"protein": keep, "train_miss": miss[keep_mask]})
    keep_df.to_csv(OUT / "keep_proteins_train_miss80.csv", index=False)
    print(f"[keep] train miss<80%: {len(keep)} proteins "
          f"(saved {OUT/'keep_proteins_train_miss80.csv'})")
    out_crop = Path("outputs/prediction_trainonly_keepcols.csv")
    df[["sample_ID"] + keep.tolist()].to_csv(out_crop, index=False)
    print(f"[saved] {out_crop} shape=({len(df)}, {len(keep)+1})")


if __name__ == "__main__":
    main()
