"""wsN13: 大阵容最终提交生成器（wsN11 稠密路由权重 → test 路由 → DEP 校准
→ CRD 有界迁移 → 列裁剪 → 5.2.4 认证）。

与 make_submission_trainonly / wsN6_deliver 的关系：同流程，权重来源换为
wsN11 的 grand_router.json（含 16 族与新角色权重），其余纪律不变：
- seen 集合 = train split；DEP 校准对照池/回退均值 train-only；
- CRD←CGD 有界迁移 β=0.35（wsN8 机制）；
- 输出 5,243 全量 + 4,422 裁剪 + 认证报告。

用法: python -m src.wsN13_final                  # 用 grand_router.json 的 best_r
      python -m src.wsN13_final --r 0.5          # 指定收缩档
      python -m src.wsN13_final --no-crd         # 不做 CRD 迁移（兜底版）
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Harness
from . import make_submission_trainonly as MST
from .make_submission import band_calibrate
from . import data as D
from .wsN8_crdtransfer import strain_modulation, apply_transfer

OUT = Path("outputs/wsN13")

# 角色 → val 划分（权重键）
ROLE_OF = {"val_chem_only": "chem_only", "val_strain_only": "strain_only",
           "val_both": "both", "val_time": "time"}

# 各族 test 预测（与 wsN11 CANDIDATES 对齐；wsG→并 wsD_s16）
TEST_NPY = {
    "wsD_s16": "outputs/wsN14/pred_test_wsD_s16.npy",
    "wsD_g2g3": "outputs/wsM/pred_test_wsD_trainonly.npy",
    "wsF": "outputs/wsN15/pred_test_wsF_s8.npy",
    "wsC_g3": "outputs/wsM/pred_test_wsC_trainonly.npy",
    "wsB": "outputs/wsN16/pred_test_wsB_s16.npy",
    "wsBg2": None,   # wsN9 未产 test 版（权重并入 wsB 同族处理见下）
    "ridge": "outputs/wsM/pred_test_ridge_trainonly.npy",
    "wsA": "outputs/wsM/pred_test_wsA_trainonly.npy",
    "fuse": "outputs/wsN16/pred_test_fuse_s16.npy",
    "fuseDeep3": "outputs/wsN20/pred_test_deep3_s16.npy",
    "fuse3e150": "outputs/wsN24/pred_test.npy",
    "wsN7": None,    # 负结果族
    "wsN1": "outputs/wsN/pred_test.npy",
    "wsD_bag0": "outputs/wsN16/pred_test_bag0.npy",
    "wsD_bag1": "outputs/wsN16/pred_test_bag1.npy",
    "wsD_deep6": "outputs/wsN10/pred_test_deep6.npy",
    "wsD_wide": "outputs/wsN10/pred_test_wide.npy",
    "wsD_mse": "outputs/wsN10/pred_test_mse.npy",
    "wsD_ep500": "outputs/wsN10/pred_test_ep500.npy",
    "wsD_drop02": "outputs/wsN10/pred_test_drop02.npy",
}
MERGE_INTO = {"wsG": "wsD_s16", "wsBg2": "wsB", "wsN7": "wsD_s16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=None)
    ap.add_argument("--no-crd", action="store_true")
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gr = json.loads(Path("outputs/wsN11/grand_router.json").read_text())
    names = gr["names"]
    r = args.r if args.r is not None else gr["best_r"]
    print(f"[final] r={r} (best_r={gr['best_r']}, scan={gr['r_scan']})")
    w_glob = np.array(gr["w_global"])
    best_w = {sp: np.array(gr["best_w_split"][sp]) for sp in ROLE_OF}
    w_use = {sp: r * best_w[sp] + (1 - r) * w_glob for sp in ROLE_OF}

    h = Harness()
    # 载入存在的 test 预测；无 test 版族的权重做合并/归零处理
    preds, idx_map = {}, {}
    for i, n in enumerate(names):
        p = TEST_NPY.get(n)
        if p and Path(p).exists():
            preds[n] = np.load(p)
    keys = MST.route_keys_trainonly(h)
    pred = np.zeros_like(next(iter(preds.values())))
    for sp, role in ROLE_OF.items():
        mask = keys == role
        if not mask.any():
            continue
        w = dict(zip(names, w_use[sp]))
        for dead, target in MERGE_INTO.items():
            if dead in w:
                w[target] = w.get(target, 0.0) + w.pop(dead)
        for dead in list(w):
            if dead not in preds:
                w.pop(dead)  # 无 test 预测族：权重丢弃（剩余归一）
        w = {k2: v for k2, v in w.items() if v > 0}
        s = sum(w.values())
        pred[mask] = (sum(wt * preds[m][mask] for m, wt in w.items())
                      / s).astype(np.float32)
        print(f"  {role:<12} n={int(mask.sum()):>5} weights="
              f"{ {k2: round(v, 3) for k2, v in w.items()} }")

    # DEP 校准（train-only）
    ctrl_hat_te, level_te = MST.build_control_hat_train(h, h.m_te)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    pred_cal = band_calibrate(pred, ctrl_hat_te, level_te, is_treat_te, g=1.3)

    # CRD 有界迁移
    if not args.no_crd:
        mod = strain_modulation(h)
        pred_cal, n_hit = apply_transfer(pred_cal, h, h.m_te, None, mod,
                                         "CGD", "CRD", args.beta)
        print(f"[crd] β={args.beta} 命中 {n_hit} 行")
    assert np.isfinite(pred_cal).all()

    tag = args.tag or (f"r{r}" + ("_nocrd" if args.no_crd else ""))
    df = pd.DataFrame(pred_cal, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    keep = pd.read_csv("outputs/wsM/keep_proteins_train_miss80.csv")["protein"]
    out_full = OUT / f"prediction_grand_{tag}.csv"
    df.to_csv(out_full, index=False)
    out_crop = OUT / f"prediction_grand_{tag}_keepcols.csv"
    df[["sample_ID"] + keep.tolist()].to_csv(out_crop, index=False)
    np.save(out_full.with_suffix(".npy"), pred_cal)

    # 5.2.4 认证
    mt = pd.read_csv("input/WAYB_WAYC_metadata_test(1).csv")
    vals = pred_cal
    ok = {"rows": len(df) == 4454,
          "id_order": bool((df["sample_ID"].to_numpy()
                            == mt["sample_ID"].to_numpy()).all()),
          "finite": bool(np.isfinite(vals).all()),
          "cols_keep": list(df.columns[1:]) == h.proteins.tolist(),
          "range": [float(vals.min()), float(vals.max())]}
    print("[certify]", ok)
    (OUT / f"certify_{tag}.json").write_text(json.dumps(ok, indent=1))
    print(f"[saved] {out_full} + {out_crop.name}")


if __name__ == "__main__":
    main()
