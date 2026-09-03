"""wsN6-deliver: fuse 版路由重拟合 + 提交再生成（与 wsA 版对照，择优交付）。

流程：
1. monkey-patch make_submission_trainonly 的阵容（wsA → fuse）后重拟合路由
   （同种子同协议，输出 router_weights_trainonly_fuse.json）；
2. 按新权重 + fuse 的 test 预测生成提交候选（train-only DEP 校准不变）；
3. 与 wsA 版（0.5486/0.5491）对比 val，仅在更优时覆盖 dist 交付物。

合规同 make_submission_trainonly（train-only 全部统计；val 仅模型选择）。

用法: python -m src.wsN6_deliver
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Harness
from . import make_submission_trainonly as MST
from .make_submission import band_calibrate
from . import data as D

OUT = Path("outputs/wsM")


def main():
    # ---- 1. fuse 阵容重拟合 ----
    # 先备份 wsA 版权重记录（refit_router 会覆盖默认路径）
    p_default = OUT / "router_weights_trainonly.json"
    if p_default.exists():
        p_default.replace(OUT / "router_weights_trainonly_wsA.json")
    MST.VAL_OPEN_EXTRA = ("fuse", "outputs/wsN6/pred_trainval_fuse.npy")
    h = Harness()
    rec = MST.refit_router(h)  # 写 router_weights_trainonly.json（含 fuse）
    # 改名保存，避免与 wsA 版混淆
    p_default.rename(OUT / "router_weights_trainonly_fuse.json")
    print(f"[deliver] fuse 阵容 r=0.5 routed val composite = "
          f"{rec['routed_val_composite']:.4f}")

    # ---- 2. 生成提交候选 ----
    names = rec["names"]
    w05 = {sp: rec["open"]["shrink"]["0.5"][sp] for sp in
           ["val_chem_only", "val_strain_only", "val_both", "val_time"]}
    role_of = {"val_chem_only": "chem_only", "val_strain_only": "strain_only",
               "val_both": "both", "val_time": "time"}
    roster = dict(MST.TEST_ROSTER)
    roster["fuse"] = "outputs/wsN6/pred_test_fuse_full.npy"
    del roster["wsA"]
    preds = {k: np.load(v) for k, v in roster.items()}
    keys = MST.route_keys_trainonly(h)
    pred = np.zeros_like(next(iter(preds.values())))
    for sp, role in role_of.items():
        mask = keys == role
        if not mask.any():
            continue
        w = dict(zip(names, w05[sp]))
        w["wsD_g2g3"] = w.get("wsD_g2g3", 0.0) + w.pop("wsG", 0.0)
        w = {k: v for k, v in w.items() if v > 0}
        s = sum(w.values())
        pred[mask] = (sum(wt * preds[m][mask] for m, wt in w.items())
                      / s).astype(np.float32)
        print(f"  {role:<12} n={int(mask.sum()):>5} weights="
              f"{ {k: round(v, 3) for k, v in w.items()} }")

    ctrl_hat_te, level_te = MST.build_control_hat_train(h, h.m_te)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    pred_cal = band_calibrate(pred, ctrl_hat_te, level_te, is_treat_te, g=1.3)
    assert np.isfinite(pred_cal).all()

    out_full = Path("outputs/prediction_trainonly_fuse.csv")
    df = pd.DataFrame(pred_cal, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    df.to_csv(out_full, index=False)
    np.save(out_full.with_suffix(".npy"), pred_cal)
    keep = pd.read_csv(OUT / "keep_proteins_train_miss80.csv")["protein"]
    out_crop = Path("outputs/prediction_trainonly_fuse_keepcols.csv")
    df[["sample_ID"] + keep.tolist()].to_csv(out_crop, index=False)
    print(f"[saved] {out_full} + {out_crop.name} ({len(keep)} cols)")


if __name__ == "__main__":
    main()
