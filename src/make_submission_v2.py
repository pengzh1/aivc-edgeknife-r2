"""v2 最终提交生成：分角色路由集成 + DEP 带门校准。

路由权重来自 wsH v2（router_weights_v2.json，r=0.5 收缩档）。
test 角色映射按"全量重训后可见性"：
- CRD（strain_role=test）× 已见化合物（train|val）→ strain_only 权重
- 已见菌株（train 4 株 + BAI）× test 化合物 → chem_only 权重
- CRD × test 化合物 → both 权重（双盲）
- 其余（双已见）→ time/global 权重

开放榜：chem_only 行换用含 wsA 的混合（wsG 权重并入 wsD_g2，同配方）。

用法:
    python -m src.make_submission_v2 --track closed
    python -m src.make_submission_v2 --track open
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluate import Harness
from .make_submission import build_control_hat_for, band_calibrate

ROSTER = {
    "wsD": "outputs/wsD/pred_test_g2.npy",
    "wsB": "outputs/wsB/pred_test.npy",
    "wsC": "outputs/wsC/pred_test.npy",
    "wsF": "outputs/wsF/pred_test.npy",
    "wsA": "outputs/wsA/pred_test.npy",
    "ridge": "outputs/pred_test_ridge.npy",
}

# wsH v2 router_weights（r=0.5 收缩后），键为模型名（wsG 权重并入 wsD）
W_CLOSED = {
    "chem_only":  {"wsD": 0.883, "wsB": 0.105, "ridge": 0.012},
    "strain_only": {"wsD": 0.444, "wsB": 0.473, "ridge": 0.073},
    "both":       {"wsD": 0.407, "wsB": 0.227, "ridge": 0.362},
    "time":       {"wsD": 0.883, "wsB": 0.105, "ridge": 0.012},
}
W_OPEN = {
    "chem_only":  {"wsD": 0.641, "wsC": 0.112, "wsF": 0.067,
                   "wsB": 0.112, "wsA": 0.055, "ridge": 0.012},
    "strain_only": {"wsD": 0.444, "wsB": 0.473, "ridge": 0.073},
    "both":       {"wsD": 0.407, "wsB": 0.227, "ridge": 0.362},
    "time":       {"wsD": 0.641, "wsC": 0.112, "wsF": 0.067,
                   "wsB": 0.112, "wsA": 0.055, "ridge": 0.012},
}

SEEN_STRAINS = {"BAH", "BAI", "CEK", "CGD", "DHY210"}  # 全量重训后已见
SEEN_CHEMS_TRAINVAL = None  # 运行时从 m_tr 取


def route_keys(m_te: pd.DataFrame, seen_chems: set) -> np.ndarray:
    keys = []
    for row in m_te.itertuples():
        s_seen = row.Strains in SEEN_STRAINS
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
    ap.add_argument("--track", choices=["closed", "open"], required=True)
    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    h = Harness()
    W = W_CLOSED if args.track == "closed" else W_OPEN
    seen_chems = set(h.m_tr["perturbation_no_concentration"].unique())

    preds = {k: np.load(v) for k, v in ROSTER.items()
             if k in set().union(*[set(w) for w in W.values()])}
    keys = route_keys(h.m_te, seen_chems)
    print("[route]", pd.Series(keys).value_counts().to_dict())

    pred = np.zeros_like(next(iter(preds.values())))
    for rk in ["chem_only", "strain_only", "both", "time"]:
        mask = keys == rk
        if not mask.any():
            continue
        w = W[rk]
        s = sum(w.values())
        blend = sum(wt * preds[m][mask] for m, wt in w.items()) / s
        pred[mask] = blend.astype(np.float32)
        print(f"  {rk:<12} n={mask.sum():>5}  weights={w}")

    # DEP 带门校准（仅 7 键命中处理行）
    ctrl_rows = np.where(
        h.m_tr["perturbation_no_concentration"].isin(D.CONTROLS).to_numpy())[0]
    ctrl_hat_te, level_te = build_control_hat_for(h, h.m_te, ctrl_rows)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    pred_cal = band_calibrate(pred, ctrl_hat_te, level_te, is_treat_te,
                              g=args.gamma)

    out = Path(args.out or f"outputs/prediction_{args.track}.csv")
    df = pd.DataFrame(pred_cal, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    assert np.isfinite(pred_cal).all()
    df.to_csv(out, index=False)
    np.save(out.with_suffix(".npy"), pred_cal)
    print(f"[saved] {out} shape={df.shape} "
          f"range=[{pred_cal.min():.2f},{pred_cal.max():.2f}]")


if __name__ == "__main__":
    main()
