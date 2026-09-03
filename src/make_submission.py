"""生成最终提交文件（封闭榜 + 开放知识榜）。

流程：
1. 按集成搜索得到的权重混合各 ws 的 pred_test.npy
2. 施加 wsE 的 DEP 带门校准（band a=0.5, γ=1.3, 仅 7 键命中处理行；
   对照组均值用全量 train_val 对照估计——test 蛋白值零接触）
3. 输出 outputs/prediction.csv（封闭）与 outputs/prediction_open.csv（开放）

权重通过参数传入（从 outputs/ensemble_{closed,open}.log 读取）。

用法:
    python -m src.make_submission --track closed --files outputs/wsD/pred_test.npy ... \
        --weights 0.45 0.25 0.15 0.15
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluate import Harness

# 对照分组层级（与 wsE_depcal.CTRL_KEY_LEVELS 同构）
CTRL_KEY_LEVELS = [
    ["data_source", "Strains", "Medium", "Temperature", "pert_time",
     "instrument", "Yeast_cell_plate"],
    ["data_source", "Strains", "Medium", "Temperature", "pert_time",
     "instrument"],
    ["data_source", "Strains", "Medium", "Temperature", "pert_time"],
    ["Strains", "Medium", "Temperature", "pert_time"],
    ["Strains", "Medium", "Temperature"],
    ["Strains"],
]


def build_control_hat_for(h: Harness, m_target: pd.DataFrame,
                          ctrl_rows: np.ndarray):
    """用给定对照行（train_val）构建分组均值，对 m_target 行取值。

    返回 (control_hat, level)；level 0=7键精确命中 … 5=1键, 6=全局回退。
    """
    mc = h.m_tr.iloc[ctrl_rows]
    Yc = h.Y_tr[ctrl_rows]
    prot_mean = np.nanmean(h.Y_tr, axis=0)
    scalar = np.nanmean(h.Y_tr)  # 全缺失蛋白的最终回退
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


def band_calibrate(pred: np.ndarray, ctrl_hat: np.ndarray, level: np.ndarray,
                   is_treat: np.ndarray, a: float = 0.5, g: float = 1.3):
    """wsE 推荐：ŷ += (γ−1)·band(|Δ̂|; a=0.5, c=1.0)·Δ̂，仅 7 键命中处理行。

    band 为梯形门：|d|<1 段从 a 上坡至 1，|d|≥1 为 0（与 wsE_depcal._band_vec 一致）。
    """
    d = pred - ctrl_hat
    ad = np.abs(d)
    gate = np.where(ad < 1.0, np.clip((ad - a) / (1.0 - a), 0.0, 1.0), 0.0)
    gate = gate.astype(np.float32)
    ok = is_treat & (level == 0)
    add = (g - 1.0) * gate * d
    out = pred.copy()
    out[ok] = pred[ok] + add[ok]
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["closed", "open"], required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert len(args.files) == len(args.weights)

    h = Harness()
    preds = np.stack([np.load(f) for f in args.files])
    w = np.array(args.weights, dtype=np.float64)
    w = w / w.sum()
    pred = np.tensordot(w, preds, axes=1).astype(np.float32)
    print(f"[blend] {args.track} weights={np.round(w,3)} files={args.files}")

    # 全量 train_val 对照 → test 对照基线
    ctrl_rows = np.where(
        h.m_tr["perturbation_no_concentration"].isin(D.CONTROLS).to_numpy())[0]
    ctrl_hat_te, level_te = build_control_hat_for(h, h.m_te, ctrl_rows)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    print(f"[control_hat] level dist: {np.bincount(level_te, minlength=7).tolist()}"
          f" | 7k-hit treat rows: {int((is_treat_te & (level_te==0)).sum())}")

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
