"""wsN18: 时间插值探针（val_time 的条件内时间近邻/GP 型回退上限）。

val_time = 每条件 6 个时间点随机留出；留出时间点的同条件其余时间点在
train 中（合法信息）。现有 MLP 用 time 特征隐式学时间曲线（time FC 0.643）。
本探针：对 val_time 处理行，直接在同条件 train 行的 Δ 时间曲线上做
核加权插值（条件键 = 菌株×化合物×培养基×温度×仪器×板号 子集，核 = 时间
高斯），测量"纯插值"在该划分的 FC 上限——判断 time 角色是否还有未挖信号。

合规：仅用 train 行；val 仅评估。

用法: python -m src.wsN18_timegp
"""
import json
import warnings
from pathlib import Path

import numpy as np

from .evaluate import Harness
from . import data as D
from .wsM_trainonly import strict_delta_train
from . import metrics as M

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN18"
KEYS = ["Strains", "perturbation_no_concentration", "Medium", "Temperature",
        "instrument", "Yeast_cell_plate"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    delta, _, valid = strict_delta_train(h)
    m_train = h.m_train.iloc[valid].reset_index(drop=True)
    d_train = delta[valid]

    # val_time 处理行
    vt_rows = np.where(
        (h.m_tr["split_final"] == "val_time").to_numpy())[0]
    vt_rows = vt_rows[h.is_treat_tr[vt_rows]]
    mv = h.m_tr.iloc[vt_rows]
    print(f"[wsN18] val_time 处理行 {len(vt_rows)} | train 处理行 {len(m_train)}")

    # 条件键（不含时间）→ train 行组
    key_tr = list(map(tuple, m_train[KEYS].to_numpy()))
    from collections import defaultdict
    groups = defaultdict(list)
    for i, k in enumerate(key_tr):
        groups[k].append(i)

    sigma_grid = [0.25, 0.5, 1.0, 2.0]
    t_tr = m_train["pert_time"].to_numpy(dtype=float)
    best = None
    for sigma in sigma_grid:
        dpred = np.full((len(vt_rows), d_train.shape[1]), np.nan)
        for i, row in enumerate(mv.itertuples()):
            k = tuple(getattr(row, c) for c in KEYS)
            idx = groups.get(k)
            if not idx:
                continue
            idx = np.array(idx)
            dt = np.abs(t_tr[idx] - float(row.pert_time))
            w = np.exp(-(dt ** 2) / (2 * sigma ** 2))
            if w.sum() < 1e-9:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                num = np.nansum(d_train[idx] * w[:, None], axis=0)
                den = np.sum(~np.isnan(d_train[idx]) * w[:, None], axis=0)
                pred = np.where(den > 1e-9, num / np.maximum(den, 1e-9),
                                np.nan)
            dpred[i] = pred
        # FC：与 Δ_true（train_val 对照池口径）逐样本 PCC
        dt_true = h.delta_tr_all[vt_rows]
        ok = ~np.isnan(dpred).all(axis=1)
        pcc = M._masked_pcc_axis1(dt_true[ok], np.nan_to_num(dpred[ok]))
        print(f"  σ={sigma}: 命中 {ok.sum()}/{len(vt_rows)} 行 | "
              f"time-kNN FC = {np.nanmean(pcc):.4f}")
        if best is None or np.nanmean(pcc) > best[1]:
            best = (sigma, float(np.nanmean(pcc)), int(ok.sum()))
    print(f"[wsN18] best σ={best[0]} FC={best[1]:.4f}（命中 {best[2]} 行）")
    print("参考：现行路由 time FC = 0.6434（val_time）")
    (OUT / "scores.json").write_text(json.dumps(
        {"best_sigma": best[0], "fc": best[1], "hit": best[2]}, indent=1))


if __name__ == "__main__":
    main()
