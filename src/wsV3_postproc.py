"""wsV3：机制轮后处理双候选（零训练）——分位数量化混合 + 秩保持条件重校准。

预注册（与 wsV1/wsV2 同一张仲裁表，本轮唯一 val 看）：
  Blend_rank：21 族路由的"值加权平均"换成"逐样本分位数量化平均"
    （权重同路由；每族行内排序→分位数，加权平均分位，再经各族分位函数
    反解回数值）。采纳门槛 composite ≥ 0.5546。
  D_cal：C1@τ0.25 输出上，处理行 Δ̂ 逐样本秩保持地重映到 train 上下文
    条件分位函数（ctx=菌株×培养基×温度×时间，分层回退；拟合全 train-only）。
    FC/resid/样本 PCC 逐位不变（样本内单调），只动保真 R2/protein_PCC 与
    DEP 过阈。采纳门槛：composite ≥ 0.5529 且 F1 > 0.2549。

合规：全部统计 train-only；val 本包一次；Y_te 零接触；新文件不改旧文件。
用法: python -m src.wsV3_postproc
"""
import json
import time
from pathlib import Path

import numpy as np

from . import wsH_router as H
from .evaluate import Harness
from .wsN11_grandrouter import CANDIDATES, SPLITS

OUT = Path("outputs/wsV3")


def blend_rank(h, preds, w_by_split):
    """逐样本分位数量化平均；preds (21, N, P)，返回与 routed 同形。"""
    out = np.empty(preds.shape[1:], np.float32)
    n = preds.shape[2]
    grid = np.linspace(0, 1, n)
    arange_n = np.arange(n)
    for sp in SPLITS:
        rows = h._fast[sp]["rows"]
        w = w_by_split[sp]
        use = np.where(w > 1e-12)[0]
        wu = w[use] / w[use].sum()
        sub = preds[use][:, rows, :]              # (k, n_r, P)
        k, n_r = sub.shape[0], sub.shape[1]
        sub_sorted = np.sort(sub, axis=2)         # (k, n_r, P)
        acc = np.zeros((n_r, n), np.float64)
        for ki in range(k):                       # 行内秩 → 加权平均分位
            order = np.argsort(sub[ki], axis=1)
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order,
                              np.broadcast_to(arange_n, order.shape), 1)
            acc += wu[ki] * ranks / (n - 1)
        val = np.zeros((n_r, n), np.float64)      # 加权平均分位函数反解
        for i in range(n_r):
            qi = acc[i]
            for ki in range(k):
                val[i] += wu[ki] * np.interp(qi, grid, sub_sorted[ki, i])
        out[rows] = val.astype(np.float32)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = Harness()
    h.prepare_fast_eval()
    gr = json.loads(Path("outputs/wsN11/grand_router.json").read_text())
    r = 0.7
    w_glob = np.array(gr["w_global"])
    w_use = {sp: r * np.array(gr["best_w_split"][sp]) + (1 - r) * w_glob
             for sp in SPLITS}

    # ---------- Blend_rank ----------
    files = [(n, p) for n, p in CANDIDATES if n in gr["names"]
             and Path(p).exists()]
    preds = H.load_preds(files, h)
    br = blend_rank(h, preds, w_use)
    br[h.tr_rows] = H.blend_rows(preds, w_glob / w_glob.sum(), h.tr_rows)
    np.save(OUT / "pred_trainval_blendrank.npy", br)
    res = h.score_val(br, verbose=False)
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    print(f"[Blend_rank] composite={res['composite']:.4f} F1={f1:.4f} "
          f"（门槛 0.5546；对照 C1 0.5541）", flush=True)

    # ---------- D_cal ----------
    base_pred = np.load("outputs/wsT9/pred_trainval_tau0.25.npy")
    ctrl_hat = np.load("outputs/wsT0/cache/control_hat.npy")
    m = h.m_tr
    treat = h.is_treat_tr
    tr_treat = h.tr_rows[treat[h.tr_rows]]
    # 条件分位函数：train 处理行 Δ_true 按 ctx 键池化排序
    ctx_tuples = list(map(tuple, m[["Strains", "Medium", "Temperature",
                                    "pert_time"]].to_numpy()))
    fb_tuples = list(map(tuple, m[["Medium", "Temperature",
                                   "pert_time"]].to_numpy()))
    dt = h.delta_tr_all[tr_treat].astype(np.float64)
    pools, fb_pools = {}, {}
    for i, idx in enumerate(tr_treat):
        v = dt[i][~np.isnan(dt[i])]
        pools.setdefault(ctx_tuples[idx], []).append(v)
        fb_pools.setdefault(fb_tuples[idx], []).append(v)
    pools = {k: np.sort(np.concatenate(v)) for k, v in pools.items()}
    fb_pools = {k: np.sort(np.concatenate(v)) for k, v in fb_pools.items()}
    glob_q = np.sort(dt[~np.isnan(dt)])

    out = base_pred.copy()
    n = base_pred.shape[1]
    rows_all = np.where(treat)[0]
    for idx in rows_all:
        key = ctx_tuples[idx]
        fq = pools.get(key)
        if fq is None:
            fq = fb_pools.get(fb_tuples[idx])
        if fq is None:
            fq = glob_q
        fgrid = np.linspace(0, 1, len(fq))    # 经验分位函数网格随池长
        d_hat = (base_pred[idx].astype(np.float64)
                 - ctrl_hat[idx].astype(np.float64))
        order = np.argsort(d_hat)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        q = ranks / (n - 1)
        new_d = np.interp(q, fgrid, fq)
        out[idx] = (ctrl_hat[idx] + new_d).astype(np.float32)
    np.save(OUT / "pred_trainval_dcal.npy", out)
    res2 = h.score_val(out, verbose=False)
    f1b = float(np.mean([res2["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    print(f"[D_cal] composite={res2['composite']:.4f} F1={f1b:.4f} "
          f"（门槛：composite≥0.5529 且 F1>0.2549）", flush=True)
    (OUT / "postproc_scores.json").write_text(json.dumps(
        {"blend_rank": res, "d_cal": res2}, indent=1, default=float))
    print(f"[done] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
