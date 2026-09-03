"""wsT5：自适应每样本旗标数（C6 候选）——F1 逐样本均值结构的决策论利用。

预注册（先于任何 val 读取）：
- 规则：旗标池 = 可用 & |Δ̂|≥0.3；每行旗标期望数 k̂_i = Σ_{j∈池} p_ij（scale=1.0
  固定，train OOF 证据：0.8→0.5186、1.0→0.5327、1.3→0.5294，峰在 1.0）；
  按 p 降序取前 ⌈k̂⌉ 条推送至 1.02（保号，同 C1 推送式）。
- 动机：F1 按样本计均值，全局 τ 对低 DEP 样本过旗标、高 DEP 样本欠旗标；
  自适应 k̂ 在 train OOF 上 +0.016 F1（零新信息，纯决策结构）。
- val 单次裁决：采纳条件 = composite ≥ 0.5541−0.0005 且 F1 均值 > 0.2442+0.003；
  否则维持 C1。与 wsT4（C5a/C5b）同表一次看，不迭代。

合规：p 来自 wsT1 的 hgb_full（train-only 拟合）；scale 由 train OOF 定；
val 一次看；Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsT5_adaptivek
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .evaluate import Harness
from .wsN11_grandrouter import SPLITS
from .wsT1_depgate import CACHE as T1C, MIN_PUSH, PUSH_TO, _predict_val

OUT = Path("outputs/wsT5")
T0C = Path("outputs/wsT0/cache")
TAU_POOL = MIN_PUSH  # 旗标池下限同 C1


def adaptive_flags(P, absDE, avail_like, scale=1.0):
    """每行按 p 降序取前 k̂_i 条（k̂_i = scale·Σp，池内）。返回布尔矩阵。"""
    pool = (absDE >= TAU_POOL) & avail_like
    Pp = np.where(pool, P, -1.0)
    k_hat = np.nansum(np.where(pool, P, 0.0), axis=1) * scale
    order = np.argsort(-Pp, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(P.shape[1]), P.shape), axis=1)
    return pool & (ranks < k_hat[:, None])


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load(T0C / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0C / "control_hat.npy")
    clf = joblib.load(T1C / "hgb_full.joblib")

    pred = routed.astype(np.float64)
    n_flag_total = 0
    for sp in SPLITS:
        vrows, Pv = _predict_val(h, clf, sp)
        d_v = (routed[vrows] - ctrl_hat[vrows]).astype(np.float64)
        # val 行全部条目视作可用（ŷ 无缺失；avail 语义仅用于 train 调参）
        avail_v = np.ones_like(d_v, dtype=bool)
        flv = adaptive_flags(Pv, np.abs(d_v), avail_v, scale=1.0)
        tgt = np.sign(d_v) * np.maximum(np.abs(d_v), PUSH_TO)
        pred[vrows] = pred[vrows] + np.where(flv, tgt - d_v, 0.0)
        n_flag_total += int(flv.sum())
        print(f"[C6] {sp} flags={int(flv.sum()):,}", flush=True)
    # train 行一致性（OOF p；不参与评分）
    rows = np.load(T1C / "train_rows.npy")
    avail = np.load(T1C / "train_avail.npy")
    d_est = np.load(T1C / "train_dest.npy").astype(np.float64)
    P = np.full(avail.shape, np.nan, np.float32)
    P[avail] = np.load(T1C / "train_p_oof.npy")
    fl = adaptive_flags(P, np.abs(d_est), avail, scale=1.0)
    tgt = np.sign(d_est) * np.maximum(np.abs(d_est), PUSH_TO)
    pred[rows] = pred[rows] + np.where(fl, tgt - d_est, 0.0)
    pred = pred.astype(np.float32)
    np.save(OUT / "pred_trainval_C6.npy", pred)
    print(f"[C6] train flags={int(fl.sum()):,} val flags={n_flag_total:,}",
          flush=True)

    res = h.score_val(pred, verbose=False)
    f1m = float(np.mean([res["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    per = {sp: round(res["per_split"][sp]["DEP_F1"], 4) for sp in SPLITS}
    print(f"[C6] val composite={res['composite']:.4f} F1={f1m:.4f} "
          f"per-split={per}", flush=True)
    print(f"      （对照 C1: composite=0.5541 F1=0.2442）", flush=True)
    (OUT / "c6_score.json").write_text(json.dumps(
        {"res": res, "train_flags": int(fl.sum()),
         "val_flags": n_flag_total}, indent=1, default=float))
    print(f"[saved] {OUT/'c6_score.json'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
