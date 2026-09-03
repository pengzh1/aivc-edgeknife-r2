"""生成 test 集预测 prediction.csv。

流程：全量 train_val（8958 样本）重训 Ridge 与 MLP → 0.55/0.45 加权集成
→ 按官方模板输出 sample_ID + 5243 蛋白列（log2 尺度，与输入蛋白顺序一致）。

    python -m src.predict_test --w_mlp 0.55 --epochs 100
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .baselines import RidgeMulti
from .evaluate import Harness
from . import train_mlp as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_mlp", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="outputs/prediction.csv")
    args = ap.parse_args()

    h = Harness()
    rows_all = np.arange(len(h.m_tr))
    stats_full = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows_all)

    t0 = time.time()
    ridge = RidgeMulti(lam=args.lam).fit(h, rows=rows_all, stats=stats_full)
    pred_ridge_te = ridge.predict(h.m_te)
    print(f"[ridge] full fit {time.time()-t0:.0f}s")

    t0 = time.time()
    preds_mlp = []
    for sd in args.seeds:
        model, enc, mean, std = T.train_model(
            h, rows_all, args.epochs, seed=sd, emb_drop=0.25,
            log_every=999, stats=stats_full)
        preds_mlp.append(T.predict(model, enc, mean, std, h.m_te))
    pred_mlp_te = np.mean(preds_mlp, axis=0)
    print(f"[mlp] full fit x{len(args.seeds)} seeds {time.time()-t0:.0f}s")

    pred = args.w_mlp * pred_mlp_te + (1 - args.w_mlp) * pred_ridge_te

    #  sanity：完整、有限、行序与 metadata 一致
    assert pred.shape == (len(h.m_te), len(h.proteins)), pred.shape
    assert np.isfinite(pred).all(), "prediction contains NaN/Inf"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pred, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    df.to_csv(out, index=False)
    print(f"[saved] {out} shape={df.shape} "
          f"range=[{pred.min():.2f},{pred.max():.2f}]")
    np.save(out.parent / "pred_test_ensemble.npy", pred)
    np.save(out.parent / "pred_test_ridge.npy", pred_ridge_te)
    np.save(out.parent / "pred_test_mlp.npy", pred_mlp_te)


if __name__ == "__main__":
    main()
