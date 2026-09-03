"""wsV1：deep3-fuse3 + listwise 排序辅助损失族（机制实验：loss 形状）。

动机：FC/残差/保真的核心成分是逐样本 Pearson 相关（排序几何），而全部
41 个历史实验只改过损失权重（已证伪），从未改过损失形状。本族在
fuse3-e150 配方上加逐样本 listwise KL 辅助（softmax 目标分布对齐），
主损失仍为 masked MSE（保绝对尺度）。

预注册裁决：单族 composite ≥ 0.538 且 wsN30 式边际扫描 Δ ≥ +0.0003
（chem 与 strain_both 两区，α∈{0.05,0.12,0.22}）才入路由；否则关闭。
16 种子 × 150ep（GPU 约 40 分钟）。val 仅本包一次看。

用法: python -m src.wsV1_rankloss
"""
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsV0_deep3train import family_eval, predict_all, train_one

OUT = Path("outputs/wsV1")
SEEDS = list(range(16))
AUX_RANK = 0.3
TABLE = "outputs/wsN23/chem_features_fuse3.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    pv = []
    for sd in SEEDS:
        t0 = time.time()
        model = train_one(h, sd, TABLE, epochs=150, aux_rank=AUX_RANK,
                          device="cuda")
        pv.append(predict_all(model, h))
        del model
        torch.cuda.empty_cache()
        print(f"[wsV1] seed {sd} ({time.time()-t0:.0f}s)", flush=True)
    family_eval(h, pv, OUT / "pred_trainval.npy", "wsV1_rank")

    # 边际扫描（≥0.538 门槛内才报告入列判定）
    res = h.score_val(np.load(OUT / "pred_trainval.npy"), verbose=False)
    if res["composite"] >= 0.538:
        h.prepare_fast_eval()
        routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
        base = h.score_val(routed, verbose=False)["composite"]
        print(f"[scan] routed 基线 {base:.4f}", flush=True)
        pred = np.load(OUT / "pred_trainval.npy")
        for tag, sps in [("strain_both", ["val_strain_only", "val_both"]),
                         ("chem", ["val_chem_only"])]:
            rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
            for a in (0.05, 0.12, 0.22):
                trial = routed.copy()
                trial[rows] = (1 - a) * routed[rows] + a * pred[rows]
                c = h.score_val(trial, verbose=False)["composite"]
                print(f"  [{tag} α={a}] composite={c:.4f} (Δ{c-base:+.4f})",
                      flush=True)


if __name__ == "__main__":
    main()
