"""wsV2：deep3-fuse3 + 条件嵌入 mixup 增广族（机制实验：输入增广）。

动机：59 实验从未试过训练期输入增广。mixup 在嵌入空间插值
（e ← λ·e_a + (1−λ)·e_b，目标同 λ 插值，掩码取交集），强迫模型学
平滑的响应面。8 种子 × 150ep（GPU 约 20 分钟，时间盒）。

预注册裁决：同 wsV1（单族 ≥0.538 + 边际扫描 Δ≥+0.0003）；否则关闭。

用法: python -m src.wsV2_mixup
"""
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsV0_deep3train import family_eval, predict_all, train_one

OUT = Path("outputs/wsV2")
SEEDS = list(range(8))
MIX_ALPHA = 0.3
TABLE = "outputs/wsN23/chem_features_fuse3.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    pv = []
    for sd in SEEDS:
        t0 = time.time()
        model = train_one(h, sd, TABLE, epochs=150, mixup_alpha=MIX_ALPHA,
                          device="cuda")
        pv.append(predict_all(model, h))
        del model
        torch.cuda.empty_cache()
        print(f"[wsV2] seed {sd} ({time.time()-t0:.0f}s)", flush=True)
    family_eval(h, pv, OUT / "pred_trainval.npy", "wsV2_mix")

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
