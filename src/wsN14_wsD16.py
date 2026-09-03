"""wsN14: wsD 扩展到 16 种子（s8-15 增量训练 + 16 种子集成 val/test 预测）。

8 种子已是最强单族（0.5431）；种子数翻倍是诚实的零风险微增益（方差缩减）。
本模块只增量训练 seeds 8-15（与 wsM 完全同配方 train-only），合并产出：
  outputs/wsN14/pred_trainval_wsD_s16.npy  （wsD_g2g3 8种子 + 新8种子 均值）
  outputs/wsN14/pred_test_wsD_s16.npy      （wsM test 8种子 + 新8种子 均值）

用法: python -m src.wsN14_wsD16        # 约 25 分钟
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsD_arch import Cfg, train_one, predict, seen_cats

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN14"

WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
NEW_SEEDS = list(range(8, 16))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    cfg = Cfg.from_dict(WSD_CFG)
    seen = seen_cats(h, h.tr_rows)
    pv, pt = [], []
    for s in NEW_SEEDS:
        t0 = time.time()
        model, enc, mean, std, _ = train_one(h, cfg, s)
        pv.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
        pt.append(predict(model, enc, mean, std, h.m_te, g3_seen=seen))
        print(f"[wsN14] seed={s} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    # 与既有 8 种子等权合并
    pv_old = np.load("outputs/wsD/pred_trainval_g2g3.npy")
    pt_old = np.load("outputs/wsM/pred_test_wsD_trainonly.npy")
    P = (pv_old * 8 + np.mean(pv, axis=0) * 8) / 16
    PT = (pt_old * 8 + np.mean(pt, axis=0) * 8) / 16
    P, PT = P.astype(np.float32), PT.astype(np.float32)
    assert np.isfinite(P).all() and np.isfinite(PT).all()
    res = h.score_val(P, verbose=False)
    print(f"[wsN14] 16种子 val composite = {res['composite']:.4f} "
          f"(8种子 0.5431)")
    np.save(OUT / "pred_trainval_wsD_s16.npy", P)
    np.save(OUT / "pred_test_wsD_s16.npy", PT)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"],
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}/pred_*_s16.npy")


if __name__ == "__main__":
    main()
