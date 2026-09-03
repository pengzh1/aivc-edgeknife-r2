"""wsN10: wsD 架构多样化变体族（集成去相关）。

动机：单架构 8 种子集成的方差缩减已到极限；架构级多样性（深度/宽度/损失/
训练时长/正则强度）提供预测层面真正的去相关，是集成工程中最可靠的增益来源。
所有变体沿用 wsD 的 g2_aug + G3 推理 + train-only 口径，仅改架构/超参。

变体（3 种子 each，先探索；胜者进全阵容路由重拟合）：
  V1 deep6:    6 层 (1024,2048×5)
  V2 wide:     4 层 (1024,3072×3)
  V3 mse:      5 层同配方但 loss=mse（收缩更小，保留大 Δ）
  V4 ep500:    5 层 500 epochs
  V5 drop02:   5 层 p_drop=0.2

对照：wsD g2g3 8 种子 0.5431 / 3 种子 0.5423。

合规：train-only 训练；val 仅模型选择；Y_te 零接触；不改既有文件。

用法: python -m src.wsN10_archdiv            # 5 变体 × 3 种子，val+test
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsD_arch import Cfg, train_one, predict, seen_cats

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN10"

BASE = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
        "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
        "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
        "loss": "huber", "film": False, "g2_aug": True}

VARIANTS = {
    "deep6":  {**BASE, "hidden": [1024, 2048, 2048, 2048, 2048, 2048]},
    "wide":   {**BASE, "hidden": [1024, 3072, 3072, 3072]},
    "mse":    {**BASE, "loss": "mse"},
    "ep500":  {**BASE, "epochs": 500},
    "drop02": {**BASE, "p_drop": 0.2},
}
SEEDS = [0, 1, 2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    seen = seen_cats(h, h.tr_rows)
    summary = {}
    for tag, cfgd in VARIANTS.items():
        out_v = OUT / f"pred_trainval_{tag}.npy"
        out_t = OUT / f"pred_test_{tag}.npy"
        if out_v.exists() and out_t.exists():
            print(f"[skip] {tag} 已存在", flush=True)
            continue
        cfg = Cfg.from_dict(cfgd)
        pv, pt = [], []
        for s in SEEDS:
            t0 = time.time()
            model, enc, mean, std, _ = train_one(h, cfg, s)
            pv.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
            pt.append(predict(model, enc, mean, std, h.m_te, g3_seen=seen))
            print(f"[{tag}] seed={s} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        P = np.mean(pv, axis=0).astype(np.float32)
        res = h.score_val(P, verbose=False)
        fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
              for sp in res["per_split"]}
        print(f"[{tag}] composite={res['composite']:.4f} FC={fc}", flush=True)
        np.save(out_v, P)
        PT = np.mean(pt, axis=0).astype(np.float32)
        assert np.isfinite(PT).all()
        np.save(out_t, PT)
        summary[tag] = {"composite": res["composite"], "FC": fc,
                        "per_split": res["per_split"], "cfg": cfgd}
        (OUT / "scores.json").write_text(json.dumps(
            summary, ensure_ascii=False, indent=1, default=float))
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
