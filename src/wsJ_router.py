"""wsJ 路由集成评估（v3 阵容）：wsJ 强描述符模型加入开放榜/封闭备选路由。

不修改 wsH；复用其 SplitScorer / run_set / 锚点。三种阵容对比：
  A) v3open-replace: v2 六模型 + wsJ（替换 wsA）——检验 wsJ 是否全面优于 wsA
  B) v3open-both   : v2 六模型 + wsA + wsJ——检验二者互补性
  C) v3closed-prep : v2 六模型 + wsJ（封闭备选；仅当组委会书面确认结构 embedding
     封闭榜合规时才可提交——本地评估先行，备好切换方案）

锚点 = v2 全局给定点（新成员权重 0）。r=0.5 交付档完整 score_val。
用法: python -m src.wsJ_router
产出: outputs/wsJ/router_v3.json + 控制台对比表
"""
import json
import time
from pathlib import Path

import numpy as np

from .evaluate import Harness
from .wsH_router import (OPEN_EXTRA, SEED, V2_CLOSED, V2_W_GLOBAL,
                         build_candidates, run_set)

OUT = Path("outputs/wsJ")
WSJ = ("wsJ", "outputs/wsJ/pred_trainval.npy")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()

    results = {}
    for tag, files in [
        ("v3open_replace", V2_CLOSED + [WSJ]),
        ("v3open_both", V2_CLOSED + [OPEN_EXTRA, WSJ]),
        ("v3closed_prep", V2_CLOSED + [WSJ]),
    ]:
        k = len(files)
        w_anchor = np.concatenate([V2_W_GLOBAL, np.zeros(k - len(V2_W_GLOBAL))])
        rec = run_set(h, files, w_anchor, tag,
                      shrink_grid=(0.7, 0.5, 0.3), full_score_grid=(0.7, 0.5, 0.3))
        comp05 = rec["shrink"]["0.5"]["full"]["composite"]
        comp07 = rec["shrink"]["0.7"]["full"]["composite"]
        comp03 = rec["shrink"]["0.3"]["full"]["composite"]
        results[tag] = {
            "names": rec["names"],
            "w_anchor": w_anchor.tolist(),
            "composite": {"r0.3": comp03, "r0.5": comp05, "r0.7": comp07},
            "splits": {sp: {"w_opt": rec["splits"][sp]["w_opt"],
                            "fast_opt": rec["splits"][sp]["fast_opt"]}
                       for sp in rec["splits"]},
            "r05_per_split": rec["shrink"]["0.5"]["full"]["per_split"],
            "r05_weights": {sp: rec["shrink"]["0.5"][sp] for sp in
                            rec["shrink"]["0.5"] if sp.startswith("val")},
        }
        # 保存 r=0.5 路由预测（供后续提交生成复用）
        np.save(OUT / f"pred_trainval_{tag}.npy",
                rec["routed_pred"].astype(np.float32))
        rec.pop("preds")
        print(f"[{tag}] r0.3={comp03:.4f} r0.5={comp05:.4f} r0.7={comp07:.4f}",
              flush=True)

    with open(OUT / "router_v3.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "results": results}, f, ensure_ascii=False,
                  indent=2, default=float)
    print(f"[saved] {OUT/'router_v3.json'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
