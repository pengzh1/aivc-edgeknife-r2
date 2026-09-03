"""wsN25: wsC 时间族扩容（3 配方 × 8→16 种子，24→48 模型）。

wsC 是初赛阵容中唯一未扩容的族（time 角色仍占权重）。
本模块增量训练 seeds 8-15（3 时间配方 × 150ep + tail_avg），与既有
outputs/wsM/pred_test_wsC_trainonly.npy（8 种子版）合并为 16 种子版；
val 版与 outputs/wsC/pred_trainval_g3.npy 合并。

合规同 wsM（train-only 训练，G3 推理）。

用法: python -m src.wsN25_wsc16        # 约 40 分钟
"""
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from . import wsC_timebatch as W

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN25"

RECIPES = [
    {"time_onehot": True, "interact": "none"},
    {"time_onehot": True, "interact": "chemstrain"},
    {"time_onehot": False, "interact": "none"},
]
NEW_SEEDS = list(range(8, 16))
EPOCHS = 150


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    seed_means_v, seed_means_t = [], []
    for ri, rec in enumerate(RECIPES):
        for sd in NEW_SEEDS:
            t0 = time.time()
            tail = []
            model, enc, mean, std, _ = W.train_model(
                h, h.tr_rows, EPOCHS, seed=sd, emb_drop=0.25, lr=1e-3,
                time_onehot=rec["time_onehot"], interact=rec["interact"],
                qc_mode="none", corrector=None, basis_kind="rbf",
                hidden=(512, 1024), p_drop=0.1, tail_states=tail,
                log_every=999)
            states = [model.state_dict()] + tail
            sv, st = [], []
            for stt in states:
                model.load_state_dict(stt)
                sv.append(W.predict(model, enc, mean, std, h, None, "none",
                                    1.0, basis_kind="rbf", df=h.m_tr,
                                    g3=True))
                st.append(W.predict(model, enc, mean, std, h, None, "none",
                                    1.0, basis_kind="rbf", df=h.m_te,
                                    g3=True))
            seed_means_v.append(np.mean(sv, axis=0))
            seed_means_t.append(np.mean(st, axis=0))
            print(f"[wsN25] recipe{ri} seed={sd} ({time.time()-t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
    # 合并：旧 24 模型 + 新 24 模型等权
    old_v = np.load("outputs/wsC/pred_trainval_g3.npy")
    old_t = np.load("outputs/wsM/pred_test_wsC_trainonly.npy")
    P = (old_v + np.mean(seed_means_v, axis=0).astype(np.float32)) / 2
    PT = (old_t + np.mean(seed_means_t, axis=0).astype(np.float32)) / 2
    for arr in (P, PT):
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_wsC_s16.npy", P)
    np.save(OUT / "pred_test_wsC_s16.npy", PT)
    res = h.score_val(P, verbose=False)
    print(f"[wsN25] wsC 48模型 val composite = {res['composite']:.4f} "
          f"(24模型 0.5385)", flush=True)
    import json
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"],
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
