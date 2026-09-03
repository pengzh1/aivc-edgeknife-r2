"""wsN5: wsA 分子描述符的 shuffle 对照验证（消融实验）。

依据（8.11 公告 + 进阶教程 5.3/5.6.2）："embedding shuffle/置零消融，可证明
表征信息被有效利用（shuffle 后性能应显著下降）"；名称标准化错配=表征完全失效。
本实验把"化合物名 → RDKit 描述符向量"的映射随机打乱（保持分布、破坏对应
关系），重训 wsA 同配方模型——若 val 成绩显著下降，证明描述符携带真实结构信息。

合规：train-only 训练；shuffle 仅作用于特征表（外部结构数据），固定种子；
不改动 wsA 原文件（monkey-patch load_chem_table 于运行进程内）。

用法: python -m src.wsN5_shuffle        # 3 种子 × 100ep，约 3 分钟
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsA_chemfeat as WSA

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN5"
SEED_SHUFFLE = 20260811
SEEDS = [0, 1, 2]
EPOCHS = 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()

    # ---- monkey-patch：打乱 化合物名 → 描述符 的映射（仅进程内生效）----
    orig_loader = WSA.load_chem_table

    def shuffled_loader(h_, out_dir, full=False):
        feat, mean_vec = orig_loader(h_, out_dir, full=full)
        names = sorted(feat.keys())
        rng = np.random.default_rng(SEED_SHUFFLE)
        perm = rng.permutation(len(names))
        shuffled = {n: feat[names[perm[i]]] for i, n in enumerate(names)}
        return shuffled, mean_vec

    WSA.load_chem_table = shuffled_loader
    print(f"[wsN5] 描述符映射已打乱（seed={SEED_SHUFFLE}）", flush=True)

    preds = []
    for s in SEEDS:
        t0 = time.time()
        model, enc, mean, std = WSA.train_model(
            h, h.tr_rows, EPOCHS, seed=s, emb_drop=0.25, chem_drop=0.25,
            lr=1e-3, bs=256, device="cuda", log_every=999)
        preds.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                 device="cuda"))
        print(f"[wsN5] seed={s} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    WSA.load_chem_table = orig_loader

    P = np.mean(preds, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"\n[wsN5 shuffle] composite={res['composite']:.4f} FC={fc} resid={rz}")
    print("对照（真实描述符 wsA）：composite=0.5102（chem FC≈0.46，见 outputs/wsA/scores.json）")
    np.save(OUT / "pred_trainval_shuffle.npy", P)
    (OUT / "scores.json").write_text(json.dumps(
        {"shuffle_seed": SEED_SHUFFLE, "seeds": SEEDS, "epochs": EPOCHS,
         "composite": res["composite"], "FC": fc, "resid": rz,
         "per_split": res["per_split"]}, ensure_ascii=False, indent=1,
        default=float))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
