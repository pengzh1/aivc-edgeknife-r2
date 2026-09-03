"""wsN27: strain-blind fuse3-deep3（both 角色专用候选）。

both 行（未见菌株 × 新化合物）的菌株信息天然缺失；现行 fuse 模型训练时
菌株嵌入只有 25% 时间被 UNK 化。本变体训练/推理全程 strain=UNK
（wsF 的 strain_blind 同款思路），把全部容量聚焦到"化合物特征×上下文"
在无菌株信息下的响应面——理论上是 both 角色的正确归纳偏置。

对照：fuse3-e150（0.5412；both FC 0.231）。16 种子 val。

用法: python -m src.wsN27_strainblind
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN27"
SEEDS = list(range(16))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot,
                  hidden=(1024, 2048, 2048), p_drop=0.2)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    df = pd.read_csv("outputs/wsN23/chem_features_fuse3.csv")
    WSA.load_chem_table = table_to_loader(df)

    try:
        pv = []
        for sd in SEEDS:
            t0 = time.time()
            # 自包含训练循环（复刻 wsA.train_model，但菌株列恒 UNK）
            from .wsA_chemfeat import ProteoMLPChem, make_chem_mat, \
                load_chem_table, masked_mse
            from .train_mlp import Encoder
            torch.manual_seed(sd)
            np.random.seed(sd)
            enc = Encoder().fit(h.m_tr)
            feat, mean_vec = load_chem_table(h, OUT, full=False)
            chem_mat = make_chem_mat(enc, feat, mean_vec)
            n_prot = h.Y_tr.shape[1]
            stats = h.stats
            X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
            X_all[:, 0] = 0  # 菌株列恒 UNK（全表）
            mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
            std = torch.tensor(stats.protein_std, dtype=torch.float32)
            Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
            M_all = ~torch.isnan(Z_all)
            Z_all = torch.nan_to_num(Z_all, nan=0.0)
            model = ProteoMLPChem(enc.n_cats, chem_mat, n_prot).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=1e-4)
            rows = h.tr_rows
            n_steps = 150 * int(np.ceil(len(rows) / 256))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_steps)
            X_dev = X_all.cuda()
            Z_dev = Z_all.cuda()
            M_dev = M_all.cuda()
            rows_dev = torch.tensor(rows).cuda()
            for ep in range(150):
                model.train()
                perm = rows_dev[torch.randperm(len(rows_dev),
                                               device="cuda")]
                for i in range(0, len(perm), 256):
                    r = perm[i:i + 256]
                    xb = X_dev[r].clone()
                    # 仅化合物特征 dropout（菌株已恒 UNK）
                    drop = torch.rand(len(r), device="cuda") < 0.25
                    xb[drop, 1] = 0
                    pred = model(xb)
                    loss = masked_mse(pred, Z_dev[r], M_dev[r].float())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
            with torch.no_grad():
                outs = []
                for i in range(0, len(X_dev), 1024):
                    outs.append(model(X_dev[i:i + 1024]).float().cpu())
            Z_pred = torch.cat(outs).numpy()
            pv.append((Z_pred * std.numpy() + mean.numpy())
                      .astype(np.float32))
            print(f"[wsN27] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval.npy", P)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsN27 strain-blind] composite={res['composite']:.4f} "
          f"(fuse3-e150 0.5412) FC={fc}")
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc,
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
