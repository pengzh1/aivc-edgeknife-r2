"""实体嵌入 MLP：条件嵌入 → 全连接 → 5243 维 log2 蛋白组预测。

- 所有类别特征用嵌入；菌株/化合物嵌入带 UNK token + 训练期随机替换
  （embedding dropout），让模型学会未见实体的回退表示
- 目标为按 train 冻结统计标准化的 log2 强度，masked MSE（跳过缺失蛋白）
- 输出层 bias 初始化为 0（对应蛋白均值），加速收敛

用法:
    python -m src.train_mlp --epochs 30                # train split 训练 + val 评分
    python -m src.train_mlp --epochs 30 --full --out outputs/mlp_full  # 全量重训+test预测
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness

CAT_COLS = ["Strains", "perturbation_no_concentration", "Medium",
            "Temperature", "pert_time", "instrument", "data_source",
            "Yeast_cell_plate"]
EMB_DIMS = {"Strains": 8, "perturbation_no_concentration": 32, "Medium": 2,
            "Temperature": 2, "pert_time": 4, "instrument": 6,
            "data_source": 3, "Yeast_cell_plate": 32}


class Encoder:
    """类别 → 索引。index 0 保留给 UNK。"""

    def fit(self, m: pd.DataFrame):
        self.maps = {}
        for c in CAT_COLS:
            vals = sorted(pd.unique(m[c]).tolist(), key=str)
            self.maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        return self

    def transform(self, m: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in CAT_COLS:
            mp = self.maps[c]
            cols.append(m[c].map(lambda v: mp.get(v, 0)).to_numpy())
        return np.stack(cols, axis=1)  # (n, n_cat)

    @property
    def n_cats(self):
        return [len(self.maps[c]) + 1 for c in CAT_COLS]


class ProteoMLP(nn.Module):
    def __init__(self, n_cats, n_prot, hidden=(512, 1024), p_drop=0.1):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[c]) for n, c in zip(n_cats, CAT_COLS)])
        d_in = sum(EMB_DIMS[c] for c in CAT_COLS)
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h),
                       nn.Dropout(p_drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)],
                      dim=1)
        return self.head(self.trunk(e))


def masked_mse(pred, target, mask):
    """pred/target (B, P)，mask 为可用位置。逐蛋白标准化空间。"""
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp_min(1.0)


def train_model(h: Harness, rows: np.ndarray, epochs: int, seed: int = 0,
                emb_drop: float = 0.15, lr: float = 1e-3, bs: int = 256,
                device: str = "cuda", log_every: int = 5, stats=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)  # 类别覆盖 train_val 全集（实体可见性由训练行控制）
    n_prot = h.Y_tr.shape[1]
    stats = stats or h.stats

    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = ProteoMLP(enc.n_cats, n_prot).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * int(np.ceil(len(rows) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev = X_all.to(device)
    Z_dev = Z_all.to(device)
    M_dev = M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)

    for ep in range(epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            xb = X_dev[r].clone()
            # embedding dropout：菌株/化合物随机替换为 UNK(0)，学 OOD 回退
            if emb_drop > 0:
                for col in [0, 1]:
                    drop = torch.rand(len(r), device=device) < emb_drop
                    xb[drop, col] = 0
            pred = model(xb)
            loss = masked_mse(pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={tot/nb:.4f}")
    return model, enc, mean, std


def predict(model, enc, mean, std, m: pd.DataFrame, device="cuda",
            bs: int = 1024) -> np.ndarray:
    model.eval()
    X = torch.tensor(enc.transform(m), dtype=torch.long, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb_drop", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--full", action="store_true",
                    help="用全部 train_val 训练并预测 test")
    ap.add_argument("--out", default="outputs/mlp")
    args = ap.parse_args()

    h = Harness()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.full:
        t0 = time.time()
        model, enc, mean, std = train_model(h, h.tr_rows, args.epochs,
                                            seed=args.seed,
                                            emb_drop=args.emb_drop,
                                            lr=args.lr)
        print(f"[train] {time.time()-t0:.0f}s")
        pred = predict(model, enc, mean, std, h.m_tr)
        np.save(out / "pred_trainval.npy", pred)
        h.score_val(pred)
    else:
        rows = np.arange(len(h.m_tr))
        t0 = time.time()
        model, enc, mean, std = train_model(h, rows, args.epochs,
                                            seed=args.seed,
                                            emb_drop=args.emb_drop,
                                            lr=args.lr)
        print(f"[train full] {time.time()-t0:.0f}s")
        pred_te = predict(model, enc, mean, std, h.m_te)
        np.save(out / "pred_test.npy", pred_te)
        torch.save(model.state_dict(), out / "model_full.pt")
        print(f"[saved] {out/'pred_test.npy'} shape={pred_te.shape}")


if __name__ == "__main__":
    main()
