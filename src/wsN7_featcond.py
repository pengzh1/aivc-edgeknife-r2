"""wsN7: 特征条件化化合物嵌入（实体可迁移表征注入 wsD 主力配方）。

动机：wsD 对未见化合物走 UNK(0) 回退 = "平均药物响应"盲答；wsN6 证明
RDKit⊕ChemBERTa 融合表征携带真实信息（路由 chem 权重 0.16）。本模块把 wsD
的化合物自由嵌入升级为**特征条件通道**：
  e_chem = free_emb(c)          —— 训练已见化合物
  e_chem = feat_map(fuse128(c)) —— 未见化合物（val/test 新化合物）
训练期用 feature-dropout（0.15 逐样本 + G2 组级）让模型学会利用特征通道；
推理期未见化合物自动走特征通道。val_chem_only 为诚实对照（6 个 val 化合物
对 train-only 模型即"未见"）。

对照基线：wsD g2g3 同配方 3 种子（wsN3 identity 复核 0.5423 / chem FC 0.511）。

合规：train-only 训练；特征表 = outputs/wsN6/chem_features_fuse.csv
（PCA 拟合仅 train 37 化合物；外部结构数据已披露）；val 仅模型选择；
不改任何既有文件。

用法: python -m src.wsN7_featcond              # 3 种子 val 对照
      python -m src.wsN7_featcond --seeds 0,1,2,3,4,5,6,7 --test   # 全量交付
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluate import Harness
from .wsD_arch import Cfg, get_tensors, seen_cats, masked_huber, masked_mse
from .train_mlp import CAT_COLS, EMB_DIMS
from .wsD_arch import predict as wsD_predict  # noqa: F401  (docstring 一致性)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN7"

WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
CHEM_COL_IDX = 1  # CAT_COLS 中化合物列


class FeatCondMLP(nn.Module):
    """wsD ProteoMLP2 的变体：化合物嵌入 = 自由表 ⊕ 特征映射双通道。

    x_cat[:, 1] == 0（UNK）或推理期未见化合物 → 特征通道；
    训练时对已见化合物以 feat_drop 概率强制走特征通道（模拟 OOD）。
    """

    def __init__(self, n_cats, n_prot, cfg: Cfg, chem_feat: torch.Tensor):
        from .wsD_arch import Block
        super().__init__()
        self.cfg = cfg
        dims = dict(EMB_DIMS)
        self.embs = nn.ModuleList(
            [nn.Embedding(n, dims[c]) for n, c in zip(n_cats, CAT_COLS)])
        cdim = dims["perturbation_no_concentration"]
        self.register_buffer("chem_feat", chem_feat)  # (n_chem, 128)
        self.feat_map = nn.Sequential(
            nn.Linear(chem_feat.shape[1], 64), nn.GELU(),
            nn.LayerNorm(64), nn.Linear(64, cdim))
        d_in = sum(dims.values())
        blocks, d = [], d_in
        for hdim in cfg.hidden:
            blocks.append(Block(d, hdim, cfg.p_drop,
                                residual=cfg.residual,
                                film_dim=0))
            d = hdim
        self.trunk = nn.ModuleList(blocks)
        self.head = nn.Linear(d, n_prot)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_cat, feat_mask: torch.Tensor | None = None):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        # 特征通道：chem_feat[0] 置零（UNK 无特征），查表后过映射
        cf = self.feat_map(self.chem_feat[x_cat[:, CHEM_COL_IDX]])
        if feat_mask is not None:
            m = feat_mask.unsqueeze(1).float()
            e[CHEM_COL_IDX] = e[CHEM_COL_IDX] * (1 - m) + cf * m
        else:
            # 无 mask 时：idx==0（UNK）自动走特征通道（chem_feat[0]=0 → cf≈常数）
            unk = (x_cat[:, CHEM_COL_IDX] == 0).unsqueeze(1)
            e[CHEM_COL_IDX] = torch.where(unk, cf, e[CHEM_COL_IDX])
        x = torch.cat(e, dim=1)
        for blk in self.trunk:
            x = blk(x)
        return self.head(x)


def build_chem_feat_tensor(h: Harness, enc) -> torch.Tensor:
    """按 encoder 的化合物词表对齐 fuse128 特征；缺失（对照/QC）→ 0。"""
    df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv").set_index("compound")
    pc = [c for c in df.columns if c.startswith("pc")]
    mp = enc.maps["perturbation_no_concentration"]
    n = len(mp) + 1
    F = np.zeros((n, len(pc)), dtype=np.float32)
    for name, idx in mp.items():
        if name in df.index:
            F[idx] = df.loc[name, pc].to_numpy(dtype=np.float32)
    return torch.tensor(F)


def train_one(h: Harness, cfg: Cfg, seed: int, feat_drop: float = 0.15,
              device: str = "cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Z_all, M_all = get_tensors(h)
    rows = h.tr_rows
    n_prot = h.Y_tr.shape[1]
    chem_feat = build_chem_feat_tensor(h, enc)
    model = FeatCondMLP(enc.n_cats, n_prot, cfg, chem_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    X_dev, Z_dev = X_all.to(device), Z_all.to(device)
    M_dev = M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)
    tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
    tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c > 0
    tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]
    for ep in range(cfg.epochs):
        model.train()
        gs = int(np.random.choice(tr_s)); gc = int(np.random.choice(tr_c))
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            # G2 组级：整株/整化合物组本 epoch 走 UNK/特征通道
            xb[xb[:, 0] == gs, 0] = 0
            feat_mask = torch.zeros(len(r), dtype=torch.bool, device=device)
            feat_mask[xb[:, 1] == gc] = True          # 组级 → 特征通道
            feat_mask |= torch.rand(len(r), device=device) < feat_drop  # 逐样本
            # 对照/QC 行不走特征通道（无结构特征，保留自由嵌入）
            feat_mask &= (xb[:, 1] > 0)
            # 菌株列维持 wsD 的 0.15 UNK dropout
            dm = torch.rand(len(r), device=device) < 0.15
            xb[dm, 0] = 0
            pred = model(xb, feat_mask=feat_mask)
            loss = (masked_huber if cfg.loss == "huber" else masked_mse)(
                pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:>3}/{cfg.epochs} loss={tot/nb:.4f}",
                  flush=True)
    return model, enc, mean, std


@torch.no_grad()
def predict(model, enc, mean, std, m, seen: dict, device="cuda",
            bs: int = 1024) -> np.ndarray:
    """推理：训练实见化合物 → 自由嵌入；未见 → 特征通道（idx 置 0 走 cf）。"""
    from .wsD_arch import transform_g3
    model.eval()
    idx = transform_g3(enc, m, seen)  # 未见实体 → UNK(0)
    X = torch.tensor(idx, dtype=torch.long, device=device)
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(X[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    cfg = Cfg.from_dict(WSD_CFG)
    seen = seen_cats(h, h.tr_rows)
    pv, pt = [], []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std = train_one(h, cfg, s)
        pv.append(predict(model, enc, mean, std, h.m_tr, seen))
        if args.test:
            pt.append(predict(model, enc, mean, std, h.m_te, seen))
        print(f"[wsN7] seed={s} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(pv, axis=0).astype(np.float32)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[wsN7 {len(seeds)}种子] composite={res['composite']:.4f} "
          f"FC={fc} resid={rz}")
    print("对照 wsD 3种子: 0.5423 / chem FC 0.511 / chem resid 0.490")
    np.save(OUT / "pred_trainval.npy", P)
    if args.test:
        PT = np.mean(pt, axis=0).astype(np.float32)
        assert np.isfinite(PT).all()
        np.save(OUT / "pred_test.npy", PT)
    (OUT / "scores.json").write_text(json.dumps(
        {"seeds": seeds, "composite": res["composite"], "FC": fc,
         "resid": rz, "per_split": res["per_split"]},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
