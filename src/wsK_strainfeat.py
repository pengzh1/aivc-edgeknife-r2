"""wsK：菌株基因组/表型特征 MLP（S2 未见菌株调制的新信息源）。

动机：wsF 的四路证据显示可迁移 chem×ctx 信号上限 ≈0.34，当前模型已贴近——
缺的不是模型容量而是**菌株身份信息**。本模块把菌株的可学习嵌入替换为
外部基因组特征（出题方确认合规路径；开放榜已登记原则覆盖，封闭榜待书面确认）：
- 特征（outputs/wsK/strain_features.csv，42 维）：
  SNP 距离 3 列 + ORF 含量距离 4 列（1011 酵母基因组计划，
  Peter et al. 2018）+ 35 条件生长表型（同来源 971 株标准化）；
  DHY210 按出题方口径用 S288C 参考基因组代理（ORF 距离精确计算，
  SNP/表型缺失列中性置 0）；BAI（val 菌株）与 CRD（test 菌株）有真实特征行。
- 结构 = wsD 最佳配方（5 层 (1024,2048×4)、Huber、300ep、p_drop 0.3），
  菌株列：42 维特征 → Linear(42→8)+GELU（替换 8 维嵌入）；
  训练时 strain_drop 0.35 置 UNK_MEAN（train 4 株均值特征）。
- 化合物列保持可学习嵌入（隔离菌株侧贡献；与 wsJ 正交，后续可叠加）。
- G3 修复：所有嵌入列（化合物/板号等）推理时未见类别 → UNK(0)；
  菌株列用真实特征（特征空间外推，无随机嵌入行问题）。

合规：训练仅用 h.tr_rows；外部特征只涉及菌株基因型/表型（无标签泄漏）；
Y_te 零接触；种子固定 0-7。val 仅评分。

用法:
    python -m src.wsK_strainfeat                  # 8 seeds × 300ep + val 评分
    python -m src.wsK_strainfeat --seeds 0 --epochs 2 --out outputs/wsK/smoke
    python -m src.wsK_strainfeat --full           # 全量重训 + pred_test.npy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness
from .train_mlp import CAT_COLS, EMB_DIMS, Encoder, masked_mse
from .wsD_arch import masked_huber, seen_cats, transform_g3

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsK"
HIDDEN = (1024, 2048, 2048, 2048, 2048)
STRAIN_COL = "Strains"


def load_strain_mat(enc: Encoder, feat_csv: Path):
    """strain_feat_mat：行 0 = UNK_MEAN，其余按 strain_features.csv 行序。
    返回 (mat (n_strain_rows, 42) float32, name→row 映射)。"""
    df = pd.read_csv(feat_csv, index_col=0)
    unk = df.loc["UNK_MEAN"].to_numpy(dtype=np.float32)
    df = df.drop(index="UNK_MEAN")
    names = df.index.tolist()
    mat = np.concatenate(
        [unk[None], df.to_numpy(dtype=np.float32)], axis=0)
    row_of = {n: i + 1 for i, n in enumerate(names)}
    return mat, row_of


class ProteoMLPStrain(nn.Module):
    """菌株列用基因组特征投影，其余列（含化合物）用可学习嵌入。"""

    def __init__(self, n_cats, strain_mat: np.ndarray, n_prot,
                 hidden=HIDDEN, p_drop=0.3):
        super().__init__()
        self.strain_col = CAT_COLS.index(STRAIN_COL)  # 0
        self.emb_cols = [i for i, c in enumerate(CAT_COLS) if c != STRAIN_COL]
        emb_n = [n for n, c in zip(n_cats, CAT_COLS) if c != STRAIN_COL]
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[CAT_COLS[i]])
            for n, i in zip(emb_n, self.emb_cols)])
        self.register_buffer(
            "strain_mat", torch.tensor(strain_mat, dtype=torch.float32))
        sdim = strain_mat.shape[1]
        self.strain_proj = nn.Sequential(
            nn.Linear(sdim, EMB_DIMS[STRAIN_COL]), nn.GELU())
        d_in = sum(EMB_DIMS[c] for c in CAT_COLS)
        layers, d = [], d_in
        for hd in hidden:
            layers += [nn.Linear(d, hd), nn.GELU(), nn.LayerNorm(hd),
                       nn.Dropout(p_drop)]
            d = hd
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat, x_strain_feat):
        parts = [self.embs[k](x_cat[:, i]) for k, i in enumerate(self.emb_cols)]
        sv = self.strain_proj(x_strain_feat)
        # CAT_COLS 顺序：Strains 在 index 0
        e = torch.cat([sv] + parts, dim=1)
        return self.head(self.trunk(e))


def train_one(h, rows, cfg, seed, strain_mat, row_of, device="cuda",
              stats=None, log_every=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)
    stats = stats or h.stats
    n_prot = h.Y_tr.shape[1]

    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    # 菌株 → 特征行号（独立于 enc，全部 train_val 菌株都有真实特征行）
    sidx = h.m_tr[STRAIN_COL].map(lambda v: row_of.get(v, 0)).to_numpy()
    S_all = torch.tensor(strain_mat[sidx], dtype=torch.float32)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = ProteoMLPStrain(enc.n_cats, strain_mat, n_prot,
                            p_drop=cfg["p_drop"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    n_steps = cfg["epochs"] * int(np.ceil(len(rows) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev, S_dev = X_all.to(device), S_all.to(device)
    Z_dev, M_dev = Z_all.to(device), M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)
    unk_feat = torch.tensor(strain_mat[0], dtype=torch.float32, device=device)

    g2 = cfg.get("g2_aug", True)
    if g2:
        tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
        tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]

    for ep in range(cfg["epochs"]):
        model.train()
        if g2:
            gs = int(np.random.choice(tr_s)); gc = int(np.random.choice(tr_c))
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg["bs"]):
            r = perm[i:i + cfg["bs"]]
            xb = X_dev[r].clone()
            sf = S_dev[r].clone()
            if g2:
                m_gs = xb[:, 0] == gs
                xb[m_gs, 0] = 0
                sf[m_gs] = unk_feat              # 整组菌株 → 均值特征
                xb[xb[:, 1] == gc, 1] = 0        # 整组化合物 → UNK 嵌入
                for col in [0, 1]:
                    dm = torch.rand(len(r), device=device) < 0.15
                    if col == 0:
                        sf[dm] = unk_feat
                    else:
                        xb[dm, col] = 0
            else:
                p = cfg["strain_drop"]
                if p > 0:
                    dm = torch.rand(len(r), device=device) < p
                    sf[dm] = unk_feat
                if cfg["emb_drop"] > 0:
                    dm = torch.rand(len(r), device=device) < cfg["emb_drop"]
                    xb[dm, 1] = 0
            pred = model(xb, sf)
            loss = (masked_huber if cfg["loss"] == "huber" else masked_mse)(
                pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
        if log_every and ((ep + 1) % log_every == 0 or ep == cfg["epochs"] - 1):
            print(f"  epoch {ep+1:>3}/{cfg['epochs']}  loss={tot/nb:.4f}",
                  flush=True)
    return model, enc, mean, std


def predict_g3(model, enc, mean, std, m, seen, strain_mat, row_of,
               device="cuda", bs=1024):
    """G3：所有嵌入列未见类别 → UNK(0)；菌株列用真实特征行。"""
    model.eval()
    X = torch.tensor(transform_g3(enc, m, seen), dtype=torch.long,
                     device=device)
    sidx = m[STRAIN_COL].map(lambda v: row_of.get(v, 0)).to_numpy()
    S = torch.tensor(strain_mat[sidx], dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs], S[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def _finalize(pred, fill):
    bad = ~np.isfinite(pred)
    if bad.any():
        r, c = np.where(bad)
        pred[r, c] = np.take(fill, c)
    assert np.isfinite(pred).all()
    return pred.astype(np.float32)


DEFAULT_CFG = dict(epochs=300, lr=1e-3, wd=1e-4, bs=256, p_drop=0.3,
                   emb_drop=0.35, strain_drop=0.35, loss="huber", g2_aug=True)


def run_val(h, out, cfg, seeds, device):
    strain_mat, row_of = load_strain_mat(None, OUT_DIR / "strain_features.csv")
    seen = seen_cats(h, h.tr_rows)
    preds = []
    for s in seeds:
        print(f"[seed {s}] {cfg['epochs']}ep ...", flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, h.tr_rows, cfg, s, strain_mat,
                                          row_of, device=device)
        print(f"[seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_tr, seen,
                                strain_mat, row_of, device=device))
    pred = _finalize(np.mean(preds, axis=0), h.stats.protein_mean)
    np.save(out / "pred_trainval.npy", pred)
    print(f"[saved] {out/'pred_trainval.npy'} {pred.shape}", flush=True)
    return h.score_val(pred)


def run_full(h, out, cfg, seeds, device):
    rows = np.arange(len(h.m_tr))
    stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)
    strain_mat, row_of = load_strain_mat(None, OUT_DIR / "strain_features.csv")
    seen = seen_cats(h, rows)
    preds = []
    for s in seeds:
        print(f"[full seed {s}] ...", flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, rows, cfg, s, strain_mat, row_of,
                                          device=device, stats=stats)
        print(f"[full seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_te, seen,
                                strain_mat, row_of, device=device))
    pred = _finalize(np.mean(preds, axis=0), stats.protein_mean)
    assert pred.shape == (len(h.m_te), h.Y_tr.shape[1])
    np.save(out / "pred_test.npy", pred)
    print(f"[saved] {out/'pred_test.npy'} {pred.shape}", flush=True)
    return {"shape": list(pred.shape), "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--no_g2", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(DEFAULT_CFG, epochs=args.epochs, g2_aug=not args.no_g2)

    h = Harness()
    if args.full:
        info = run_full(h, out, cfg, seeds, device)
        (out / "pred_test_info.json").write_text(json.dumps(info, indent=1))
        return
    res = run_val(h, out, cfg, seeds, device)
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    scores[f"wsK{'_nog2' if args.no_g2 else ''}"] = res
    scores_path.write_text(json.dumps(scores, indent=1))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
