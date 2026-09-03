"""wsL：双外部表征模型（wsJ 化合物描述符 × wsK 菌株基因组特征，开放榜主力候选）。

= wsD 最佳配方（5 层 (1024,2048×4)、Huber、300ep、p_drop 0.3）
+ 化合物列：64 维结构描述符投影（wsJ，PubChem SMILES→RDKit，PCA-all54）
+ 菌株列：42 维基因组/表型特征投影（wsK，1011 计划 + 35 条件生长）
两列训练期各自 0.35 概率回退（chem→均值描述符，strain→UNK_MEAN 特征），
G2 组级增强默认开；推理：化合物用真实描述符、菌株用真实特征，
其余嵌入列（板号等）未见类别 → UNK(0)（G3 修复）。

开放榜合规：外部数据 = PubChem + RDKit（wsA 已登记）+ 1011 基因组/表型
（需在开放榜登记中补充 Peter et al. 2018 数据集条目）。
封闭榜：待组委会书面确认结构/基因组 embedding 合规后方可并入。

用法:
    python -m src.wsL_dualfeat                    # 8 seeds × 300ep + val 评分
    python -m src.wsL_dualfeat --seeds 0 --epochs 2 --out outputs/wsL/smoke
    python -m src.wsL_dualfeat --full             # 全量重训 + pred_test.npy
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
from .wsA_chemfeat import CHEM_COL, CHEM_DIM, CHEM_HID, make_chem_mat
from .wsD_arch import masked_huber, seen_cats, transform_g3
from .wsJ_chemboost import load_chem_table_j
from .wsK_strainfeat import load_strain_mat

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsL"
HIDDEN = (1024, 2048, 2048, 2048, 2048)
STRAIN_COL = "Strains"


class ProteoMLPDual(nn.Module):
    """化合物描述符 + 菌株基因组特征；其余列可学习嵌入。"""

    def __init__(self, n_cats, chem_mat, strain_mat, n_prot,
                 hidden=HIDDEN, p_drop=0.3):
        super().__init__()
        keep = {STRAIN_COL, CHEM_COL}
        self.emb_cols = [i for i, c in enumerate(CAT_COLS) if c not in keep]
        emb_n = [n for n, c in zip(n_cats, CAT_COLS) if c not in keep]
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[CAT_COLS[i]])
            for n, i in zip(emb_n, self.emb_cols)])
        self.register_buffer("chem_mat",
                             torch.tensor(chem_mat, dtype=torch.float32))
        self.register_buffer("strain_mat",
                             torch.tensor(strain_mat, dtype=torch.float32))
        self.chem_proj = nn.Sequential(nn.Linear(CHEM_DIM, CHEM_HID), nn.GELU())
        self.strain_proj = nn.Sequential(
            nn.Linear(strain_mat.shape[1], EMB_DIMS[STRAIN_COL]), nn.GELU())
        d_in = (sum(EMB_DIMS[CAT_COLS[i]] for i in self.emb_cols)
                + CHEM_HID + EMB_DIMS[STRAIN_COL])
        layers, d = [], d_in
        for hd in hidden:
            layers += [nn.Linear(d, hd), nn.GELU(), nn.LayerNorm(hd),
                       nn.Dropout(p_drop)]
            d = hd
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat, chem_idx, strain_feat):
        # CAT_COLS 顺序：0=Strains, 1=chem, 其余依次
        parts = [self.embs[k](x_cat[:, i]) for k, i in enumerate(self.emb_cols)]
        sv = self.strain_proj(strain_feat)
        cv = self.chem_proj(self.chem_mat[chem_idx])
        return self.head(self.trunk(torch.cat([sv, cv] + parts, dim=1)))


def _encode(h, enc, row_of, df):
    X = torch.tensor(enc.transform(df), dtype=torch.long)
    sidx = df[STRAIN_COL].map(lambda v: row_of.get(v, 0)).to_numpy()
    return X, sidx


def train_one(h, rows, cfg, seed, chem_mat, strain_mat, row_of,
              device="cuda", stats=None, log_every=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)
    stats = stats or h.stats
    n_prot = h.Y_tr.shape[1]

    X_all, sidx = _encode(h, enc, row_of, h.m_tr)
    S_all = torch.tensor(strain_mat[sidx], dtype=torch.float32)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = ProteoMLPDual(enc.n_cats, chem_mat, strain_mat, n_prot,
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
                sf[m_gs] = unk_feat
                xb[xb[:, 1] == gc, 1] = 0   # chem → chem_mat[0] 均值描述符
                dm = torch.rand(len(r), device=device) < 0.15
                sf[dm] = unk_feat
                dm = torch.rand(len(r), device=device) < 0.15
                xb[dm, 1] = 0
            else:
                dm = torch.rand(len(r), device=device) < cfg["strain_drop"]
                sf[dm] = unk_feat
                dm = torch.rand(len(r), device=device) < cfg["chem_drop"]
                xb[dm, 1] = 0
            pred = model(xb, xb[:, 1], sf)
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
    """化合物列保留描述符映射（不做 G3 置 0）；其余嵌入列 G3。"""
    model.eval()
    cols = []
    for c in CAT_COLS:
        mp = enc.maps[c]
        if c == CHEM_COL:
            cols.append(m[c].map(lambda v: mp.get(v, 0)).to_numpy())
        else:
            keep = seen[c]
            cols.append(
                m[c].map(lambda v: mp.get(v, 0) if v in keep else 0).to_numpy())
    X = torch.tensor(np.stack(cols, axis=1), dtype=torch.long, device=device)
    sidx = m[STRAIN_COL].map(lambda v: row_of.get(v, 0)).to_numpy()
    S = torch.tensor(strain_mat[sidx], dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs], X[i:i + bs, 1],
                              S[i:i + bs]).float().cpu())
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
                   chem_drop=0.35, strain_drop=0.35, loss="huber", g2_aug=True)


def run_val(h, out, cfg, seeds, device):
    feat, mean_vec = load_chem_table_j(h, OUT_DIR, pca="all54")
    enc0 = Encoder().fit(h.m_tr)
    chem_mat = make_chem_mat(enc0, feat, mean_vec)
    strain_mat, row_of = load_strain_mat(
        None, OUT_DIR.parent / "wsK" / "strain_features.csv")
    seen = seen_cats(h, h.tr_rows)
    preds = []
    for s in seeds:
        print(f"[seed {s}] {cfg['epochs']}ep ...", flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, h.tr_rows, cfg, s, chem_mat,
                                          strain_mat, row_of, device=device)
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
    feat, mean_vec = load_chem_table_j(h, OUT_DIR, full=True)
    enc0 = Encoder().fit(h.m_tr)
    mp = enc0.maps[CHEM_COL]
    for c in sorted(set(h.m_te[CHEM_COL]) - set(mp), key=str):
        mp[c] = len(mp) + 1
    chem_mat = make_chem_mat(enc0, feat, mean_vec)
    strain_mat, row_of = load_strain_mat(
        None, OUT_DIR.parent / "wsK" / "strain_features.csv")
    seen = seen_cats(h, rows)
    preds = []
    for s in seeds:
        print(f"[full seed {s}] ...", flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, rows, cfg, s, chem_mat,
                                          strain_mat, row_of, device=device,
                                          stats=stats)
        enc.maps = enc0.maps
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
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(DEFAULT_CFG, epochs=args.epochs)

    h = Harness()
    if args.full:
        info = run_full(h, out, cfg, seeds, device)
        (out / "pred_test_info.json").write_text(json.dumps(info, indent=1))
        return
    res = run_val(h, out, cfg, seeds, device)
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    scores["wsL_dual"] = res
    scores_path.write_text(json.dumps(scores, indent=1))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
