"""wsJ：强配方化合物描述符 MLP（wsA 特征 × wsD 配方 × G3 修复）。

动机：wsA 证明分子描述符对未见化合物有真实外推增益（chem FC +0.035 同预算消融），
但 wsA 只是 (512,1024)×100ep 的小模型，且菌株侧存在 G3 未训练嵌入行问题
（val_strain FC 0.286 反而低于消融版 0.292）。本模块把描述符方案嫁接到 wsD 最佳配方
（5 层 (1024,2048×4)、Huber、300ep、dropout 0.3、emb_drop 0.35、G2 组级增强），
并对非化合物列做 G3 修复（推理时未见类别 → UNK(0)），目标是把 chem 侧的结构化外推
推到 wsD 强度，供开放榜（已登记外部数据）路由集成使用；若组委会书面确认结构
embedding 封闭榜合规，亦可并入封闭榜。

设计（与 wsA 的差异）：
1. 配方 = wsD 最佳：hidden (1024,2048,2048,2048,2048)、masked Huber(β=1)、
   300 epochs cosine、p_drop 0.3、strain emb_drop 0.35、chem 描述符 drop 0.35
   （置 train 化合物描述符均值，即 chem_mat[0]）。
2. G3 修复：strain / 其余类别列在推理时，凡训练行未出现的类别强制 UNK(0)
   （wsA 的 Encoder 在 train_val 全集 fit，BAI 会分到未训练随机嵌入行——本模块修复）。
   化合物列不修：未见化合物使用真实描述符（这正是描述符方案的意义）。
3. PCA 拟合集可选：train37（仅 train 化合物，与 wsA 同口径）或 all54
   （全部化合物，纯结构信息、无标签泄漏；n_comp 提升到 53，描述符信息更全）。
4. G2 组级增强默认开（wsG 证据：+0.0036）：每 epoch 随机 1 菌株整组 UNK +
   1 化合物整组置均值描述符，其余逐样本 0.15。

合规：训练仅用 h.tr_rows；描述符标准化/PCA 拟合只涉及化合物结构（无标签）；
Y_te 零接触；种子固定。val 划分仅用于评分。

用法:
    python -m src.wsJ_chemboost                 # 8 seeds × 300ep + val 评分（主实验）
    python -m src.wsJ_chemboost --seeds 0 --epochs 2 --out outputs/wsJ/smoke  # 冒烟
    python -m src.wsJ_chemboost --pca all54     # PCA 用全部 54 化合物拟合（默认 train37）
    python -m src.wsJ_chemboost --full          # 全量 train_val 重训 + pred_test.npy
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
from .wsA_chemfeat import (CHEM_COL, CHEM_DIM, CHEM_HID, EXCLUDE,
                           build_chem_features, make_chem_mat)
from .wsD_arch import masked_huber

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsJ"
HIDDEN = (1024, 2048, 2048, 2048, 2048)


# ---------------------------------------------------------------- 特征（支持 all54）

def load_chem_table_j(h: Harness, out_dir: Path, pca: str = "train37",
                      full: bool = False):
    """{化合物: (64,) 向量} + 拟合集均值向量。

    pca="train37": 仅 train 化合物拟合标准化/PCA（wsA 同口径）。
    pca="all54"  : 全部 54 个非对照化合物拟合（纯结构信息，无标签）。
    full=True    : 全部 train_val 化合物（43 个；test-only 仅 transform）。
    smiles.csv 从 wsA 目录复制一份到本目录（不修改他人产物）。
    """
    if not (out_dir / "smiles.csv").exists():
        import shutil
        shutil.copy(OUT_DIR.parent / "wsA" / "smiles.csv", out_dir / "smiles.csv")
    if full:
        fit_names = set(h.m_tr[CHEM_COL].unique()) - EXCLUDE
        df = build_chem_features(fit_names, out_dir, "chem_features_full.csv")
    elif pca == "all54":
        import csv
        with open(out_dir / "smiles.csv", encoding="utf-8") as f:
            all_names = {r["compound"] for r in csv.DictReader(f)} - EXCLUDE
        df = build_chem_features(all_names, out_dir, "chem_features_all54.csv")
    else:
        fit_names = set(
            h.m_train.loc[h.m_train["chemical_role"] == "train",
                          CHEM_COL].unique()) - EXCLUDE
        df = build_chem_features(fit_names, out_dir, "chem_features_train37.csv")
    feat = {r.compound: r.drop("compound").to_numpy(dtype=np.float32)
            for _, r in df.iterrows()}
    tr_fit = set(h.m_train.loc[h.m_train["chemical_role"] == "train",
                               CHEM_COL].unique()) - EXCLUDE
    tr_vecs = [feat[c] for c in tr_fit if c in feat]
    mean_vec = np.mean(tr_vecs, axis=0).astype(np.float32)
    return feat, mean_vec


# ---------------------------------------------------------------- 模型

class ProteoMLPChem2(nn.Module):
    """wsA 描述符输入 + wsD 深度配方。"""

    def __init__(self, n_cats, chem_mat: np.ndarray, n_prot,
                 hidden=HIDDEN, p_drop=0.3):
        super().__init__()
        self.emb_cols = [i for i, c in enumerate(CAT_COLS) if c != CHEM_COL]
        self.chem_col = CAT_COLS.index(CHEM_COL)
        emb_n = [n for n, c in zip(n_cats, CAT_COLS) if c != CHEM_COL]
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[CAT_COLS[i]])
            for n, i in zip(emb_n, self.emb_cols)])
        self.register_buffer(
            "chem_mat", torch.tensor(chem_mat, dtype=torch.float32))
        self.chem_proj = nn.Sequential(nn.Linear(CHEM_DIM, CHEM_HID), nn.GELU())
        d_in = sum(EMB_DIMS[CAT_COLS[i]] for i in self.emb_cols) + CHEM_HID
        layers, d = [], d_in
        for hd in hidden:
            layers += [nn.Linear(d, hd), nn.GELU(), nn.LayerNorm(hd),
                       nn.Dropout(p_drop)]
            d = hd
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat):
        parts = [self.embs[k](x_cat[:, i]) for k, i in enumerate(self.emb_cols)]
        cv = self.chem_proj(self.chem_mat[x_cat[:, self.chem_col]])
        e = torch.cat([parts[0], cv] + parts[1:], dim=1)  # chem 列原位置 index 1
        return self.head(self.trunk(e))


# ---------------------------------------------------------------- 训练 / 推理

def train_one(h: Harness, rows: np.ndarray, cfg: dict, seed: int,
              chem_mat: np.ndarray, device: str = "cuda",
              stats=None, log_every: int = 20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)   # 全类别词表（化合物列供描述符映射；G3 在推理侧修）
    stats = stats or h.stats
    n_prot = h.Y_tr.shape[1]

    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = ProteoMLPChem2(enc.n_cats, chem_mat, n_prot,
                           p_drop=cfg["p_drop"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n_steps = cfg["epochs"] * int(np.ceil(len(rows) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev = X_all.to(device)
    Z_dev = Z_all.to(device)
    M_dev = M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)

    g2 = cfg.get("g2_aug", True)
    if g2:
        tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
        tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]

    for ep in range(cfg["epochs"]):
        model.train()
        if g2:
            gs = int(np.random.choice(tr_s))
            gc = int(np.random.choice(tr_c))
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg["bs"]):
            r = perm[i:i + cfg["bs"]]
            xb = X_dev[r].clone()
            if g2:
                # 组级 UNK：模拟整个实体未见
                xb[xb[:, 0] == gs, 0] = 0
                xb[xb[:, 1] == gc, 1] = 0   # → chem_mat[0] = 均值描述符
                for col in [0, 1]:
                    dm = torch.rand(len(r), device=device) < 0.15
                    xb[dm, col] = 0
            else:
                for col, p in [(0, cfg["emb_drop"]), (1, cfg["chem_drop"])]:
                    if p > 0:
                        dm = torch.rand(len(r), device=device) < p
                        xb[dm, col] = 0
            pred = model(xb)
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


def predict_g3(model, enc, mean, std, m: pd.DataFrame, seen: dict,
               device="cuda", bs: int = 1024) -> np.ndarray:
    """G3 修复推理：非化合物列中训练未见的类别 → UNK(0)；化合物列保留描述符。"""
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
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def _seen_cats(h: Harness, rows: np.ndarray) -> dict:
    mt = h.m_tr.iloc[rows]
    return {c: set(mt[c].unique()) for c in CAT_COLS}


def _finalize(pred: np.ndarray, fill_mean: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"[warn] {bad.sum()} non-finite -> protein mean")
        r, c = np.where(bad)
        pred[r, c] = np.take(fill_mean, c)
    assert np.isfinite(pred).all()
    return pred.astype(np.float32)


# ---------------------------------------------------------------- 主流程

DEFAULT_CFG = dict(epochs=300, lr=1e-3, wd=1e-4, bs=256, p_drop=0.3,
                   emb_drop=0.35, chem_drop=0.35, loss="huber", g2_aug=True)


def run_val(h: Harness, out: Path, cfg: dict, seeds: list[int],
            pca: str, device: str) -> dict:
    feat, mean_vec = load_chem_table_j(h, out, pca=pca)
    # 词表覆盖 train_val（化合物描述符映射需要），chem_mat 默认均值
    enc0 = Encoder().fit(h.m_tr)
    chem_mat = make_chem_mat(enc0, feat, mean_vec)
    seen = _seen_cats(h, h.tr_rows)
    preds = []
    for s in seeds:
        print(f"[seed {s}] {cfg['epochs']}ep ...", flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, h.tr_rows, cfg, s, chem_mat,
                                          device=device)
        assert enc.maps is enc0.maps or enc.maps == enc0.maps
        print(f"[seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_tr, seen,
                                device=device))
    pred = _finalize(np.mean(preds, axis=0), h.stats.protein_mean)
    np.save(out / "pred_trainval.npy", pred)
    print(f"[saved] {out/'pred_trainval.npy'} {pred.shape}", flush=True)
    return h.score_val(pred)


def run_full(h: Harness, out: Path, cfg: dict, seeds: list[int],
             device: str) -> dict:
    """全量 train_val 重训 + test 预测（开放榜提交用；Y_te 零接触）。"""
    rows = np.arange(len(h.m_tr))
    stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)
    feat, mean_vec = load_chem_table_j(h, out, full=True)
    enc0 = Encoder().fit(h.m_tr)
    # 化合物词表扩展 test-only 名称（仅用 m_te 元数据）
    mp = enc0.maps[CHEM_COL]
    for c in sorted(set(h.m_te[CHEM_COL]) - set(mp), key=str):
        mp[c] = len(mp) + 1
    chem_mat = make_chem_mat(enc0, feat, mean_vec)
    seen = _seen_cats(h, rows)
    preds = []
    for s in seeds:
        print(f"[full seed {s}] {cfg['epochs']}ep on {len(rows)} rows ...",
              flush=True)
        t0 = time.time()
        model, enc, mean, std = train_one(h, rows, cfg, s, chem_mat,
                                          device=device, stats=stats)
        enc.maps = enc0.maps  # 与扩展词表一致
        print(f"[full seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_te, seen,
                                device=device))
    pred = _finalize(np.mean(preds, axis=0), stats.protein_mean)
    assert pred.shape == (len(h.m_te), h.Y_tr.shape[1])
    np.save(out / "pred_test.npy", pred)
    print(f"[saved] {out/'pred_test.npy'} {pred.shape}", flush=True)
    return {"shape": list(pred.shape), "seeds": seeds, "cfg": cfg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--pca", choices=["train37", "all54"], default="all54")
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
    res = run_val(h, out, cfg, seeds, args.pca, device)
    scores_path = out / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    scores[f"wsJ_{args.pca}{'_nog2' if args.no_g2 else ''}"] = res
    scores_path.write_text(json.dumps(scores, indent=1))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
