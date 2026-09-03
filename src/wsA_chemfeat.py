"""化合物分子描述符 MLP（wsA）：用结构描述符替换可学习化合物嵌入。

思路：未见化合物（val_chem_only / test_chem_only）在原 MLP 中只能回退 UNK 嵌入。
这里把化合物身份表示为"分子结构描述符向量"（PubChem SMILES → RDKit Morgan 指纹
r=2/2048bit + 16 个理化描述符 → 标准化 → PCA 64 维），模型学"结构 → 响应"映射，
从而对从未见过的化合物具备结构化外推能力。

- 化合物表示：64 维描述符 → Linear(64→48) → GELU（替换原 32 维可学习嵌入）
- 训练时 25% 概率将化合物列索引置 0（chem_mat[0] = train 化合物描述符均值），
  学"未知/缺失描述符"回退；菌株列保留 emb_drop=0.25
- 其余结构/超参与 src/train_mlp.py 相同：trunk (512,1024) + LayerNorm + Dropout(0.1)，
  AdamW(lr=1e-3, wd=1e-4) + cosine，batch 256，masked MSE，100 epochs，仅 h.tr_rows 训练
- 标准化与 PCA 均只用 train_val 中 chemical_role=='train' 的 37 个化合物拟合；
  PCA 后逐维再标准化（同样仅 train 拟合）保证网络输入尺度 ~1
- SMILES 缓存：outputs/wsA/smiles.csv（PubChem PUG-REST，54/54 覆盖；
  盐类取母核 FragmentParent，过渡金属配合物如顺铂保留整体）

用法:
    python -m src.wsA_chemfeat                    # 3 seeds × 100 epochs + val 评分
    python -m src.wsA_chemfeat --ablation         # 可学习嵌入消融（退化为原 MLP 配置）
    python -m src.wsA_chemfeat --full --seeds 0,1,2,3,4
        # 最终 test 提交：全部 train_val 行重训（FrozenStats 全量重估、
        # 描述符标准化/PCA 用全部 train_val 43 个化合物拟合，缓存
        # chem_features_full.csv），预测 h.m_te -> outputs/wsA/pred_test.npy；
        # 化合物词表扩展 test-only 名称使其使用真实描述符（仅用 m_te 元数据）
    python -m src.wsA_chemfeat --epochs 2 --seeds 0 --out outputs/wsA/smoke  # 冒烟
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
from .train_mlp import CAT_COLS, Encoder, masked_mse

CHEM_COL = "perturbation_no_concentration"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsA"

# 除化合物列外的嵌入维度（与 train_mlp.EMB_DIMS 一致）
EMB_DIMS = {"Strains": 8, "Medium": 2, "Temperature": 2, "pert_time": 4,
            "instrument": 6, "data_source": 3, "Yeast_cell_plate": 32}
CHEM_DIM = 64
CHEM_HID = 48

EXCLUDE = {"Water", "DMSO", "Quality Control"}
PHYSCHEM_NAMES = ["MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds",
                  "RingCount", "FractionCSP3", "HeavyAtomCount",
                  "NumAromaticRings", "NumAliphaticRings", "NumHeteroatoms",
                  "NHOHCount", "NOCount", "MolMR", "FormalCharge"]
# 过渡金属配合物（顺铂）不脱盐，避免丢失金属中心
TRANSITION_METALS = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                     39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                     72, 73, 74, 75, 76, 77, 78, 79, 80}


# ---------------------------------------------------------------- 特征构建

def _parent_mol(mol):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    if not any(a.GetAtomicNum() in TRANSITION_METALS for a in mol.GetAtoms()):
        mol = rdMolStandardize.FragmentParent(mol)
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # 重建 ring info


def build_chem_features(fit_names: set, out_dir: Path,
                        cache_name: str = "chem_features.csv") -> pd.DataFrame:
    """从 smiles.csv 构建 64 维化合物特征；有缓存直接用。

    标准化/PCA 仅用 fit_names 中的化合物拟合：
    - val 版：chemical_role=='train' 的 37 个化合物（chem_features.csv）
    - full 版：全部 train_val 化合物 43 个（chem_features_full.csv）
    """
    cache = out_dir / cache_name
    if cache.exists():
        return pd.read_csv(cache)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    from sklearn.decomposition import PCA
    RDLogger.DisableLog("rdApp.*")

    physchem = [
        Descriptors.MolWt, Descriptors.MolLogP, rdMolDescriptors.CalcTPSA,
        Lipinski.NumHDonors, Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds,
        rdMolDescriptors.CalcNumRings, rdMolDescriptors.CalcFractionCSP3,
        Lipinski.HeavyAtomCount, rdMolDescriptors.CalcNumAromaticRings,
        rdMolDescriptors.CalcNumAliphaticRings, rdMolDescriptors.CalcNumHeteroatoms,
        Lipinski.NHOHCount, Lipinski.NOCount, Descriptors.MolMR,
        lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms()),
    ]

    smi = pd.read_csv(out_dir / "smiles.csv")
    names = smi["compound"].tolist()
    train_idx = [i for i, c in enumerate(names) if c in fit_names]
    print(f"[chemfeat] {cache_name}: {len(names)} compounds | fit {len(train_idx)}")

    rows = []
    for r in smi.itertuples():
        mol = _parent_mol(Chem.MolFromSmiles(r.smiles))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        bits = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, bits)
        pc = np.array([f(mol) for f in physchem], dtype=np.float32)
        rows.append(np.concatenate([bits, pc]))
    F = np.stack(rows)

    mu, sd = F[train_idx].mean(0), F[train_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (F - mu) / sd
    n_comp = min(CHEM_DIM, len(train_idx) - 1)  # 秩 ≤ n_fit-1，避免退化方向
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(Z[train_idx])
    P = pca.transform(Z)
    pm, ps = P[train_idx].mean(0), P[train_idx].std(0)
    ps[ps < 1e-8] = 1.0
    P = ((P - pm) / ps).astype(np.float32)
    if n_comp < CHEM_DIM:
        P = np.concatenate(
            [P, np.zeros((len(P), CHEM_DIM - n_comp), np.float32)], axis=1)

    df = pd.DataFrame(P, columns=[f"pc{i}" for i in range(CHEM_DIM)])
    df.insert(0, "compound", names)
    df.to_csv(cache, index=False)
    print(f"[chemfeat] saved {cache} {df.shape}")
    return df


def load_chem_table(h: Harness, out_dir: Path, full: bool = False):
    """返回 {化合物名: (64,) 向量} 与拟合集化合物均值向量。

    full=False：标准化/PCA 拟合 chemical_role=='train' 的 37 个 train 化合物（val 版）。
    full=True ：拟合全部 train_val 化合物（43 个；test-only 化合物不参与拟合，
    仅被 transform，供 test 预测使用）。
    """
    if full:
        fit_names = set(h.m_tr[CHEM_COL].unique()) - EXCLUDE
        df = build_chem_features(fit_names, out_dir, "chem_features_full.csv")
    else:
        fit_names = set(
            h.m_train.loc[h.m_train["chemical_role"] == "train",
                          CHEM_COL].unique()) - EXCLUDE
        df = build_chem_features(fit_names, out_dir, "chem_features.csv")
    feat = {r.compound: r.drop("compound").to_numpy(dtype=np.float32)
            for _, r in df.iterrows()}
    tr_vecs = [feat[c] for c in fit_names if c in feat]
    mean_vec = np.mean(tr_vecs, axis=0).astype(np.float32)
    return feat, mean_vec


# ---------------------------------------------------------------- 模型

class ProteoMLPChem(nn.Module):
    """化合物列用描述符投影，其余类别列用嵌入（结构同 train_mlp.ProteoMLP）。"""

    def __init__(self, n_cats, chem_mat: np.ndarray, n_prot,
                 hidden=(512, 1024), p_drop=0.1):
        super().__init__()
        # 嵌入列（保持 CAT_COLS 顺序，跳过化合物列）
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
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h),
                       nn.Dropout(p_drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat):
        parts = [self.embs[k](x_cat[:, i]) for k, i in enumerate(self.emb_cols)]
        cv = self.chem_proj(self.chem_mat[x_cat[:, self.chem_col]])
        # 按 CAT_COLS 原始顺序拼接：化合物列位于 index 1
        e = torch.cat([parts[0], cv] + parts[1:], dim=1)
        return self.head(self.trunk(e))


def make_chem_mat(enc: Encoder, feat: dict, mean_vec: np.ndarray) -> np.ndarray:
    """chem_mat[idx] = 化合物 64 维描述符；UNK(0)/对照/缺失 = train 均值。"""
    mp = enc.maps[CHEM_COL]
    mat = np.tile(mean_vec, (len(mp) + 1, 1)).astype(np.float32)
    for name, idx in mp.items():
        if name in feat:
            mat[idx] = feat[name]
    return mat


def train_model(h: Harness, rows: np.ndarray, epochs: int, seed: int = 0,
                emb_drop: float = 0.25, chem_drop: float = 0.25,
                lr: float = 1e-3, bs: int = 256, device: str = "cuda",
                log_every: int = 10, stats=None, full: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)
    if full:
        # 化合物词表扩展 test-only 化合物（仅用 m_te 元数据名称，不涉及 Y_te），
        # 让其在 chem_mat 中使用真实描述符；其余列（菌株等）保持 m_tr 词表，
        # test 独有值 → UNK(0)，回退表示经 emb_drop 训练过
        mp = enc.maps[CHEM_COL]
        for c in sorted(set(h.m_te[CHEM_COL]) - set(mp), key=str):
            mp[c] = len(mp) + 1
    feat, mean_vec = load_chem_table(h, OUT_DIR, full=full)
    chem_mat = make_chem_mat(enc, feat, mean_vec)
    n_prot = h.Y_tr.shape[1]
    stats = stats or h.stats

    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = ProteoMLPChem(enc.n_cats, chem_mat, n_prot).to(device)
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
            # 菌株 embedding dropout（学 OOD 回退）
            if emb_drop > 0:
                drop = torch.rand(len(r), device=device) < emb_drop
                xb[drop, 0] = 0
            # 化合物描述符 → train 均值（chem_mat[0] 即均值向量）
            if chem_drop > 0:
                drop = torch.rand(len(r), device=device) < chem_drop
                xb[drop, 1] = 0
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


# ---------------------------------------------------------------- 主流程

def run_main(h: Harness, out: Path, epochs: int, seeds: list[int],
             emb_drop: float, chem_drop: float, lr: float, bs: int,
             device: str) -> dict:
    preds = []
    for s in seeds:
        print(f"[seed {s}] training {epochs} epochs ...")
        t0 = time.time()
        model, enc, mean, std = train_model(
            h, h.tr_rows, epochs, seed=s, emb_drop=emb_drop,
            chem_drop=chem_drop, lr=lr, bs=bs, device=device)
        print(f"[seed {s}] train {time.time()-t0:.0f}s")
        preds.append(predict(model, enc, mean, std, h.m_tr, device=device))
    pred = np.mean(preds, axis=0).astype(np.float32)
    # 交付契约：无 NaN/Inf（缺失处用 train 蛋白均值填）
    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"[warn] {bad.sum()} non-finite preds -> train protein mean")
        r, c = np.where(bad)
        pred[r, c] = np.take(h.stats.protein_mean, c)
    assert np.isfinite(pred).all()
    np.save(out / "pred_trainval.npy", pred)
    print(f"[saved] {out/'pred_trainval.npy'} {pred.shape}")
    res = h.score_val(pred)
    return res


def run_ablation(h: Harness, out: Path, epochs: int, seeds: list[int],
                 lr: float, bs: int, device: str) -> dict:
    """消融：化合物表示换回可学习嵌入（退化为原 MLP，emb_drop=0.15 原配置）。"""
    from .train_mlp import predict as mlp_predict
    from .train_mlp import train_model as mlp_train
    preds = []
    for s in seeds:
        print(f"[ablation seed {s}] learnable embedding, {epochs} epochs ...")
        t0 = time.time()
        model, enc, mean, std = mlp_train(h, h.tr_rows, epochs, seed=s,
                                          emb_drop=0.15, lr=lr, bs=bs,
                                          device=device, log_every=10)
        print(f"[ablation seed {s}] train {time.time()-t0:.0f}s")
        preds.append(mlp_predict(model, enc, mean, std, h.m_tr, device=device))
    pred = np.mean(preds, axis=0).astype(np.float32)
    r, c = np.where(~np.isfinite(pred))
    if len(r):
        pred[r, c] = np.take(h.stats.protein_mean, c)
    np.save(out / "pred_ablation_mlp.npy", pred)
    print(f"[saved] {out/'pred_ablation_mlp.npy'} {pred.shape}")
    return h.score_val(pred)


def run_full(h: Harness, out: Path, epochs: int, seeds: list[int],
             emb_drop: float, chem_drop: float, lr: float, bs: int,
             device: str) -> dict:
    """最终 test 提交：全部 train_val 行重训（冻结统计全量重估），预测 h.m_te。

    - rows = np.arange(len(h.m_tr))；蛋白均值/std 用全量 train_val 冻结
    - 描述符标准化/PCA 用全部 train_val 化合物（43 个）拟合
    - 其余超参与 val 版一致；预测取 seeds 均值
    """
    rows = np.arange(len(h.m_tr))
    stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)  # 全量 train_val 冻结统计
    preds = []
    for s in seeds:
        print(f"[full seed {s}] training {epochs} epochs on {len(rows)} rows ...")
        t0 = time.time()
        model, enc, mean, std = train_model(
            h, rows, epochs, seed=s, emb_drop=emb_drop, chem_drop=chem_drop,
            lr=lr, bs=bs, device=device, stats=stats, full=True)
        print(f"[full seed {s}] train {time.time()-t0:.0f}s")
        preds.append(predict(model, enc, mean, std, h.m_te, device=device))
    pred = np.mean(preds, axis=0).astype(np.float32)
    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"[warn] {bad.sum()} non-finite preds -> full-trainval protein mean")
        r, c = np.where(bad)
        pred[r, c] = np.take(stats.protein_mean, c)
    assert np.isfinite(pred).all()
    assert pred.shape == (len(h.m_te), h.Y_tr.shape[1]), pred.shape
    np.save(out / "pred_test.npy", pred)
    print(f"[saved] {out/'pred_test.npy'} {pred.shape} dtype={pred.dtype}")
    return {"shape": list(pred.shape), "seeds": seeds, "epochs": epochs,
            "rows": int(len(rows)), "n_test_compounds_with_descriptors":
            len(set(h.m_te[CHEM_COL].unique()) - EXCLUDE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--emb_drop", type=float, default=0.25)
    ap.add_argument("--chem_drop", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="全量 train_val 重训 + 预测 test -> pred_test.npy")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h = Harness()
    if args.full:
        info = run_full(h, out, args.epochs, seeds, args.emb_drop,
                        args.chem_drop, args.lr, args.bs, device)
        (out / "pred_test_info.json").write_text(json.dumps(info, indent=1))
        print(f"[saved] {out/'pred_test_info.json'}")
        return
    scores_path = OUT_DIR / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    if args.ablation:
        scores["ablation_mlp_100ep"] = run_ablation(
            h, out, args.epochs, seeds, args.lr, args.bs, device)
    else:
        scores["chemfeat"] = run_main(
            h, out, args.epochs, seeds, args.emb_drop, args.chem_drop,
            args.lr, args.bs, device)
    scores_path.write_text(json.dumps(scores, indent=1))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
