"""wsP: 化学基因组学机制先验族（STITCH5 化合物→酵母蛋白关联特征）。

动机：chem 侧全部既有特征都是"结构空间"（RDKit FP/描述符 + 3×分子 LM，
已互冗余，fuse4 边际 Δ=0 实证）。STITCH v5（stitch-db.org，2026-08-14 下载
4932 专株文件，extref/stitch/）提供"活性空间"的全新信息类型：化合物 →
3,315 个酵母蛋白的关联打分（0-999，实验/数据库/文本挖掘多通道），即作用
机制指纹。覆盖我们 54 化合物的 47 个（rapamycin 954 基因/999、wortmannin
427、TSA 356、nocodazole/MMS/HU 335±、geldanamycin 241…；cisplatin、
G418、SDS 等 7 个无 STITCH 记录 → 零向量）。

外部数据披露：STITCH v5（Szklarczyk et al. 2016, NAR 44:D380；
stitch.embl.de/stitch-db.org 直下，已探活校验 gzip 完整）；
化合物→PubChem CID 映射经 PUG-REST（2026-08-14，母核 SMILES 对齐 STITCH
flat-CID 口径；Tunicamycin 用 smiles.csv 自带 CID 11104835）。

变体（协议全对齐 wsN23，公平对照）：
- sig   ：STITCH 矩阵(54×3315, score/1000) → train 拟合标准化 → PCA64
- fuse5 ：STITCH 3315 ⊕ fuse3 四源 3584（RDKit+MLM+MolFormer+MTR）
          → 合并 6,899 维 → 同口径 PCA64（结构+活性全部信息一锅）
对照基线：fuse3-100ep 0.5397 / fuse3-e150 0.5412 / fuse3-e150-s32 0.5420。

合规：train-only 训练；val 仅模型选择；Y_te 零接触；新族新文件。

用法: python -m src.wsP_chemgenom              # sig + fuse5，100ep×16 种子
      python -m src.wsP_chemgenom --epochs 150 # e150 版
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import data as D
from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader, chemberta_embed
from .wsN23_fuse3 import molformer_embed
from .wsN29_fuse4 import mtr_embed

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsP"
SEEDS = list(range(16))
HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2


def _rdkit_fps(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(str(s).strip())
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        bits = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, bits)
        fps.append(bits)
    return np.stack(fps)


def build_stitch_matrix():
    """54 化合物 × 3315 基因 STITCH 打分矩阵（score/1000，无记录=0）。"""
    hits = json.load(open("extref/stitch/our_compounds_stitch.json"))
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    genes = sorted({g for v in hits.values() for g in v})
    gidx = {g: i for i, g in enumerate(genes)}
    X = np.zeros((len(names), len(genes)), dtype=np.float32)
    for i, c in enumerate(names):
        for g, s in hits.get(c, {}).items():
            X[i, gidx[g]] = s / 1000.0
    n_cov = sum(1 for c in names if c in hits)
    print(f"[wsP] STITCH 矩阵 {X.shape}（覆盖 {n_cov}/54 化合物）", flush=True)
    return X


def build_fuse_raw(smi):
    """fuse4 四源原始矩阵（RDKit 2048 + MLM 384 + MolFormer 768 + MTR 384）。"""
    smiles = smi["smiles"].tolist()
    return np.concatenate([_rdkit_fps(smiles), chemberta_embed(smiles),
                           molformer_embed(smiles), mtr_embed(smiles)], axis=1)


def pca64_table(E, names, fit_names, cache, tag):
    from sklearn.decomposition import PCA
    train_idx = np.array([i for i, c in enumerate(names) if c in fit_names])
    mu, sd = E[train_idx].mean(0), E[train_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (E - mu) / sd
    n_comp = min(64, len(train_idx) - 1, Z.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(Z[train_idx])
    P = pca.transform(Z)
    pm, ps = P[train_idx].mean(0), P[train_idx].std(0)
    ps[ps < 1e-8] = 1.0
    P = ((P - pm) / ps).astype(np.float32)
    if n_comp < 64:
        P = np.concatenate([P, np.zeros((len(P), 64 - n_comp), np.float32)],
                           axis=1)
    df = pd.DataFrame(P, columns=[f"pc{i}" for i in range(64)])
    df.insert(0, "compound", names)
    df.to_csv(cache, index=False)
    print(f"[wsP] {tag} 特征表 {df.shape} → {cache}", flush=True)
    return df


def run_variant(h, tag, df, epochs, test):
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot, hidden=HIDDEN,
                  p_drop=P_DROP)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    try:
        pv, pt = [], []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, epochs, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=test)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            if test:
                pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                      device="cuda"))
            print(f"[{tag}] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
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
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[{tag}-e{epochs}-s16 val] composite={res['composite']:.4f} "
          f"FC={fc}", flush=True)
    suffix = f"e{epochs}" if epochs != 100 else ""
    np.save(OUT / f"pred_trainval_{tag}{suffix}.npy", P)
    if test:
        PT = np.mean(pt, axis=0).astype(np.float32)
        bad = ~np.isfinite(PT)
        if bad.any():
            r, c = np.where(bad)
            PT[r, c] = np.take(h.stats.protein_mean, c)
        np.save(OUT / f"pred_test_{tag}{suffix}.npy", PT)
    return {"composite": res["composite"], "FC": fc,
            "per_split": res["per_split"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--variant", choices=["sig", "fuse5", "both"],
                    default="both")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    if args.test:
        fit_names = set(h.m_tr.loc[
            h.m_tr["perturbation_no_concentration"].notna(),
            "perturbation_no_concentration"].unique()) - D.CONTROLS - {D.QC}
        tag_s = "_full"
    else:
        fit_names = set(h.m_tr.loc[h.m_tr["chemical_role"] == "train",
                                   "perturbation_no_concentration"].unique())
        tag_s = ""

    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    summary = {}
    variants = ["sig", "fuse5"] if args.variant == "both" else [args.variant]
    for v in variants:
        cache = OUT / f"chem_features_{v}{tag_s}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
        else:
            E_sig = build_stitch_matrix()
            E = E_sig if v == "sig" else np.concatenate(
                [build_fuse_raw(smi), E_sig], axis=1)
            df = pca64_table(E, names, fit_names, cache, v)
        summary[v] = run_variant(h, f"wsP_{v}", df, args.epochs, args.test)
    suffix = f"e{args.epochs}" if args.epochs != 100 else ""
    (OUT / (f"scores_test{suffix}.json" if args.test
            else f"scores{suffix}.json")).write_text(json.dumps(
        {**summary,
         "baselines": {"fuse3_100ep": 0.5397, "fuse3_e150": 0.5412,
                       "fuse3_e150_s32": 0.5420}},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
