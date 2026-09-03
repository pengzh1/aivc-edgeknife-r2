"""wsN32: 双块 128 维融合（绕开 PCA64 联合瓶颈的集成方式对照）。

fuse5/6/7（结构⊕签名联合 PCA64）全部零边际——一个未排除的解释是联合 PCA
把签名信息在 6899+ 结构维度里稀释掉了。本变体改**分块 PCA 再拼接**：
结构四源 3584 → PCA64（块 A）‖ HOP⊕Hill 联合签名 12666 → PCA64（块 B）
→ 128 维输入；chem_proj 改 Linear(128→48)。wsN6 fuse 同构（RDKit64⊕BERT64）。

对照基线：fuse3-100ep 0.5397 / fuse4 0.5401 / fuse5 0.5381 / fuse6 0.5398 /
fuse7 0.5385（全部联合 PCA 变体）。若双块 ≥0.5410 说明瓶颈在 PCA 合并方式，
签名信息值得保留并考虑入列。

合规：train-only 训练；val 仅模型选择；Y_te 零接触；新族新文件。

用法: python -m src.wsN32_dualfuse
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
from .wsP_chemgenom import build_fuse_raw
from .wsP3_hill import build_combined

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN32"
SEEDS = list(range(16))
HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2
N_BLOCK = 64


def pca_block(E, fit_idx, n=N_BLOCK, seed=0, clip=8.0):
    """分块标准化→PCA→再标准化；clip 防稀疏块尾部 PC 方差近零导致的
    val/test 行数值爆炸（wsP2/wsP3 sig 表曾因此出现 1e6 级异常值）。"""
    from sklearn.decomposition import PCA
    mu, sd = E[fit_idx].mean(0), E[fit_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (E - mu) / sd
    n_comp = min(n, len(fit_idx) - 1, Z.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=seed)
    pca.fit(Z[fit_idx])
    P = pca.transform(Z)
    pm, ps = P[fit_idx].mean(0), P[fit_idx].std(0)
    ps[ps < 1e-8] = 1.0
    P = ((P - pm) / ps).astype(np.float32)
    P = np.clip(P, -clip, clip)
    if n_comp < n:
        P = np.concatenate([P, np.zeros((len(P), n - n_comp), np.float32)],
                           axis=1)
    return P


def table_to_loader_128(df):
    pc_cols = [f"pc{i}" for i in range(2 * N_BLOCK)]
    indexed = df.set_index("compound")
    feat = {name: indexed.loc[name, pc_cols].to_numpy(dtype=np.float32)
            for name in indexed.index}
    mean_vec = np.stack(list(feat.values())).mean(0)

    def loader(h_, out_dir, full=False):
        return feat, mean_vec
    return loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    fit_names = set(h.m_tr.loc[h.m_tr["chemical_role"] == "train",
                               "perturbation_no_concentration"].unique())

    cache = OUT / "chem_features_dual128.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        smi = pd.read_csv("outputs/wsA/smiles.csv")
        names = smi["compound"].tolist()
        fit_idx = np.array([i for i, c in enumerate(names)
                            if c in fit_names])
        print("[dual] 结构块…", flush=True)
        E_struct = build_fuse_raw(smi)           # 3584
        print("[dual] 签名块…", flush=True)
        E_sig, n_cov = build_combined()           # 12666（wsP3 缓存）
        PA = pca_block(E_struct, fit_idx)
        PB = pca_block(E_sig, fit_idx)
        P = np.concatenate([PA, PB], axis=1)      # (54, 128)
        df = pd.DataFrame(P, columns=[f"pc{i}" for i in range(2 * N_BLOCK)])
        df.insert(0, "compound", names)
        df.to_csv(cache, index=False)
        print(f"[dual] 特征表 {df.shape} → {cache}", flush=True)

    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        import torch.nn as nn
        orig_init(self, n_cats, chem_mat, n_prot, hidden=HIDDEN,
                  p_drop=P_DROP)
        self.chem_proj = nn.Sequential(
            nn.Linear(chem_mat.shape[1], WSA.CHEM_HID), nn.GELU())
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader_128(df)
    try:
        pv = []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, args.epochs, seed=sd, emb_drop=0.25,
                chem_drop=0.25, lr=1e-3, bs=256, device="cuda",
                log_every=999)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            print(f"[dual128] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
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
    print(f"[dual128-e{args.epochs}-s16 val] composite={res['composite']:.4f} "
          f"FC={fc}")
    np.save(OUT / "pred_trainval.npy", P)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc,
         "per_split": res["per_split"],
         "baselines": {"fuse3": 0.5397, "fuse4": 0.5401, "fuse6": 0.5398,
                       "fuse7": 0.5385}},
        default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
