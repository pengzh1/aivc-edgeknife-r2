"""wsP4: clip 修复后重测 Hoepfner HOP 签名单用的真实基线（复赛记账用）。

wsP2/wsP3 的 sig 表存在 PCA 尾部 PC train 方差近零导致的 val 行数值爆炸
（极值 3.4e6/1.1e6）——0.4840/0.4570 是被毒化输入压低后的读数。本脚本
用 winsorize(±8) 重建 Hoepfner 单平台签名表，重跑同协议（deep3 16 种子
100ep），给复赛留**正确的单用基线**。不改变任何初赛判定（fuse6/fuse7 联合
表未受影响，结论已生效）。

用法: python -m src.wsP4_sigclip
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsA_chemfeat as WSA
from .wsP_chemgenom import run_variant
from .wsP2_hopfeld import build_hop_matrix

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsP4"


def pca64_clip(E, names, fit_names, cache):
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
    P = np.clip(P, -8.0, 8.0)  # winsorize：防尾部 PC 爆炸
    if n_comp < 64:
        P = np.concatenate([P, np.zeros((len(P), 64 - n_comp), np.float32)],
                           axis=1)
    df = pd.DataFrame(P, columns=[f"pc{i}" for i in range(64)])
    df.insert(0, "compound", names)
    df.to_csv(cache, index=False)
    print(f"[wsP4] clip 版特征表 {df.shape} 极值={np.abs(P).max():.1f}",
          flush=True)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    fit_names = set(h.m_tr.loc[h.m_tr["chemical_role"] == "train",
                               "perturbation_no_concentration"].unique())
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    cache = OUT / "chem_features_sig_clip.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        E = build_hop_matrix()
        df = pca64_clip(E, names, fit_names, cache)
    summary = run_variant(h, "wsP4_sigclip", df, 100, False)
    summary["note"] = ("Hoepfner 单用 clip 修复版；对照毒化版 0.4840/"
                       "chem 0.4058；fuse3 0.5397")
    (OUT / "scores.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
