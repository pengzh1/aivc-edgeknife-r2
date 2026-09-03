"""wsO：菌株基因组原始序列 k-mer 特征 → wsK 式菌株特征模型（S2 上限第八路检验）。

动机：wsK（1011 计划距离矩阵+35 表型，42 维）已证伪 strain 侧增益
（composite 0.5330 / strain FC 0.3102，S2 上限 7 路证据）。但 wsK 用的是
**二手统计量**（距离/表型矩阵），原始序列本身未直接入模。本模块用
k-mer(k=4,5,6) 频率从 5 株 de novo 组装 + S288C 参考直接构建菌株特征，
回答"原始序列组成是否携带距离矩阵之外的菌株身份信息"。

数据（outputs/wsK/genomes/fasta/，1011Assemblies.tar.gz 提取，官方 MD5
10cd6ced9cd9a1064ee35583d644c458 已校验）：
  BAH/BAI/CEK/CGD/CRD.fasta（de novo contigs，11.9-12.3 Mb）
  DHY210 = S288C_reference_genomic.fna.gz 代理（同 wsK 口径）

特征：canonical k-mer 频率（双链计数；k=4,5,6 → 136+512+2080=2728 维）
  + GC 含量/N 比例/基因组长度/contig 数 → 6 株间标准化 → PCA（6 株秩上限
  → 5 成分）→ 再标准化。拟合涉及全部 6 株（基因型外部数据，无标签泄漏，
  与 wsK 对 BAI/CRD 用真实特征行同口径）。UNK_MEAN = train 4 株均值。

模型：复用 wsK_strainfeat 全套（ProteoMLPStrain + wsD 配方 300ep Huber
  G2 增强），仅替换特征表。对照基线：wsK composite 0.5330 /
  strain FC 0.3102 / both FC 0.1897；路由 strain FC 基线 0.351。

注意：4 训练菌株样本瓶颈——val_strain_only(BAI) 的 FC 若仍 ≤0.31 平台，
即关闭并归档为复赛素材（复赛预案 §5.1：DNABERT-2/NT embedding）。

用法:
    python -m src.wsO_genomefeat --build-only          # 只建特征表（CPU）
    python -m src.wsO_genomefeat                       # 8 种子 val 对照
    python -m src.wsO_genomefeat --variant combined    # kmer5 + wsK42 拼接版
    python -m src.wsO_genomefeat --full                # 全量重训 + pred_test.npy
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .evaluate import Harness
from .wsK_strainfeat import (DEFAULT_CFG, load_strain_mat, predict_g3,
                             train_one, _finalize)
from .wsD_arch import seen_cats

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsO"
FASTA_DIR = Path(__file__).resolve().parent.parent / \
    "outputs" / "wsK" / "genomes" / "fasta"
STRAIN_FASTA = {
    "BAH": "BAH.fasta", "BAI": "BAI.fasta", "CEK": "CEK.fasta",
    "CGD": "CGD.fasta", "CRD": "CRD.fasta",
    "DHY210": "S288C_reference_genomic.fna.gz",  # S288C 代理
}
TRAIN_STRAINS = ["BAH", "CEK", "CGD", "DHY210"]
KS = (4, 5, 6)
N_PCA = 5  # 6 株中心化的秩上限

_B2I = np.full(256, -1, dtype=np.int8)
_B2I[ord("A")] = _B2I[ord("a")] = 0
_B2I[ord("C")] = _B2I[ord("c")] = 1
_B2I[ord("G")] = _B2I[ord("g")] = 2
_B2I[ord("T")] = _B2I[ord("t")] = 3


def _read_fasta(path: Path) -> list[str]:
    op = gzip.open if path.suffix == ".gz" else open
    seqs, cur = [], []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def _kmer_counts_one(seq: str, ks=KS) -> dict[int, np.ndarray]:
    """单条序列的 canonical k-mer 计数（正链+反向互补链各计一遍）。"""
    enc = _B2I[np.frombuffer(seq.encode("ascii", "latin1"), dtype=np.uint8)]
    valid = enc >= 0
    enc = np.concatenate([enc, (3 - enc[::-1])])  # 拼反向互补
    valid = np.concatenate([valid, valid[::-1]])  # 掩码同步镜像（N 补码≠合法碱基）
    out = {}
    for k in ks:
        n = 4 ** k
        idx = np.zeros(len(enc) - k + 1, dtype=np.int64)
        ok = np.ones(len(idx), dtype=bool)
        for j in range(k):
            idx += enc[j:j + len(idx)].astype(np.int64) << (2 * (k - 1 - j))
            ok &= valid[j:j + len(idx)]
        out[k] = np.bincount(idx[ok], minlength=n)
    return out


def build_kmer_table() -> pd.DataFrame:
    """6 株 k-mer 频率 + 基础统计 → 株间标准化 → PCA5 → 再标准化。"""
    from sklearn.decomposition import PCA
    rows, meta = {}, {}
    for strain, fn in STRAIN_FASTA.items():
        t0 = time.time()
        seqs = _read_fasta(FASTA_DIR / fn)
        tot = {k: np.zeros(4 ** k, dtype=np.int64) for k in KS}
        n_bp = n_n = 0
        gc = 0
        for s in seqs:
            for k, c in _kmer_counts_one(s, KS).items():
                tot[k] += c
            n_bp += len(s)
            su = s.upper()
            gc += su.count("G") + su.count("C")
            n_n += len(su) - su.count("A") - su.count("C") - \
                su.count("G") - su.count("T")
        feats = np.concatenate(
            [tot[k] / max(tot[k].sum(), 1) for k in KS]).astype(np.float64)
        meta[strain] = [gc / max(n_bp - n_n, 1), n_n / n_bp, n_bp, len(seqs)]
        rows[strain] = feats
        print(f"[kmer] {strain}: {n_bp:,} bp {len(seqs)} contigs "
              f"({time.time()-t0:.0f}s)", flush=True)
    names = list(rows)
    F = np.stack([rows[n] for n in names])            # (6, 2728)
    M = np.array([meta[n] for n in names])            # (6, 4)
    X = np.concatenate([F, M], axis=1)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    pca = PCA(n_components=N_PCA, svd_solver="full", random_state=0)
    P = pca.fit_transform(Z)
    pm, ps = P.mean(0), P.std(0)
    ps[ps < 1e-12] = 1.0
    P = (P - pm) / ps
    print(f"[kmer] PCA 解释率: "
          f"{np.round(pca.explained_variance_ratio_, 3).tolist()}")
    df = pd.DataFrame(P, index=names,
                      columns=[f"kmer_pc{i}" for i in range(N_PCA)])
    df.loc["UNK_MEAN"] = df.loc[TRAIN_STRAINS].mean(0)
    return df


def build_combined_table(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """kmer PCA5 ⊕ wsK 42 维（距离/表型）→ 47 维联合表（再分别标准化）。"""
    wsk = pd.read_csv("outputs/wsK/strain_features.csv", index_col=0)
    unk = wsk.loc["UNK_MEAN"]
    wsk = wsk.drop(index="UNK_MEAN")
    km = kmer_df.drop(index="UNK_MEAN")
    Z = (wsk - wsk.mean(0)) / wsk.std(0).replace(0, 1.0)
    comb = pd.concat([km, Z.loc[km.index]], axis=1)
    comb.loc["UNK_MEAN"] = comb.loc[TRAIN_STRAINS].mean(0)
    return comb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--variant", choices=["kmer", "combined"], default="kmer")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache = OUT_DIR / f"strain_features_{args.variant}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
    else:
        km = build_kmer_table()
        km.to_csv(OUT_DIR / "strain_features_kmer.csv")
        print(f"[saved] {OUT_DIR/'strain_features_kmer.csv'} {km.shape}")
        df = km if args.variant == "kmer" else build_combined_table(km)
        if args.variant != "kmer":
            df.to_csv(cache)
            print(f"[saved] {cache} {df.shape}")
    print(f"[wsO] variant={args.variant} 特征表 {df.shape}")
    if args.build_only:
        return

    cfg = dict(DEFAULT_CFG, epochs=args.epochs)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = Harness()
    strain_mat, row_of = load_strain_mat(None, cache)

    if args.full:
        rows = np.arange(len(h.m_tr))
        stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)
        seen = seen_cats(h, rows)
        preds = []
        for s in seeds:
            t0 = time.time()
            model, enc, mean, std = train_one(
                h, rows, cfg, s, strain_mat, row_of, device=device,
                stats=stats, log_every=999)
            print(f"[wsO full seed {s}] {time.time()-t0:.0f}s", flush=True)
            preds.append(predict_g3(model, enc, mean, std, h.m_te, seen,
                                    strain_mat, row_of, device=device))
        pred = _finalize(np.mean(preds, axis=0), stats.protein_mean)
        np.save(OUT_DIR / f"pred_test_{args.variant}.npy", pred)
        print(f"[saved] {OUT_DIR/f'pred_test_{args.variant}.npy'} {pred.shape}")
        return

    seen = seen_cats(h, h.tr_rows)
    preds = []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std = train_one(
            h, h.tr_rows, cfg, s, strain_mat, row_of, device=device,
            log_every=999)
        print(f"[wsO seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_tr, seen,
                                strain_mat, row_of, device=device))
    pred = _finalize(np.mean(preds, axis=0), h.stats.protein_mean)
    np.save(OUT_DIR / f"pred_trainval_{args.variant}.npy", pred)
    res = h.score_val(pred, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsO {args.variant}] composite={res['composite']:.4f} FC={fc}")
    print("[wsO] 基线: wsK 0.5330 (strain FC 0.3102) | "
          "判定: strain FC 无显著提升即关闭归档")
    scores_path = OUT_DIR / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    scores[f"wsO_{args.variant}"] = {
        "composite": res["composite"], "FC": fc,
        "per_split": res["per_split"],
        "baselines": {"wsK_composite": 0.5330, "wsK_strain_FC": 0.3102}}
    scores_path.write_text(json.dumps(scores, ensure_ascii=False,
                                      indent=1, default=float))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
