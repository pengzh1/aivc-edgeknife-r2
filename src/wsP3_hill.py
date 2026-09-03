"""wsP3: Hillenmeyer 2008 ⊕ Hoepfner 2014 双平台定量 fitness 签名族（收官战）。

与 wsP2（Hoepfner 单平台，17/54 覆盖，集成边际零）的差别：
1. Hillenmeyer 2008（YeastPhenome 再处理版，PMID 18420932，
   extref/hop/hillenmeyer_2008/：hom 4,770 株×419 条件 + het 5,985 株×726 条件）
   条件为经典探针屏，覆盖我们 25/54 化合物（含 Hoepfner 缺的 cisplatin/
   wortmannin/staurosporine/tunicamycin/brefeldin A/amphotericin B…）；
2. 双平台联合覆盖 **30/54**（重叠 11 个取两平台均值），特征是史上最强
   化学基因组学签名覆盖；
3. 符号口径统一为"z 分数负=敏感"（Hillenmeyer 原值 log2(control/treatment)
   取负后按条件 z 化；Hoepfner 用官方 z-score 列）。

变体（协议全对齐 wsP/wsP2，公平对照）：
- sig   ：双平台联合签名矩阵(54×~7k 基因) → train 拟合标准化 → PCA64
- fuse7 ：四源结构 3584 ⊕ 联合签名 → PCA64
对照基线：fuse3-100ep 0.5397 / fuse4 0.5401 / wsP_fuse5(STITCH) 0.5381 /
wsP2_fuse6(Hoepfner 单平台) 0.5398。边际判定线 +0.0003。

合规：train-only 训练；val 仅模型选择；Y_te 零接触；新族新文件。

用法: python -m src.wsP3_hill
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import data as D
from .wsP_chemgenom import build_fuse_raw, pca64_table, run_variant
from .wsP2_hopfeld import parse_hop_matrix, map_our_compounds, HOP_DIR

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsP3"
HILL_DIR = HOP_DIR / "hillenmeyer_2008"

ALIAS = {
    "Rapamycin": ["rapamycin"], "Fluconazole": ["fluconazole"],
    "Clotrimazole": ["clotrimazole"], "Tunicamycin": ["tunicamycin"],
    "MMS": ["methyl methanesulfonate", "mms"],
    "Hydroxyurea": ["hydroxyurea"], "Nocodazole": ["nocodazole"],
    "CHX": ["cycloheximide"], "G418": ["geneticin", "g418"],
    "Hygromycin B": ["hygromycin b", "hygromycin"],
    "Wortmannin": ["wortmannin"], "Geldanamycin": ["geldanamycin"],
    "Cisplatin": ["cisplatin"], "Amphotericin B": ["amphotericin b"],
    "Nystatin dihydrate": ["nystatin"], "Brefeldin A": ["brefeldin a"],
    "Staurosporine": ["staurosporine"],
    "Trichostatin A": ["trichostatin a"],
    "(S)-(+)-Camptothecin": ["camptothecin"],
    "Oligomycin": ["oligomycin"], "FCCP": ["fccp"],
    "Valinomycin": ["valinomycin"], "Nigericin": ["nigericin"],
    "Anisomycin": ["anisomycin"], "H2O2": ["hydrogen peroxide", "h2o2"],
    "NaCl": ["sodium chloride", "nacl"], "Sorbitol": ["sorbitol"],
    "Tamoxifen": ["tamoxifen"],
    "4-Hydroxytamoxifen": ["4-hydroxytamoxifen"],
    "Haloperidol": ["haloperidol"],
    "Trifluoperazine dihydrochloride": ["trifluoperazine"],
    "Artemisinin": ["artemisinin"], "Emodin": ["emodin"],
    "Parthenolide": ["parthenolide"], "Plumbagin": ["plumbagin"],
    "Harmine hydrochloride": ["harmine"],
    "Pentamidine isethionate": ["pentamidine"],
    "Sulfometuron methyl": ["sulfometuron methyl"], "EDTA": ["edta"],
    "SDS": ["sodium dodecyl sulfate", "sds"],
    "Doxycycline hyclate": ["doxycycline"],
    "Clomiphene citrate": ["clomiphene"],
    "Raloxifene hydrochloride": ["raloxifene"],
    "Desipramine hydrochloride": ["desipramine"],
    "Amiodarone hydrochloride": ["amiodarone"],
    "LY 294002 hydrochloride": ["ly294002"],
    "U-73122": ["u-73122", "u73122"],
    "Cyclopiazonic acid": ["cyclopiazonic acid"],
    "Abietic acid": ["abietic acid"],
    "Hoechst 33258": ["hoechst 33258", "hoechst"],
    "Dyclonine hydrochloride": ["dyclonine"], "Neomycin B": ["neomycin"],
    "1-10 Phenanthroline monohydrate": ["1,10-phenanthroline",
                                        "o-phenanthroline"],
    "(1R, 2S, 5R) - (-) - Menthol": ["menthol"],
}


def parse_hillenmeyer():
    """hom+het → 取负 → 条件内 z 化 → 化合物聚合 → compound × gene。"""
    cache = HILL_DIR / "hill_compound_gene.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["compounds"].tolist(), z["genes"], z["X"]
    t0 = time.time()
    mats = []
    genes_ref = None
    for fn in ["hom.ratio_result_nm.pub", "het.ratio_result_nm.pub"]:
        df = pd.read_csv(HILL_DIR / fn, sep="\t")
        genes = df.iloc[:, 0].astype(str).tolist()
        if genes_ref is None:
            genes_ref = genes
        assert genes == genes_ref or len(genes) > 0
        V = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        V = -V  # log2(control/treatment) → log2(treatment/control)：负=敏感
        conds = df.columns[1:]
        cname = [c.split(":")[1].strip().lower() if ":" in c else c
                 for c in conds]
        mats.append((np.array(genes), conds, cname, V))
        print(f"[wsP3] {fn}: {V.shape}（{time.time()-t0:.0f}s）", flush=True)
    # 基因全集（hom/het 株名可能不同序）
    all_genes = sorted(set().union(*[set(m[0]) for m in mats]))
    gidx = {g: i for i, g in enumerate(all_genes)}
    compounds = sorted({a for als in ALIAS.values() for a in als} |
                       set())  # 条件名是小写原名
    comp_hits = {}
    X = np.zeros((len(ALIAS), len(all_genes)), dtype=np.float64)
    cnt = np.zeros(len(ALIAS), dtype=np.int32)
    our_names = list(ALIAS)
    our_idx = {}
    for i, c in enumerate(our_names):
        for a in ALIAS[c]:
            our_idx.setdefault(a, i)
    for genes, conds, cname, V in mats:
        grow = np.array([gidx[g] for g in genes])
        for j, cn in enumerate(cname):
            i = our_idx.get(cn)
            if i is None:
                continue
            col = V[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 100:
                continue
            mu, sd = np.nanmean(col), np.nanstd(col)
            if sd < 1e-8:
                continue
            z = np.where(ok, (col - mu) / sd, 0.0)
            X[i, grow] += z
            cnt[i] += 1
            comp_hits[our_names[i]] = comp_hits.get(our_names[i], 0) + 1
    for i in range(len(our_names)):
        if cnt[i]:
            X[i] /= cnt[i]
    X = X.astype(np.float32)
    print(f"[wsP3] Hillenmeyer 覆盖 "
          f"{sum(1 for c in cnt if c > 0)}/{len(our_names)} "
          f"（{time.time()-t0:.0f}s）", flush=True)
    np.savez(cache, compounds=np.array(our_names),
             genes=np.array(all_genes), X=X)
    return our_names, np.array(all_genes), X


def build_combined():
    """Hoepfner ∪ Hillenmeyer 联合签名矩阵（双覆盖取均值）。"""
    cache = OUT / "combined_sig.npz"
    if cache.exists():
        z = np.load(cache)
        return z["E"], z["n_cov"].item()
    OUT.mkdir(parents=True, exist_ok=True)
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()

    # Hoepfner 部分
    cmb, genes_h, XC = parse_hop_matrix()
    mapping = map_our_compounds()
    cidx = {int(c): i for i, c in enumerate(cmb)}
    E_hop = {c: np.nan_to_num(XC[cidx[mapping[c]]], nan=0.0)
             for c in names if mapping.get(c) in cidx}

    # Hillenmeyer 部分
    hnames, genes_l, XL = parse_hillenmeyer()
    hidx = {c: i for i, c in enumerate(hnames)}
    E_hill = {c: XL[hidx[c]] for c in names
              if c in hidx and np.abs(XL[hidx[c]]).sum() > 0}

    genes_all = sorted(set(genes_h) | set(genes_l))
    ga = {g: i for i, g in enumerate(genes_all)}
    gh = np.array([ga[g] for g in genes_h])
    gl = np.array([ga[g] for g in genes_l])
    E = np.zeros((len(names), len(genes_all)), dtype=np.float32)
    n_cov = 0
    for i, c in enumerate(names):
        vecs = []
        if c in E_hop:
            v = np.zeros(len(genes_all), np.float32)
            v[gh] = E_hop[c]
            vecs.append(v)
        if c in E_hill:
            v = np.zeros(len(genes_all), np.float32)
            v[gl] = E_hill[c]
            vecs.append(v)
        if vecs:
            E[i] = np.mean(vecs, axis=0)
            n_cov += 1
    print(f"[wsP3] 联合矩阵 {E.shape}（覆盖 {n_cov}/54：Hoepfner "
          f"{len(E_hop)} + Hillenmeyer {len(E_hill)}）", flush=True)
    np.savez(cache, E=E, n_cov=n_cov)
    return E, n_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    E_sig, n_cov = build_combined()

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

    summary = {"coverage": n_cov}
    for v in ["sig", "fuse7"]:
        cache = OUT / f"chem_features_{v}{tag_s}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
        else:
            E = E_sig if v == "sig" else np.concatenate(
                [build_fuse_raw(smi), E_sig], axis=1)
            df = pca64_table(E, names, fit_names, cache, v)
        summary[v] = run_variant(h, f"wsP3_{v}", df, args.epochs, args.test)
    suffix = f"e{args.epochs}" if args.epochs != 100 else ""
    (OUT / (f"scores_test{suffix}.json" if args.test
            else f"scores{suffix}.json")).write_text(json.dumps(
        {**summary,
         "baselines": {"fuse3_100ep": 0.5397, "fuse4_100ep": 0.5401,
                       "wsP2_fuse6": 0.5398, "wsP_fuse5": 0.5381}},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
