"""wsP2: Hoepfner 2014 Novartis HOP 定量 fitness 签名族（化学基因组学第二战）。

与 wsP（STITCH 关联置信度，已证伪）的本质区别：本数据是**统一平台的定量实验
测定**（HIP/HOP 缺失库生长 z 分数），不是文献关联打分。Hillenmeyer 路线情报见
docs/外部数据下载清单_20260814.md B1+。

数据：Hoepfner et al. 2014（Microbiol Res 169:107，Dryad doi:10.5061/dryad.v5m8v，
2026-08-14 经 Anubis PoW 破解下载，extref/hop/，md5 校验见 dl.log）：
- HOP_scores.txt（505MB，5,846 实验列 × ~6k 基因，"Ad. scores for Exp.
  <CMB_ID>_<dose>_HOP_<板号>[ z-score]"双型列）
- Table_S1.xls（CMB ID ↔ 通用名/SMILES 注释）
特征：z-score 列按 CMB ID 聚合（重复/剂量取均值）→ 化合物 × 基因签名矩阵
（我们的 54 化合物经 Table_S1 名字命中 23 + SMILES 命中 7 → CMB ID；
未覆盖 = 零向量）→ wsA 协议 PCA64 → deep3 16 种子 val。
变体：sig（HOP 单用）/ fuse6（四源结构 3584 ⊕ HOP → PCA64）。
对照：fuse3-100ep 0.5397 / fuse4 0.5401 / wsP_fuse5 0.5381（STITCH 证伪版）。

合规：train-only 训练；val 仅模型选择；Y_te 零接触；新族新文件。

用法: python -m src.wsP2_hopfeld                # 解析+双变体 val（16 种子 100ep）
      python -m src.wsP2_hopfeld --parse-only   # 只解析矩阵与映射
      python -m src.wsP2_hopfeld --epochs 150
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
from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader
from .wsP_chemgenom import build_fuse_raw, pca64_table, run_variant

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsP2"
HOP_DIR = Path(__file__).resolve().parent.parent / "extref" / "hop"


# ---------------------------------------------------------------- 解析

def parse_hop_matrix():
    """HOP_scores.txt → CMB ID × 基因 z-score 矩阵（缓存 npz）。"""
    cache = HOP_DIR / "hop_cmb_gene.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["cmb"], z["genes"], z["X"]
    src = HOP_DIR / "HOP_scores.txt"
    print("[wsP2] 解析 HOP_scores.txt（505MB，耐心）…", flush=True)
    t0 = time.time()
    header = pd.read_csv(src, sep="\t", nrows=0)
    cols = header.columns.tolist()
    use, meta = [], []
    for i, c in enumerate(cols):
        m = re.match(r'Ad\. scores for Exp\. (\d+)_([\d.]+)_HOP_\d+[AB]?'
                     r'( z-score)?', c)
        if m and m.group(3):  # 仅 z-score 列
            use.append(i)
            meta.append(int(m.group(1)))
    print(f"[wsP2] z-score 实验列 {len(use)}（{time.time()-t0:.0f}s）",
          flush=True)
    df = pd.read_csv(src, sep="\t", usecols=[0] + use, dtype=str,
                     na_values=["", "NA", "nan"])
    genes = df.iloc[:, 0].astype(str).tolist()
    X = np.full((len(use), len(genes)), np.nan, dtype=np.float32)
    for j, c in enumerate(df.columns[1:]):
        X[j] = pd.to_numeric(df[c], errors="coerce").values
    # CMB 聚合（重复/剂量均值）
    meta = np.array(meta)
    cmbs = sorted(set(meta))
    XC = np.full((len(cmbs), len(genes)), np.nan, dtype=np.float32)
    for k, cb in enumerate(cmbs):
        rows = X[meta == cb]
        with np.errstate(invalid="ignore"):
            XC[k] = np.nanmean(rows, axis=0)
    cov = np.isfinite(XC).sum(1)
    print(f"[wsP2] CMB {len(cmbs)} | 基因 {len(genes)} | "
          f"中位覆盖 {np.median(cov):.0f} | {time.time()-t0:.0f}s", flush=True)
    np.savez(cache, cmb=np.array(cmbs), genes=np.array(genes), X=XC)
    return np.array(cmbs), np.array(genes), XC


def map_our_compounds():
    """54 化合物 → CMB ID（Table_S1 名字 + SMILES 双通道）。"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        try:
            m = Chem.MolFromSmiles(str(s).strip())
            return Chem.MolToSmiles(m) if m else None
        except Exception:
            return None

    x = pd.ExcelFile(HOP_DIR / "Table_S1.xls")
    ref = pd.read_excel(x, "Reference Substances known MoA")
    st = pd.read_excel(x, "All Structures")
    st["canon"] = st["SMILE string"].map(canon)
    name2cmb = {}
    for r in ref.itertuples():
        if isinstance(r._2, str) and r._2.strip():
            name2cmb[r._2.strip().lower()] = int(r._1)
    alias = {
        "CHX": "cycloheximide", "MMS": "methyl methanesulfonate",
        "H2O2": "hydrogen peroxide", "NaCl": "sodium chloride",
        "G418": "geneticin", "SDS": "sodium dodecyl sulfate",
        "(S)-(+)-Camptothecin": "camptothecin",
        "1-10 Phenanthroline monohydrate": "1,10-phenanthroline",
    }
    out = {}
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    for r in smi.itertuples():
        q = alias.get(r.compound, r.compound).lower()
        q = re.sub(r"\s*(hydrochloride|citrate|isethionate|dihydrochloride|"
                   r"dihydrate|monohydrate|hyclate)\s*$", "", q)
        if q in name2cmb:
            out[r.compound] = name2cmb[q]
            continue
        cs = canon(r.smiles)
        if cs:
            m = st[st["canon"] == cs]
            if len(m):
                out[r.compound] = int(m.iloc[0]["CMB ID"])
    return out


def build_hop_matrix():
    """54 化合物 × 基因 HOP 签名矩阵。"""
    cmb, genes, XC = parse_hop_matrix()
    mapping = map_our_compounds()
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    cidx = {int(c): i for i, c in enumerate(cmb)}
    E = np.zeros((len(names), len(genes)), dtype=np.float32)
    for i, c in enumerate(names):
        cb = mapping.get(c)
        if cb in cidx:
            row = XC[cidx[cb]]
            E[i] = np.nan_to_num(row, nan=0.0)
    n_cov = sum(1 for c in names if c in mapping and mapping[c] in cidx)
    print(f"[wsP2] HOP 矩阵 {E.shape}（覆盖 {n_cov}/54 化合物）", flush=True)
    json.dump(mapping, open(HOP_DIR / "our_compounds_cmb.json", "w"),
              ensure_ascii=False, indent=1)
    return E


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    E_sig = build_hop_matrix()
    if args.parse_only:
        return

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

    summary = {}
    for v in ["sig", "fuse6"]:
        cache = OUT / f"chem_features_{v}{tag_s}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
        else:
            E = E_sig if v == "sig" else np.concatenate(
                [build_fuse_raw(smi), E_sig], axis=1)
            df = pca64_table(E, names, fit_names, cache, v)
        summary[v] = run_variant(h, f"wsP2_{v}", df, args.epochs, args.test)
    suffix = f"e{args.epochs}" if args.epochs != 100 else ""
    (OUT / (f"scores_test{suffix}.json" if args.test
            else f"scores{suffix}.json")).write_text(json.dumps(
        {**summary,
         "baselines": {"fuse3_100ep": 0.5397, "fuse4_100ep": 0.5401,
                       "wsP_fuse5_stitch": 0.5381}},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
