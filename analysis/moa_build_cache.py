# -*- coding: utf-8 -*-
"""MoA 叙事分析 —— 重解析/对齐步骤固化脚本。

产出 analysis/cache_moa/ 下的轻量缓存，供 moa_narrative.ipynb 直接读取：
  - common_axis.npz          共同基因轴（我们蛋白 <-> Hoepfner HOP 基因）
  - signatures_mapped.npz    14 个 CMB 映射且在 train_val 出现的化合物：
                             真值签名 / 预测签名 / HOP z 签名（共同轴上）
  - signatures_all.npz       全部 train_val 处理化合物的预测/真值签名（共同轴）
  - moa_annotations.csv      17 个 CMB 映射化合物的 MoA 文本与粗分类
  - consistency_table.csv    逐化合物三项相关（脚本内先算一份，notebook 复核）

合规：只用 train_val（Y_tr）+ 我们的 trainval 预测；Y_test 零接触；不改任何既有文件。
运行：.venv/Scripts/python.exe analysis/moa_build_cache.py   （从项目根目录）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仅用于定位，不修改 src

META = ROOT / "input" / "WAYB_WAYC_metadata_train_val(1).csv"
PROT_CACHE = ROOT / "cache" / "proteome_log2.npz"
PRED_CANDIDATES = [
    ROOT / "outputs" / "wsT0" / "cache" / "routed_r07_band13_trainval.npy",
    ROOT / "outputs" / "wsT0" / "cache" / "routed_r07_trainval.npy",
    ROOT / "outputs" / "wsN11" / "pred_trainval_routed_r05.npy",
]
HOP_NPZ = ROOT / "extref" / "hop" / "hop_cmb_gene.npz"
GENE2LOCUS = ROOT / "extref" / "hop" / "gene2locus.json"
CMB_MAP = ROOT / "extref" / "hop" / "our_compounds_cmb.json"
TABLE_S1 = ROOT / "extref" / "hop" / "Table_S1.xls"
OUTDIR = ROOT / "analysis" / "cache_moa"
OUTDIR.mkdir(parents=True, exist_ok=True)

CONTROLS = {"DMSO", "Water"}
QC = "Quality Control"
MATCH_KEYS = ["data_source", "Strains", "Medium", "Temperature",
              "pert_time", "instrument", "Yeast_cell_plate"]

# 粗粒度 MoA 分类（用于聚类对照；test-only 三个一并标注，便于报告中说明）
MOA_CLASS = {
    "Anisomycin": "Protein synthesis",
    "CHX": "Protein synthesis",
    "Rapamycin": "TOR signaling",
    "Clotrimazole": "Ergosterol/lipid",
    "Fluconazole": "Ergosterol/lipid",
    "Geldanamycin": "Hsp90 chaperone",
    "Nocodazole": "Microtubule",
    "Hydroxyurea": "DNA synthesis/damage",
    "Hoechst 33258": "DNA binding",
    "MMS": "DNA alkylation",
    "(S)-(+)-Camptothecin": "Topoisomerase I",
    "Trichostatin A": "HDAC/chromatin",
    "Nigericin": "Ionophore (K+)",
    "Valinomycin": "Ionophore (K+)",
    "NaCl": "Osmotic/salt stress",
    "EDTA": "Metal chelation",
    "SDS": "Membrane/surfactant",
}


def log(msg):
    print(f"[moa_build] {msg}", flush=True)


def pick_prediction():
    for p in PRED_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"no prediction file found among {PRED_CANDIDATES}")


def build_common_axis(proteins, hop_genes, g2l):
    """把我们蛋白轴对齐到 HOP 基因轴。返回共同基因的信息。"""
    hopset = set(hop_genes)
    hopidx = {g: i for i, g in enumerate(hop_genes)}
    kept_prot, kept_hop, loci, std_names = [], [], [], []
    seen = set()
    for i, p in enumerate(proteins):
        p = str(p)
        loc = None
        if p in g2l and g2l[p] in hopset:
            loc = g2l[p]
        elif p in hopset:
            loc = p
        if loc and loc not in seen:
            seen.add(loc)
            kept_prot.append(i)
            kept_hop.append(hopidx[loc])
            loci.append(loc)
            std_names.append(p)
    return (np.array(kept_prot, dtype=np.int64),
            np.array(kept_hop, dtype=np.int64),
            np.array(loci), np.array(std_names))


def main():
    log("loading metadata / proteome / prediction ...")
    m = pd.read_csv(META)
    z = np.load(PROT_CACHE)
    Y_tr = z["Y_tr"].astype(np.float32)          # (8958, 5243) log2 绝对强度
    proteins = z["proteins"].astype(str)
    pred_path = pick_prediction()
    log(f"prediction file: {pred_path.relative_to(ROOT)}")
    PRED = np.load(pred_path).astype(np.float32)
    assert PRED.shape == Y_tr.shape, f"PRED {PRED.shape} != Y_tr {Y_tr.shape}"
    assert len(m) == Y_tr.shape[0]

    # ---- 共同基因轴 ----
    hop = np.load(HOP_NPZ)
    hop_genes = hop["genes"].astype(str)
    hop_cmb = hop["cmb"].astype(np.int64)
    hop_X = hop["X"].astype(np.float32)          # (1818, 6681) z
    g2l = json.load(open(GENE2LOCUS))
    prot_idx, hop_idx, loci, std_names = build_common_axis(proteins, hop_genes, g2l)
    n_common = len(loci)
    log(f"common gene axis: {n_common} genes "
        f"(of {len(proteins)} proteins, {len(hop_genes)} HOP genes)")
    np.savez_compressed(OUTDIR / "common_axis.npz",
                        prot_idx=prot_idx, hop_idx=hop_idx,
                        locus=loci, std_name=std_names)

    # ---- 对照匹配（仅 train_val 内）----
    ctrl_mask = m["perturbation_no_concentration"].isin(CONTROLS).to_numpy()
    ctrl_df = m[ctrl_mask]
    # 组 -> 对照行号列表
    groups = ctrl_df.groupby(MATCH_KEYS)["sample_ID"].apply(list).to_dict()
    # 组 -> 对照 log2 均值向量（用行号索引回 Y_tr）
    sid2row = {s: i for i, s in enumerate(m["sample_ID"])}
    grp_mean = {}
    for key, sids in groups.items():
        rows = [sid2row[s] for s in sids]
        grp_mean[key] = np.nanmean(Y_tr[rows], axis=0)  # (5243,)

    is_treat = ~m["perturbation_no_concentration"].isin(CONTROLS | {QC})
    treat_idx = np.where(is_treat.to_numpy())[0]
    treat_comp = m["perturbation_no_concentration"].to_numpy()[treat_idx]
    # 每个处理样本的对照均值矩阵
    keys = [tuple(m.iloc[i][k] for k in MATCH_KEYS) for i in treat_idx]
    miss = sum(1 for k in keys if k not in grp_mean)
    log(f"treated samples: {len(treat_idx)}, unmatched controls: {miss}")
    CTRL = np.stack([grp_mean[k] for k in keys]).astype(np.float32)  # (n_treat,5243)

    # 真值/预测 delta（共同轴蛋白列）
    Ysub = Y_tr[treat_idx][:, prot_idx] - CTRL[:, prot_idx]      # 真值 Δ
    Psub = PRED[treat_idx][:, prot_idx] - CTRL[:, prot_idx]      # 预测 Δ̂

    # ---- 14 个映射化合物签名 ----
    cmb_map = json.load(open(CMB_MAP))            # name -> cmb_id
    present = [c for c in cmb_map if c in set(m["perturbation_no_concentration"])]
    present = sorted(present)
    log(f"CMB-mapped compounds present in train_val: {len(present)} / {len(cmb_map)}")

    # HOP cmb_id -> 行号；个别化合物（Clotrimazole 1101）无 HOP 行 -> NaN
    hop_row_of = {int(cid): i for i, cid in enumerate(hop_cmb)}
    sig_true, sig_pred, sig_hop = [], [], []
    n_list, cid_list, has_hop = [], [], []
    for c in present:
        rows = np.where(treat_comp == c)[0]
        n_list.append(len(rows))
        cid = int(cmb_map[c])
        cid_list.append(cid)
        sig_true.append(np.nanmedian(Ysub[rows], axis=0))
        sig_pred.append(np.nanmedian(Psub[rows], axis=0))
        if cid in hop_row_of:
            sig_hop.append(hop_X[hop_row_of[cid]][hop_idx])  # 共同轴上的 HOP z
            has_hop.append(True)
        else:
            sig_hop.append(np.full(n_common, np.nan, dtype=np.float32))
            has_hop.append(False)
            log(f"  WARNING: {c} (CMB {cid}) has no HOP row -> sig_hop=NaN")
    sig_true = np.stack(sig_true).astype(np.float32)
    sig_pred = np.stack(sig_pred).astype(np.float32)
    sig_hop = np.stack(sig_hop).astype(np.float32)

    moa_txt = [MOA_CLASS.get(c, "") for c in present]
    np.savez_compressed(OUTDIR / "signatures_mapped.npz",
                        comp_names=np.array(present),
                        cmb_ids=np.array(cid_list, dtype=np.int64),
                        n_samples=np.array(n_list, dtype=np.int64),
                        has_hop=np.array(has_hop, dtype=bool),
                        moa_class=np.array(moa_txt),
                        locus=loci, std_name=std_names,
                        sig_true=sig_true, sig_pred=sig_pred, sig_hop=sig_hop)
    log(f"saved signatures_mapped.npz  sig shape {sig_true.shape}")

    # ---- 全部 train_val 处理化合物签名（共同轴）----
    all_comp = sorted(set(treat_comp))
    all_true, all_pred, all_n = [], [], []
    for c in all_comp:
        rows = np.where(treat_comp == c)[0]
        all_n.append(len(rows))
        all_true.append(np.nanmedian(Ysub[rows], axis=0))
        all_pred.append(np.nanmedian(Psub[rows], axis=0))
    all_true = np.stack(all_true).astype(np.float32)
    all_pred = np.stack(all_pred).astype(np.float32)
    is_mapped = np.array([c in cmb_map for c in all_comp])
    np.savez_compressed(OUTDIR / "signatures_all.npz",
                        comp_names=np.array(all_comp),
                        n_samples=np.array(all_n, dtype=np.int64),
                        is_mapped=is_mapped,
                        moa_class=np.array([MOA_CLASS.get(c, "") for c in all_comp]),
                        locus=loci, std_name=std_names,
                        sig_true=all_true, sig_pred=all_pred)
    log(f"saved signatures_all.npz  {len(all_comp)} compounds")

    # ---- MoA 注释表（Table_S1）----
    known = pd.read_excel(TABLE_S1, sheet_name="Reference Substances known MoA", header=0)
    known.columns = ["CMB_ID", "Common", "IC30", "Wiki", "PMID", "MoA"]
    novel = pd.read_excel(TABLE_S1, sheet_name="Substances novel MoA", header=0)
    novel.columns = ["CMB_ID", "Common", "IC30", "MoA", "Structures"]
    kmap = dict(zip(known["CMB_ID"], known["MoA"]))
    nmap = dict(zip(novel["CMB_ID"], novel["MoA"]))
    rows = []
    for name, cid in sorted(cmb_map.items(), key=lambda x: x[1]):
        src = "known" if cid in kmap else ("novel" if cid in nmap else "none")
        rows.append({
            "compound": name, "cmb_id": int(cid),
            "moa_class": MOA_CLASS.get(name, ""),
            "moa_text": str(kmap.get(cid, nmap.get(cid, ""))),
            "moa_src": src,
            "present_in_trainval": bool(name in set(m["perturbation_no_concentration"])),
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "moa_annotations.csv", index=False)
    log("saved moa_annotations.csv")

    # ---- 逐化合物一致性表（notebook 会复核重算）----
    from scipy.stats import spearmanr, pearsonr
    recs = []
    for j, c in enumerate(present):
        a, b, cc = sig_true[j], sig_pred[j], sig_hop[j]
        def safe(x, y):
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 20:
                return np.nan, np.nan, int(ok.sum())
            return (spearmanr(x[ok], y[ok]).statistic,
                    pearsonr(x[ok], y[ok]).statistic, int(ok.sum()))
        sab, pab, nab = safe(a, b)
        sac, pac, nac = safe(a, cc)
        sbc, pbc, nbc = safe(b, cc)
        recs.append({"compound": c, "cmb_id": cid_list[j], "n_samples": n_list[j],
                     "moa_class": moa_txt[j],
                     "sp_true_pred": sab, "sp_true_hop": sac, "sp_pred_hop": sbc,
                     "pe_true_pred": pab, "pe_true_hop": pac, "pe_pred_hop": pbc,
                     "n_genes_pred_hop": nbc})
    pd.DataFrame(recs).to_csv(OUTDIR / "consistency_table.csv", index=False)
    log("saved consistency_table.csv")
    log("DONE.")


if __name__ == "__main__":
    main()
