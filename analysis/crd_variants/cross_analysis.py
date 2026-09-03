"""交叉：CGD/CRD 序列差异基因 × 响应谱（CGD 响应活跃 + 菌株背景敏感）。

集合定义（蛋白组覆盖的基因为总体）:
  V  = 高置信差异位点涉及基因（hom_diff==1, 双株 DP>=4 且 GQ>=20;
       效应等级: missense/frameshift/stop/inframe/promoter/terminator/synonymous 均收,
       主表给全量, 富集检验用 effect_rank>=2 即功能/调控相关）
  R1 = CGD 响应活跃: cgd_mean_abs_delta 处于总体前 10%（可观测蛋白中）
  R2 = 菌株背景敏感: sens_absmedian_diff 前 10%（要求 sens_n_pairs>=5）
富集: scipy 超几何; 量小则叙述性说明。

输出:
  analysis/crd_variants/cross_candidates.csv  — V 中全部基因 × 响应指标（排序）
  analysis/crd_variants/cross_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

OUT = Path("analysis/crd_variants")

EFFECT_RANK = {"frameshift": 5, "stop_gained": 5, "stop_lost": 4, "missense": 3,
               "inframe_indel": 3, "promoter": 2, "terminator": 1, "synonymous": 1,
               "heterozygous_cds": 2, "nc_or_intron": 1, "intergenic": 0,
               "heterozygous_other": 0, "cds_undef": 1, "cds_other": 1,
               "gene_nonCDS": 1}


def main():
    ann = pd.read_csv(OUT / "cgd_crd_diff_sites_annotated.tsv", sep="\t")
    prof = pd.read_csv(OUT / "cgd_response_profile.csv")
    g2l = json.load(open("extref/hop/gene2locus.json"))
    l2g = {v: k for k, v in g2l.items()}

    # ---- 高置信过滤 ----
    hq = ann[(ann.hom_diff == 1) & (ann.dp_cgd >= 4) & (ann.gq_cgd >= 20)
             & (ann.dp_crd >= 4) & (ann.gq_crd >= 20)].copy()
    print(f"annotated={len(ann)} high_conf_hom={len(hq)}")
    ann["effect_rank"] = ann.effect.map(EFFECT_RANK).fillna(0)
    hq["effect_rank"] = hq.effect.map(EFFECT_RANK).fillna(0)

    # 每基因汇总（全量 + 高置信两套）
    def gene_table(df, tag):
        g = df[df.locus != ""].groupby("locus").agg(
            n_sites=("pos", "size"),
            worst_rank=("effect_rank", "max"),
            worst_effect=("effect", lambda s: s.iloc[s.map(EFFECT_RANK).fillna(0).argmax()]),
            n_missense=("effect", lambda s: (s == "missense").sum()),
            n_synonymous=("effect", lambda s: (s == "synonymous").sum()),
            n_frameshift=("effect", lambda s: (s == "frameshift").sum()),
            n_stop=("effect", lambda s: s.isin(["stop_gained", "stop_lost"]).sum()),
            n_promoter=("effect", lambda s: (s == "promoter").sum()),
            n_terminator=("effect", lambda s: (s == "terminator").sum()),
            aa_changes=("aa_change", lambda s: ";".join(sorted({x for x in s if isinstance(x, str) and x}))),
        ).reset_index()
        g.columns = ["locus"] + [f"{tag}_{c}" for c in g.columns[1:]]
        return g

    g_all = gene_table(ann, "all")
    g_hq = gene_table(hq, "hq")
    gt = g_all.merge(g_hq, on="locus", how="left")

    # 映射到蛋白组蛋白名
    gt["protein"] = gt.locus.map(l2g).fillna(gt.all_worst_effect.map(lambda x: ""))
    # GFF name 可能与 proteome 名一致；优先 gene2locus 反查，否则用 GFF name
    gff_name = ann[ann.locus != ""].drop_duplicates("locus").set_index("locus").gene_name
    gt["gff_name"] = gt.locus.map(gff_name)
    gt["protein"] = np.where(gt.protein.isin(["", None]), gt.gff_name, gt.protein)

    # 合并响应谱（prof 以 protein 名为主键）
    prof = prof.rename(columns={c: c for c in prof.columns})
    merged = gt.merge(prof, on="protein", how="left")
    merged["in_proteome"] = merged.cgd_n_obs.notna()

    # ---- 响应集定义 ----
    obs = prof[prof.cgd_n_obs.notna() & (prof.cgd_n_obs > 0)]
    N = len(obs)
    thr_r1 = obs.cgd_mean_abs_delta.quantile(0.9)
    obs_s = obs[obs.sens_n_pairs >= 5]
    thr_r2 = obs_s.sens_absmedian_diff.quantile(0.9)
    R1 = set(obs.protein[obs.cgd_mean_abs_delta >= thr_r1])
    R2 = set(obs_s.protein[obs_s.sens_absmedian_diff >= thr_r2])
    print(f"proteome genes N={N} |R1|={len(R1)} thr={thr_r1:.3f} |R2|={len(R2)} thr={thr_r2:.3f}")

    merged["in_R1"] = merged.protein.isin(R1)
    merged["in_R2"] = merged.protein.isin(R2)

    # ---- 富集检验: V_func(高置信 rank>=2 且在蛋白组) vs R1/R2 ----
    Vp = merged[(merged.in_proteome) & (merged.hq_worst_rank.fillna(0) >= 2)]
    V_all_p = merged[merged.in_proteome]
    res = {}
    for name, Vset in (("V_func_hq", set(Vp.protein)), ("V_any", set(V_all_p.protein))):
        for rname, R in (("R1_cgd_active", R1), ("R2_strain_sensitive", R2)):
            K = len(R)
            n = len(Vset)
            k = len(Vset & R)
            p = hypergeom.sf(k - 1, N, K, n) if n > 0 and k > 0 else 1.0
            res[f"{name}_x_{rname}"] = {"N": N, "K": K, "n_V": n, "overlap": k,
                                        "expected": n * K / N, "p_hypergeom": p}
    print(json.dumps(res, indent=2))

    # ---- 排序输出 ----
    merged["score"] = (merged.hq_worst_rank.fillna(0)
                       + merged.in_R1.astype(int) * 2
                       + merged.in_R2.astype(int) * 2
                       + np.log1p(merged.hq_n_sites.fillna(0)))
    merged = merged.sort_values(["score", "hq_worst_rank", "sens_absmedian_diff"],
                                ascending=False)
    merged.to_csv(OUT / "cross_candidates.tsv", index=False, sep="\t")

    top = merged[(merged.in_R1 | merged.in_R2) & (merged.hq_worst_rank.fillna(0) >= 2)]
    summ = {
        "n_genes_with_any_diff": int(len(gt)),
        "n_genes_with_hq_func_diff": int((merged.hq_worst_rank.fillna(0) >= 2).sum()),
        "n_V_in_proteome": int(len(Vp)),
        "thr_R1_mean_abs_delta": float(thr_r1),
        "thr_R2_sens_absmedian": float(thr_r2),
        "enrichment": res,
        "n_top_candidates": int(len(top)),
    }
    with open(OUT / "cross_summary.json", "w") as fp:
        json.dump(summ, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print("\nTop candidates:")
    cols = ["protein", "gff_name", "locus", "hq_worst_effect", "hq_n_sites",
            "all_n_sites", "hq_aa_changes", "hq_n_promoter",
            "cgd_mean_abs_delta", "cgd_max_abs_delta", "sens_absmedian_diff",
            "sens_n_pairs", "in_R1", "in_R2"]
    print(top[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
