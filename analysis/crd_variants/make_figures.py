"""绘图: (1) CGD/CRD 35 条件表型对比; (2) 差异位点染色体分布与注释分类。
用父级 venv (matplotlib): C:/Users/31564/ai-workspace/.venv/Scripts/python.exe
输出 analysis/crd_variants/fig_phenotype.png, fig_variant_landscape.png
"""
import gzip, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("analysis/crd_variants")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Fig 1: 35-condition phenotype ----------
with gzip.open("outputs/wsK/genomes/phenoMatrix_35ConditionsNormalizedByYPD.tab.gz", "rt") as f:
    ph = pd.read_csv(f, sep="\t", index_col=0)
sub = ph.loc[["CGD", "CRD"]].T.sort_values("CGD")
conds = sub.index.tolist()
y = np.arange(len(conds))

fig, ax = plt.subplots(figsize=(6.5, 6.2))
ax.barh(y - 0.2, sub["CGD"], height=0.4, color="#4C78A8", label="CGD")
ax.barh(y + 0.2, sub["CRD"], height=0.4, color="#E45756", label="CRD")
ax.set_yticks(y)
ax.set_yticklabels(conds, fontsize=6.5)
ax.set_xlabel("Growth ratio vs YPD 30°C (1011 collection)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("CGD vs CRD: condition growth phenotypes (1011 phenoMatrix)")
for name in ["YPDCAFEIN50", "YPDCHX1", "YPDNYSTATIN", "YPDSODIUMMETAARSENITE", "YPD42"]:
    i = conds.index(name)
    ax.annotate("", xy=(max(sub.iloc[i]), i), xytext=(0, 0))
fig.tight_layout()
fig.savefig(OUT / "fig_phenotype.png", dpi=200)
plt.close(fig)

# ---------- Fig 2: variant landscape ----------
ann = pd.read_csv(OUT / "cgd_crd_diff_sites_annotated.tsv", sep="\t")
CHROM_LEN = {
    "chromosome1": 230218, "chromosome2": 813184, "chromosome3": 316620,
    "chromosome4": 1531933, "chromosome5": 576874, "chromosome6": 270161,
    "chromosome7": 1090940, "chromosome8": 562643, "chromosome9": 439888,
    "chromosome10": 745751, "chromosome11": 666816, "chromosome12": 1078177,
    "chromosome13": 924431, "chromosome14": 784333, "chromosome15": 1091291,
    "chromosome16": 948066,
}
order = [f"chromosome{i}" for i in range(1, 17)]
eff_order = ["missense", "synonymous", "frameshift", "stop_gained", "stop_lost",
             "inframe_indel", "promoter", "terminator", "heterozygous_cds",
             "nc_or_intron", "intergenic", "heterozygous_other"]
colors = {"missense": "#E45756", "synonymous": "#9D9DA5", "frameshift": "#B279A2",
          "stop_gained": "#7E1E9C", "stop_lost": "#C5A3D8", "inframe_indel": "#F2A7C3",
          "promoter": "#4C78A8", "terminator": "#72B7B2", "heterozygous_cds": "#F2CF5B",
          "nc_or_intron": "#59A14F", "intergenic": "#D6D6D6", "heterozygous_other": "#EDEDED"}

tab = ann.groupby(["chrom", "effect"]).size().unstack(fill_value=0).reindex(order).fillna(0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), height_ratios=[1.1, 1])
x = np.arange(16)
bottom = np.zeros(16)
for eff in eff_order:
    if eff not in tab:
        continue
    v = tab[eff].to_numpy()
    ax1.bar(x, v, bottom=bottom, color=colors[eff], label=eff, width=0.75)
    bottom += v
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace("chromosome", "chr") for c in order], rotation=45, ha="right", fontsize=7)
ax1.set_ylabel("# CGD≠CRD sites")
ax1.legend(fontsize=6, ncol=3, frameon=False, loc="upper right")
ax1.set_title("Sequence differences between CGD and CRD by chromosome and annotation")

density = (tab.sum(axis=1) / pd.Series(CHROM_LEN)[order] * 1e5)
ax2.bar(x, density, color="#4C78A8", width=0.75)
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace("chromosome", "chr") for c in order], rotation=45, ha="right", fontsize=7)
ax2.set_ylabel("diff sites / 100 kb")
fig.tight_layout()
fig.savefig(OUT / "fig_variant_landscape.png", dpi=200)
plt.close(fig)
print("figures written")
