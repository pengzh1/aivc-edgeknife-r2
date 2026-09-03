"""可选增强: DNABERT-2 调控/功能序列差异强度（±1kb 等位窗口 embedding 余弦距离）。

方法: 对目标基因取覆盖高置信纯合差异位点最密集的 2001bp 参考窗口,
按 HQ 纯合差异位点分别替换为 CGD / CRD 等位（indel 从右往左应用），
得到两株等位序列 -> DNABERT-2 last-hidden mean-pool -> 余弦距离。
对照: 5 个仅同义差异基因窗 + 5 个无差异基因窗（后者两株序列相同，距离应为 0，
用于验证管线正确性）。

输出: analysis/crd_variants/dnabert_window_scores.tsv
"""
import gzip, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

OUT = Path("analysis/crd_variants")
WIN = 2001  # ±1kb
SEED = 7

TARGETS = ["GAS1", "PDR5", "SLN1", "ENA1", "SCW10", "QDR3", "PDR11", "HPF1",
           "HXT1", "SIT1", "SNQ2", "UTH1", "ZRT3", "PSD2", "CMR1", "COS8",
           "HXT5", "GAS3"]

CH = {'NC_001133.9': 'chromosome1', 'NC_001134.8': 'chromosome2', 'NC_001135.5': 'chromosome3',
      'NC_001136.10': 'chromosome4', 'NC_001137.3': 'chromosome5', 'NC_001138.5': 'chromosome6',
      'NC_001139.3': 'chromosome7', 'NC_001140.6': 'chromosome8', 'NC_001141.2': 'chromosome9',
      'NC_001142.9': 'chromosome10', 'NC_001143.9': 'chromosome11', 'NC_001144.5': 'chromosome12',
      'NC_001145.3': 'chromosome13', 'NC_001146.8': 'chromosome14', 'NC_001147.6': 'chromosome15',
      'NC_001148.4': 'chromosome16'}
CH_INV = {v: k for k, v in CH.items()}
CHROM_LEN = {
    "chromosome1": 230218, "chromosome2": 813184, "chromosome3": 316620,
    "chromosome4": 1531933, "chromosome5": 576874, "chromosome6": 270161,
    "chromosome7": 1090940, "chromosome8": 562643, "chromosome9": 439888,
    "chromosome10": 745751, "chromosome11": 666816, "chromosome12": 1078177,
    "chromosome13": 924431, "chromosome14": 784333, "chromosome15": 1091291,
    "chromosome16": 948066,
}


def load_ref():
    seqs = {}
    name = None
    buf = []
    with gzip.open("outputs/wsK/genomes/fasta/S288C_reference_genomic.fna.gz", "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name:
            seqs[name] = "".join(buf).upper()
    return seqs


def main():
    ann = pd.read_csv(OUT / "cgd_crd_diff_sites_annotated.tsv", sep="\t")
    hq = ann[(ann.hom_diff == 1) & (ann.dp_cgd >= 4) & (ann.gq_cgd >= 20)
             & (ann.dp_crd >= 4) & (ann.gq_crd >= 20)].copy()
    ref = load_ref()
    rng = np.random.default_rng(SEED)

    # 对照组
    g2l = json.load(open("extref/hop/gene2locus.json"))
    l2g = {v: k for k, v in g2l.items()}
    syn_only = (hq[(hq.effect == "synonymous") & (hq.locus != "")]
                .groupby("locus").size()
                .rename("n").reset_index())
    miss_loci = set(hq[hq.effect.isin(["missense", "frameshift", "stop_gained", "stop_lost",
                                       "inframe_indel", "promoter"])].locus)
    syn_only = syn_only[~syn_only.locus.isin(miss_loci) & syn_only.locus.isin(l2g)]
    syn_ctrl = rng.choice(syn_only.locus.to_numpy(), size=5, replace=False).tolist()
    # 无差异对照: 有注释基因但完全不在差异表里
    all_diff_loci = set(ann[ann.locus != ""].locus)
    nodiff = [l for l in l2g if l not in all_diff_loci]
    nodiff_ctrl = rng.choice(nodiff, size=5, replace=False).tolist()

    # 目标基因 locus
    name2locus = {}
    sub = ann[ann.gene_name.isin(TARGETS)][["gene_name", "locus"]].drop_duplicates()
    for r in sub.itertuples():
        name2locus[r.gene_name] = r.locus

    jobs = []  # (label, group, locus)
    for t in TARGETS:
        if t in name2locus:
            jobs.append((t, "target", name2locus[t]))
    for l in syn_ctrl:
        jobs.append((l2g.get(l, l), "synonymous_ctrl", l))
    for l in nodiff_ctrl:
        jobs.append((l2g.get(l, l), "nodiff_ctrl", l))

    # 基因坐标
    genes = {}
    with gzip.open("outputs/wsK/genomes/GCF_000146045.2_R64_genomic.gff.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            kv = dict(i.split("=", 1) for i in c[8].split(";") if "=" in i)
            loc = kv.get("locus_tag", kv.get("Name", ""))
            genes[loc] = (c[0], int(c[3]), int(c[4]), c[6])

    # 选窗口 + 构建等位序列
    records = []
    seqs_to_embed = []
    for label, grp, locus in jobs:
        if locus not in genes:
            print("skip(no gff):", label)
            continue
        nc, gs, ge, strand = genes[locus]
        ch = CH.get(nc)
        if ch is None:
            continue
        refseq = ref[nc]
        gsites = hq[hq.locus == locus]
        if len(gsites) == 0:
            center = (gs + ge) // 2
        else:
            # 位点最密窗口中心: 以每个位点为中心计数
            pos = np.sort(gsites.pos.to_numpy())
            best_c, best_n = pos[0], -1
            for p in pos:
                n_in = np.searchsorted(pos, p + 1000) - np.searchsorted(pos, p - 1000, side="right")
                if n_in > best_n:
                    best_n, best_c = n_in, p
            center = best_c
        w0 = max(1, center - 1000)
        w1 = min(len(refseq), center + 1000)
        wseq = refseq[w0 - 1:w1]
        sites = hq[(hq.chrom == ch) & (hq.pos >= w0) & (hq.pos <= w1)]
        # 构建两株等位序列（从右往左应用 indel/snp）
        alleles = {}
        n_skipped = 0
        for strain, gtcol in (("CGD", "gt_cgd"), ("CRD", "gt_crd")):
            s = wseq
            for r in sites.sort_values("pos", ascending=False).itertuples():
                gt = getattr(r, gtcol)
                if gt == "0/0":
                    continue  # 与参考一致
                if gt != "1/1":
                    continue  # 跳过杂合/其他（HQ 已保证纯合差异）
                alt = r.alt.split(",")[0]
                off = r.pos - w0
                if s[off:off + len(r.ref)].upper() != r.ref.upper():
                    n_skipped += 1  # 与相邻 indel 锚定区重叠的位点，放弃该位点
                    continue
                s = s[:off] + alt + s[off + len(r.ref):]
            alleles[strain] = s
        n_diff_bases = sum(a != b for a, b in zip(alleles["CGD"], alleles["CRD"])) \
            if len(alleles["CGD"]) == len(alleles["CRD"]) else -1
        records.append({"label": label, "group": grp, "locus": locus, "chrom": ch,
                        "win_start": w0, "win_end": w1, "strand": strand,
                        "n_hq_sites_in_window": int(len(sites)),
                        "n_skipped_overlap": n_skipped,
                        "len_cgd": len(alleles["CGD"]), "len_crd": len(alleles["CRD"]),
                        "n_diff_bases_aligned": n_diff_bases})
        seqs_to_embed.append((alleles["CGD"], alleles["CRD"]))
        print(f"{label:12s} {grp:15s} {ch} {w0}-{w1} sites={len(sites)} "
              f"lenCGD={len(alleles['CGD'])} lenCRD={len(alleles['CRD'])}", flush=True)

    # DNABERT-2 embedding
    tok = AutoTokenizer.from_pretrained("cache/dnabert2", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("cache/dnabert2", trust_remote_code=True)
    model = model.cuda().eval()

    @torch.no_grad()
    def embed(seq):
        inp = {k: v.cuda() for k, v in tok(seq, return_tensors="pt").items()}
        out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0]  # (L,768)
        mask = inp["attention_mask"][0].unsqueeze(-1).float()
        return (h * mask).sum(0).div(mask.sum()).cpu().numpy()

    cos = torch.nn.CosineSimilarity(dim=0)
    for rec, (s1, s2) in zip(records, seqs_to_embed):
        e1 = torch.tensor(embed(s1))
        e2 = torch.tensor(embed(s2))
        rec["dnabert_cosine_dist"] = float(1 - cos(e1, e2))
        rec["dnabert_cosine_sim"] = float(cos(e1, e2))

    df = pd.DataFrame(records)
    df.to_csv(OUT / "dnabert_window_scores.tsv", sep="\t", index=False)
    print(df[["label", "group", "n_hq_sites_in_window", "n_diff_bases_aligned",
              "dnabert_cosine_dist"]].to_string(index=False))


if __name__ == "__main__":
    main()
