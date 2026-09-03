"""注释 CGD/CRD 差异位点：GFF 区域分类 + CDS 内 SNP 同义/错义判定 + indel 移码判定。

输入: analysis/crd_variants/cgd_crd_diff_sites_raw.csv
输出: analysis/crd_variants/cgd_crd_diff_sites_annotated.csv
      analysis/crd_variants/annotate_summary.json

坐标系: gVCF 用 chromosomeN，GFF/FASTA 用 NC_*；按 contig 长度建立映射。
区域分类优先级: CDS > promoter(基因5'上游<=500bp) > terminator(基因3'下游<=200bp) > ncRNA/tRNA/rRNA/snoRNA > intergenic
CDS 内: snp -> synonymous / missense / stop_gained|lost；indel -> frameshift / inframe_indel
错义判定仅对纯合差异（0/0 vs 1/1）的双等位 SNP 做；杂合位点标 heterozygous。
"""
import gzip, json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

GEN = Path("outputs/wsK/genomes")
OUT = Path("analysis/crd_variants")

CHROM_LEN = {
    "chromosome1": 230218, "chromosome2": 813184, "chromosome3": 316620,
    "chromosome4": 1531933, "chromosome5": 576874, "chromosome6": 270161,
    "chromosome7": 1090940, "chromosome8": 562643, "chromosome9": 439888,
    "chromosome10": 745751, "chromosome11": 666816, "chromosome12": 1078177,
    "chromosome13": 924431, "chromosome14": 784333, "chromosome15": 1091291,
    "chromosome16": 948066,
}

CODON = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','AGT':'S','AGC':'S','CCT':'P','CCC':'P',
    'CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A',
    'GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H',
    'CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D',
    'GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R',
    'CGA':'R','CGG':'R','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(s):
    return s.translate(COMP)[::-1]


def load_contig_map():
    """chromosomeN -> NC 名（按 region 行长度匹配）。"""
    nc_len = {}
    with gzip.open(GEN / "GCF_000146045.2_R64_genomic.gff.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "region":
                continue
            nc_len[c[0]] = int(c[4])
    m = {}
    for ch, ln in CHROM_LEN.items():
        for nc, l2 in nc_len.items():
            if l2 == ln:
                m[ch] = nc
                break
    assert len(m) == 16, f"contig map incomplete: {len(m)}"
    return m


def load_gff():
    """返回 genes: list of dict(chrom_nc,start,end,strand,locus,name,product,cds[(s,e,phase)])"""
    genes = {}
    rna2gene = {}
    with gzip.open(GEN / "GCF_000146045.2_R64_genomic.gff.gz", "rt") as f:
        cds_rows = []
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            feat, attrs = c[2], c[8]
            kv = {}
            for item in attrs.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    kv[k] = v
            if feat == "gene":
                locus = kv.get("locus_tag", kv.get("Name", ""))
                genes[kv.get("ID", "")] = {
                    "chrom": c[0], "start": int(c[3]), "end": int(c[4]),
                    "strand": c[6], "locus": locus,
                    "name": kv.get("Name", locus),
                    "biotype": kv.get("gene_biotype", ""),
                    "product": "", "cds": [],
                }
            elif feat == "mRNA" or feat == "transcript":
                par = kv.get("Parent", "")
                if par.startswith("gene-"):
                    rna2gene[kv.get("ID", "")] = par
            elif feat == "CDS":
                cds_rows.append((kv.get("Parent", ""), c[0], int(c[3]), int(c[4]), c[7]))
            # product 从 mRNA 的 product= 字段不易取，改用 gene 的 Note? R64 RefSeq GFF
            # 的 protein 名在 CDS 的 Name/Note。简单起见后面用 locus/name 注释。
    # CDS 归属: Parent=rna-* -> gene
    n_orphan = 0
    for par, ch, s, e, ph in cds_rows:
        g = rna2gene.get(par)
        if g is None or g not in genes:
            n_orphan += 1
            continue
        genes[g]["cds"].append((s, e))
    # 去重 CDS 段
    for g in genes.values():
        g["cds"] = sorted(set(g["cds"]))
    print(f"genes={len(genes)} orphan_cds={n_orphan}")
    return genes


def load_ref():
    """NC -> 序列字符串。"""
    seqs = {}
    name = None
    buf = []
    with gzip.open(GEN / "fasta/S288C_reference_genomic.fna.gz", "rt") as f:
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


def build_cds_seq(gene, refseq):
    """拼接 CDS（编码方向），返回 (seq, list of (cds_idx_in_seq -> genomic_pos))。"""
    segs = sorted(gene["cds"])
    if gene["strand"] == "-":
        segs = segs[::-1]
    pieces = []
    posmap = []  # 每个编码碱基对应的基因组 1-based 坐标
    for (s, e) in segs:
        sub = refseq[s - 1:e]
        coords = list(range(s, e + 1))
        if gene["strand"] == "-":
            sub = revcomp(sub)
            coords = coords[::-1]
        pieces.append(sub)
        posmap.extend(coords)
    return "".join(pieces), posmap


def main():
    df = pd.read_csv(OUT / "cgd_crd_diff_sites_raw.tsv", sep="\t")
    print("diff sites:", len(df))
    ch2nc = load_contig_map()
    genes = load_gff()
    ref = load_ref()

    # 每条染色体上的区间索引（简单排序+扫描，酵母规模足够小）
    by_chrom = defaultdict(list)
    for g in genes.values():
        if g["chrom"] in ch2nc.values():
            by_chrom[g["chrom"]].append(g)

    # CDS 序列缓存（仅在有位点落入时构建）
    cds_cache = {}

    def cds_seq_of(gene):
        key = gene["locus"]
        if key not in cds_cache:
            cds_cache[key] = build_cds_seq(gene, ref[gene["chrom"]])
        return cds_cache[key]

    ann_rows = []
    summary = Counter()
    gene_effect = {}  # locus -> 最差效应等级（用于交叉表）

    EFFECT_RANK = {"frameshift": 5, "stop_gained": 5, "stop_lost": 4, "missense": 3,
                   "inframe_indel": 3, "synonymous": 1, "promoter": 2, "terminator": 1,
                   "ncRNA": 1, "intergenic": 0, "heterozygous_cds": 2, "heterozygous_other": 0}

    for row in df.itertuples():
        nc = ch2nc[row.chrom]
        pos = int(row.pos)
        region = "intergenic"
        hit_gene = None
        effect = "intergenic"
        aa_change = ""
        codon_change = ""
        detail = ""

        for g in by_chrom.get(nc, []):
            # CDS?
            in_cds = any(s <= pos <= e for (s, e) in g["cds"])
            if in_cds:
                hit_gene = g
                region = "CDS"
                break
            if g["start"] <= pos <= g["end"]:
                hit_gene = g
                region = "gene_nonCDS"  # 内含子/UTR（酵母 UTR 注释缺）
                break
            # 启动子/终止子（链感知）
            if g["strand"] == "+":
                if g["start"] - 500 <= pos < g["start"]:
                    hit_gene = g; region = "promoter"; break
                if g["end"] < pos <= g["end"] + 200:
                    hit_gene = g; region = "terminator"; break
            else:
                if g["end"] < pos <= g["end"] + 500:
                    hit_gene = g; region = "promoter"; break
                if g["start"] - 200 <= pos < g["start"]:
                    hit_gene = g; region = "terminator"; break

        if region == "CDS" and hit_gene is not None:
            gt1, gt2 = row.gt_cgd, row.gt_crd
            hom = row.hom_diff == 1
            if row.vtype == "snp" and hom:
                seq, posmap = cds_seq_of(hit_gene)
                try:
                    i = posmap.index(pos)
                except ValueError:
                    i = -1
                if i >= 0 and i < len(seq):
                    ci = (i // 3) * 3
                    codon_ref = seq[ci:ci + 3]
                    if len(codon_ref) == 3 and codon_ref in CODON:
                        # 变异碱基 = ALT（纯合差异，一株 0/0 一株 1/1）
                        alt_base = row.alt.split(",")[0]
                        if hit_gene["strand"] == "-":
                            alt_base = revcomp(alt_base)
                        codon_alt = codon_ref[:i % 3] + alt_base + codon_ref[i % 3 + 1:]
                        aa0, aa1 = CODON.get(codon_ref, "?"), CODON.get(codon_alt, "?")
                        codon_change = f"{codon_ref}>{codon_alt}"
                        aa_change = f"{aa0}{ci // 3 + 1}{aa1}"
                        if aa0 == aa1:
                            effect = "synonymous"
                        elif aa0 == "*":
                            effect = "stop_lost"
                        elif aa1 == "*":
                            effect = "stop_gained"
                        else:
                            effect = "missense"
                    else:
                        effect = "cds_undef"
            elif row.vtype == "snp":
                effect = "heterozygous_cds"
            elif row.vtype in ("indel", "multiallelic"):
                if hom:
                    delta_len = len(row.alt.split(",")[0]) - len(row.ref)
                    effect = "frameshift" if delta_len % 3 != 0 else "inframe_indel"
                else:
                    effect = "heterozygous_cds"
            else:
                effect = "cds_other"
        elif region in ("promoter", "terminator"):
            effect = region
        elif region == "gene_nonCDS":
            effect = "nc_or_intron"
        elif row.vtype != "snp" and row.hom_diff != 1:
            effect = "heterozygous_other"

        if hit_gene is not None and region not in ("CDS",):
            # 非 CDS 命中但可能有产物注释
            pass
        summary[effect] += 1
        if hit_gene is not None:
            locus = hit_gene["locus"]
            r = EFFECT_RANK.get(effect, 0)
            if locus not in gene_effect or r > gene_effect[locus][0]:
                gene_effect[locus] = (r, effect, hit_gene["name"])
        ann_rows.append({
            "chrom": row.chrom, "pos": pos, "ref": row.ref, "alt": row.alt,
            "vtype": row.vtype, "gt_cgd": row.gt_cgd, "gt_crd": row.gt_crd,
            "dp_cgd": row.dp_cgd, "gq_cgd": row.gq_cgd,
            "dp_crd": row.dp_crd, "gq_crd": row.gq_crd,
            "hom_diff": row.hom_diff, "region": region, "effect": effect,
            "locus": hit_gene["locus"] if hit_gene else "",
            "gene_name": hit_gene["name"] if hit_gene else "",
            "strand": hit_gene["strand"] if hit_gene else "",
            "codon_change": codon_change, "aa_change": aa_change,
        })

    adf = pd.DataFrame(ann_rows)
    adf.to_csv(OUT / "cgd_crd_diff_sites_annotated.tsv", index=False, sep="\t")
    out = {
        "n_sites": len(adf),
        "effect_counts": dict(summary),
        "n_genes_affected": len(gene_effect),
        "gene_worst_effect": {k: {"effect": v[1], "name": v[2]} for k, v in gene_effect.items()},
    }
    with open(OUT / "annotate_summary.json", "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "gene_worst_effect"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
