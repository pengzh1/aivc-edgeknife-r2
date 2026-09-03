"""流式扫描 1011Matrix.gvcf.gz，提取 CGD/CRD 基因型不一致位点。

只读 outputs/wsK/genomes/1011Matrix.gvcf.gz（gzip 流，不落盘解压）。
输出:
  analysis/crd_variants/cgd_crd_diff_sites_raw.csv  — 两株均 called 且 GT 不同的位点
  analysis/crd_variants/scan_stats.json             — 扫描统计（总数、染色体分布、SNP 距离复核）
"""
import gzip, json, time
from collections import Counter, defaultdict
from pathlib import Path

GVCF = Path("outputs/wsK/genomes/1011Matrix.gvcf.gz")
OUT = Path("analysis/crd_variants")
CGD_I, CRD_I = 761, 920  # 全行 split 后的列下标（含前 9 个固定列）

# R64 16 条染色体长度（VCF header contig 行）
CHROM_LEN = {
    "chromosome1": 230218, "chromosome2": 813184, "chromosome3": 316620,
    "chromosome4": 1531933, "chromosome5": 576874, "chromosome6": 270161,
    "chromosome7": 1090940, "chromosome8": 562643, "chromosome9": 439888,
    "chromosome10": 745751, "chromosome11": 666816, "chromosome12": 1078177,
    "chromosome13": 924431, "chromosome14": 784333, "chromosome15": 1091291,
    "chromosome16": 948066,
}
GENOME_SIZE = sum(CHROM_LEN.values())


def parse_sample(field):
    """返回 (GT, DP, GQ)；缺失返回 ('.', -1, -1)。"""
    p = field.split(":")
    gt = p[0]
    if gt.startswith("."):
        return ".", -1, -1
    dp = -1
    gq = -1
    if len(p) > 2 and p[2] not in ("", "."):
        dp = int(p[2])
    if len(p) > 3 and p[3] not in ("", "."):
        gq = int(p[3])
    return gt, dp, gq


def vtype(ref, alt):
    if alt == "*" or "<" in alt:
        return "other"
    alts = alt.split(",")
    if len(alts) > 1:
        return "multiallelic"
    a = alts[0]
    if len(ref) == 1 and len(a) == 1:
        return "snp"
    return "indel"


def main():
    t0 = time.time()
    n = 0
    both_called = 0
    diff = 0
    diff_by_chrom = Counter()
    called_by_chrom = Counter()
    type_counts = Counter()
    hom_diff = 0  # 0/0 vs 1/1 型纯合差异
    het_involved = 0
    # SNP 距离复核：双等位 SNP 位点上加权差异（纯合 1.0，杂合 0.5）
    snp_weighted_diff = 0.0
    biallelic_snp_sites = 0

    out_path = OUT / "cgd_crd_diff_sites_raw.csv"
    with gzip.open(GVCF, "rt") as f, open(out_path, "w") as out:
        out.write("chrom,pos,ref,alt,vtype,gt_cgd,dp_cgd,gq_cgd,gt_crd,dp_crd,gq_crd,hom_diff\n")
        for line in f:
            if line.startswith("#"):
                continue
            n += 1
            if n % 200000 == 0:
                el = time.time() - t0
                print(f"[{el:7.1f}s] lines={n} both_called={both_called} diff={diff}", flush=True)
            c = line.split("\t", 921)
            gt1, dp1, gq1 = parse_sample(c[CGD_I])
            gt2, dp2, gq2 = parse_sample(c[CRD_I])
            if gt1 == "." or gt2 == ".":
                continue
            both_called += 1
            chrom = c[0]
            called_by_chrom[chrom] += 1
            ref, alt = c[3], c[4]
            vt = vtype(ref, alt)
            # SNP 距离复核（双等位 SNP，两株均 called 的位点）
            if vt == "snp":
                biallelic_snp_sites += 1
                if gt1 != gt2:
                    if ("1" in gt1.split("/")) != ("1" in gt2.split("/")):
                        # 一株纯合 ref、一株纯合 alt
                        if gt1 in ("0/0", "1/1") and gt2 in ("0/0", "1/1"):
                            snp_weighted_diff += 1.0
                        else:
                            snp_weighted_diff += 0.5
                    else:
                        snp_weighted_diff += 0.5  # 0/1 vs 1/1 等
            if gt1 == gt2:
                continue
            diff += 1
            diff_by_chrom[chrom] += 1
            type_counts[vt] += 1
            is_hom = 1 if (gt1 in ("0/0", "1/1") and gt2 in ("0/0", "1/1")) else 0
            hom_diff += is_hom
            if not is_hom:
                het_involved += 1
            out.write(f"{chrom},{c[1]},{ref},{alt},{vt},{gt1},{dp1},{gq1},{gt2},{dp2},{gq2},{is_hom}\n")

    stats = {
        "total_variant_lines": n,
        "both_called": both_called,
        "diff_sites": diff,
        "diff_rate_among_called": diff / max(both_called, 1),
        "hom_diff": hom_diff,
        "het_involved": het_involved,
        "type_counts": dict(type_counts),
        "diff_by_chrom": dict(diff_by_chrom),
        "called_by_chrom": dict(called_by_chrom),
        "biallelic_snp_sites_both_called": biallelic_snp_sites,
        "snp_weighted_diff_bases": snp_weighted_diff,
        "snp_distance_pct_of_genome": 100.0 * snp_weighted_diff / GENOME_SIZE,
        "genome_size": GENOME_SIZE,
        "elapsed_sec": time.time() - t0,
    }
    with open(OUT / "scan_stats.json", "w") as fp:
        json.dump(stats, fp, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
