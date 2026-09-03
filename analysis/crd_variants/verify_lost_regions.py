"""验证 CGD/CRD 各自 lost 区（对方 CN=0）在 gVCF 中的 call 率 vs 全基因组。

判定缺失解释: 若 lost 区 call 率远低于全基因组 -> reads 无法比对 -> 真缺失或深度趋异;
若 call 率正常且多为 1/1 -> 序列存在但趋异（拷贝注释为 0 源于组装注释口径）。
输出: analysis/crd_variants/lost_region_callrate.json
"""
import gzip, json, re
import pandas as pd

CH = {'NC_001133.9': 'chromosome1', 'NC_001134.8': 'chromosome2', 'NC_001135.5': 'chromosome3',
      'NC_001136.10': 'chromosome4', 'NC_001137.3': 'chromosome5', 'NC_001138.5': 'chromosome6',
      'NC_001139.3': 'chromosome7', 'NC_001140.6': 'chromosome8', 'NC_001141.2': 'chromosome9',
      'NC_001142.9': 'chromosome10', 'NC_001143.9': 'chromosome11', 'NC_001144.5': 'chromosome12',
      'NC_001145.3': 'chromosome13', 'NC_001146.8': 'chromosome14', 'NC_001147.6': 'chromosome15',
      'NC_001148.4': 'chromosome16'}
CGD_I, CRD_I = 761, 920


def main():
    with gzip.open('outputs/wsK/genomes/GCF_000146045.2_R64_genomic.gff.gz', 'rt') as f:
        genes = {}
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) < 9 or c[2] != 'gene':
                continue
            kv = dict(i.split('=', 1) for i in c[8].split(';') if '=' in i)
            loc = kv.get('locus_tag', kv.get('Name', ''))
            genes[loc] = (c[0], int(c[3]), int(c[4]))

    with gzip.open('outputs/wsK/genomes/genesMatrix_CopyNumber.tab.gz', 'rt') as f:
        cn = pd.read_csv(f, sep='\t', index_col=0)
    sub = cn.loc[['CGD', 'CRD']].T
    yloc = [c for c in cn.columns if re.search(r'\.Y[A-P][LR]\d{3}', c)]
    s = sub.loc[yloc]
    cgd_lost = {o.split('.')[1] for o in s[(s['CGD'] == 0) & (s['CRD'] >= 1)].index}
    crd_lost = {o.split('.')[1] for o in s[(s['CRD'] == 0) & (s['CGD'] >= 1)].index}

    regions = {'CGD': [], 'CRD': []}
    for strain, locs in (('CGD', cgd_lost), ('CRD', crd_lost)):
        for loc in locs:
            if loc in genes:
                nc, a, b = genes[loc]
                ch = CH.get(nc)
                if ch:
                    regions[strain].append((ch, a, b))
    chrom_pos = {}
    for st, ivs in regions.items():
        byc = {}
        for ch, a, b in ivs:
            byc.setdefault(ch, []).append((a, b))
        for ch in byc:
            byc[ch].sort()
        chrom_pos[st] = byc

    def in_regions(st, ch, pos):
        for a, b in chrom_pos[st].get(ch, []):
            if a <= pos <= b:
                return True
            if a > pos:
                break
        return False

    stats = {st: {'in_called': 0, 'in_total': 0, 'out_called': 0, 'out_total': 0,
                  'in_hom_alt': 0} for st in ('CGD', 'CRD')}
    n = 0
    with gzip.open('outputs/wsK/genomes/1011Matrix.gvcf.gz', 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            n += 1
            if n % 300000 == 0:
                print(f"lines={n}", flush=True)
            c = line.split('\t', 921)
            ch, pos = c[0], int(c[1])
            for st, idx in (('CGD', CGD_I), ('CRD', CRD_I)):
                fld = c[idx]
                called = not fld.startswith('.')
                stt = stats[st]
                if in_regions(st, ch, pos):
                    stt['in_total'] += 1
                    stt['in_called'] += called
                    if called and fld.startswith('1/1'):
                        stt['in_hom_alt'] += 1
                else:
                    stt['out_total'] += 1
                    stt['out_called'] += called
    out = {}
    for st in ('CGD', 'CRD'):
        s2 = stats[st]
        out[st] = {
            **s2,
            'own_lost_call_rate': s2['in_called'] / max(s2['in_total'], 1),
            'genomewide_call_rate': s2['out_called'] / max(s2['out_total'], 1),
            'own_lost_hom_alt_frac_of_called': s2['in_hom_alt'] / max(s2['in_called'], 1),
            'n_lost_orfs': len(cgd_lost) if st == 'CGD' else len(crd_lost),
        }
        print(st, out[st])
    json.dump(out, open('analysis/crd_variants/lost_region_callrate.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
