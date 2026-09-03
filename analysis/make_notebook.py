# -*- coding: utf-8 -*-
"""生成 analysis/moa_narrative.ipynb（nbformat 保证合法、可重跑）。
运行：.venv/Scripts/python.exe analysis/make_notebook.py
notebook 只读 analysis/cache_moa/ 缓存（由 moa_build_cache.py 生成）。"""
import json
from pathlib import Path

nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
cells = nb["cells"]
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {},
                               "execution_count": None, "outputs": [], "source": s})

# ---------------- Markdown: title ----------------
md("""# 蛋白组预测 → 药物 MoA：与 Hoepfner/Hillenmeyer 化学基因组学签名的一致性分析

**复赛"机理叙事"专项。** 对 17 个可映射到 Hoepfner 2014 CMB 的化合物，构建三条基因轴对齐的签名：

- **(a) 数据真值签名** Δ_true：train_val 内该化合物全部处理样本、相对七键匹配对照的中位数（逐蛋白）；
- **(b) 模型预测签名** Δ̂：同一批样本的预测 log2 强度 − 匹配对照真值均值，取中位数；
- **(c) Hoepfner HOP z 签名**：该 CMB 化合物在 HOP 筛选中的逐基因 z 分数。

随后做三方向一致性、(b) 的 MoA 层次聚类（以 HOP (c) 聚类为参照）、以及全部 train_val 化合物的聚类与 MoA 推测。

> 合规：仅用 train_val（Y_tr）与本模型 trainval 预测；Y_test 零接触。基因轴经 `gene2locus.json`（标准名→locus）对齐到 HOP。
> 数据来自 `analysis/cache_moa/`（由 `moa_build_cache.py` 生成）。""")

# ---------------- imports & load ----------------
code("""import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, hypergeom, rankdata
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path

CACHE = Path('analysis/cache_moa'); FIGS = Path('analysis/figs')
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi':110,'savefig.dpi':160,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','savefig.facecolor':'white',
                     'savefig.bbox':'tight'})

sm  = np.load(CACHE/'signatures_mapped.npz', allow_pickle=True)
sa  = np.load(CACHE/'signatures_all.npz',  allow_pickle=True)
moa_ann = pd.read_csv(CACHE/'moa_annotations.csv')

comp      = list(sm['comp_names']);  moa  = list(sm['moa_class'])
cmb_ids   = sm['cmb_ids']; n_samples = sm['n_samples']; has_hop = sm['has_hop']
sig_true, sig_pred, sig_hop = sm['sig_true'], sm['sig_pred'], sm['sig_hop']
locus, std_name = sm['locus'], sm['std_name']

all_comp = list(sa['comp_names']); all_moa = list(sa['moa_class'])
all_pred, all_true = sa['sig_pred'], sa['sig_true']
all_mapped = sa['is_mapped']; all_n = sa['n_samples']

print(f'{len(comp)} CMB-mapped compounds in train_val; common axis = {len(locus)} genes')
print(f'all train_val treatment compounds = {len(all_comp)}')

# ---- helpers ----
def pairwise_corr(A, min_ok=20):
    A = np.asarray(A, float); n = len(A); C = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            ok = ~(np.isnan(A[i]) | np.isnan(A[j]))
            if ok.sum() >= min_ok:
                C[i, j] = C[j, i] = pearsonr(A[i][ok], A[j][ok]).statistic
            else:
                C[i, j] = C[j, i] = np.nan
    return C

def zscore_rows(A):
    A = np.asarray(A, float)
    mu = np.nanmean(A, 1, keepdims=True); sd = np.nanstd(A, 1, keepdims=True)
    sd = np.where((sd < 1e-9) | np.isnan(sd), 1.0, sd)
    return (A - mu) / sd

def avg_linkage(C):
    D = 1 - np.nan_to_num(C, nan=0.0); np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, 2)
    return linkage(squareform(D, checks=False), method='average')

MOA_COLORS = {'Protein synthesis':'#d62728','Ionophore (K+)':'#1f77b4','TOR signaling':'#2ca02c',
 'Ergosterol/lipid':'#9467bd','Hsp90 chaperone':'#ff7f0e','Microtubule':'#8c564b',
 'DNA synthesis/damage':'#e377c2','DNA binding':'#bcbd22','HDAC/chromatin':'#17becf',
 'Osmotic/salt stress':'#7f7f7f','Metal chelation':'#4c72b0','Membrane/surfactant':'#55a868',
 'DNA alkylation':'#e377c2','Topoisomerase I':'#e377c2','':'#333333'}
def mc(m): return MOA_COLORS.get(m, '#333333')""")

# ---------------- Section 1 ----------------
md("""## 1. 签名构建与基因轴对齐

- 我们的蛋白轴是**标准基因名**，HOP 基因轴是**locus 系统名**，经 `gene2locus.json` 桥接，得到 **5182 个共同基因**（覆盖我们 5243 蛋白的 98.8%）。
- 对照匹配：同 `data_source/Strains/Medium/Temperature/pert_time/instrument/Yeast_cell_plate` 七键组内 DMSO/Water 均值（train_val 内覆盖率 100%）。
- 17 个 CMB 映射化合物中，**Camptothecin / Fluconazole / MMS 为 test-only**（train_val 无样本）；**Clotrimazole (CMB 1101) 无 HOP 行**。故三签名分析覆盖 **14 个化合物**，其中 13 个可做 HOP 对比。""")

code("""info = pd.DataFrame({'compound': comp, 'cmb_id': cmb_ids, 'n_trainval': n_samples,
                     'moa_class': moa, 'has_HOP': has_hop})
test_only = moa_ann[~moa_ann['present_in_trainval']]
print('test-only CMB compounds (excluded, no train_val rows):',
      ', '.join(test_only['compound']))
info""")

# ---------------- Section 2 ----------------
md("""## 2. 三方向一致性：(a)真值 × (b)预测 × (c)HOP

对每个化合物在 5182 共同基因上计算 Spearman / Pearson 相关。
- **(a)×(b)**：模型对化合物特异蛋白组签名的保真度（本模型核心能力）。
- **(a)×(c)、(b)×(c)**：蛋白组丰度变化 vs 化学基因组学适应性（HOP z）——两个**不同分子层**。
并给出 (a)×(b) 的置换零模型（打乱基因标签）作基线。""")

code("""def corr_pair(x, y, min_ok=20):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < min_ok: return np.nan, np.nan, int(ok.sum())
    return (spearmanr(x[ok], y[ok]).statistic,
            pearsonr(x[ok], y[ok]).statistic, int(ok.sum()))

rng = np.random.default_rng(0)
rows = []
for j, c in enumerate(comp):
    a, b, cc = sig_true[j], sig_pred[j], sig_hop[j]
    sab, pab, _ = corr_pair(a, b)
    sac, pac, _ = corr_pair(a, cc)
    sbc, pbc, nbc = corr_pair(b, cc)
    # permutation null for (a)x(b): shuffle predicted gene order
    null = [corr_pair(a, b[rng.permutation(len(b))])[0] for _ in range(50)]
    rows.append(dict(compound=c, moa=moa[j], n=int(n_samples[j]),
                     sp_true_pred=sab, sp_true_hop=sac, sp_pred_hop=sbc,
                     pe_true_pred=pab, pe_true_hop=pac, pe_pred_hop=pbc,
                     null_ab_mean=np.nanmean(null), null_ab_max=np.nanmax(null)))
cons = pd.DataFrame(rows)
cons.to_csv(CACHE/'consistency_table.csv', index=False)
print(f"(a)x(b) median Spearman = {cons['sp_true_pred'].median():.3f} "
      f"(range {cons['sp_true_pred'].min():.3f}-{cons['sp_true_pred'].max():.3f}); "
      f"permutation null ~ {cons['null_ab_mean'].mean():.3f}")
print(f"(b)x(c) median Spearman = {cons['sp_pred_hop'].median():.3f}; "
      f"(a)x(c) median = {cons['sp_true_hop'].median():.3f}")
cons[['compound','moa','n','sp_true_pred','sp_true_hop','sp_pred_hop',
      'pe_true_pred','pe_pred_hop']].round(3)""")

md("""**读法**：(a)×(b) 全部显著为正（中位 ~0.72，远高于置换零 ~0），说明模型真实复现了化合物特异的蛋白组响应。
(a)×(c)/(b)×(c) 全基因组相关≈0（甚至略负）——这是**预期**的：HOP 找的是"缺失后改变药物敏感性"的**缓冲/靶点基因**（适应性层），
而我们的读出是"蛋白丰度变化"（表达层）；丰度与适应性在文献中本就弱耦合。MoA 信号需到 **top-|z| 基因重叠** 与 **聚类** 中找。""")

code("""# FIG 1: per-compound consistency bars
d = cons.sort_values('sp_true_pred', ascending=False).reset_index(drop=True)
x = np.arange(len(d)); w = 0.38
fig, ax = plt.subplots(figsize=(11.5, 4.4))
b1 = ax.bar(x - w/2, d['sp_true_pred'], w, label='(a) truth × (b) prediction', color='#1f77b4')
b2 = ax.bar(x + w/2, d['sp_pred_hop'], w, label='(b) prediction × (c) HOP', color='#ff7f0e')
ax.axhline(0, color='k', lw=0.8)
ax.axhline(d['sp_true_pred'].median(), color='#1f77b4', ls='--', lw=1, alpha=0.6)
ax.axhline(d['sp_pred_hop'].median(), color='#ff7f0e', ls='--', lw=1, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(d['compound'], rotation=38, ha='right', fontsize=9)
ax.set_ylabel('Spearman correlation'); ax.set_ylim(-0.25, 1.0)
ax.set_title('Per-compound signature consistency (5182 common genes)')
ax.legend(frameon=False, loc='lower left', fontsize=9)
fig.savefig(FIGS/'fig1_consistency_bars.png'); plt.show()
print('median (a)x(b)=%.3f  median (b)x(c)=%.3f' % (d['sp_true_pred'].median(), d['sp_pred_hop'].median()))""")

# ---------------- Section 3 ----------------
md("""## 3. Top-|z| 基因集重叠（超几何富集）

全基因组相关≈0 时，信号集中在强响应基因。取每个化合物 top-200 |Δ|（真值/预测）与 top-200 |z|（HOP），
计算重叠数、富集倍数与超几何 p 值。背景期望重叠 = 200×200/5182 ≈ 7.7。""")

code("""def topx(v, k=200):
    v = np.where(np.isnan(v), -np.inf, np.abs(v)); return set(np.argsort(-v)[:k])
M = len(locus); K = 200
def enr(A, B):
    o = len(A & B); return o, o/(K*K/M), hypergeom.sf(o-1, M, K, K)
ov_rows = []
for j, c in enumerate(comp):
    if not has_hop[j]: continue
    hop, tr, pr = topx(sig_hop[j]), topx(sig_true[j]), topx(sig_pred[j])
    oT, eT, pT = enr(tr, hop); oP, eP, pP = enr(pr, hop)
    ov_rows.append(dict(compound=c, ov_true=oT, enr_true=eT, p_true=pT,
                        ov_pred=oP, enr_pred=eP, p_pred=pP))
ov = pd.DataFrame(ov_rows).sort_values('p_pred')
ov.to_csv(CACHE/'topk_overlap.csv', index=False)
def star(p): return '***' if p<1e-3 else '**' if p<1e-2 else '*' if p<5e-2 else ''
ov['sig_pred'] = ov['p_pred'].map(star); ov['sig_true'] = ov['p_true'].map(star)
ov.round(3)""")

code("""# FIG 2: top-|z| overlap enrichment
d = ov.sort_values('enr_pred', ascending=False).reset_index(drop=True)
x = np.arange(len(d)); w = 0.38
fig, ax = plt.subplots(figsize=(11.5, 4.2))
ax.bar(x - w/2, d['enr_pred'], w, label='prediction × HOP', color='#ff7f0e')
ax.bar(x + w/2, d['enr_true'], w, label='truth × HOP', color='#1f77b4')
for i, r in d.iterrows():
    if r['sig_pred']: ax.text(i - w/2, r['enr_pred']+0.05, r['sig_pred'], ha='center', fontsize=9)
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(d['compound'], rotation=38, ha='right', fontsize=9)
ax.set_ylabel('Top-200 |z| overlap enrichment (fold)')
ax.set_title('Enrichment of HOP top-|z| genes among proteomically-affected genes')
ax.legend(frameon=False, fontsize=9)
fig.savefig(FIGS/'fig2_topk_overlap.png'); plt.show()""")

# ---------------- Section 4 ----------------
md("""## 4. MoA 聚类：预测蛋白组签名 (b)，以 HOP 适应性签名 (c) 为参照

对 14 个化合物的 z-score 化预测签名做相关距离 + average linkage 层次聚类，标注已知 MoA。
已知应成簇的对：**Anisomycin+CHX（蛋白合成）**、**Nigericin+Valinomycin（K+ 离子载体）**。
HOP（金标准，稀疏、背景低）作参照——它应干净回收这些对。""")

code("""def cluster_fig(C, labels, colors, title, fname, vmax=None):
    Z = avg_linkage(C); n = len(labels)
    fig = plt.figure(figsize=(8.0, 6.6))
    axd = fig.add_axes([0.03, 0.08, 0.19, 0.80])
    axh = fig.add_axes([0.235, 0.08, 0.60, 0.80])
    dd = dendrogram(Z, orientation='left', ax=axd, no_labels=True,
                    color_threshold=None, above_threshold_color='#999999',
                    link_color_func=lambda k: '#999999')
    axd.invert_xaxis(); axd.axis('off')
    idx = dd['leaves']
    Cr = C[np.ix_(idx, idx)]
    if vmax is None:
        vmax = np.nanmax(np.abs(Cr[~np.eye(n, dtype=bool)]))
    im = axh.imshow(Cr, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    origin='lower', aspect='auto')
    axh.set_xticks(range(n)); axh.set_yticks(range(n))
    axh.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=8.5)
    axh.set_yticklabels([labels[i] for i in idx], fontsize=8.5)
    for t, i in zip(axh.get_yticklabels(), idx): t.set_color(colors[i])
    for t, i in zip(axh.get_xticklabels(), idx): t.set_color(colors[i])
    axh.tick_params(length=0)
    for s in axh.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.14); cb.set_label('Pearson r', fontsize=9)
    axh.set_title(title, fontsize=11, pad=12)
    fig.savefig(fname); plt.show()
    return Z, idx

# (b) predicted proteome signatures, 14 compounds
Zp = zscore_rows(sig_pred); Cp = pairwise_corr(Zp)
colors = [mc(m) for m in moa]
Zb, idxb = cluster_fig(Cp, comp, colors,
    'Predicted proteome signatures — MoA clustering (14 compounds)',
    FIGS/'fig3_pred_cluster.png')
off = Cp[~np.eye(len(Cp), dtype=bool)]
print(f'predicted: median background r = {np.nanmedian(off):.3f}')""")

code("""# (c) HOP fitness signatures, 13 compounds (reference)
hidx = [i for i in range(len(comp)) if has_hop[i]]
hcomp = [comp[i] for i in hidx]; hmoa = [moa[i] for i in hidx]
Zh = zscore_rows(sig_hop[hidx]); Ch = pairwise_corr(Zh)
hcolors = [mc(m) for m in hmoa]
Zc, idxc = cluster_fig(Ch, hcomp, hcolors,
    'HOP fitness signatures — MoA clustering (reference, 13 compounds)',
    FIGS/'fig4_hop_cluster.png')
offh = Ch[~np.eye(len(Ch), dtype=bool)]
print(f'HOP: median background r = {np.nanmedian(offh):.3f}')

def gpair(C, labs, a, b):
    return C[labs.index(a), labs.index(b)]
print(f'PRED  ANI-CHX={gpair(Cp,comp,"Anisomycin","CHX"):.3f}  '
      f'NIG-VAL={gpair(Cp,comp,"Nigericin","Valinomycin"):.3f}')
print(f'HOP   ANI-CHX={gpair(Ch,hcomp,"Anisomycin","CHX"):.3f}  '
      f'NIG-VAL={gpair(Ch,hcomp,"Nigericin","Valinomycin"):.3f}')""")

md("""**对比即故事**：
- 预测蛋白组里 **Nigericin×Valinomycin（均为 K+ 离子载体）强成簇**（r≈0.48，远超 ~0.23 的高背景）；
- **Anisomycin×CHX（蛋白合成）在蛋白组仅弱相关，但在 HOP 适应性层清晰成簇**（r≈0.35，HOP 第 3 高对）；
- 蛋白组两两相关**背景很高**（共享的酵母应激/生长速率程序，PC1 占 ~40% 方差），会淹没 MoA 特异信号；
  HOP 适应性谱稀疏、背景近零，因此更适合直接做 MoA 聚类。
这正说明：**丰度响应 ≠ 适应性响应**，模型学到的是前者，二者在 MoA 层面互补。""")

# ---------------- Section 5 ----------------
md("""## 5. 全部 train_val 化合物（43）聚类 + 未映射化合物 MoA 推测

用 z-score 化预测签名对全部 43 个化合物聚类，标注 14 个已知 MoA 锚点（粗体着色）。
**重要 caveat**：蛋白组响应被共享应激/生长程序主导，两两相关基线很高（中位 ~0.6），
因此"最近锚点"的原始相关普遍虚高、不代表特异 MoA。我们改用 **specificity = 锚点相关 − 该化合物背景相关** 来排序，
特异性高者才具备 MoA 转移价值；以下仅作**假设生成**（叙事素材），需后续实验/外部证据佐证。""")

code("""Za = zscore_rows(all_pred); Ca = pairwise_corr(Za)
Zall = avg_linkage(Ca)
fig, ax = plt.subplots(figsize=(9.5, 9.0))
dd = dendrogram(Zall, orientation='left', ax=ax, no_labels=True,
                color_threshold=None, above_threshold_color='#bbbbbb',
                link_color_func=lambda k: '#bbbbbb')
ax.invert_xaxis()
order = dd['leaves']
for k, li in enumerate(order):
    nm = all_comp[li]; known = all_mapped[li]
    ax.text(-0.02, (k*10+5), nm, va='center', ha='right', fontsize=7.5,
            color=mc(all_moa[li]) if known else '#999999',
            fontweight='bold' if known else 'normal')
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1]*1.02)
ax.axis('off')
ax.set_title('All 43 train_val compounds — predicted-signature dendrogram\\n'
             '(bold color = 14 known-MoA anchors; grey = unmapped)', fontsize=11)
fig.savefig(FIGS/'fig5_all43_dendrogram.png'); plt.show()""")

code("""# nearest known-MoA anchor, ranked by SPECIFICITY over each compound's background correlation.
# 高背景（共享应激程序）会让"最近锚点 r"普遍虚高 -> 用 specificity = r - background 区分
# "独特的锚点关系"（有 MoA 转移价值）与"仅是同一应激簇成员"（无特异 MoA 价值）。
known_idx = np.where(all_mapped)[0]
rows = []
for i in range(len(all_comp)):
    if all_mapped[i]: continue
    base = np.nanmedian(np.delete(Ca[i], i))   # background = median r to all other compounds
    cs = sorted([(j, Ca[i, j]) for j in known_idx if not np.isnan(Ca[i, j])], key=lambda x: -x[1])
    j1, r1 = cs[0]; j2, r2 = cs[1]
    rows.append(dict(compound=all_comp[i], n=int(all_n[i]),
                     anchor=all_comp[j1], anchor_moa=all_moa[j1], r=round(r1, 3),
                     baseline=round(base, 3), specificity=round(r1 - base, 3),
                     anchor2=all_comp[j2], moa2=all_moa[j2], r2=round(r2, 3)))
anchor = pd.DataFrame(rows).sort_values('specificity', ascending=False)
anchor.to_csv(CACHE/'unmapped_moa_hypotheses.csv', index=False)
print('MoA-transfer hypotheses ranked by specificity (r - background); caveat: proteome')
print('clusters reflect shared stress program, so treat as hypothesis-generating only.')
anchor.head(14)""")

code("""# FIG 6: PCA embedding of all 43 predicted signatures
from sklearn.decomposition import PCA
X = np.where(np.isnan(Za), 0, Za)
pc = PCA(n_components=2).fit_transform(X)
var = PCA(n_components=10).fit(X).explained_variance_ratio_
fig, ax = plt.subplots(figsize=(9.0, 7.2))
for i in range(len(all_comp)):
    if all_mapped[i]:
        ax.scatter(pc[i,0], pc[i,1], s=42, color=mc(all_moa[i]), zorder=3,
                   edgecolor='k', linewidth=0.4)
    else:
        ax.scatter(pc[i,0], pc[i,1], s=22, color='#cccccc', zorder=2)
for i in range(len(all_comp)):
    if all_mapped[i] or i in np.argsort(-np.linalg.norm(pc,axis=1))[:8]:
        ax.annotate(all_comp[i], (pc[i,0], pc[i,1]), fontsize=7,
                    xytext=(3,3), textcoords='offset points',
                    color=mc(all_moa[i]) if all_mapped[i] else '#888888')
import matplotlib.lines as mlines
handles=[mlines.Line2D([],[],marker='o',ls='',color=mc(m),label=m,markersize=7)
         for m in dict.fromkeys(all_moa) if m]
handles.append(mlines.Line2D([],[],marker='o',ls='',color='#cccccc',label='unmapped',markersize=7))
ax.legend(handles=handles, frameon=False, fontsize=7.5, loc='best', ncol=2)
ax.set_xlabel(f'PC1 ({var[0]*100:.0f}% var)'); ax.set_ylabel(f'PC2 ({var[1]*100:.0f}% var)')
ax.set_title('PCA of predicted proteome signatures (43 compounds)')
fig.savefig(FIGS/'fig6_pca_all43.png'); plt.show()
print('PCA var explained (first 5):', np.round(var[:5],3))""")

# ---------------- Section 6 ----------------
md("""## 6. 结论

1. **模型高保真复现化合物特异蛋白组签名**：(a)×(b) 中位 Spearman ≈ 0.72（全部化合物显著为正，远高于置换零 ~0）。
2. **蛋白组丰度响应与 HOP 适应性是互补的两个分子层**：全基因组相关≈0（预期）；仅 EDTA、Hoechst 在 top-|z| 基因集上呈弱富集（p<0.05）。
3. **MoA 信号确实存在但层面不同**：K+ 离子载体 Nigericin/Valinomycin 在预测蛋白组强成簇；蛋白合成抑制剂 Anisomycin/CHX 在 HOP 适应性层清晰成簇——说明模型读出（丰度）与化学基因组学（适应性）在 MoA 层面互补，且蛋白组响应被共享应激/生长程序主导。

产物：`analysis/figs/*.png`、`analysis/cache_moa/*.csv|npz`、`analysis/moa_report.md`。""")

nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3'}}
out = Path('analysis/moa_narrative.ipynb')
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', out, 'with', len(cells), 'cells')
