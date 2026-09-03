# 蛋白组预测 → 药物 MoA：与 Hoepfner/Hillenmeyer 化学基因组学签名的一致性分析

> 复赛"机理叙事"专项报告。所有分析仅用 train_val（Y_tr）与本模型 trainval 预测，**Y_test 零接触**。
> 重解析/对齐固化在 `analysis/moa_build_cache.py`，notebook `analysis/moa_narrative.ipynb` 只读缓存、可重跑。

## 1. 方法

对 17 个可映射到 Hoepfner 2014 CMB 的化合物，在**对齐的基因轴**上构建三条签名并两两比对：

- **(a) 数据真值签名 Δ_true**：train_val 内该化合物全部处理样本，相对七键（data_source/Strains/Medium/Temperature/pert_time/instrument/Yeast_cell_plate）组内 DMSO/Water 对照的 log2 差，按蛋白取中位数；
- **(b) 模型预测签名 Δ̂**：同一批样本的预测 log2 强度 − 匹配对照真值均值，按蛋白取中位数；
- **(c) Hoepfner HOP z 签名**：该 CMB 化合物在 HOP 筛选中的逐基因 z 分数。

基因轴对齐：我们的蛋白轴为**标准基因名**，HOP 为 **locus 系统名**，经 `gene2locus.json` 桥接，得 **5182 个共同基因**（覆盖 5243 个蛋白的 98.8%；4753 经映射 + 429 直接 locus）。一致性用 Spearman/Pearson 相关 + top-200 |z| 基因集超几何富集；MoA 结构用相关距离 + average linkage 层次聚类（z-score 化签名）。

## 2. 数据覆盖与坑（如实记录）

| 事项 | 结果 |
|---|---|
| CMB 映射化合物 | 17 个 |
| 其中 test-only（train_val 无样本） | 3 个：(S)-(+)-Camptothecin、Fluconazole、MMS → 无法构建 (a)/(b) |
| 其中无 HOP 行 | 1 个：Clotrimazole (CMB 1101) → 仅缺 (c) |
| **实际进入三签名分析** | **14 个化合物**，其中 13 个可做 HOP 对比 |
| 共同基因 | 5182 |
| 对照匹配覆盖 | 7884/7884 = 100% |
| 预测文件 | `outputs/wsT0/cache/routed_r07_band13_trainval.npy`（行序与 metadata 对齐，已校验 sample_ID） |

说明：任务设定"train split 内"构建真值签名，但 **Hydroxyurea 在 train split 有 0 个样本**（356 个全在 val split）。为不丢失该化合物并提高中位数稳健性，改用**全部 train_val 处理样本**；这也让 (a)vs(b) 包含留出 split 上的真正外样本预测，证据更强。

## 3. 三方向一致性（每化合物，5182 共同基因）

| 化合物 | MoA 类 | n | (a)×(b) Sp. | (a)×(c) Sp. | (b)×(c) Sp. | (a)×(b) Pe. |
|---|---|---|---|---|---|---|
| Hoechst 33258 | DNA binding | 115 | **0.929** | 0.025 | 0.013 | 0.898 |
| Trichostatin A | HDAC/chromatin | 119 | **0.910** | -0.042 | -0.038 | 0.840 |
| NaCl | Osmotic/salt | 349 | **0.900** | -0.106 | -0.112 | 0.818 |
| Nigericin | Ionophore (K+) | 119 | **0.867** | -0.087 | -0.077 | 0.844 |
| CHX | Protein synthesis | 327 | **0.827** | -0.048 | -0.061 | 0.658 |
| EDTA | Metal chelation | 699 | 0.744 | -0.103 | -0.102 | 0.665 |
| Valinomycin | Ionophore (K+) | 114 | 0.743 | -0.050 | -0.045 | 0.597 |
| Nocodazole | Microtubule | 117 | 0.693 | -0.049 | -0.048 | 0.617 |
| Rapamycin | TOR signaling | 346 | 0.643 | 0.000 | -0.034 | 0.523 |
| Geldanamycin | Hsp90 chaperone | 118 | 0.639 | -0.007 | -0.034 | 0.534 |
| Clotrimazole | Ergosterol/lipid | 117 | 0.637 | — | — | 0.557 |
| Anisomycin | Protein synthesis | 238 | 0.597 | -0.035 | -0.021 | 0.518 |
| SDS | Membrane/surfactant | 346 | 0.553 | -0.060 | -0.023 | 0.349 |
| Hydroxyurea | DNA synth/damage | 356 | 0.438 | -0.031 | -0.019 | 0.172 |

**汇总**：(a)×(b) 中位 Spearman **0.718**（0.438–0.929），**14/14 全部远超置换零模型**（打乱基因标签，null 均值 ~0、max ≤0.05）。(a)×(c) 中位 **-0.048**、(b)×(c) 中位 **-0.038** —— 全基因组层面≈0。

**解读**：(a)×(b) 强且一致 → 模型真实复现化合物特异蛋白组响应；(a/b)×(c)≈0 是**预期**——HOP 找的是"缺失后改变药物敏感性"的缓冲/靶点基因（适应性层），我们读出的是"蛋白丰度变化"（表达层），两层在文献中本就弱耦合。

## 4. Top-|z| 基因集重叠（top-200，超几何）

背景期望重叠 ≈ 200×200/5182 ≈ 7.7。仅 2 个化合物达显著：

| 化合物 | 重叠(pred) | 富集(pred) | p(pred) | 富集(true) | p(true) |
|---|---|---|---|---|---|
| **EDTA** | 15 | 1.94× | **0.010** | 2.07× | **0.004** |
| **Hoechst 33258** | 13 | 1.68× | **0.044** | 1.04× | 0.511 |

其余化合物富集≈1、p>0.2（不显著）。→ 丰度与适应性在基因集层面也基本正交，仅零星收敛。

## 5. MoA 聚类 vs 已知 MoA（核心对照）

对 z-score 化签名做相关距离 + average linkage 聚类。**背景相关**：预测蛋白组中位 r=0.233（共享应激/生长程序，PC1 占 ~32–40% 方差），HOP 中位 r=0.028（稀疏、特异）。

| 已知同 MoA 对 | 预测蛋白组 r | HOP 适应性 r | 是否成簇 |
|---|---|---|---|
| **Nigericin – Valinomycin**（K+ 离子载体） | **0.478** | 0.200 | **蛋白组强成簇（最亮点）**，远超 0.233 背景 |
| **Anisomycin – CHX**（蛋白合成） | 0.184 | **0.348** | 蛋白组弱（被应激程序淹没）；**HOP 干净回收（第 3 高对）** |
| Hydroxyurea – Hoechst 33258（DNA 相关） | 0.198 | -0.003 | 两层都不强（亚 MoA 不同，预期内） |

**对比即结论**：同一批化合物，HOP 适应性谱能干净分开 MoA（低背景），而蛋白组丰度响应被一套共享的应激/生长程序主导（高背景、两大反相关超级簇），只有生理效应特异的离子载体对 Nigericin/Valinomycin 突出。模型读出（丰度）与化学基因组学（适应性）**在 MoA 层面互补**，并非模型失败。

## 6. 未映射化合物 MoA 推测（假设生成，需谨慎）

全部 43 个 train_val 化合物聚类（图 fig5/fig6）。因基线相关高，"最近锚点"原始 r 普遍虚高，故按 **specificity = 锚点 r − 该化合物背景 r** 排序。生物学上**自洽**的假设（叙事素材，需外部佐证）：

| 未映射化合物 | 推测锚点（MoA） | r | specificity | 自洽性 |
|---|---|---|---|---|
| **Sorbitol** | NaCl（渗透压/盐） | 0.898 | 0.72 | ★ 教科书级正确（山梨醇即渗透胁迫） |
| Clomiphene citrate | Clotrimazole（麦角固醇/膜） | 0.843 | 0.31 | ★ 阳离子两亲药，已知扰酵母膜/麦角固醇 |
| Trifluoperazine | Clotrimazole（麦角固醇/膜） | 0.853 | 0.39 | ★ 同上 |
| Tunicamycin | Anisomycin（蛋白稳态/ER 胁迫） | 0.792 | 0.30 | ○ 均涉蛋白稳态/N-糖基化-ER 胁迫 |
| Haloperidol / Brefeldin A | Geldanamycin（Hsp90/分泌途径） | 0.94/0.92 | 0.42/0.38 | ○ ER/分泌途径-蛋白折叠 |
| Nystatin | Nigericin（膜离子载体） | 0.689 | 0.19 | ○ 多烯类成孔抗真菌，膜离子通透 |

⚠️ 反面教材（**mega-cluster 伪影，不可作 MoA 结论**）：Amphotericin B→Hydroxyurea（r=0.98）、FCCP→Hydroxyurea、Cisplatin→SDS——这些仅因同属一个强应激超级簇，MoA 实际不同。说明蛋白组层面的 MoA 迁移**只适用于生理效应相近**的情形。

## 7. 结论（三条）

1. **模型高保真复现化合物特异蛋白组签名**：(a)×(b) 中位 Spearman 0.72，14/14 化合物显著高于置换零（~0）。模型学到的是真实、化合物特异的调控信号。
2. **蛋白组丰度响应与 HOP 化学基因组学适应性是两个互补分子层**：全基因组相关≈0（预期），仅 EDTA/Hoechst 在 top-|z| 基因集弱富集。HOP 揭示"缓冲/靶点基因"，我们读出"丰度变化"，二者弱耦合符合转录-适应性解耦的已知结论。
3. **MoA 信号确实存在但层面不同**：K+ 离子载体 Nigericin/Valinomycin 在预测蛋白组强成簇（r=0.48）；蛋白合成抑制剂 Anisomycin/CHX 在 HOP 适应性层清晰成簇（r=0.35）但在蛋白组被共享应激程序（PC1 ~40%）淹没。蛋白组聚类反映"广谱生理/应激相似性"，适应性聚类反映"特异 MoA"，二者互为补充。

## 8. 产物清单

- `analysis/moa_narrative.ipynb` — 可重跑 notebook（只读缓存，单元格自包含）
- `analysis/moa_build_cache.py` — 重解析/对齐 → 缓存脚本；`analysis/make_notebook.py`（生成 ipynb）；`analysis/_run_notebook.py`（无依赖验证器）
- `analysis/cache_moa/` — `common_axis.npz`、`signatures_mapped.npz`、`signatures_all.npz`、`moa_annotations.csv`、`consistency_table.csv`、`topk_overlap.csv`、`unmapped_moa_hypotheses.csv`
- `analysis/figs/` — `fig1_consistency_bars.png`、`fig2_topk_overlap.png`、`fig3_pred_cluster.png`、`fig4_hop_cluster.png`、`fig5_all43_dendrogram.png`、`fig6_pca_all43.png`（160 dpi，英文标注）
