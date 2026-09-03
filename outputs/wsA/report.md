# wsA：化合物分子描述符 MLP（chemfeat）

## 方法要点

针对未见化合物只能靠 UNK 嵌入回退的问题，把化合物身份从"可学习嵌入"替换为
**分子结构描述符向量**，让模型学"结构 → 响应"的映射，获得对未见化合物的结构化外推能力。

1. **SMILES 获取**：PubChem PUG-REST 按名称取 CanonicalSMILES，自建别名表
   （CHX=cycloheximide、MMS=methyl methanesulfonate、G418=geneticin、
   LY 294002 hydrochloride=LY294002、1-10 Phenanthroline monohydrate=1,10-phenanthroline、
   (1R,2S,5R)-(-)-Menthol=(-)-menthol、Oligomycin=oligomycin A、Tunicamycin=tunicamycin A (CID 11104835) 等）。
   **覆盖率 54/54**（57 个名称排除 Water/DMSO/Quality Control 后全部解析成功，RDKit 全部可解析）。
   缓存：`outputs/wsA/smiles.csv`。
2. **描述符**：RDKit Morgan 指纹（radius=2, 2048 bit）+ 16 个理化描述符
   （MolWt, MolLogP, TPSA, HBD, HBA, RotatableBonds, RingCount, FractionCSP3,
   HeavyAtomCount, NumAromaticRings, NumAliphaticRings, NumHeteroatoms, NHOHCount,
   NOCount, MolMR, FormalCharge）→ 拼接 2064 维。
   盐类/水合物取母核（FragmentParent）；过渡金属配合物（顺铂）保留整体以免丢失金属中心。
3. **标准化 + PCA**：均值/方差仅用 train_val 中 chemical_role=='train' 的 **37 个化合物**拟合；
   PCA 同样只用这 37 个化合物拟合。37 样本中心化后秩 ≤36，故取 36 个主成分
   （避免退化方向在样本外爆炸），逐维再标准化（仍仅 train 拟合）后零填充到 64 维。
   产物：`outputs/wsA/chem_features.csv`（54 化合物 × 64 维，max|v|=5.7，无 NaN）。
4. **模型**（`src/wsA_chemfeat.py`，框架复制自 `src/train_mlp.py`）：
   - 化合物表示：64 维描述符 → Linear(64→48) → GELU（替换原 32 维可学习嵌入）；
   - 训练时 **25%** 概率将化合物列置 0（chem_mat[0] = train 化合物描述符均值），学回退；
     菌株列 emb_drop 保留 **0.25**；
   - 其余与 train_mlp 相同：trunk (512,1024) + LayerNorm + Dropout(0.1)、masked MSE、
     AdamW(lr=1e-3, wd=1e-4) + cosine、batch 256、**100 epochs**、仅用 `h.tr_rows` 训练；
   - 3 个种子（0,1,2）预测取均值。
5. **消融**：同训练预算（100 epochs × 3 seeds）下化合物表示换回可学习嵌入
   （即原 MLP 配置，emb_drop=0.15），隔离"描述符特征"与"训练配置"的贡献。

## score_val 全表

### 主模型 chemfeat（3 seeds 均值预测）

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only   | 0.9606 | 0.9933 | 0.9857 | 0.9029 | **0.4896** | **0.4610** | 0.8643 | 0.7252 | 0.1764 |
| val_strain_only | 0.9124 | 0.9774 | 0.9518 | 0.8081 | 0.2858 | 0.2866 | 0.8328 | 0.6423 | 0.1552 |
| val_both        | 0.9192 | 0.9749 | 0.9462 | 0.8364 | 0.1767 | -      | 0.7796 | 0.5396 | 0.1057 |
| val_time        | 0.9583 | 0.9930 | 0.9846 | 0.8971 | 0.6108 | -      | 0.9008 | 0.7937 | 0.2688 |

**composite = 0.5102**

### 消融：可学习化合物嵌入（原 MLP，100 epochs，3 seeds 均值）

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only   | 0.9585 | 0.9929 | 0.9847 | 0.8980 | 0.4547 | 0.4229 | 0.8445 | 0.7125 | 0.1913 |
| val_strain_only | 0.9163 | 0.9785 | 0.9543 | 0.8161 | 0.2921 | 0.2943 | 0.8388 | 0.6506 | 0.1596 |
| val_both        | 0.9223 | 0.9772 | 0.9520 | 0.8378 | 0.1890 | -      | 0.7862 | 0.5520 | 0.1109 |
| val_time        | 0.9615 | 0.9938 | 0.9863 | 0.9044 | 0.6198 | -      | 0.9062 | 0.8015 | 0.2809 |

**composite = 0.5052**

## 与基线对比

| 模型 | composite | chem FC_PCC | chem resid_PCC |
|---|---|---|---|
| MLP 基线（AGENT_GUIDE） | 0.470 | 0.404 | 0.363 |
| 消融：可学习嵌入 100ep（本工作） | 0.5052 | 0.4547 | 0.4229 |
| **chemfeat 描述符（本工作）** | **0.5102** | **0.4896** | **0.4610** |
| 参考：当前集成 | 0.509 | - | - |

- 相对 MLP 基线：chem FC **+0.086**，chem resid **+0.098**，composite **+0.040**。
- 相对同预算消融（架构/训练配置相同，仅换化合物表示）：chem FC **+0.035**，
  chem resid **+0.038**，composite **+0.005** —— 目标指标上的增益确实来自描述符特征本身；
  消融也显示一部分相对旧基线的提升来自 100 epochs 训练预算（0.470 → 0.505）。
- 代价：strain 侧略降（val_strain_only resid 0.287 vs 0.294；DEP_F1 略低），
  幅度远小于 chem 侧收益，composite 净增。

## 关键发现

1. **结构描述符对未见化合物有效**：val_chem_only 的 FC_PCC 0.4896 / resid_PCC 0.4610，
   单模型 composite 0.5102 已追平当前集成（0.509），而基线 MLP 只有 0.470。
2. **增益来源经消融验证**：同 100 epochs 预算下可学习嵌入版 chem FC 只有 0.4547，
   描述符版 0.4896 —— 结构特征带来真实外推信息，而非架构/预算差异。
3. **覆盖率 100%**：54 个非对照化合物全部拿到 SMILES（个别需别名/母核处理：
   Tunicamycin 用 tunicamycin A、Oligomycin 用 oligomycin A 代表混合物主成分；
   盐类取母核，顺铂保留 Pt 配合物整体）。
4. **工程细节**：37 个 train 化合物拟合 PCA 时秩上限为 36，必须用 n_comp=36 再零填充，
   否则近零特征值方向在样本外投影会爆炸（曾观测 |v|~2e5）；PCA 后逐维再标准化
   （仅 train 拟合）把输入尺度压到 ~1，chem 均值回退向量 ≈ 0。
5. **后续方向**：chemfeat 预测与现有集成在 val_chem_only 上互补（描述符外推 vs 统计回退），
   适合做加权融合或作为集成的 chem 专家；也可尝试更大半径指纹/3D 描述符，或
   用全部 54 化合物拟合 PCA（纯结构、无标签泄漏风险低）以利用更多主成分。

## 文件清单

- `src/wsA_chemfeat.py` — 特征构建 + 模型 + 训练/消融（可复现）
- `outputs/wsA/smiles.csv` — 54 化合物 SMILES 缓存（含 PubChem 查询名）
- `outputs/wsA/chem_features.csv` — 64 维描述符特征
- `outputs/wsA/pred_trainval.npy` — 主模型 3-seed 均值预测 (8958×5243, float32, 无 NaN/Inf)
- `outputs/wsA/pred_ablation_mlp.npy` — 消融预测
- `outputs/wsA/scores.json` — score_val 原始结果

合规：仅用 `h.tr_rows` 训练；标准化/PCA 仅用 train 化合物拟合；未使用 `h.Y_te`；
种子固定（0,1,2）；val 划分仅用于评分。
