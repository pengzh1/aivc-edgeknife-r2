# wsB 两阶段建模报告：control_hat(上下文+批次) + Δ̂(扰动效应)

## 方法要点

- **Stage 1 control_hat**：仅 train split 对照样本（DMSO/Water，751 个；任务书中 956 为 train_val 总数，按红线只用 train split）。两个互补模型按权重融合：
  - 条件嵌入 MLP（菌株/培养基/温度/时间/仪器/来源/板号，无化合物输入；512-1024，菌株 UNK+25% embedding dropout，masked MSE，100 epochs，3 种子均值）；
  - 分层组均值（5 键→逐蛋白级联回退→菌株→全局，对照版 DeltaAdditive 控制端）。
- **Stage 2 Δ̂**：train split 处理样本（5078 个），目标 = `h.delta_tr_all` 对应行（raw log2 Δ，masked MSE，不标准化以保持蛋白间方差权重与 FC_PCC 口径一致）。输入 = 菌株(UNK+25% drop)+化合物(UNK+25% drop)+培养基/温度/时间/仪器/来源/板号嵌入，MLP 512-1024，100 epochs，3 种子均值。
- **合成**：处理样本 ŷ = control_hat + Δ̂；对照样本 ŷ = control_hat；QC 样本 = train QC 按 (仪器×来源)→仪器→全局 级联组均值。
- **编码器差异**：encoder 只在各模型训练行上 fit，val 独有菌株/化合物/板号 → UNK(0)，不用未训练的随机嵌入（与 src/train_mlp.py 的差异，属有意改进）。
- **合规说明**：185 个 train 处理样本的 Δ_true 匹配对照落在 val 划分（harness 自带冻结参照 μ_ctx/μ_drug 与 DeltaAdditive 基线同口径，且 `h.delta_tr_all` 为 AGENT_GUIDE 明示接口）。pred_trainval.npy 为非锚定版，无 NaN。


**融合权重选择（一次性模型选择）**：w*=1.0，composite=0.5306


## 主提交（非锚定）score_val 全表

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only | 0.9543 | 0.9925 | 0.9844 | 0.8859 | 0.4839 | 0.4410 | 0.8568 | 0.7138 | 0.2520 |
| val_strain_only | 0.9215 | 0.9836 | 0.9658 | 0.8150 | 0.3527 | 0.3523 | 0.8561 | 0.6919 | 0.1958 |
| val_both | 0.9338 | 0.9834 | 0.9657 | 0.8524 | 0.2427 | - | 0.8087 | 0.5983 | 0.1313 |
| val_time | 0.9544 | 0.9928 | 0.9841 | 0.8864 | 0.5895 | - | 0.8889 | 0.7830 | 0.3395 |


**composite = 0.5306**


## 融合权重对比（control_hat = w·MLP + (1−w)·组均值，Δ̂ 相同）

| w | composite | FC_PCC 均值 |
|---|---|---|
| w=1.0 | 0.5306 | 0.4172 |
| w=0.5 | 0.5110 | 0.3967 |
| w=0.0 | 0.4248 | 0.3370 |


## 与基线对比

| 模型 | split | fidelity | FC_PCC | resid_PCC | composite |
|---|---|---|---|---|---|
| Ridge | val_chem_only | 0.9427 | 0.3385 | 0.2834 | 0.4608 |
| Ridge | val_strain_only | 0.9168 | 0.3159 | 0.3056 | 0.4608 |
| Ridge | val_both | 0.9296 | 0.2453 | - | 0.4608 |
| Ridge | val_time | 0.9412 | 0.4401 | - | 0.4608 |
| MLP-3seed | val_chem_only | 0.9588 | 0.4581 | 0.4267 | 0.5056 |
| MLP-3seed | val_strain_only | 0.9162 | 0.2914 | 0.2950 | 0.5056 |
| MLP-3seed | val_both | 0.9220 | 0.1867 | - | 0.5056 |
| MLP-3seed | val_time | 0.9612 | 0.6151 | - | 0.5056 |
| wsB-twostage | val_chem_only | 0.9543 | 0.4839 | 0.4410 | 0.5306 |
| wsB-twostage | val_strain_only | 0.9215 | 0.3527 | 0.3523 | 0.5306 |
| wsB-twostage | val_both | 0.9338 | 0.2427 | - | 0.5306 |
| wsB-twostage | val_time | 0.9544 | 0.5895 | - | 0.5306 |


## 诊断（不进提交）：锚定真实匹配对照的上界

把 val 处理样本的 ŷ 换成「匹配对照真值均值 + Δ̂」（对照取自 train_val 池，含 val 划分自身的对照），其余行不变。此时评分侧 Δ_pred ≡ Δ̂，用于量化『若组委会允许用 test 对照原始值锚定』的提分上限，**仅供向组委会提问参考，不可用于提交**。

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only | 0.9567 | 0.9930 | 0.9854 | 0.8916 | 0.4384 | 0.4271 | 0.8247 | 0.6547 | 0.1114 |
| val_strain_only | 0.9263 | 0.9892 | 0.9770 | 0.8126 | 0.2627 | 0.2391 | 0.6660 | 0.3638 | 0.1561 |
| val_both | 0.9432 | 0.9905 | 0.9802 | 0.8589 | 0.1194 | - | 0.5835 | 0.1672 | 0.0416 |
| val_time | 0.9557 | 0.9930 | 0.9846 | 0.8894 | 0.5828 | - | 0.8731 | 0.7569 | 0.2279 |


**锚定上界 composite = 0.4818**（非锚定 0.5306，差值 -0.0489）


### 逐划分 FC/resid 对比（非锚定 → 锚定）

| split | FC_PCC 非锚定 | FC_PCC 锚定 | resid_PCC 非锚定 | resid_PCC 锚定 |
|---|---|---|---|---|
| val_chem_only | 0.4839 | 0.4384 | 0.4410 | 0.4271 |
| val_strain_only | 0.3527 | 0.2627 | 0.3523 | 0.2391 |
| val_both | 0.2427 | 0.1194 | - | - |
| val_time | 0.5895 | 0.5828 | - | - |


## 关键发现

- **Δ 解耦有效**：composite 0.5306，显著超过 Ridge 0.4608 / MLP-3seed 0.5056（及指南所载集成 0.509）。val_strain_only FC 0.3527（Ridge 0.3159 / MLP 0.2914）、resid 0.3523（MLP 0.2950）；val_both FC 0.2427（MLP 0.1867，Ridge 0.2453 持平）；val_chem_only FC 0.4839 / resid 0.4410 均为最优；val_time FC 0.5895 略低于 MLP 基线 0.6151（时间外推上直接绝对值预测略占优）。
- **锚定诊断的反直觉结论（重要）**：锚定版 composite 0.4818 **低于**非锚定版 0.5306——"用 test 对照原始值锚定"不会提分，无需向组委会争取。机制：评分侧 Δ_true = y_treat − 对照真值含有对照重复测量噪声（负号）；非锚定版 Δ_pred = Δ̂ + (control_hat − 对照真值) 中的 (control_hat − 对照真值) ≈ −对照噪声，与 Δ_true 的 −对照噪声分量正相关，抬高逐样本 FC_PCC；锚定版把该分量完全消去（Δ_pred ≡ Δ̂），FC/resid/DEP 全面下降。fidelity 则相反（锚定版略高，绝对空间受益于真实对照）。此机制在官方 test 评分（同样对匹配对照求 Δ）下同样成立，非本地 val 伪影。
- **control_hat 选型**：纯组均值（w=0）在 val_strain_only/val_both 上蛋白间结构崩塌（protein_PCC 0.069/−0.071，未见菌株只能回退全局均值）；MLP 的 UNK 菌株嵌入回退显著更优，w=1.0（纯 MLP）最佳，提交采用之。
- 板号在 val 处理样本中 100% 见于 train（对照端亦 100%），批次效应可转移；val_strain_only/val_both 的菌株在 train 对照与 train 处理样本中均完全缺失（0% 键命中），只能依赖 UNK 回退，是这两个划分 FC 偏低的主因。
