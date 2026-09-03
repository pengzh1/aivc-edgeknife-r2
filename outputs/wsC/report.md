# wsC 报告：连续时间建模 + QC/批次校正（MLP 框架）

代码：`src/wsC_timebatch.py`（复用 `src/train_mlp.py` 的 Encoder/训练循环框架，未修改任何已有文件）。
交付：`outputs/wsC/pred_trainval.npy`，float32 (8958,5243)，无 NaN/Inf，模型仅用 train split 训练。

## 最终模型（FINAL，24 个 MLP 的等权集成）

三个时间表示组 × 8 种子（seeds 0-7）× tail-checkpoint 预测平均（最后 30% 每 10 epoch 取一次），
三组预测等权平均：
- **G1 onehot**：pert_time 6 档嵌入（原基线表示）
- **G2 both+interact**：one-hot + 连续时间基（log2(t/15) 归一 + 5 中心 RBF）+ 化合物/菌株嵌入 × 时间基双线性外积
- **G3 cont**：仅连续时间基（去掉 time one-hot 嵌入）

公共超参：hidden (512,1024)，emb_drop 0.25（仅菌株/化合物列），epochs 150，AdamW lr 1e-3 wd 1e-4，bs 256，cosine。

### score_val 全表（最终集成）

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only | 0.9608 | 0.9933 | 0.9857 | 0.9035 | 0.4785 | 0.4420 | 0.8608 | 0.7269 | 0.1843 |
| val_strain_only | 0.9252 | 0.9824 | 0.9632 | 0.8300 | 0.3324 | 0.3287 | 0.8530 | 0.6811 | 0.1792 |
| val_both | 0.9323 | 0.9828 | 0.9644 | 0.8499 | 0.2274 | - | 0.8016 | 0.5814 | 0.1252 |
| val_time | 0.9621 | 0.9939 | 0.9864 | 0.9059 | **0.6308** | - | 0.9084 | 0.8076 | 0.2765 |

**composite = 0.5258**（目标 >=0.48；基线 MLP 单模 0.470；现有 3 种子集成 0.5056）
**val_time FC = 0.6308**（目标 >=0.62；基线单模 0.591；现有集成 0.6151）

## 子方向 1 消融：连续时间 vs one-hot

单种子（epochs 100，emb_drop 0.25，同配置仅改时间表示）：

| 变体 | composite | val_time FC | 说明 |
|---|---|---|---|
| onehot（基线） | 0.4760 | 0.5919 | 与 src.train_mlp 一致 |
| cont（RBF 基） | 0.4744 | 0.5914 | 去掉 time 嵌入，只用连续基 |
| cont（poly3 基） | 0.4723 | 0.5877 | 基函数形式无影响 |
| cont + 化合物x时间交互 | 0.4673 | 0.5887 | 双线性外积 appended |
| both + 化合物/菌株x时间交互 | 0.4608 | 0.5928 | one-hot+连续基+交互 |

集成后（epochs 150 + tail 平均）：

| 组 | 5 种子 comp / vtFC | 8 种子 comp / vtFC |
|---|---|---|
| G1 onehot | 0.5135 / 0.6248 | 0.5245 / 0.6269 |
| G2 both+interact | 0.5089 / 0.6253 | 0.5195 / 0.6266 |
| G3 cont | 0.5132 / 0.6230 | 0.5232 / 0.6259 |
| **G1+G2+G3 等权** | 0.5164 / 0.6303 | **0.5258 / 0.6308** |

**消融结论（时间）**：
1. 单模型上连续时间基与 one-hot 完全打平（0.474-0.476 vs 0.476），双线性交互单独使用反而略降 composite（对 chem/strain 残差划分有轻微负作用），val_time FC 不变。
2. 但**时间表示多样性对集成有真实增益**：三组等权混合把 val_time FC 从单一表示的 0.625-0.627 推到 0.6308，composite +0.001-0.006。连续时间变体的价值体现在集成多样性，而非单模精度。
3. 其他划分未因时间表示改变而明显变差（各组 fidelity/FC 差异 <0.005）。

**关键诊断（解释为何时间建模收益有限）**：val_time 的 139 个处理行中 52 行其 (菌株x化合物x培养基x温度x时间) 五键组合在 train 中有**另一 data_source 批次**的完全相同样本——这些 train Δ 与 val Δ 的逐样本 PCC 仅 **0.176**；而 MLP 集成在同一批行上达 0.63。即响应的"批次特异成分"巨大且不可预测，模型只能靠多学次平均逼近"全局共享响应型"，这构成 val_time FC 的信息上限。自身时间曲线的检索式插值（同四键其他时刻 Δ 加权平均）PCC 也仅 0.256，远低于模型——所以"平滑时间曲线"先验能提供的额外信息很少，这解释了连续时间单模无增益的现象。

## 子方向 2 消融：QC/批次校正

估计（仅用 train split 的 91 个 QC 样本，覆盖 70/144 板）：
plate_effect = mean(QC_板) − mean(QC_全部train)，按有效观测数收缩 n/(n+2)；
缺失板回退到同 instrument 板效应均值（x0.5 收缩），再回退 0。
QC 板效应量级可观（跨板 per-protein std≈0.32，同 instrument 板效应相关 0.35）。

| 用法 | composite | val_time FC | 结论 |
|---|---|---|---|
| 无校正（最终集成 α=0） | **0.5258** | **0.6308** | |
| post-hoc ŷ+0.25·effect | 0.5247 | 0.6242 | 单调变差 |
| post-hoc ŷ+0.50·effect | 0.5199 | 0.6046 | |
| post-hoc ŷ+1.00·effect | 0.5044 | 0.5480 | |
| target 模式（训练目标去板效应+预测加回，单模） | 0.4747 | 0.5903 | ≈无校正单模 0.4760/0.5919 |
| feature 模式（板效应 PCA 得分作输入，单模） | 0.4668 | 0.5902 | 略差 |

在旧 3 种子集成（无本任务改进）上同样单调变差：α=0→1 composite 0.5056→0.4857，mean sample_R2 0.972→0.970（详见 `qc_post_ablation_*.json`）。

**消融结论（QC）**：对该 MLP，任何形式的 QC 板效应校正（后验加减/目标校正/特征注入）都是中性到有害。
原因：基线 MLP 的 **Yeast_cell_plate 嵌入（32 维）已经吸收了板效应**——每板约 62 个 train 样本提供的信息远多于 1-2 个 QC；且全部 val/test 板都在 train 中出现，板嵌入可直接迁移。实测模型逐板平均残差与 QC 板效应向量基本不相关（train 上 |corr|<0.1），证明板效应已被学到。QC 校正只对"无板感知能力"的模型（如 Ridge/全局均值）或有全新板的场景才有价值。

## 与基线对比

| 模型 | composite | val_time FC |
|---|---|---|
| Ridge | 0.461 | - |
| MLP 单模（对照基线） | 0.470 | 0.591 |
| 现有 3 种子集成 | 0.5056 | 0.6151 |
| **wsC 最终（24 模型三时间表示集成）** | **0.5258** | **0.6308** |

提升来源分解：epochs 150+emb_drop 0.25 调参（单模 0.470→0.477）、5→8 种子 + tail 平均（→0.524）、三时间表示混合（→0.5258，val_time FC 0.6269→0.6308）。

## 合规

- 训练仅用 `h.tr_rows`；QC 统计、FrozenStats、PCA 全部来自 train split；val 仅用于 score_val 评分与模型选择；未接触 `h.Y_te` 任何数值；种子固定（0-7）。
- 复现：`python -m src.wsC_timebatch run --tag E1 --time_mode onehot --epochs 150 --emb_drop 0.25 --tail_avg --seeds 0,1,2,3,4,5,6,7`（G2 用 `--time_mode both --interact chemstrain`，G3 用 `--time_mode cont`），三组预测等权平均即得 `pred_trainval.npy`。
