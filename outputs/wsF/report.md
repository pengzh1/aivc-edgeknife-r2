# wsF：显式 chem×ctx 交互 + 低秩响应程序（主攻 strain_only/both）

代码：`src/wsF_interact.py`（独立模块，复用 `src/train_mlp.py` 的 Encoder/masked_mse、`src/evaluate.py` 的 Harness、`src/wsE_depcal.py` 的 build_control_hat；未修改任何已有文件）。
合规：训练只用 `h.tr_rows`；Δ 训练目标的对照池**严格限 train split 行**（比 harness 默认更严）；val 仅评分；`h.Y_te` 零接触；种子固定（0,1,2）。

## 交付物（均已校验）

| 文件 | 形状/类型 | NaN/Inf | 说明 |
|---|---|---|---|
| `outputs/wsF/pred_trainval.npy` | (8958, 5243) float32 | 无 | 方向1最佳：ctx5L，train split 训练，3 种子均值 |
| `outputs/wsF/pred_test.npy` | (4454, 5243) float32 | 无 | 同配置全量 train_val 重训（FrozenStats 全量重估），3 种子均值 |
| `outputs/wsF/final_score.json` / `results.jsonl` | - | - | 交付评分 / 全部 22 组实验记录 |

**val composite = 0.5414**（单模型口径；参照：当前最佳集成 0.5413、wsD 8 种子 0.5381、基线 MLP 0.470、Ridge 0.461）。

## 交付模型（方向1，d1_ctx5L）score_val 全表

| split | fidelity | sample_PCC | sample_R2 | protein_PCC | FC_PCC | resid_PCC | DEP_dir_acc | DEP_PCC | DEP_F1 |
|---|---|---|---|---|---|---|---|---|---|
| val_chem_only | 0.9625 | 0.9937 | 0.9866 | 0.9073 | 0.5114 | 0.4948 | 0.8623 | 0.7180 | 0.1814 |
| val_strain_only | 0.9229 | 0.9823 | 0.9632 | 0.8231 | 0.3322 | 0.3370 | 0.8505 | 0.6744 | 0.1806 |
| val_both | 0.9345 | 0.9834 | 0.9658 | 0.8543 | 0.2341 | - | 0.8047 | 0.5899 | 0.1249 |
| val_time | 0.9616 | 0.9940 | 0.9863 | 0.9044 | 0.6355 | - | 0.9070 | 0.8010 | 0.2905 |

**[composite] 0.5414**

配置：`(1024,2048,2048,2048,2048)` 5 层 GELU+LayerNorm，Huber，300ep cosine，dropout 0.3，emb_drop 0.35，bs 256，lr 1e-3，wd 1e-4；显式交互通路 `chem_emb(32d) ⊙ W_ctx·[medium,temp,time]`（8→32 双线性）；**unk_unseen**：推理期把 train 行未出现的类别（BAI 菌株、6 个 val 专属化合物等）映射到 UNK(0)。

## 两方向对比

### 方向1：显式交互 MLP（3 种子均值预测，完整评分）

| 配置 | composite | strain FC | strain resid | both FC | chem resid | time FC |
|---|---|---|---|---|---|---|
| base3L（无交互，对照） | 0.5402 | 0.3292 | 0.3332 | 0.2329 | 0.4954 | 0.6337 |
| base3L + nounk（消融 unk_unseen） | 0.5356 | 0.3255 | 0.3329 | 0.2207 | 0.4829 | 0.6337 |
| ctx3L | 0.5405 | 0.3293 | 0.3335 | 0.2328 | 0.4956 | 0.6352 |
| ctx+strain 3L | 0.5403 | 0.3288 | 0.3330 | 0.2330 | 0.4957 | 0.6351 |
| **ctx5L（交付）** | **0.5414** | 0.3322 | 0.3370 | 0.2341 | 0.4948 | 0.6355 |
| ctx3L + aux(β=0.5, fc+ctx+drug) | 0.5388 | 0.3349 | 0.3377 | 0.2376 | 0.4783 | 0.6332 |
| ctx5L + aux(β=0.5, fc+drug) | 0.5399 | 0.3374 | 0.3404 | 0.2385 | 0.4786 | 0.6347 |
| ctx5L + aux(β=1.0, fc+drug) | 0.5398 | 0.3377 | 0.3406 | 0.2389 | 0.4773 | 0.6360 |
| ctx5L + strain_blind | 0.4605 | 0.3323 | 0.3370 | 0.2337 | 0.2728 | 0.4288 |

要点：
1. **unk_unseen 是最大单项增益**（+0.0046 composite；both FC 0.2207→0.2329）。Encoder 给 train_val 全类别分配索引，但 val 专属类别的嵌入从未获梯度（保持随机初始化）；emb_drop 训练的是 UNK(0) 回退。推理时把未见类别显式映射到 UNK 严格更优。现有管线（wsD 等）未做此映射，可直接迁移此 trick。
2. **显式交互通路本身无显著增益**（±0.001）：chem×ctx、chem×strain 在 3L/5L 上都被深网隐式学到（与 wsD 的 FiLM/残差结论一致）。
3. **aux_delta**（自研指标对齐辅助损失：对评分口径 Δ̂=ŷ−control_hat 做逐样本中心化，直接优化 FC/ctx_resid/drug_resid 三个目标）：strain resid +0.003~0.007、both FC +0.004~0.005、DEP F1 +0.03，但 chem resid −0.016（20% 权重），net composite −0.0015。固有权衡，未交付。
4. **strain_blind**（训练时 strain 恒 UNK 的纯 chem×ctx 模型）在未见菌株上**不优于** strain-aware 模型（0.3403 vs 0.3404），其余划分崩坏——strain-aware + UNK 回退已是最优，无需路由。

### 方向2：低秩响应程序（3 种子均值，完整评分）

| 配置 | composite | strain FC | strain resid | both FC | chem resid | time FC |
|---|---|---|---|---|---|---|
| L16S（rank16 含strain） | 0.3690 | 0.2098 | 0.2140 | 0.1578 | 0.2479 | 0.3472 |
| L32S | 0.3796 | 0.2186 | 0.2197 | 0.1596 | 0.2616 | 0.3906 |
| L64S | 0.3902 | 0.2235 | 0.2233 | 0.1610 | 0.2797 | 0.4345 |
| L32N（不含strain） | 0.3709 | 0.2191 | 0.2202 | 0.1599 | 0.2456 | 0.3523 |
| L64N | 0.3794 | 0.2241 | 0.2237 | 0.1615 | 0.2591 | 0.3880 |
| L64 路由（未见菌株→N版） | 0.3904 | 0.2241 | 0.2237 | 0.1615 | 0.2797 | 0.4345 |

**方向2 明确失败**，三个叠加原因：
1. **锚点保真度崩塌**：ŷ = control_hat + Δ̂ 中，未见菌株（BAI）行的 control_hat 只能回退到 3 键（Medium×Temp×time 跨菌株对照均值），strain_only fidelity 0.70（protein_PCC 0.26）vs MLP 0.92/0.82。MLP 通过 plate/instrument 嵌入学到了查表法无法提供的逐样本基线变化。
2. **低秩天花板**：train 处理样本（5066 行，对照池严格 train split）带缺失掩膜岭最小二乘重构的 in-sample PCC 仅 0.444/0.528/0.608（rank 16/32/64）——Δ 的逐样本模式大半是噪声/秩外成分。
3. **可迁移性更差**：权重预测器（条件→32 维 w）学到的跨上下文调制弱于直接 MLP（strain resid 0.223 vs 0.337）。含/不含 strain、路由、混合均无差异（strain 输入对未见菌株本就无用）。

### 方向判定：**交付方向1**（0.5414 vs 0.3902，差距 0.15）

## strain/both 目标分析（目标 strain resid ≥0.37、both FC ≥0.25）

未达标，最佳观测 strain resid 0.3406（auxFD10_5L）/ 0.3412（与集成混合），both FC 0.2389。多方法证据指向**信号天花板**而非模型不足：

| 方法（strain_only resid） | 值 |
|---|---|
| μ_ctx 参照预测 | 0.118 |
| 双向加性统计估计 E[Δ\|chem,ctx]（train 4 菌株 cell 均值，含 strain 主效应修正） | 0.136 |
| 低秩响应程序 rank64 | 0.223 |
| 直接 MLP（交付） | 0.337 |
| MLP + 指标对齐辅助损失 | 0.341 |
| 与当前最佳集成 50/50 混合 | 0.341 |

朴素 chem×ctx cell 均值只有 0.136——MLP 已从中蒸馏出远超统计估计的可迁移结构；辅助损失直接把 strain resid 当目标优化也只能再挤 +0.003。未见菌株的菌株特异响应按定义不可迁移，残差中可解释的 chem×ctx 调制部分约占总模式方差的 1/3 上限。both FC 同理（0.234 交付 / 0.239 aux 版，未达 0.25）。

## 是否建议并入集成：**建议**

`0.5·ensemble_best + 0.5·wsF_ctx5L` → **composite 0.5430**（vs 集成单独 0.5413，+0.0017），strain resid 0.3412、chem resid 0.4936、time FC 0.6390 均不降。wsF 与 wsD/wsB 架构血缘近但 unk_unseen 处理不同、误差相关性低，混合稳定获益。精细权重可由集成搜索决定（粗网格显示 0.5 附近平坦）。

## 复现

```bash
python -m src.wsF_interact d1 --name ctx5L --cfg '{"inter_ctx":true,"inter_strain":false,"hidden":[1024,2048,2048,2048,2048]}' --seeds 0,1,2
python -m src.wsF_interact deliver --src outputs/wsF/cache/d1_ctx5L_mean.npy
python -m src.wsF_interact d1test --name ctx5L_full --cfg '{"inter_ctx":true,"inter_strain":false,"hidden":[1024,2048,2048,2048,2048]}' --seeds 0,1,2
```
（d2prep/d2/d2route/blend/d1route 各 stage 见模块 docstring；全部实验记录在 `results.jsonl`。）
