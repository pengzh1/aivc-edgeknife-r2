# wsT0：DEP 选择器前置快检报告（T1.4 跨族分歧 × DEP 漏检）

**日期**：2026-08-25 晚 → 08-26 凌晨 | **模块**：`src/wsT0_varcheck.py` | **产出**：`outputs/wsT0/varcheck.json`、`outputs/wsT0/cache/`

## 裁决：GREEN → 上 DEP 选择器（wsT1_depgate）

预注册规则：train Q3（边际带内 FN 判别 AUC）≥0.60 或 Q4≥Q1+0.02，且 val 同向。
实测 train Q3=**0.7702**，val Q3 均值=**0.6522**（≥0.58 绝对条款）→ GREEN。

## 基线锚定（复现确认）

| 管线 | composite | DEP_F1 均值 |
|---|---|---|
| 21 族路由 r=0.7（重建） | 0.5528 | 0.1985 |
| + band γ=1.3（现行最终管线 trainval 侧） | **0.5536** | **0.2305** |

与 handoff 8.15 的 0.5540 一致（尾差来自 CRD val 代理，与本检无关）。
train 处理行 5,078，可用条目 18.8M，DEP 基准率 **2.93%**，现行预测过阈率仅
**1.50%**——召回饥饿确诊（检出量只有真值的一半）。

## 核心发现

| 信号 | train AUC（→\|Δ_true\|>1） | val chem | val strain | val both | val time |
|---|---|---|---|---|---|
| \|Δ̂_ens\|（现行唯一依据） | 0.9534 | 0.8051 | 0.6127 | 0.5430 | 0.8902 |
| **跨族 std** | 0.9210 | **0.8226** | **0.6940** | **0.6431** | 0.8686 |
| 蛋白级 train DEP 率 | 0.7791 | 0.8009 | **0.8047** | **0.7985** | 0.7734 |
| 三特征 logistic（样本组 CV） | 0.9675（Q1+0.014） | — | — | — | — |
| **边际带内 std → FN**（Q3） | **0.7702** | 0.6618 | 0.6427 | 0.6209 | 0.6835 |

三个超出预期的结论：

1. **跨族 std 在 OOD 划分上反超 |Δ̂_ens|**：strain_only 0.694 vs 0.613、
   both 0.643 vs 0.543。模型在未见菌株上的 |Δ̂| 排序能力塌缩时，21 族的
   分歧程度仍保留了 DEP 可分性——这正是 band 结构够不到的 strain/both
   两划分（F1≈0.13）的可攻击面。
2. **蛋白级 train DEP 率全划分稳态 ~0.80**：蛋白身份先验（该蛋白在 train
   里多常成为 DEP）是最抗 OOD 的特征，与"4 训练菌株瓶颈"判语一致——
   蛋白轴统计不依赖菌株轴泛化。
3. 边际带（|Δ̂|∈[0.3,1)）内 FN 率 9.24%，std 判别 AUC train 0.77 /
   val 0.65——选择器在此带有真实可分信息，不是纯噪声赌博。

## 缓存（wsT1 及后续直接复用）

- `routed_r07_trainval.npy` — 21 族 r=0.7 路由基线（8958×5243）
- `routed_r07_band13_trainval.npy` — +band γ=1.3（现行管线）
- `x_std.npy` / `x_std_w.npy` — 跨族标准差（无权/w_global 加权）
- `control_hat.npy` / `control_level.npy` — wsE 分层对照锚点重建（wsE cache
  的 npy 未随报告留存，已按 `src/wsE_depcal.py::build_control_hat` 原样重建）
- `prot_dep_rate.npy` — 蛋白级 train DEP 率

## 合规

特征/统计全部 train-only；Δ_true 口径同 wsE 调参惯例（train_val 对照池）；
val 仅本报告一张表用于路线裁决；Y_te 零接触；未修改任何既有文件。
