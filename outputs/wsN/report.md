# wsN 优化轮报告（2026-08-11，出题人分享会驱动）

> 输入：outref/meeting.txt（周梓卓答疑）→ outref/meeting_结论总结.md。
> 结论：**全部 7 条探索路径均为零/负结果，提交不变**（`outputs/prediction_trainonly*.csv`）。
> 所有实验 train-only 训练、val 模型选择、Y_te 零接触。

## 结果总览

| # | 实验（模块） | 结果 | 结论 |
|---|---|---|---|
| N1 | 化合物近邻 Δ 迁移（Morgan Tanimoto kNN，`wsN_transfer.py`） | 单族 composite 0.328；chem FC 0.097 | **强负结果**。深模型的 UNK 回退（chem FC 0.51）远胜结构相似度 Δ 迁移；"深模型打不过统计模型"不适用于本管线强度 |
| N2 | 菌株近邻 Δ 迁移（1011 SNP，CRD←CGD 0.398%，BAI←CEK） | nn1 ≈ none（0.3283 vs 0.3244 单族） | **S2 信息上限第六路独立证据**：即便遗传最近邻，菌株特异 Δ 调制也不迁移 |
| N3 | 高变化蛋白 loss 加权（`wsN3_hubw.py`，sqrt/linear 两档 × 3 种子） | 0.5427 / 0.5426 vs 对照 0.5423 | **零结果**（噪声内）；Δstd 加权不改变学习内容；DEP_F1 无实质变化 |
| N4 | wsD + tail-checkpoint 平均（`wsN4_tailavg.py`，8 种子） | 0.5427 vs wsD 基线 0.5431 | **零结果**；cosine 调度尾部已收敛，快照近乎相同 |
| N5 | DEP 校准 γ 分划分/层级放宽 | γ=1.3 统一 level-0 已最优（0.5492）；level 1-2 无命中行、level 6 反降 | **路径穷尽**；未见菌株行无 train 对照（7 键），结构性不可校准 |
| N6 | 均匀 Δ 缩放 post-hoc（chem ×1.2/×1.5、strain/both ×0.8 等） | 全部 ≤ 0.5462 < 0.5492 | **负结果**：现行收缩是 MSE 最优偏差-方差权衡；PCC 尺度不变、保真度受损 |
| N7 | wsN 入路由闭环 | chem 0.001 / strain 0.000 / time 0.000 / both 0.080（收缩 0.040）；fast composite 0.5473 < 0.5486 | **路由拒绝**，正式关闭 |

## 关键诊断（val 路由预测）

| 划分 | Δ_pred/Δ_true 行均 std | \|Δ\|>1 幅度恢复 |
|---|---|---|
| chem_only | 0.642（收缩） | 0.313 |
| strain_only | 1.348（过胀） | 0.594 |
| both | 1.320（过胀） | 0.438 |
| time | 0.718（收缩） | 0.516 |

解读：chem/time 的 Δ 收缩是 MSE 意义最优（N6 证明均匀拉伸反而掉分）；strain/both 的 std 过胀+幅度恢复不足 = 预测在错误位置产生中等幅度 Δ（噪声），非校准可治——与 S2 上限互证。

## 对复赛的意义

- 初赛层面模型侧已平台期：8.8（wsJ/K/L）+ 8.11（wsN×7）两轮共 10+ 独立路径全部零/负；
- 剩余上行空间在需要新信息源的方向：基因组/蛋白预训练 embedding（DNABERT-2/ESM-2 级，出题人亦推荐 CPA/ESM-2）、机理层评估叙事；
- 出题人在说明中明确：隐藏评估看**高变化蛋白/DEP 质量**——我们 DEP_F1（val 0.13~0.37、test 自评 0.14~0.41）是相对短板，复赛应作为主攻点（需要真能预测大 Δ 的模型，而非后处理）。

## 产物

- `src/wsN_transfer.py` / `src/wsN3_hubw.py` / `src/wsN4_tailavg.py`
- `outputs/wsN/`：pred_trainval/pred_test（迁移模型）、scores.json、router_test.log
- `outputs/wsN3/`、`outputs/wsN4/`：各变体 val 预测与 scores.json
