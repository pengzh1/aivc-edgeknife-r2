# wsJ：强配方化合物描述符 MLP（wsA 特征 × wsD 配方 × G3 修复）

代码：`src/wsJ_chemboost.py`（独立模块，import `src/wsA_chemfeat` 特征构建、`src/wsD_arch` masked_huber，未修改任何已有文件）。
合规：仅用 `h.tr_rows` 训练；描述符标准化/PCA 只涉化合物结构（无标签）；Y_te 零接触；种子 0-7 固定。

## 动机
wsA 证明描述符对未见化合物有真实增益（同预算消融 chem FC +0.035），但只是
(512,1024)×100ep 小模型，且菌株侧带 G3 随机嵌入行问题。假设：**把描述符方案放大到
wsD 强度可以拿到更大的 chem 侧增益**。

## 设计
- 配方 = wsD 最佳：hidden (1024,2048,2048,2048,2048)、masked Huber(β=1)、300ep cosine、
  p_drop 0.3、strain emb_drop 0.35、chem 描述符 drop 0.35（置 train 均值描述符）、G2 组级增强。
- 化合物表示 = wsA 的 Morgan FP+理化描述符 → PCA（**all54 拟合变体**：全部 54 个非对照
  化合物拟合标准化/PCA，n_comp 36→53，纯结构信息无标签泄漏）→ Linear(64→48)+GELU。
- G3 修复：非化合物列未见类别 → UNK(0)；化合物列用真实描述符（特征空间外推）。

## 结果（8 种子均值，train split 训练，完整 score_val）

| split | fidelity | protein_PCC | FC_PCC | resid_PCC | DEP_F1 |
|---|---|---|---|---|---|
| val_chem_only | 0.9619 | 0.9057 | 0.5013 | 0.4838 | 0.1773 |
| val_strain_only | 0.9220 | 0.8207 | 0.3339 | 0.3383 | 0.1813 |
| val_both | 0.9338 | 0.8529 | 0.2263 | - | 0.1245 |
| val_time | 0.9592 | 0.8992 | 0.6398 | - | 0.2911 |

**composite = 0.5382**（单族）；对比：wsD_g2g3 0.5431 / wsG 0.5422 / wsA(弱描述符) 0.5102。

- G3 修复有效：strain FC 0.3339（wsA 同指标 0.2858 → 追平嵌入模型），且 both FC 0.2263 正常。
- **但 chem FC 0.5013 < wsD_g2g3 的 0.5106**：wsA 消融中"描述符 vs 可学习嵌入"的优势
  （+0.035）在 5 层 300ep 强度下被 UNK 回退反超——深网把"平均药物响应"学得更准，
  描述符的外推优势被稀释。

## v3 路由评估（`src/wsJ_router.py`，复用 wsH 方法，r=0.5 交付档完整评分）

| 阵容 | r=0.5 composite | 对照 |
|---|---|---|
| v2 封闭（现提交口径） | 0.5481 | — |
| v2 开放（现提交口径） | 0.5479 | — |
| v3 open-replace（v2+wsJ 替换 wsA） | 0.5479 | 与 v2 开放持平 |
| v3 open-both（v2+wsA+wsJ） | 0.5481 | +0.0002（噪声级） |
| v3 closed-prep（v2+wsJ） | 0.5479 | −0.0002（噪声级） |

chem_only 划分最优权重中 wsJ 仅得 0.001（wsA 得 0.064）——**wsJ 在路由中无增量**。

## 结论：不并入，记为零结果
1. 描述符方案在弱模型上有效（wsA），放大后相对优势消失；深 MLP 的 UNK 回退已是
   更强的"未见化合物先验"。
2. 若后续组委会书面确认结构 embedding 封闭榜合规：**也不建议因此切换封闭提交**
   （v3closed_prep 无增益）；开放榜维持 wsA 即可。
3. 交付物：`outputs/wsJ/pred_trainval.npy`（0.5382，可用）、`router_v3.json`、
   三种阵容的 r=0.5 路由预测（`pred_trainval_v3*.npy`）。
