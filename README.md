# AIVC-edgeknife-r2

**English** | [中文](#中文)

**Virtual yeast cell: predicting proteome-wide perturbation responses across unseen strains and compounds.**
GOAI 2026 · Track 3 (AI for Research) · Virtual Cell direction · second-round submission · MIT License · release tag `r2-submit`.

## Overview

- **Task.** Given a perturbation condition (strain × compound × medium × temperature × time × instrument × plate), predict the full log2 proteome response (4,422 proteins after the official train-only missing-rate filter). The test split contains one unseen strain and eleven unseen compounds — entity-level out-of-distribution generalization.
- **Results** (local four-split OOD validation, official-weight approximation): composite score **0.5541**, versus 0.277 (global-mean baseline) and 0.461 (strongest first-order Ridge baseline). Detection F1 for high-effect proteins (|Δ|>1, the organizer-designated evaluation focus) improved from 0.2305 to **0.2442**, while every other metric moved by no more than 0.0012.
- **Method.**
  1. *The detection threshold as a priced decision variable* — an entry-level significance classifier (18.8M protein entries, sample-grouped 5-fold OOF AUC 0.973) followed by a sign-preserving additive push past the threshold; because the push is additive, it does not depend on matched-control anchors and covers unseen-strain rows for the first time.
  2. *Role-routed ensemble* — 21 model families assigned weights per OOD role (unseen compound / unseen strain / both / time), with the search objective aligned to the composite score's partial derivatives and shrinkage r = 0.7.
  3. *Falsification-driven research discipline* — a 59-experiment verdict archive, a 14-item falsified-direction list, and a pre-registered single-look validation protocol (6 tables, one look, no iteration).
- **Mechanism.** Predicted proteome signatures agree with independent chemogenomic measurements (median Spearman 0.718 across 14 compounds, all above permutation null); variant-level analysis of the CGD/CRD near-clone pair (2.5× amplification of the ARR1/2/3 arsenic-resistance cluster; PDR efflux-network deletions in CGD) supports the bounded strain-transfer design.

## Reproduce

Three commands regenerate everything from the official training data (details: `vc_repro/README.md`):

```bash
python vc_repro/scripts/build_embeddings.py   # external compound features
python vc_repro/scripts/train.py              # ~5 h on an RTX 3070-class GPU
python vc_repro/scripts/predict.py            # → prediction.csv (4,454 × 4,422, log2)
```

The submitted prediction has SHA256 `1bca69053769bacb4d7a6cf695b46d5ec7a01d918c3194224389af7056927e3a`
(manifest: `vc_repro/REPRODUCIBILITY_MANIFEST.json`). Competition raw CSVs are provided by the
organizer and are not redistributed here.

## Compliance

- Labels, statistics and hyperparameters are fitted on `split_final == train` only; validation splits are used solely for pre-registered model selection.
- Test proteome ground truth is never read. The whole chain runs under `GOAI_NO_TEST_GT=1`, which replaces the test matrix with an all-NaN placeholder (the file may even be absent); re-running inference with the test proteome file hidden reproduces the submitted prediction with an identical SHA256.
- All external data are public and registered (source / version / license / checksum) in `vc_repro/external_data/source_manifest.json`; licenses are collected in `vc_repro/LICENSES/`.

## License

MIT for the code in this repository. External datasets and model weights retain their original licenses.

---

## 中文

**虚拟酵母：跨菌株、跨化合物的扰动蛋白质组响应预测。**
GOAI 2026 · 赛道三（前沿探索 AI for Research）· 虚拟细胞方向 · 复赛提交 · MIT 许可 · 冻结版本 tag `r2-submit`。

- **任务**：给定扰动条件（菌株×化合物×培养基×温度×时间×仪器×孔板），预测完整 log2 蛋白质组响应（按官方 train 缺失率过滤后 4,422 个蛋白）。测试集含 1 株未见菌株与 11 种未见化合物——实体级分布外泛化。
- **结果**（本地四划分分布外验证，官方权重近似）：综合分 **0.5541**（均值基线 0.277，最强一阶基线 Ridge 0.461）；高效应蛋白（|Δ|>1，出题方明示的评估重点）检出 F1 由 0.2305 提升至 **0.2442**，其余指标扰动均不超过 0.0012。
- **方法**：① 把检出阈当作可定价的决策变量——条目级显著性分类器（1,880 万条目，样本分组 5 折 OOF AUC 0.973）加保号加性推送过阈，加性操作不依赖对照锚点，首次覆盖未见菌株；② 分角色路由集成——21 个模型族按 OOD 角色分配权重，搜索目标按综合分偏导对齐，收缩 r=0.7；③ 证伪式研究纪律——59 项实验判语档案、14 类已证伪方向清单、验证集预注册单次裁决（6 张表，一次观看，不迭代）。
- **机理**：预测蛋白组签名与独立化学基因组学实测一致（14 个化合物中位 Spearman 0.718，全部高于置换零）；CGD/CRD 近克隆对的变异级分析（ARR1/2/3 砷抗性簇 2.5 倍扩增、CGD 的 PDR 外排网络缺失）为菌株有界迁移提供机制支撑。

### 复现

三条命令从官方训练数据重建全部产物（详见 `vc_repro/README.md`）：

```bash
python vc_repro/scripts/build_embeddings.py   # 外部化合物表征
python vc_repro/scripts/train.py              # RTX 3070 级 GPU 约 5 小时
python vc_repro/scripts/predict.py            # → prediction.csv（4,454 × 4,422，log2）
```

提交件 prediction.csv 的 SHA256 为 `1bca69053769bacb4d7a6cf695b46d5ec7a01d918c3194224389af7056927e3a`。
官方原始数据经赛事渠道获取，本仓库不分发。

### 合规

- 标签、统计量与超参数仅用 `split_final == train` 拟合；验证集仅用于预注册的模型选择。
- 测试蛋白真值全程零接触：复现链在 `GOAI_NO_TEST_GT=1` 守卫下运行，test 矩阵被替换为全 NaN 占位壳（文件可缺席）；在隐藏 test 蛋白文件的环境下重跑推理，产物与提交件 SHA256 逐位一致。
- 外部数据均为公开资源，来源/版本/许可/校验和逐项登记于 `vc_repro/external_data/source_manifest.json`，许可证文本见 `vc_repro/LICENSES/`。

### 许可证

本仓库代码以 MIT 协议开源；外部数据与模型权重遵循其原始许可证。
