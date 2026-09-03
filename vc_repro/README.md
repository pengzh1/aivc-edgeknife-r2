# AIVC-edgeknife-r2 — 虚拟细胞方向复赛复现包

> **English**: Reproduction pack for the second-round submission. Three commands —
> `build_embeddings.py` → `train.py` → `predict.py` — regenerate the final
> prediction from the official training data. All statistics are fitted on the
> train split only; the chain runs under `GOAI_NO_TEST_GT=1`, so test proteome
> values are never read (the file may be absent) and re-running inference
> reproduces the submitted prediction bit-identically. Details below in Chinese.

> GOAI 大赛 · 赛道三方向一（虚拟细胞）· 算法赛题 · 复赛提交
> 作品编号：无（官网提交系统未预分配；队伍 edgeknife）| 最终模型：21 族分角色路由集成 + 条目级高效应蛋白检出层 + CRD 有界迁移
> prediction SHA256：`1bca69053769bacb4d7a6cf695b46d5ec7a01d918c3194224389af7056927e3a`
> （outputs/wsT2/prediction_wst2_final_keepcols.csv，4,454 行 × sample_ID+4,422 蛋白，log2）
> 配置 SHA256：`b26addd777b6d9d8…`（vc_repro/configs/final.yaml）| 代码版本：release tag `r2-submit`
> 负责人联系方式：周鹏 18115152829 | 已知限制：见 §6

## 0. 合规声明

- 训练标签仅来自官方 `split_final == train` 样本；验证集仅用于模型选择
  （路由权重、收缩档 r=0.7、集成成员入列、DEP 阈值档位裁定，详见报告实验设置节）；
- **测试蛋白真值全程零接触**：训练、统计量、调参、早停、后处理均不使用；
  推理脚本只读取测试 metadata；
- **结构性隔离守卫（GOAI_NO_TEST_GT）**：本包 `_driver.py` 对全部阶段注入
  `GOAI_NO_TEST_GT=1`；`src/data.py` 在该守卫下对 test 蛋白矩阵只构造
  全 NaN 占位壳——数值零读取、文件可缺席（对应主办方"推理侧仅挂载
  test metadata"的复现方式）。**位级证明**：在 test 蛋白文件与缓存同时隐藏的
  重跑中，本包再生的提交件与上列 SHA256 完全一致（212 秒，五项认证全过）；
- 所有统计量（蛋白过滤、均值/方差、PCA 基、类别词表、对照锚点、检出层特征、
  迁移调制）的拟合范围均为 train 划分，逐项见 `configs/final.yaml` 的
  `contract` 节；
- 菌株处理声明：最终模型**未使用任何菌株基因组外部数据**（1011 基因组计划
  仅作机理对照，见报告披露附录）；DHY210 等全部菌株仅按官方 metadata 类别
  原样处理，**不存在 S288c 代理等假设**；
- 外部数据均公开可获取，来源/版本/许可/校验和见
  `vc_repro/external_data/source_manifest.json`；最终模型仅使用 4 项外部
  输入（PubChem 化合物结构、RDKit、ChemBERTa-77M-MLM、MoLFormer-XL 权重）。

## 1. 环境

- Windows 11 / Linux · Python 3.12 · NVIDIA GPU（≥8GB 显存；RTX 3070 级全流程约 5 小时，
  CPU 仅可跑通训练收尾两阶段，族训练必须 GPU）
- 安装（uv 示例）：

```bash
uv venv .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r vc_repro/requirements.txt
```

- 已知边界：pandas 3.0.5 的 `Series.map().to_numpy()` 返回只读数组，代码已规避；
  请勿降级 pandas（族训练脚本依赖 3.x 行为）。

## 2. 数据放置（官方渠道获取，随包不含）

```
input/WAYB_WAYC_metadata_train_val(1).csv
input/WAYB_WAYC_metadata_test(1).csv
input/WAYB_WAYC_proteome_raw_train_val.csv
input/WAYB_WAYC_proteome_raw_test.csv   # 评测侧文件：本包任何脚本不读取
```

> 本包三条主命令在 `GOAI_NO_TEST_GT=1` 守卫下运行（见 §0）：即便
> `proteome_raw_test.csv` 缺席，训练与推理也照常完成——推理侧仅需
> test metadata。

## 3. 三条主命令（仓库根目录执行）

```bash
# ① 构建外部化合物表征（未使用外部数据以外的输入；约 10 分钟，含权重下载则更久）
python vc_repro/scripts/build_embeddings.py

# ② 从头训练最终模型（21 族成员 + 路由 + DEP 检出层；RTX 3070 约 5 小时，可断点续跑）
python vc_repro/scripts/train.py

# ③ 冻结模型推理 → prediction.csv（约 10 分钟；含五项格式认证 + SHA256）
python vc_repro/scripts/predict.py

# 独立校验（可选双重复核）
python vc_repro/scripts/validate_submission.py outputs/wsT2/prediction_wst2_final_keepcols.csv

# 冒烟测试（纯标准库，无需数据/GPU）
python vc_repro/tests/test_smoke.py
```

提交文件：`outputs/wsT2/prediction_wst2_final_keepcols.csv`
（4,454 行 × sample_ID + 4,422 蛋白，log2 尺度，`prediction_scale=log2`）。

## 4. 模型与冻结配置

- 集成：21 个模型族（深度 MLP 系/两阶段 Δ 系/时间批次系/三源分子融合系等，
  成员与训练入口 = 第②条命令的阶段链），分角色（chem/strain/both/time）
  稠密路由，权重向全局锚点收缩 r=0.7；
- 高效应蛋白检出层：条目级梯度提升分类器（P(|Δ|>1)，10 维 train-only 特征，
  样本分组 5 折交叉拟合），对高置信条目做保号加性推送至 |Δ̂|=1.02
  （阈值 τ=0.25，冻结；定价依据见报告方法节）；
- CRD←CGD 有界迁移：β=0.35（train Δ 调制的有界叠加，机制依据见报告）；
- 全部超参、种子、权重、后处理规则冻结于 `vc_repro/configs/final.yaml`。

## 5. 外部数据与权重

- 化合物表征缓存（54 化合物 × ≤64 维，合计 <1MB）随包携带于
  `vc_repro/artifacts/embeddings/`（SHA256 见 `artifacts/ARTIFACTS_MANIFEST.json`），
  离线可用；训练时由阶段①在 `outputs/` 下重建同名缓存；
- 重建需下载 ChemBERTa-77M-MLM（约 150MB）与 MoLFormer-XL（约 400MB）权重，
  URL/SHA256/许可见 `external_data/source_manifest.json`；网络受限时可经
  HF 镜像（`HF_ENDPOINT=https://hf-mirror.com`，`HF_HUB_DISABLE_XET=1`）。
- PubChem SMILES 由 `build_embeddings.py` 按冻结 CID 表经 PUG-REST 拉取
  （54/54 覆盖，失败自动重试 3 次）。

**checkpoint 策略**：最终模型的全部 checkpoint 由第②条命令从头重建
（主办方复现路径即从头训练）；推理侧冻结小件（路由权重 / DEP 分类器 /
蛋白契约 / embedding 缓存）的认证副本随包于 `vc_repro/artifacts/`；
各族大体积 checkpoint（数十 GB）不随包，如评审需要可提供下载链接。

## 6. 已知限制

- 双未见划分（未见菌株 × 未见化合物）的高效应检出受限于训练菌株覆盖
  （4 株），val 上该划分 DEP F1 ≈ 0.13，为如实报告的短板（报告 §限制）；
- 全流程重训约 5 小时 GPU 时间；中途断点可从 `.state` 标记续跑；
- 数值 reproducibility：同配置 GPU 重跑预期 max|diff| < 1e-3（浮点非确定性），
  不影响指标至小数点后三位。

## 7. 许可证

见 `vc_repro/LICENSES/`（外部数据与模型权重各自的许可证与署名文本）；
本包代码以 MIT 许可开源。
