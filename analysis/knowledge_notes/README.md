# AI 知识笔记（outref/index.html）提取分析

> 生成日期：2026-09-02 · 分析对象：`outref/index.html`（7.1 MB，"notework · 笔记库"单页应用导出）
> 配套文件：`entries.csv`（1845 条结构化全表）

## 0. 文件解剖

这不是浏览器书签导出，而是**自研笔记库（notework）的全量快照**：HTML 外壳 + 内嵌 JSON 数据块
（`<script type="application/json" id="notes-data">`，约 6.6 MB）。提取方式：正则定位数据块 →
`json.loads` → 得到 9 个顶层集合。原始 JSON 留档于 `_notes_raw.json`，重建脚本为 `_build_csv.py`。

| 集合 | 数量 | 说明 |
|---|---|---|
| **notes** | **32** | 精读笔记（2026-07-27 ~ 08-29），每条含标题/标签/摘要/精华（essence）/正文（4k-26k 字）/来源清单 |
| **inbox** | **1813** | 收藏文章条目，含标题、原文链接、来源、语言、评分（1-5）、处理状态（深入/已吸收/略过/暂存/待处理）、AI 摘要 |
| topics | 34 | 研究主题线索（tier1-4），多数已"处理"并挂到某条 note |
| questions | 4 | 顶层研究问题：A 验证与判断（含 AI 与科学）、B Agent 工程与可靠性、C 成本算力经济学、D 组织责任商业化 |
| books | 209 | 读书卡片（与本项目基本无关） |
| signals | 17 | 灵感速记；其中 2026-08-29 一条 = `aifors 判断力 做对 熵增 信息论 干湿实验 物理循环 迭代自我进化`——正是复赛报告的立意关键词 |

## 1. 主题统计

**32 条精读笔记（人工归类）**：验证与判断力 5 · AI4Science 与生物信息 2 · Agent 工程与可靠性 8 ·
模型与训练前沿 3 · AI 产业与经济 4 · 创业与产品 4 · 组织与职业 2 · 其余（编码/安全/系统/算力）4

**1813 条 inbox（规则归类，`entries.csv` 的 category 列）**：Agent 工程与可靠性 660 ·
模型与训练前沿 333 · AI 编码与软件工程 194 · AI 产业与经济 182 · **AI4Science 与生物信息 135** ·
算力与硬件 111 · 组织与职业 53 · AI 安全与治理 38 · 验证评测与复现 20 · 写作与叙事 16 · 其他 90

**相关性分级（`entries.csv` 的 relevance 列）**：
- **高（146）**：标题/摘要命中项目关键词（虚拟细胞/蛋白/扰动/因果数据/AI4Science/复现/竞赛等）——直接可读
- **中（840）**：用户自评 r≥4 或状态为"深入/已吸收"，且命中工程/模型/评估类关键词——间接方法论
- **背景（859）**：其余——行业视野
- 注意：inbox 分级为规则自动打标，有误判（如"营养冷食"因含"蛋白"被标高）；32 条笔记的分级为人工判定，以笔记为准。

## 2. 32 条精读笔记清单（按主题分组，★=与复赛相关性）

### AI4Science 与生物信息 ★高
| 笔记 | 日期 | 一行摘要 |
|---|---|---|
| **验证才是天花板** ★★★ | 08-03 | AI4S 瓶颈从生成移向验证；描述性数据训的模型做**扰动/反事实预测打不过线性基线**，必须真去扰动细胞造因果数据；含 Xaira X-Cell、腾讯 Cell 虚拟试药、Lila/A-Lab |
| **AI 做科学** ★★★ | 08-25 | 能证明能设计，谁来验谁来担；反馈回路"快准廉"决定可信度；**可审计性**是把人救回判断者的唯一杠杆；scaling 之差在数据出身（自然选择的蛋白 vs 人类攒的小分子） |

### 验证与判断力
| 笔记 | 日期 | 一行摘要 |
|---|---|---|
| **神谕在哪里，AI 就能改到哪里** ★高 | 08-08 | 递归自我改进的工程现实：神谕（可验证目标）在哪 AI 才能改到哪；改运行系统已跑通、改参数全员押注 |
| **神谕的延迟** ★高 | 08-11 | 能力-可靠性鸿沟、reward hacking、反馈延迟：评估信号滞后时系统必钻空子 |
| 验证器，是这个时代的新代码 ★中 | 07-30 | 代码生产成本塌了，瓶颈沿流程上下游同时移到「判断」 |
| 什么样的判断撑得住 ★背景 | 08-03 | 具身智能路线之争外壳下，一流选手在未知领域下注的内核判断 |

### Agent 工程与可靠性
| 笔记 | 日期 | 一行摘要 |
|---|---|---|
| 弹 AI 这把吉他 ★中 | 07-27 | 可靠 agent 大多是确定性软件，只在刻意选择的决策点调用模型 |
| 当执行不再稀缺 ★中 | 07-29 | Shippy/阿福/Claude Code 同一结论：高风险场景靠评测驱动与零容忍容错 |
| 赢在模型之外 ★中 | 08-18 | Model+Harness=Agent 右半边首次被计量：同模型换外壳成功率差 20 点 |
| 把活外包，把负责留下 ★背景 | 07-30 | 执行外包后责任归属；共识塌掉没有警报 |
| 当基础设施开始为新物种重建 ★背景 | 08-03 | Agent 是 Loop 不是 One Shot |
| 当管控骨架长出来 ★背景 | 08-10 | OpenSandbox / Agent Sandbox / MCP 管控骨架 |
| 确定的地面，是概率铺的 ★背景 | 08-10 | 上下文工程 / 记忆系统 / 本体论 / 语义层 |

### 模型与训练前沿
| 笔记 | 日期 | 一行摘要 |
|---|---|---|
| 模型前沿 2026 年中：五个值得记住的坐标 ★中 | 08-04 | 小模型/万亿 RL/路线追问/社会智能五坐标 |
| Agent 持续学习与自进化 ★中 | 08-27 | "把已知题修好不等于学会"；外围系统只改输入不改参数；闭环/燃料/边界 |
| 世界模型与物理 AI ★中 | 08-26 | 语言是智能的薄皮；AMI Labs 组织设计（"物理循环"素材） |

### 其余（背景）
智能在变成商品聪明不会（07-31）· 过程不死（08-01）· 出不了错的 AI（08-03）· 执行正在过期（08-03）·
AI 的稀缺、终局与人作为参照物（08-04）· 当成本塌方之后（08-05）· 算力换岗（08-05）·
越便宜越烧钱（08-19）· 不是技术是连接（08-23）· 办公 Agent 主战场（08-23）· 分布式硬骨头（08-25）·
《妈妈测试》深读（08-25）· 实现成了少数（08-25）· AI 冲击下的经济秩序（08-29）·
ToB Agent 落地（08-29）· 智能稀缺退潮后什么还在牌桌上（08-29）

## 3. 精选 inbox 条目（对复赛直接可引，均为高相关）

### 3.1 虚拟细胞 / 扰动预测直接相关（报告"相关工作"必引）
| 条目 | 链接 | 要点 |
|---|---|---|
| 国产AI登上《Cell》主刊：UniPert–G2CP 虚拟试药（腾讯+中南，r4 已吸收） | https://mp.weixin.qq.com/s/fKDRskz3olzSKwmJXRAqHw | 国内首个 Cell 主刊 AI 虚拟细胞研究；用密集廉价的**基因扰动**（CRISPR）知识撬动稀疏昂贵的**化学扰动**预测——与本赛题"给定扰动预测蛋白质组响应"同构 |
| Causal Models Need Causal Data — Xaira X-Cell（Latent Space，r4 已吸收） | https://www.latent.space/p/xaira | 测试损失 15 亿参数后走平、训练损失继续降 = **被数据信息量卡住**；CELLxGENE 1.68 亿细胞 × 2-3 万基因 ≈ 4 万亿条目矩阵催生虚拟细胞模型生态 |
| Drug Discovery Has No Magic Wands（a16z，Koller，r5 已吸收） | https://www.a16z.news/p/drug-discovery-has-no-magic-wands | 90%+ 临床失败不是分子没做好而是**机制选错**（"擅长造钥匙，大多开错锁"）；agentic 闭环实验室只在有"快准廉"评分卡时有效 |
| The BioAI Phase Shift — Chai Discovery（Latent Space，r4 深入） | https://www.latent.space/p/chai-discovery | AI×医药工具的"可信"拐点：2026/1 JPM 四笔大单；工具好用到值得信任后药企才规模化采用 |
| Claude 自主设计新蛋白质（新智元，r5 暂存） | https://mp.weixin.qq.com/s/QHS-Fp19XoulMkdyDwDgjg | 1.6 万词提示词编排现成开源工具（RFdiffusion/ProteinMPNN/ESMFold2），26.8% 成功率 vs 行业 10-15%；关键是编排不是新工具 |
| 4 个月 1 亿美元、3 座"数据发电厂"：寻明生科生物世界模型（机器之心，r4 已吸收） | https://mp.weixin.qq.com/s/h0djQRTTX2Synlwnu5bjeg | 用真实实验持续反馈让模型从结果中学习；2026 AI 制药从小分子转向生成式蛋白/抗体 |
| 中国版"生物 DeepSeek"：GeneLLM 多组学大模型（新智元，r3） | https://mp.weixin.qq.com/s/vbp6JAiuD9UOj28CvARWng | 不依赖基因注释，原始测序数据当语料做 next-token 预测 |

### 3.2 AI4Science 方法论与验证（报告"讨论/展望"素材）
| 条目 | 链接 | 要点 |
|---|---|---|
| The Lab of the Future — Lila Sciences（r4 已吸收） | https://www.latent.space/p/the-lab-of-the-future-should-feel | 10 万亿实验验证 token；全押"实验室即无限 token 生成器"（笔记 10 批评其只敢押生成） |
| 17 天 36 种材料：GNoME + A-Lab（r4 已吸收） | https://mp.weixin.qq.com/s/anlz9rssnC9O_FXJyCaRbg | 自主实验室标志案例；笔记 10 指出其验证与生成同源（共享"材料皆有序"盲区） |
| AI 能接管实验室了？中国科大压力测试（r4 已吸收） | https://mp.weixin.qq.com/s/M9uPW-PQ8tI0fwMDHqQafA | 48 配置 4608 试次：仅 3.3% 工作流免人工修复；"会调参数 ≠ 会重新规划" |
| 上智院"神珍"11B 科学多模态基础模型（r4 已吸收） | https://mp.weixin.qq.com/s/9kC37uYyOlO_e4a7gbEv1A | 把结构化科学数据压成文本会系统性损失结构信息——支持"尊重数据结构"的特征工程叙事 |
| Mechanist：AI 作为发现智能机制的科学仪器（HF，r4） | https://huggingface.co/papers/2608.12036 | 1.3 万篇可解释性论文知识图谱 + 32 个因果干预与验证方法库 |
| Intern-S2-Preview：科学 agentic 基础模型（HF，r4） | https://huggingface.co/papers/2608.13505 | 科学多模态预训练 + 统一后训练管线（SFT/多任务 RL/agentic RL） |
| Have We Seen an Acceleration in Discoveries?（METR，r4 已吸收） | https://metr.org/notes/2026-08-14-llm-contribution-to-discoveries/ | 实证：LLM 是否让"发现"加速——分三档的诚实结论，答辩引用显严谨 |
| Anthropic AI for Science Program / Claude Science（r2/r3） | https://www.anthropic.com/news/ai-for-science-program · /claude-science-ai-workbench | 面向基因组学/单细胞/**蛋白质组学**的 60+ 预置工具与可审计生成历史 |
| AI for Science 开始"动手"：机器人进国家级实验室（量子位，r3） | https://mp.weixin.qq.com/s/ruiMQBBJ-IBIy_lVJ-wYXg | 干湿闭环通量瓶颈的物理侧现状 |
| 深势科技"玻尔·科学空间"（量子位，r4） | https://mp.weixin.qq.com/s/ePdboR4QIIrDBKQDgPjmaQ | 科研全流程桌面端：调研/假设/验证/复现/成图/审稿 |
| Scientific computing in the age of agentic AI（OpenAI，r3） | https://openai.com/index/scientific-computing-agentic-ai | 以基因组学为案例的田野报告 |

### 3.3 复现与竞赛方法论（复现包 + 答辩）
| 条目 | 链接 | 要点 |
|---|---|---|
| **What We Learned by Reproducing 2,200 papers from ICML（HF，r5）** | https://huggingface.co/blog/icml-2026-open-reproductions | 1200+ 人用 coding agent 逐 claim 复现；**不信任自评**；23% 论文有 claim 被证伪；"理论写 reverse KL 代码跑 forward KL"类陷阱清单——复现包设计圣经 |
| KDD Cup 26 DataAgents 冠军思路解读（美团，r4） | https://tech.meituan.com/2026/08/13/KDD-2026-meituan-papers.html | 复杂数据分析竞赛冠军方案复盘（同为大厂数据赛，可参考其写作结构） |
| KDD 2026 腾讯广告算法大赛双冠军（新智元，r3） | https://mp.weixin.qq.com/s/ufpLPtmuW0h1cCjfgNBv_Q | 1.4 万选手大规模数据赛；"序列建模与特征交互统一化"赛题的冠军做法 |
| 竞赛编程 Agent Solvita 进全球前十（新智元，r3） | https://mp.weixin.qq.com/s/_VaYlATQ9Ootfj2Q03sU7Q | 不微调底座，在 Agent 外部构建可训练知识网络持续积累经验 |
| 不换模型效果提升 104%：Self-Harness（量子位，r4） | https://mp.weixin.qq.com/s/7dqygHsvbdsa0J7xaHF6Lw | held-in/held-out 一升一不降才采纳改动——与"双候选留档、以认证拍板"同思路 |
| Scaling Law 一招鲜？晶体结构基准 AtomWorld（新智元，r4） | https://mp.weixin.qq.com/s/T7ZSsP3kI8e2ogvLZ4p9hg | 规模扩大只提升规则清晰任务；物理约束任务翻车——"堆容量有限"论据 |

## 4. 对复赛提交的 8 条具体建议

> 提交物 = 代码仓库 + 非代码材料 zip（报告+PPT）+ 代码材料 zip（复现包 vc_repro/）。

1. **报告·研究背景**：用 Koller 三段论（疾病→机制→药物→患者）+"90% 临床失败源于机制选错"（3.1 第 3 条）
   立起扰动响应预测的生物学价值；再用腾讯 UniPert–G2CP（Cell 主刊）坐实"虚拟细胞"是已被顶刊验证的方向。
2. **报告·相关工作**：把 3.1 全组整理成一张谱系图——数据层（CELLxGENE 4 万亿条目）→
   模型层（X-Cell / UniPert-G2CP / GeneLLM / 神珍）→ 闭环层（Lila / A-Lab / 寻明生科数据发电厂），
   本方案定位在"模型层的工程化逼近"。
3. **报告·方法设计原则**：引 X-Cell"测试损失 15 亿参数后走平 = 数据信息量瓶颈"（3.1 第 2 条）+
   AtomWorld"规模只提升规则清晰任务"（3.3 末条），论证为什么 DEP 选择器 C1 + 21 族路由 +
   特征工程优先于堆模型容量——这是 0.5541 方案的理论底座，比"我们试了所以行"高一个层级。
4. **报告·评估可信度**：用笔记「神谕的延迟」（reward hacking/反馈滞后）与「验证才是天花板」
   （"验证若与生成同源，就是给错误盖章"）论证：独立 test 备件、五项认证、SHA256 终版校验不是
   工程洁癖，而是防止"验证者与生成者共享盲区"的必要设计。
5. **复现包 README**：照搬 HF 2200 篇复现黑客松的三条原则——逐 claim 验证（对应三主命令+
   certify JSON）、不信任自评（独立校验 7/7 PASS）、环境即文档；并把"理论 reverse KL /
   代码 forward KL"陷阱做成提交前自查表（报告数字 vs 代码输出逐一对账）。
6. **PPT·工程叙事**：用「赢在模型之外」（同模型换 harness 成功率差 20 点）+「当执行不再稀缺」
   （评测驱动）讲 68 个实验档案、双候选留档（C1 0.5541 vs band 0.5536）的决策纪律——
   "我们的优势在 harness 层，不在某个单点 trick"。
7. **PPT·展望页**：用笔记「AI 做科学」的"反馈回路快准廉"框架 + 中国科大压力测试
   （"会调参数 ≠ 会重新规划"）收束：本预测模型是干湿闭环里的"神谕"环节，
   下一步价值在于接入实验反馈回路——呼应用户 8/29 signal 的"干湿实验 物理循环 迭代自我进化"。
8. **答辩 Q&A 预案**：评委若问"为什么信任你的预测"，用 questions[A] 的"神谕三要素
   （客观/独立/准确几乎只能取两个）"作答：公开榜验证牺牲独立性，我们的备件认证牺牲实时性换独立与准确——
   主动展示这个权衡比声称"我们分数高"更有说服力。

## 5. 空白提示

库中**几乎没有**直接讲"竞赛答辩技巧 / PPT 制作 / 科技报告写作"的条目（写作与叙事类仅 16 条且多为
商业文案），KDD 冠军复盘仅 2 条可参考写作结构。报告与 PPT 的叙事打磨不要指望此库，应另找
Datawhale 进阶教程与初赛模板对标。

## 6. 文件说明

- `entries.csv`：1845 行全表。列 = `kind`(note/inbox) `,title,url,category,relevance,date,summary,note`。
  note 行的 url 为库内路径（`notes/*.md`，在 index.html 页面内搜索标题即可打开）；inbox 行为原文外链。
  UTF-8-BOM 编码，Excel 可直接打开。
- `_notes_raw.json`：从 HTML 抽出的原始 JSON（6.6 MB，留档备查）。
- `_build_csv.py` / `_dump_picks.py`：本次分析的可重跑脚本。
