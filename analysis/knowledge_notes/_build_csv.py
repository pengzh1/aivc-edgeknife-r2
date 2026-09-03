# -*- coding: utf-8 -*-
"""Build entries.csv from extracted notework JSON. One-off analysis script."""
import json, re, csv, collections

ROOT = r"C:\Users\31564\ai-workspace\projects\goai"
data = json.load(open(ROOT + r"\analysis\knowledge_notes\_notes_raw.json", encoding="utf-8"))
notes = data["notes"]
inbox = data["inbox"]

note_meta = {
 "弹 AI 这把吉他": ("Agent工程与可靠性","中","可靠 agent 大多是确定性软件,只在刻意选择的决策点调用模型——工程章节写 agent 管线的原则"),
 "当执行不再稀缺": ("Agent工程与可靠性","中","三个产品同一结论:高风险场景靠评测驱动与零容忍容错——支持复现包的评测驱动叙事"),
 "把活外包，把负责留下": ("人机协作与责任","背景","执行外包后责任归属的反思,共识塌掉没有警报"),
 "验证器，是这个时代的新代码": ("验证与判断力","中","代码生产成本塌了,瓶颈沿流程上下游移到「判断」——报告问题定义的通用框架"),
 "智能在变成商品，聪明不会": ("AI产业与经济","背景","智能价值链地图:造智能/包结果/用智能三段各自的赢法"),
 "过程不死": ("创业与产品","背景","15 篇合读:工程过程与品味的价值不随生成成本下降而消失"),
 "什么样的判断撑得住": ("验证与判断力","背景","具身智能路线之争外壳下,一流选手在未知领域下注的内核判断"),
 "出不了错的 AI": ("AI安全与战略","背景","Anthropic 三年安全路线:high-stakes 场景里无辜的错"),
 "当基础设施开始为新物种重建": ("Agent工程与可靠性","背景","数据库/云平台/记忆基准共同撞墙:Agent 是 Loop 不是 One Shot"),
 "执行正在过期": ("组织与职业","背景","执行过期的实证与人和组织的迁移方向"),
 "验证才是天花板": ("AI4Science与生物信息","高","AI4S 瓶颈从生成移向验证;描述性数据训的模型做扰动/反事实预测打不过线性基线,必须造因果数据——直接呼应虚拟细胞赛题;含 Xaira X-Cell 与腾讯 Cell 虚拟试药"),
 "AI 的稀缺、终局与人作为参照物": ("AI产业与经济","背景","AI 替代劳动的机制、终局坐标与「参照物」框架"),
 "模型前沿 2026 年中:五个值得记住的坐标": ("模型与训练前沿","中","小模型/万亿RL/路线追问/社会智能五坐标——报告相关工作章节的前沿定位素材"),
 "当成本塌方之后": ("AI产业与经济","背景","Codex 合体访谈与阶跃 Step AOS:价值上移与抢入口"),
 "算力换岗": ("算力与硬件","背景","芯片/整机/模型三层看推理算力与开源模型落地经济性"),
 "神谕在哪里，AI 就能改到哪里": ("验证与判断力","高","递归自我改进的工程现实:「神谕(可验证目标)在哪,AI 才能改到哪」——报告可把扰动预测定位为给细胞建模提供神谕;含熵/信息论视角"),
 "当管控骨架长出来": ("Agent工程与可靠性","背景","OpenSandbox/Agent Sandbox/MCP 管控骨架同月齐发"),
 "确定的地面，是概率铺的": ("Agent工程与可靠性","背景","上下文工程/记忆系统/本体论/语义层"),
 "神谕的延迟": ("验证与判断力","高","能力-可靠性鸿沟、reward hacking、反馈延迟:评估信号滞后时系统会钻空子——支撑评估设计(独立测试备件、防过拟合公开榜)讨论"),
 "赢在模型之外": ("Agent工程与可靠性","中","Model+Harness=Agent 右半边首次被计量:同模型换外壳成功率差 20 点——工程章节论证管线工程的价值"),
 "越便宜越烧钱": ("成本与算力","背景","token 降价与多 agent 失控账单的反常识经济学"),
 "不是技术，是连接": ("创业与产品","背景","做出来便宜到极致后,稀缺的是分销与连接"),
 "办公 Agent 主战场": ("Agent产品","背景","OpenAI 一年真金白银证明入口不是想的那样"),
 "AI 做科学": ("AI4Science与生物信息","高","能证明能设计,谁来验谁来担:反馈回路快准廉决定 AI 产出可信度;可审计性是把人救回判断者的杠杆;scaling 之差在数据出身——报告讨论章核心素材"),
 "分布式硬骨头": ("系统工程","背景","一致性/幂等/多区域/共识算法参考资料式整理"),
 "《妈妈测试》深读笔记 · 信息无损版": ("创业与产品","背景","验证点子别问「好不好」,只信过去行为与当下代价"),
 "实现成了少数": ("AI编码与软件工程","中","AI 写的代码读不读是坏问题,信任锚按「四层代码」分层施策——复现代码可信度论述"),
 "世界模型与物理 AI": ("模型与训练前沿","中","语言是智能的薄皮,难题在感知运动与世界模型;AMI Labs 组织设计——「物理循环/干湿闭环」展望素材"),
 "Agent 持续学习与自进化:闭环、燃料与边界": ("模型与训练前沿","中","把已知题修好不等于学会:外围系统只改输入不改参数;持续学习的闭环/燃料/边界——「迭代自我进化」叙事素材"),
 "AI 冲击下的经济秩序:就业、流量、组织、职业与制度": ("组织与职业","背景","AI 冲击绕开失业率先动 JD 与定价;盯住一手数据的方法论"),
 "ToB Agent 落地:商业化、FDE 与垂直场景": ("创业与产品","背景","开源 Coding Agent 一年 1300 万月活:成功主要来自定位"),
 "智能稀缺退潮后，什么还在牌桌上": ("AI产业与经济","背景","2026/7 回调是旧估值语法压力测试,不是 AI 终结"),
}

cat_rules = [
 ("AI4Science与生物信息", re.compile(r"蛋白|酵母|细胞|基因|生物|药|虚拟试药|虚拟细胞|组学|protein|proteom|genom|drug|BioAI|X-Cell|Chai|Xaira|AlphaFold|科学发现|科学基础模型|AI.?for.?Science|自主实验|材料|分子|临床|湿实验|因果数据|causal", re.I)),
 ("Agent工程与可靠性", re.compile(r"agent|harness|上下文工程|记忆系统|MCP|工具调用|多智能体|编排|sandbox", re.I)),
 ("AI编码与软件工程", re.compile(r"coding|代码|claude code|cursor|软件工程|程序员|copilot", re.I)),
 ("模型与训练前沿", re.compile(r"RL|强化学习|预训练|微调|蒸馏|scaling|transformer|注意力|扩散模型|世界模型|具身|机器人|多模态|embedding|架构", re.I)),
 ("算力与硬件", re.compile(r"算力|芯片|GPU|显存|推理成本|英伟达|nvidia|TPU|硬件", re.I)),
 ("AI产业与经济", re.compile(r"融资|估值|商业|创业|产品|SaaS|变现|市场|投资|股价|裁员|就业|经济|公司|战略|竞争|格局", re.I)),
 ("组织与职业", re.compile(r"组织|团队|管理|职业|工作|招聘|面试|协作", re.I)),
 ("AI安全与治理", re.compile(r"安全|治理|监管|对齐|红队|攻击|漏洞|隐私|风险", re.I)),
 ("写作与叙事", re.compile(r"写作|叙事|故事|演讲|表达|沟通|文案", re.I)),
 ("验证评测与复现", re.compile(r"复现|reproduc|评测|评估|基准|benchmark|验证|eval|测试集|leaderboard|打榜", re.I)),
]

proj_hi = re.compile(r"虚拟细胞|蛋白|酵母|扰动|虚拟试药|Cell.{0,4}主刊|BioAI|X-Cell|Chai|Xaira|生物世界模型|因果数据|Causal Models|AI.?for.?Science|科学基础模型|自主实验|Reproducing|复现|干湿|药物发现|Drug Discovery|protein|蛋白质设计|科学发现|信息论|熵|Kaggle|kaggle|竞赛|比赛复盘|金牌|答辩|hackathon|Hackathon", re.I)
proj_mid = re.compile(r"agent|harness|验证|评测|评估|benchmark|判断力|工程|上下文|embedding|表征|特征|机器学习|深度学习|模型|数据|scaling|因果|causal|可复现|复现包", re.I)

def categorize(it):
    txt = (it.get("title") or "") + " " + (it.get("summary") or "") + " " + " ".join(it.get("tags") or [])
    for name, rx in cat_rules:
        if rx.search(txt):
            return name
    return "其他"

def relevance_inbox(it):
    txt = (it.get("title") or "") + " " + (it.get("summary") or "") + " " + " ".join(it.get("tags") or [])
    r = it.get("rating") or 0
    st = it.get("status") or ""
    curated = (r >= 4) or (st in ("深入", "已吸收"))
    if proj_hi.search(txt):
        return "高"
    if curated and proj_mid.search(txt):
        return "中"
    return "背景"

def one_line(s, n=160):
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:n] + ("…" if len(s) > n else "")

rows = []
for nt in notes:
    cat, rel, why = note_meta.get(nt["title"], ("其他", "背景", ""))
    rows.append({
        "kind": "note",
        "title": nt["title"],
        "url": nt.get("source") or "",
        "category": cat,
        "relevance": rel,
        "date": nt.get("date", ""),
        "summary": one_line(nt.get("essence") or nt.get("summary")),
        "note": why,
    })

for it in inbox:
    rows.append({
        "kind": "inbox",
        "title": it.get("title", ""),
        "url": it.get("link", ""),
        "category": categorize(it),
        "relevance": relevance_inbox(it),
        "date": it.get("date", ""),
        "summary": one_line(it.get("summary") or it.get("excerpt")),
        "note": "",
    })

print("total rows:", len(rows))
print("by relevance:", collections.Counter(r["relevance"] for r in rows))
print("by category:", collections.Counter(r["category"] for r in rows).most_common())
hi = [r for r in rows if r["kind"] == "inbox" and r["relevance"] == "高"]
print("HIGH inbox:", len(hi))
for r in sorted(hi, key=lambda x: x["date"]):
    t = r["title"][:60]
    print(" ", r["date"], "|", t)

with open(ROOT + r"\analysis\knowledge_notes\entries.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["kind", "title", "url", "category", "relevance", "date", "summary", "note"])
    w.writeheader()
    w.writerows(rows)
print("CSV written")
