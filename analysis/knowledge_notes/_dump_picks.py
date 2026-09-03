# -*- coding: utf-8 -*-
"""Print full details of hand-picked inbox items for README featuring."""
import json, re

ROOT = r"C:\Users\31564\ai-workspace\projects\goai"
data = json.load(open(ROOT + r"\analysis\knowledge_notes\_notes_raw.json", encoding="utf-8"))
inbox = data["inbox"]

picks = [
 "X-Cell",
 "Cell",
 "Magic Wands",
 "BioAI Phase",
 "自主设计新蛋白质",
 "生物世界模型",
 "Reproducing 2,200",
 "接管实验室",
 "自动材料工厂",
 "神珍",
 "深势科技",
 "Lab of the Future",
 "中国版「生物DeepSeek」",
 "KDD Cup 26",
 "300万全归中国学生",
 "竞赛编程Agent进入全球前十",
 "Introducing Anthropic's AI for Science Program",
 "Claude Science",
 "How scientists are using Claude",
 "Scientific computing in the age of agentic",
 "Mechanist",
 "Intern-S2",
 "Have We Seen an Acceleration",
 "AI for Science开始",
 "网球教练",
 "IR2Solve",
 "不换模型，效果提升104",
 "Scale 上力压群雄",  # may not exist
 "Scaling Law一招鲜",
 "甲骨文",
 "让沉默三千年的甲骨",
 "AI首次拿下IMO",
]
seen = set()
out = []
for p in picks:
    for it in inbox:
        if p.lower() in (it["title"] or "").lower() and it["path"] not in seen:
            seen.add(it["path"])
            s = re.sub(r"\s+", " ", (it.get("summary") or "")).strip()
            out.append("TITLE: %s\n  date=%s r=%s status=%s src=%s\n  link=%s\n  sum=%s" % (
                it["title"], it["date"], it.get("rating"), it.get("status"),
                it.get("source"), it.get("link"), s[:260]))
            break
print("\n\n".join(out))
