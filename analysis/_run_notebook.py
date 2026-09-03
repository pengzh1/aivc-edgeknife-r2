# -*- coding: utf-8 -*-
"""无 jupyter 依赖的 notebook 验证器：用 Agg 后端逐格 exec moa_narrative.ipynb 的代码单元。
运行：.venv/Scripts/python.exe analysis/_run_notebook.py （从项目根目录）"""
import json, os, sys, traceback
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path

nb = json.loads(Path("analysis/moa_narrative.ipynb").read_text(encoding="utf-8"))
g = {"__name__": "__main__"}
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"{len(code_cells)} code cells")
ok = True
for i, c in enumerate(code_cells):
    src = c["source"]
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
        print(f"[cell {i}] OK")
    except Exception:
        ok = False
        print(f"[cell {i}] FAILED")
        traceback.print_exc()
        break
print("ALL OK" if ok else "STOPPED ON ERROR")
