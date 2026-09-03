"""driver 公共件：从 vc_repro 编排 src 模块的冻结阶段链。

设计：所有阶段 = (名称, [argv...], 期望产物)。阶段按序执行，产物已存在则跳过
（幂等，支持断点续跑）；任一阶段失败立即停并给出重跑命令。工作目录一律为
仓库根（脚本内 chdir 到 vc_repro 的上级）。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable

# 复现链全程以 GOAI_NO_TEST_GT=1 运行：src/data.py 在该守卫下对 test 蛋白
# 矩阵只构造全 NaN 占位壳（数值零读取、文件可缺席），从结构上排除
# test 真值（含 test 对照）进入任何训练/推理路径的可能。
ENV = {**os.environ, "GOAI_NO_TEST_GT": "1"}


def sh(*argv, desc=""):
    print(f"\n===== [{time.strftime('%H:%M:%S')}] {desc or ' '.join(argv)} =====",
          flush=True)
    t0 = time.time()
    r = subprocess.run([PY, "-m", *argv], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        print(f"[FATAL] 阶段失败（exit {r.returncode}）："
              f"{' '.join(argv)}\n重跑：python -m {' '.join(argv)}", flush=True)
        sys.exit(r.returncode)
    print(f"[ok] {time.time()-t0:.0f}s", flush=True)


STATE = ROOT / "outputs" / ".repro_state"  # 随 outputs/ 一起不进提交包


def sh_if(marker: str, *argv, desc=""):
    """以 .state/<marker>.done 为幂等标记（覆盖型阶段的产物路径不可靠）。"""
    STATE.mkdir(parents=True, exist_ok=True)
    flag = STATE / f"{marker}.done"
    if flag.exists():
        print(f"[skip] 阶段 {marker} 已完成", flush=True)
        return
    sh(*argv, desc=desc or marker)
    flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
