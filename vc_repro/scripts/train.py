"""主命令 ②：从指定训练数据从头训练最终模型（含路由与 DEP 检出层拟合）。

冻结阶段链（args 全部固化，取自 configs/final.yaml 对应的模块默认值）：
  ①-⑪  21 族成员训练 + test 侧推理（train-only；val 仅 wsN11 模型选择用）
  ⑫    21 族分角色稠密路由（wsN11，权重存 grand_router.json）
  ⑬    路由基线缓存 + 跨族方差 + 对照锚点（wsT0，train-only 统计）
  ⑭    DEP 条目级分类器拟合（wsT1 build+fit；τ 不在此调定——
        最终 τ=0.25 已在配置冻结，OOF 产物留档备查）
预计：RTX 3070 级 GPU 约 5 小时；CPU 仅阶段⑬⑭（约 20 分钟）。

用法（仓库根目录）:  python vc_repro/scripts/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from vc_repro.scripts._driver import sh, sh_if  # noqa: E402

# (标记, 命令, 说明)；标记 = vc_repro/.state/<标记>.done（幂等续跑）
STAGES = [
    ("s01_wsM", ("src.wsM_trainonly",), "六族 train-only + test 预测"),
    ("s02_chem", ("src.wsN6_chemberta", "--test"), "分子表征族 full"),
    ("s03_wsD16", ("src.wsN14_wsD16",), "wsD 16 种子"),
    ("s04_boost8", ("src.wsN15_seedboost",), "wsB/wsF/fuse 8 种子"),
    ("s05_more16", ("src.wsN16_more",), "fuse/wsB 16 种子 + bagged wsD"),
    ("s06_deepv", ("src.wsN19_fusedeep",), "fuse 深度变体"),
    ("s07_deep3", ("src.wsN20_deep3full",), "deep3 16 种子"),
    ("s08_fuse3v", ("src.wsN23_fuse3",), "fuse3 三源 16 种子 val"),
    ("s09_fuse3t", ("src.wsN23_fuse3", "--test"), "fuse3 full 协议 test"),
    ("s10_e150v", ("src.wsN24_fuse3e150",), "fuse3-e150 16 种子 val"),
    ("s11_e150t", ("src.wsN24_fuse3e150", "--test"), "fuse3-e150 16 种子 test"),
    ("s12_s32v", ("src.wsN28_fuse3e150_s32",), "fuse3-e150 32 种子合并 val"),
    ("s13_s32t", ("src.wsN28_fuse3e150_s32", "--test"),
     "fuse3-e150 32 种子合并 test"),
    ("s14_router", ("src.wsN11_grandrouter",), "21 族稠密路由"),
    ("s15_wsT0", ("src.wsT0_varcheck",), "路由基线缓存/跨族方差/对照锚点"),
    ("s16_wsT1b", ("src.wsT1_depgate", "--stage", "build"), "DEP 特征/张量"),
    ("s17_wsT1f", ("src.wsT1_depgate", "--stage", "fit"), "DEP 分类器终模"),
]


def main():
    for marker, cmd, desc in STAGES:
        sh_if(marker, *cmd, desc=f"{desc} [{marker}]")
    print("\n[done] train：最终模型全部成员 + 路由 + DEP 检出层就绪。\n"
          "下一步：python vc_repro/scripts/predict.py", flush=True)


if __name__ == "__main__":
    main()
