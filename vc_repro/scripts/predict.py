"""主命令 ③：冻结模型推理 → prediction.csv（只读 test metadata，零真值接触）。

管线（全部冻结，见 configs/final.yaml）：
  21 族 test 预测按 grand_router.json 权重 r=0.7 收缩路由
  → DEP 条目级分类器推送（τ=0.25, min_push=0.3, push_to=1.02）
  → CRD←CGD 有界迁移（β=0.35）
  → 4,422 蛋白列裁剪 + 五项认证 + SHA256 记录
产物：outputs/wsT2/prediction_wst2_final.csv / _keepcols.csv /
      prediction_wst2_final.npy / certify_final.json

用法（仓库根目录）:  python vc_repro/scripts/predict.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from vc_repro.scripts._driver import sh  # noqa: E402


def main():
    sh("src.wsT2_finaltest", "--name", "C1", "--tau", "0.25",
       "--tag", "final", desc="冻结推理 → prediction.csv（五项认证）")
    print("\n[done] predict：prediction_wst2_final_keepcols.csv 即为提交文件"
          "（4,454 行 × sample_ID+4,422 蛋白，log2，认证在 certify_final.json）。",
          flush=True)


if __name__ == "__main__":
    main()
