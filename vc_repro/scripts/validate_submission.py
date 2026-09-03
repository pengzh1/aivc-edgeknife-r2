"""提交文件独立校验（不依赖 predict 内认证，供主办方/选手双重复核）。

检查项对齐《虚拟细胞方向材料提交说明》§6：
  行数 4,454；sample_ID 集合与顺序 == 官方 test metadata；
  列 = sample_ID + 4,422 官方蛋白（名称与顺序严格匹配）；
  全值有限；scale=log2（值域合理性）；无重复列/行；
  输出 SHA256。
用法:  python vc_repro/scripts/validate_submission.py [csv路径]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT = ROOT / "outputs/wsT2/prediction_wst2_final_keepcols.csv"
CONTRACT = ROOT / "outputs/wsM/keep_proteins_train_miss80.csv"
META = ROOT / "input/WAYB_WAYC_metadata_test(1).csv"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    df = pd.read_csv(path)
    meta = pd.read_csv(META)
    keep = pd.read_csv(CONTRACT)["protein"].tolist()

    checks = {
        "rows_4454": len(df) == 4454,
        "id_unique": df["sample_ID"].is_unique,
        "id_order_match_official":
            bool((df["sample_ID"].to_numpy() == meta["sample_ID"].to_numpy()).all()),
        "cols_exactly_contract": list(df.columns[1:]) == keep
        and len(df.columns) == 4423,
        "no_dup_cols": len(set(df.columns)) == len(df.columns),
        "all_finite": bool(np.isfinite(df.iloc[:, 1:].to_numpy()).all()),
        "log2_range_plausible":
            (df.iloc[:, 1:].to_numpy().min() > 0)
            and (df.iloc[:, 1:].to_numpy().max() < 45),
    }
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"file: {path}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  sha256: {sha}")
    ok = all(checks.values())
    print(f"\n{'VALIDATION PASSED' if ok else 'VALIDATION FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
