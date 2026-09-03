"""复现包冒烟测试（纯标准库，无需 GPU/数据即可运行）。

用法（仓库根目录）:  python vc_repro/tests/test_smoke.py
覆盖：冻结配置关键值、实体映射行数、复现清单字段、README 三主命令、
      随包小件 artifact 的 SHA256 一致性、隐私/密钥泄漏粗检。
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PKG = ROOT / "vc_repro"
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    print("== vc_repro smoke tests ==")

    # 1. 冻结配置关键值
    cfg = (PKG / "configs/final.yaml").read_text(encoding="utf-8")
    for key in ["shrink_r: 0.7", "tau: 0.25", "beta: 0.35",
                "prediction_scale: log2", "n_output_cols: 4422"]:
        check(f"final.yaml 含冻结值 `{key}`", key in cfg)

    # 2. 实体映射 54 化合物
    with open(PKG / "external_data/entity_mapping.csv",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check("entity_mapping.csv = 54 行", len(rows) == 54, f"实际 {len(rows)}")

    # 3. 复现清单
    man = json.loads((PKG / "REPRODUCIBILITY_MANIFEST.json").read_text(
        encoding="utf-8"))
    fp = man["final_prediction"]
    check("manifest prediction SHA256 为 64 位十六进制",
          bool(re.fullmatch(r"[0-9a-f]{64}", fp["sha256"])))
    check("manifest 行列契约 4454×4423",
          fp["rows"] == 4454 and fp["cols"] == 4423)
    check("manifest scale=log2", fp.get("scale") == "log2")
    check("manifest config sha256_16 已填",
          bool(man["config"].get("sha256_16")))

    # 4. README 三主命令
    readme = (PKG / "README.md").read_text(encoding="utf-8")
    for cmd in ["build_embeddings.py", "train.py", "predict.py"]:
        check(f"README 含主命令 {cmd}", cmd in readme)

    # 5. 随包 artifact 清单与 SHA256
    aman = json.loads((PKG / "artifacts/ARTIFACTS_MANIFEST.json").read_text(
        encoding="utf-8"))
    for rel, h in aman["files"].items():
        p = PKG / "artifacts" / rel
        check(f"artifact {rel}", p.exists() and sha256(p) == h["sha256"])

    # 6. 密钥/绝对路径粗检（包内代码）
    bad = []
    for p in list(PKG.rglob("*.py")) + list(PKG.rglob("*.yaml")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"[A-Z]:[\\/]Users[\\/]|access_token\s*=\s*['\"]\w", t):
            bad.append(str(p.relative_to(PKG)))
    check("无硬编码本机绝对路径/密钥", not bad, ";".join(bad))

    print(f"\n{'SMOKE TESTS PASSED' if not FAIL else 'SMOKE TESTS FAILED: ' + ', '.join(FAIL)}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
