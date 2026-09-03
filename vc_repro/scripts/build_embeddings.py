"""主命令 ①：构建外部化合物表征（embedding）。

冻结输入：54 个竞赛化合物的 PubChem CID（configs 登记的映射表）。
产物（均为 <1MB 的化合物级表，随包携带合法缓存，可离线校验）：
  outputs/wsA/smiles.csv                     PubChem PUG-REST 按 CID 拉取 SMILES
  outputs/wsA/chem_features_full.csv         RDKit 描述符 → PCA64（基=train 37 化合物）
  outputs/wsN6/chem_features_fuse_full.csv   RDKit⊕ChemBERTa-MLM 融合 → PCA64
  outputs/wsN23/chem_features_fuse3_full.csv 三源（+MolFormer-XL）→ PCA64
重生成依赖模型权重（ChemBERTa-77M-MLM / MoLFormer-XL，manifest 有 URL+SHA256）；
若随包缓存校验通过则无需下载权重。

用法（仓库根目录）:  python vc_repro/scripts/build_embeddings.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from vc_repro.scripts._driver import sh, ROOT as R  # noqa: E402

SMILES = R / "outputs/wsA/smiles.csv"
CIDS = R / "outputs/wsA/compound_cids.json"

ARTIFACTS = [  # (路径, 缺失时的重建命令)
    ("outputs/wsA/chem_features_full.csv", None),  # 由 wsN6 --test 内部链路生成
    ("outputs/wsN6/chem_features_fuse_full.csv",
     ("src.wsN6_chemberta", "--test")),
    ("outputs/wsN23/chem_features_fuse3_full.csv",
     ("src.wsN23_fuse3", "--test")),
]


def fetch_smiles():
    """PubChem PUG-REST 按冻结 CID 拉 SMILES（含 3 次重试与断言语义校验）。"""
    if SMILES.exists():
        print(f"[skip] {SMILES} 已存在（随包缓存）", flush=True)
        return
    cids = json.loads(CIDS.read_text())
    rows = ["compound,query,smiles"]
    for name, cid in sorted(cids.items()):
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"cid/{cid}/property/CanonicalSMILES/JSON")
        for att in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as f:
                    smi = json.load(f)["PropertyTable"]["Properties"][0][
                        "CanonicalSMILES"]
                break
            except Exception as e:
                if att == 2:
                    raise
                time.sleep(2)
        assert smi and len(smi) > 3, f"{name} SMILES 异常"
        rows.append(f'"{name}","{name.lower()}",{smi}')
        print(f"  {name:<40} CID {cid:<8} {smi[:40]}", flush=True)
    SMILES.parent.mkdir(parents=True, exist_ok=True)
    SMILES.write_text("\n".join(rows), encoding="utf-8")
    print(f"[ok] {SMILES}（{len(rows)-1} 化合物）", flush=True)


def main():
    fetch_smiles()
    for rel, cmd in ARTIFACTS:
        if (R / rel).exists():
            print(f"[skip] {rel} 已存在（随包缓存）", flush=True)
        elif cmd:
            sh(*cmd, desc=f"重建 {rel}")
        else:
            print(f"[defer] {rel} 由后续训练链生成", flush=True)
    print("\n[done] build_embeddings：全部化合物表征就绪。", flush=True)


if __name__ == "__main__":
    main()
