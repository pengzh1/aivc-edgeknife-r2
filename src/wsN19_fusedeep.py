"""wsN19: fuse 深度变体（深 fuse MLP；fuse 是唯一持续上涨的族）。

fuse 轨迹：5种子 0.5142 → 8种子 0.5246 → 16种子 0.5279。试架构加深：
  deep3: hidden=(1024,2048,2048) p_drop=0.2
  deep4: hidden=(1024,2048,2048,2048) p_drop=0.3
monkey-patch ProteoMLPChem 默认 hidden/p_drop（不改原文件），8 种子 val 对照。

用法: python -m src.wsN19_fusedeep
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN19"
SEEDS = list(range(8))

VARIANTS = {
    "deep3": ((1024, 2048, 2048), 0.2),
    "deep4": ((1024, 2048, 2048, 2048), 0.3),
}


def run(h, tag, hidden, p_drop):
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot, hidden=hidden,
                  p_drop=p_drop)
    WSA.ProteoMLPChem.__init__ = new_init
    df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv")
    orig_loader = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    try:
        pv = []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=False)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            print(f"[{tag}] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[{tag}] composite={res['composite']:.4f} FC={fc} resid={rz}",
          flush=True)
    np.save(OUT / f"pred_trainval_{tag}.npy", P)
    return {"composite": res["composite"], "FC": fc, "resid": rz,
            "per_split": res["per_split"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    summary = {}
    for tag, (hidden, p_drop) in VARIANTS.items():
        summary[tag] = run(h, tag, hidden, p_drop)
    (OUT / "scores.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, default=float))
    print("对照 fuse-s8 (512,1024): 0.5246 | fuse-s16: 0.5279")


if __name__ == "__main__":
    main()
