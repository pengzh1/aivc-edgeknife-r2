"""wsN20: fuse-deep3（1024,2048,2048 / p_drop 0.2）16 种子交付版 + test 预测。

wsN19 探索结果：deep3 单族 0.5367（8 种子），超越 fuse-s16（0.5279）成为
第二强族。本模块：16 种子 val（full=False 协议）+ 16 种子 test（full=True，
43 化合物 PCA 表）。

用法: python -m src.wsN20_deep3full
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

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN20"
SEEDS = list(range(16))
HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot, hidden=HIDDEN,
                  p_drop=P_DROP)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    try:
        # val 协议（37 化合物 PCA 表）
        df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv")
        WSA.load_chem_table = table_to_loader(df)
        pv = []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=False)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            print(f"[deep3-s16-val] seed={sd} ({time.time()-t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        # test 协议（43 化合物 PCA 表 + test 化合物词表扩展）
        df_full = pd.read_csv("outputs/wsN6/chem_features_fuse_full.csv")
        WSA.load_chem_table = table_to_loader(df_full)
        pt = []
        for sd in SEEDS:
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=True)
            pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                  device="cuda"))
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    PT = np.mean(pt, axis=0).astype(np.float32)
    for arr in (P, PT):
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_deep3_s16.npy", P)
    np.save(OUT / "pred_test_deep3_s16.npy", PT)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[deep3-s16] composite={res['composite']:.4f} FC={fc} resid={rz}")
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "resid": rz,
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}/pred_*_s16.npy")


if __name__ == "__main__":
    main()
