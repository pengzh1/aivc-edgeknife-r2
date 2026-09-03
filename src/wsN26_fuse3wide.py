"""wsN26: fuse3-deep3 150 epochs（容量族训练量加倍）。

chem 特征族一直用 wsA 默认 100 epochs；deep3 容量翻倍后训练量可能欠配。
16 种子 val 对照 fuse3-deep3-100ep 的 0.5397。

用法: python -m src.wsN26_fuse3e150 [--test]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN26"
SEEDS = list(range(16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot,
                  hidden=(1536, 3072, 3072), p_drop=0.2)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    table = ("outputs/wsN23/chem_features_fuse3_full.csv" if args.test
             else "outputs/wsN23/chem_features_fuse3.csv")
    df = pd.read_csv(table)
    WSA.load_chem_table = table_to_loader(df)
    try:
        pv, pt = [], []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 150, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999,
                full=args.test)
            if args.test:
                pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                      device="cuda"))
            else:
                pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                      device="cuda"))
            print(f"[fuse3-wide] seed={sd} ({time.time()-t0:.0f}s)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    if args.test:
        PT = np.mean(pt, axis=0).astype(np.float32)
        bad = ~np.isfinite(PT)
        if bad.any():
            r, c = np.where(bad)
            PT[r, c] = np.take(h.stats.protein_mean, c)
        np.save(OUT / "pred_test.npy", PT)
        print(f"[saved] {OUT/'pred_test.npy'}")
        return
    P = np.mean(pv, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval.npy", P)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[fuse3-wide] composite={res['composite']:.4f} "
          f"(fuse3-e100: 0.5397) FC={fc}")
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc,
         "per_split": res["per_split"]}, default=float, indent=1))


if __name__ == "__main__":
    main()
