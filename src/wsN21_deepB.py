"""wsN21: 容量深化第二梯队（deep-wsB + deep4-s16）。

wsN19/20 教训：浅层（512,1024）是特征族的容量瓶颈（deep3: +0.011）。
同法加深 wsB 的 CondMLP（默认 512,1024）与 fuse-deep4：
  - deepB: wsB ctrl/Δ̂ MLP hidden=(1024,2048,2048) p_drop=0.2，8 种子
  - deep4s16: fuse (1024,2048×4) p_drop=0.3，16 种子（wsN19 探索 8 种子 0.5330）

用法: python -m src.wsN21_deepB
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import wsB_twostage as B
from . import wsA_chemfeat as WSA
from . import data as D
from .wsM_trainonly import strict_delta_train
from .wsN6_chemberta import table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN21"


def deep_wsB(h):
    m_train, Y_train = h.m_train, h.Y_train
    n_prot = h.Y_tr.shape[1]
    pert = m_train["perturbation_no_concentration"]
    is_ctrl = pert.isin(D.CONTROLS).to_numpy()
    is_qc = (pert == D.QC).to_numpy()
    ctrl_rows = np.where(is_ctrl)[0]
    qc_rows = np.where(is_qc)[0]
    delta, treat_rows_all, treat_rows = strict_delta_train(h)
    stats = h.stats
    mean_t = torch.tensor(stats.protein_mean, dtype=torch.float32,
                          device="cuda")
    std_t = torch.tensor(stats.protein_std, dtype=torch.float32,
                         device="cuda")
    Z = (torch.tensor(Y_train, dtype=torch.float32, device="cuda")
         - mean_t) / std_t
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    Dmat = torch.tensor(delta, dtype=torch.float32, device="cuda")
    Md = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)
    qc_model = B.QCGroupMean().fit(
        m_train.iloc[qc_rows].reset_index(drop=True), Y_train[qc_rows],
        stats.protein_mean)
    HIDDEN = (1024, 2048, 2048)

    enc_c = B.Encoder(B.CTRL_COLS).fit(m_train.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xc_v = torch.tensor(enc_c.transform(h.m_tr), dtype=torch.long,
                        device="cuda")
    ctrl_tr = np.zeros((len(m_train), n_prot), np.float32)
    ctrl_v = np.zeros((len(h.m_tr), n_prot), np.float32)
    for sd in range(8):
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot,
                          hidden=HIDDEN, p_drop=0.2)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, 100, sd,
                              drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        ctrl_tr += B.predict_model(model, Xc, "cuda").astype(np.float32)
        ctrl_v += B.predict_model(model, Xc_v, "cuda").astype(np.float32)
        del model; torch.cuda.empty_cache()
    ctrl_tr = ctrl_tr / 8 * stats.protein_std + stats.protein_mean
    ctrl_v = ctrl_v / 8 * stats.protein_std + stats.protein_mean
    print("[deepB] control done", flush=True)

    enc_d = B.Encoder(B.DLT_COLS).fit(m_train.iloc[treat_rows_all])
    Xd = torch.tensor(enc_d.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xd_v = torch.tensor(enc_d.transform(h.m_tr), dtype=torch.long,
                        device="cuda")
    dlt_tr = np.zeros((len(m_train), n_prot), np.float32)
    dlt_v = np.zeros((len(h.m_tr), n_prot), np.float32)
    for sd in range(8):
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot,
                          hidden=HIDDEN, p_drop=0.2)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows, 100,
                              sd, drop_cols=(0, 1), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        dlt_tr += B.predict_model(model, Xd, "cuda").astype(np.float32)
        dlt_v += B.predict_model(model, Xd_v, "cuda").astype(np.float32)
        del model; torch.cuda.empty_cache()
    dlt_tr /= 8
    dlt_v /= 8
    pert_v = h.m_tr["perturbation_no_concentration"]
    is_treat_v = ~pert_v.isin(D.CONTROLS | {D.QC}).to_numpy()
    is_qc_v = (pert_v == D.QC).to_numpy()
    pred = ctrl_v.copy()
    pred[is_treat_v] += dlt_v[is_treat_v]
    if is_qc_v.any():
        pred[is_qc_v] = qc_model.predict(h.m_tr)[is_qc_v]
    pred = stats.impute(pred).astype(np.float32)
    np.save(OUT / "pred_trainval_deepB.npy", pred)
    res = h.score_val(pred, verbose=False)
    print(f"[deepB] composite={res['composite']:.4f} (wsB_s16 0.5359)",
          flush=True)
    return res["composite"]


def deep4_s16(h):
    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot,
                  hidden=(1024, 2048, 2048, 2048), p_drop=0.3)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    try:
        df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv")
        WSA.load_chem_table = table_to_loader(df)
        pv = []
        for sd in range(16):
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=False)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            print(f"[deep4-s16] seed={sd} ({time.time()-t0:.0f}s)",
                  flush=True)
            del model; torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_deep4_s16.npy", P)
    res = h.score_val(P, verbose=False)
    print(f"[deep4-s16] composite={res['composite']:.4f} (deep3_s16 0.5390)",
          flush=True)
    return res["composite"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    out = {"deepB": deep_wsB(h), "deep4_s16": deep4_s16(h)}
    (OUT / "scores.json").write_text(json.dumps(out, indent=1))
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
