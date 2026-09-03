"""wsN16: 第二轮扩容（fuse 8→16、wsB 8→16）+ bagged wsD 变体。

fuse 5→8 曾 +0.0104（方差缩减未饱和）；wsB 3→8 曾 +0.0039。继续翻倍。
bagged wsD：80% 行子采样 × 2 个 bag × 3 种子（集成去相关）。

全部 train-only、原配方；产物落 outputs/wsN16/。

用法: python -m src.wsN16_more
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from . import wsB_twostage as B
from . import wsA_chemfeat as WSA
from . import data as D
from .wsM_trainonly import strict_delta_train
from .wsN6_chemberta import table_to_loader
from .wsD_arch import Cfg, train_one, predict, seen_cats

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN16"
NEW = list(range(8, 16))


def extend_fuse(h):
    import pandas as pd
    df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv")
    orig = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    pv, pt = [], []
    try:
        for sd in NEW:
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=False)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            del model; torch.cuda.empty_cache()
        df_full = pd.read_csv("outputs/wsN6/chem_features_fuse_full.csv")
        WSA.load_chem_table = table_to_loader(df_full)
        for sd in NEW:
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=True)
            pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                  device="cuda"))
            del model; torch.cuda.empty_cache()
    finally:
        WSA.load_chem_table = orig
    old_v = np.load("outputs/wsN15/pred_trainval_fuse_s8.npy")
    old_t = np.load("outputs/wsN15/pred_test_fuse_s8.npy")
    P = (old_v + np.mean(pv, axis=0).astype(np.float32)) / 2
    PT = (old_t + np.mean(pt, axis=0).astype(np.float32)) / 2
    for arr in (P, PT):
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_fuse_s16.npy", P)
    np.save(OUT / "pred_test_fuse_s16.npy", PT)
    res = h.score_val(P, verbose=False)
    print(f"[fuse-s16] composite={res['composite']:.4f} (s8 0.5246)",
          flush=True)
    return res["composite"]


def extend_wsB(h):
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
    views = {"val": h.m_tr, "test": h.m_te}
    enc_c = B.Encoder(B.CTRL_COLS).fit(m_train.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xc_v = {k: torch.tensor(enc_c.transform(mv), dtype=torch.long,
                            device="cuda") for k, mv in views.items()}
    ctrl = {k: np.zeros((len(mv), n_prot), np.float32)
            for k, mv in views.items()}
    for sd in NEW:
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, 100, sd,
                              drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        for k in views:
            ctrl[k] += B.predict_model(model, Xc_v[k], "cuda") \
                .astype(np.float32)
        del model; torch.cuda.empty_cache()
    for k in views:
        ctrl[k] = ctrl[k] / len(NEW) * stats.protein_std + stats.protein_mean
    enc_d = B.Encoder(B.DLT_COLS).fit(m_train.iloc[treat_rows_all])
    Xd = torch.tensor(enc_d.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xd_v = {k: torch.tensor(enc_d.transform(mv), dtype=torch.long,
                            device="cuda") for k, mv in views.items()}
    dlt = {k: np.zeros((len(mv), n_prot), np.float32)
           for k, mv in views.items()}
    for sd in NEW:
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows, 100,
                              sd, drop_cols=(0, 1), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        for k in views:
            dlt[k] += B.predict_model(model, Xd_v[k], "cuda") \
                .astype(np.float32)
        del model; torch.cuda.empty_cache()
    for k in views:
        dlt[k] /= len(NEW)
    outs = {}
    for k, mv in views.items():
        pert_v = mv["perturbation_no_concentration"]
        is_treat_v = ~pert_v.isin(D.CONTROLS | {D.QC}).to_numpy()
        is_qc_v = (pert_v == D.QC).to_numpy()
        pred = ctrl[k].copy()
        pred[is_treat_v] += dlt[k][is_treat_v]
        if is_qc_v.any():
            pred[is_qc_v] = qc_model.predict(mv)[is_qc_v]
        outs[k] = stats.impute(pred).astype(np.float32)
    old_v = np.load("outputs/wsN15/pred_trainval_wsB_s8.npy")
    old_t = np.load("outputs/wsN15/pred_test_wsB_s8.npy")
    P = (old_v + outs["val"]) / 2
    PT = (old_t + outs["test"]) / 2
    np.save(OUT / "pred_trainval_wsB_s16.npy", P)
    np.save(OUT / "pred_test_wsB_s16.npy", PT)
    res = h.score_val(P, verbose=False)
    print(f"[wsB-s16] composite={res['composite']:.4f} (s8 0.5347)",
          flush=True)
    return res["composite"]


def bag_wsD(h):
    cfg = Cfg.from_dict({"hidden": [1024, 2048, 2048, 2048, 2048],
                         "epochs": 300, "lr": 0.001, "wd": 0.0001,
                         "emb_drop": 0.35, "p_drop": 0.3, "bs": 256,
                         "chem_emb": 32, "residual": False, "lowrank": 0,
                         "loss": "huber", "film": False, "g2_aug": True})
    seen = seen_cats(h, h.tr_rows)
    for bag in (0, 1):
        pv, pt = [], []
        for sd in (0, 1, 2):
            rng = np.random.default_rng(1000 + bag)
            rows = np.sort(rng.choice(h.tr_rows,
                                      size=int(len(h.tr_rows) * 0.8),
                                      replace=False))
            t0 = time.time()
            model, enc, mean, std, _ = train_one(h, cfg, sd, rows=rows)
            pv.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
            pt.append(predict(model, enc, mean, std, h.m_te, g3_seen=seen))
            print(f"[bag{bag}] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()
        P = np.mean(pv, axis=0).astype(np.float32)
        np.save(OUT / f"pred_trainval_bag{bag}.npy", P)
        np.save(OUT / f"pred_test_bag{bag}.npy",
                np.mean(pt, axis=0).astype(np.float32))
        res = h.score_val(P, verbose=False)
        print(f"[bag{bag}] composite={res['composite']:.4f}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    extend_fuse(h)
    extend_wsB(h)
    bag_wsD(h)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
