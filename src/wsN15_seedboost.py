"""wsN15: 种子扩容（wsB 3→8、wsF 3→8、fuse 5→8）。

小种子族的方差缩减是诚实增益：wsB 承担 strain 角色 60% 权重却只有 3 种子。
全部沿用 train-only 口径与原配方，仅增加种子数。
产物（outputs/wsN15/）：
  pred_trainval_wsB_s8.npy / pred_test_wsB_s8.npy
  pred_trainval_wsF_s8.npy / pred_test_wsF_s8.npy
  pred_trainval_fuse_s8.npy / pred_test_fuse_s8.npy

用法: python -m src.wsN15_seedboost        # 约 25 分钟
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from . import wsB_twostage as B
from . import wsF_interact as WF
from . import wsA_chemfeat as WSA
from . import data as D
from .wsM_trainonly import strict_delta_train
from .wsN6_chemberta import build_bert_table, table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN15"
SEEDS = list(range(8))


def boost_wsB(h: Harness):
    """wsB strict（同 wsM.run_wsB），seeds=8。"""
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

    t0 = time.time()
    enc_c = B.Encoder(B.CTRL_COLS).fit(m_train.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xc_v = {k: torch.tensor(enc_c.transform(mv), dtype=torch.long,
                            device="cuda") for k, mv in views.items()}
    ctrl = {k: np.zeros((len(mv), n_prot), np.float32)
            for k, mv in views.items()}
    for sd in SEEDS:
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, 100, sd,
                              drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999,
                              tag=f"s8-ctrl s{sd}")
        for k in views:
            ctrl[k] += B.predict_model(model, Xc_v[k], "cuda") \
                .astype(np.float32)
        del model; torch.cuda.empty_cache()
    for k in views:
        ctrl[k] = ctrl[k] / len(SEEDS) * stats.protein_std \
            + stats.protein_mean
    print(f"[wsB-s8] control ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    enc_d = B.Encoder(B.DLT_COLS).fit(m_train.iloc[treat_rows_all])
    Xd = torch.tensor(enc_d.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xd_v = {k: torch.tensor(enc_d.transform(mv), dtype=torch.long,
                            device="cuda") for k, mv in views.items()}
    dlt = {k: np.zeros((len(mv), n_prot), np.float32)
           for k, mv in views.items()}
    for sd in SEEDS:
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows, 100,
                              sd, drop_cols=(0, 1), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999,
                              tag=f"s8-delta s{sd}")
        for k in views:
            dlt[k] += B.predict_model(model, Xd_v[k], "cuda") \
                .astype(np.float32)
        del model; torch.cuda.empty_cache()
    for k in views:
        dlt[k] /= len(SEEDS)
    print(f"[wsB-s8] delta ({time.time()-t0:.0f}s)", flush=True)

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
    np.save(OUT / "pred_trainval_wsB_s8.npy", outs["val"])
    np.save(OUT / "pred_test_wsB_s8.npy", outs["test"])
    res = h.score_val(outs["val"], verbose=False)
    print(f"[wsB-s8] composite={res['composite']:.4f} (3种子 0.5308)",
          flush=True)
    return res["composite"]


def boost_wsF(h: Harness):
    senc = WF.build_senc(h)
    from .wsM_trainonly import _senc_transform
    cfg = WF.CfgD1.from_dict({"hidden": [1024, 2048, 2048, 2048, 2048],
                              "inter_ctx": True, "inter_strain": False})
    X_all = torch.tensor(_senc_transform(senc, h.m_tr, True),
                         dtype=torch.long)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    Mk = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    tensors = (X_all, mean, std, Z, Mk)
    X_te = torch.tensor(_senc_transform(senc, h.m_te, True),
                        dtype=torch.long)
    pv, pt = [], []
    for sd in SEEDS:
        t0 = time.time()
        model, mean_t, std_t = WF.train_d1(h, cfg, sd, senc, log_every=999,
                                           tensors=tensors)
        pv.append(WF.predict_d1(model, X_all, mean_t, std_t))
        pt.append(WF.predict_d1(model, X_te, mean_t, std_t))
        print(f"[wsF-s8] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()
    P = np.mean(pv, axis=0).astype(np.float32)
    PT = np.mean(pt, axis=0).astype(np.float32)
    np.save(OUT / "pred_trainval_wsF_s8.npy", P)
    np.save(OUT / "pred_test_wsF_s8.npy", PT)
    res = h.score_val(P, verbose=False)
    print(f"[wsF-s8] composite={res['composite']:.4f} (3种子 0.5414)",
          flush=True)
    return res["composite"]


def boost_fuse(h: Harness):
    df = pd_read = __import__("pandas").read_csv(
        "outputs/wsN6/chem_features_fuse.csv")
    orig = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    try:
        pv, pt = [], []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=False)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            del model; torch.cuda.empty_cache()
            print(f"[fuse-s8] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
        # test 用 full 协议 43 化合物表
        df_full = pd_read = __import__("pandas").read_csv(
            "outputs/wsN6/chem_features_fuse_full.csv")
        WSA.load_chem_table = table_to_loader(df_full)
        for sd in SEEDS:
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=True)
            pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                  device="cuda"))
            del model; torch.cuda.empty_cache()
    finally:
        WSA.load_chem_table = orig
    P = np.mean(pv, axis=0).astype(np.float32)
    PT = np.mean(pt, axis=0).astype(np.float32)
    for arr in (P, PT):
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_fuse_s8.npy", P)
    np.save(OUT / "pred_test_fuse_s8.npy", PT)
    res = h.score_val(P, verbose=False)
    print(f"[fuse-s8] composite={res['composite']:.4f} (5种子 0.5142)",
          flush=True)
    return res["composite"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    out = {}
    out["wsB_s8"] = boost_wsB(h)
    out["wsF_s8"] = boost_wsF(h)
    out["fuse_s8"] = boost_fuse(h)
    (OUT / "scores.json").write_text(json.dumps(out, indent=1))
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
