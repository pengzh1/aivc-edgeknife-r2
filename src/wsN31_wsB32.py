"""wsN31: wsB 两阶段 16→32 种子（strain 侧主锚族的方差收缩）。

wsB 在路由 strain 划分权重 0.86（r=0.7 口径 0.669），是集成中杠杆最高的单族；
其种子曲线未饱和（3→8 +0.0039、8→16 +0.0012 估）。16→32 预期单族 +0.0005~0.001，
经 strain/both 权重折算 composite +0.0002~0.0004——当前所剩不多杠杆最大的安全增益。
机制与 wsN28（fuse3e150 s32）同构：增量训练 seeds 16-31，与既有 s16 等权合并。

用法: python -m src.wsN31_wsB32
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from . import wsB_twostage as B
from . import data as D
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN31"
NEW = list(range(16, 32))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
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
        t0 = time.time()
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, 100, sd,
                              drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        for k in views:
            ctrl[k] += B.predict_model(model, Xc_v[k], "cuda") \
                .astype(np.float32)
        print(f"[wsB-s32] ctrl seed={sd} ({time.time()-t0:.0f}s)", flush=True)
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
        t0 = time.time()
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows, 100,
                              sd, drop_cols=(0, 1), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999, tag="")
        for k in views:
            dlt[k] += B.predict_model(model, Xd_v[k], "cuda") \
                .astype(np.float32)
        print(f"[wsB-s32] dlt seed={sd} ({time.time()-t0:.0f}s)", flush=True)
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
    # 与 s16 等权合并
    old_v = np.load("outputs/wsN16/pred_trainval_wsB_s16.npy")
    old_t = np.load("outputs/wsN16/pred_test_wsB_s16.npy")
    P = (old_v + outs["val"]) / 2
    PT = (old_t + outs["test"]) / 2
    for arr in (P, PT):
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / "pred_trainval_wsB_s32.npy", P)
    np.save(OUT / "pred_test_wsB_s32.npy", PT)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsB-s32] composite={res['composite']:.4f} (s16 0.5359) FC={fc}")
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc,
         "per_split": res["per_split"],
         "baseline_s16": 0.5359}, default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
