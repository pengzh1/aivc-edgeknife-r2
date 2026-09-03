"""wsN9: wsB-strict 的 G2 组级增强变体（Δ̂ 模型组级 UNK 训练）。

wsD 的 G2（每 epoch 整组 UNK）是其 OOD 稳健性的关键之一；wsB 的 Δ̂ 模型
当前只有逐样本 emb_drop=0.25。本变体在 strict-Δ（train-only 对照池）口径下
给 Δ̂ 加 G2 组级增强（g2_cols=(0,1)），其余与 wsM 版 wsB 完全一致。

对照：wsM strict wsB val composite 0.5308（strain FC 0.3555）。

用法: python -m src.wsN9_wsBg2        # 3 种子，约 2 分钟
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .evaluate import Harness
from . import wsB_twostage as B
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN9"
SEEDS = [0, 1, 2]
EPOCHS = 100


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

    # Stage 1: control MLP（与 wsM 版相同）
    t0 = time.time()
    enc_c = B.Encoder(B.CTRL_COLS).fit(m_train.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xc_v = torch.tensor(enc_c.transform(h.m_tr), dtype=torch.long,
                        device="cuda")
    ctrl_tr = np.zeros((len(m_train), n_prot), np.float32)
    ctrl_v = np.zeros((len(h.m_tr), n_prot), np.float32)
    for sd in SEEDS:
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, EPOCHS,
                              sd, drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=999,
                              tag=f"wsN9-ctrl s{sd}")
        ctrl_tr += B.predict_model(model, Xc, "cuda").astype(np.float32)
        ctrl_v += B.predict_model(model, Xc_v, "cuda").astype(np.float32)
        del model; torch.cuda.empty_cache()
    ctrl_tr = ctrl_tr / len(SEEDS) * stats.protein_std + stats.protein_mean
    ctrl_v = ctrl_v / len(SEEDS) * stats.protein_std + stats.protein_mean
    print(f"[wsN9] control done ({time.time()-t0:.0f}s)", flush=True)

    # Stage 2: Δ̂ with G2（唯一变更点）
    t0 = time.time()
    enc_d = B.Encoder(B.DLT_COLS).fit(m_train.iloc[treat_rows_all])
    Xd = torch.tensor(enc_d.transform(m_train), dtype=torch.long,
                      device="cuda")
    Xd_v = torch.tensor(enc_d.transform(h.m_tr), dtype=torch.long,
                        device="cuda")
    dlt_tr = np.zeros((len(m_train), n_prot), np.float32)
    dlt_v = np.zeros((len(h.m_tr), n_prot), np.float32)
    for sd in SEEDS:
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows,
                              EPOCHS, sd, drop_cols=(0, 1), emb_drop=0.25,
                              lr=1e-3, bs=256, device="cuda", log_every=999,
                              tag=f"wsN9-delta s{sd}", g2_cols=(0, 1))
        dlt_tr += B.predict_model(model, Xd, "cuda").astype(np.float32)
        dlt_v += B.predict_model(model, Xd_v, "cuda").astype(np.float32)
        del model; torch.cuda.empty_cache()
    dlt_tr /= len(SEEDS)
    dlt_v /= len(SEEDS)
    print(f"[wsN9] delta(G2) done ({time.time()-t0:.0f}s)", flush=True)

    # 合成 val 预测（与 wsM 版同流程）
    pert_v = h.m_tr["perturbation_no_concentration"]
    is_treat_v = ~pert_v.isin(D.CONTROLS | {D.QC}).to_numpy()
    is_qc_v = (pert_v == D.QC).to_numpy()
    pred = ctrl_v.copy()
    pred[is_treat_v] += dlt_v[is_treat_v]
    if is_qc_v.any():
        pred[is_qc_v] = qc_model.predict(h.m_tr)[is_qc_v]
    pred = stats.impute(pred).astype(np.float32)
    res = h.score_val(pred, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[wsN9 g2] composite={res['composite']:.4f} FC={fc} resid={rz}")
    print("对照 wsM-strict wsB: 0.5308 | strain FC 0.3555 resid 0.3530")
    np.save(OUT / "pred_trainval.npy", pred)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "resid": rz,
         "per_split": res["per_split"]}, ensure_ascii=False, indent=1,
        default=float))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
