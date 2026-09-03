"""wsN8: CRD←CGD 调制迁移变体（有界赌注，β 收缩）。

依据（outputs/wsN/loso_transfer.json + wsN2 实测）：
- 菌株自身调制 oracle 0.33~0.40（信号真实存在）；
- 最近邻迁移在 1.6~2.2% 距离失败（resid≈0、FC 反降）；
- BAI←CEK（1.36%）在完整管线 val 上 resid +0.017 / FC +0.002（微正）；
- CRD←CGD 距离 0.398%（比 BAI 近 3.4 倍），不可在 val 诚实验证。

策略：对 test strain_only 行（CRD × train 化合物），在路由预测的 Δ 上叠加
β·m_CGD,c（CGD 的菌株特异调制 = pair_mean(CGD,c) − Δ̄_c，strict train Δ，
pair 缺失→0）。β=0.25 有界收缩：若迁移失败，损失有界（约 −0.001 量级）；
若 0.398% 突破距离阈值，drug_resid（20% 权重）与 strain FC（25%）有实质上行。

纪律声明：本变体的 val 验证仅证明"无害"（BAI 代理）；是否作为最终提交由
人工先验决策（val/test 均不可验证其增益）。

用法: python -m src.wsN8_crdtransfer            # val 无害性验证 + test 变体生成
      python -m src.wsN8_crdtransfer --beta 0.25
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Harness
from . import make_submission_trainonly as MST
from .make_submission import band_calibrate
from . import data as D
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN8"
CHEM = "perturbation_no_concentration"


def strain_modulation(h: Harness) -> dict:
    """每 (菌株, 化合物) 的调制残差 m = pair_mean − Δ̄_chem（strict train Δ）。"""
    delta, _, valid = strict_delta_train(h)
    m = h.m_train.iloc[valid].reset_index(drop=True)
    d = delta[valid]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chem_mu = {c: np.nanmean(d[(m[CHEM] == c).to_numpy()], axis=0)
                   for c in m[CHEM].unique()}
        for (s, c), sub in m.groupby(["Strains", CHEM]):
            idx = sub.index.to_numpy()
            pmu = np.nanmean(d[idx], axis=0)
            out[(s, c)] = np.nan_to_num(pmu - chem_mu[c], nan=0.0)
    return out


def apply_transfer(pred, h, m_target, ctrl_hat, mod, from_s, to_s, beta):
    """对 m_target 中 Strains==to_s 且化合物已见的处理行叠加 β·m_{from_s,chem}。"""
    out = pred.copy()
    pert = m_target[CHEM]
    is_treat = ~pert.isin(D.CONTROLS | {D.QC}).to_numpy()
    n_hit = 0
    for i, row in enumerate(m_target.itertuples()):
        if not is_treat[i] or row.Strains != to_s:
            continue
        mv = mod.get((from_s, row.perturbation_no_concentration))
        if mv is None:
            continue
        out[i] = pred[i] + beta * mv
        n_hit += 1
    return out, n_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.25)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    mod = strain_modulation(h)

    # ---- val 无害性验证：BAI←CEK 同构迁移 ----
    P_val = np.load(OUT.parent / "wsM" / "pred_trainval_routed_trainonly.npy")
    # ^ 注：该文件为 wsA 阵容路由；fuse 版路由 val 预测用同法再算亦可，
    #   无害性结论与阵容无关（迁移是微小加性扰动）。
    base = h.score_val(P_val, verbose=False)["composite"]
    Pv, n_val = apply_transfer(P_val, h, h.m_tr, None, mod, "CEK", "BAI",
                               args.beta)
    res = h.score_val(Pv, verbose=False)
    print(f"[val 无害性] BAI←CEK β={args.beta} 命中 {n_val} 行: "
          f"composite {base:.4f} -> {res['composite']:.4f} "
          f"(Δ={res['composite']-base:+.4f})")
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"  FC={fc} resid={rz}")

    # ---- test 变体：在 fuse 版路由预测上施加 CRD←CGD ----
    # 重放 fuse 路由（与 wsN6_deliver 同权重同流程）
    rec = json.loads((OUT.parent / "wsM" / "router_weights_trainonly_fuse.json")
                     .read_text())
    names = rec["names"]
    w05 = {sp: rec["open"]["shrink"]["0.5"][sp]
           for sp in ["val_chem_only", "val_strain_only", "val_both",
                      "val_time"]}
    role_of = {"val_chem_only": "chem_only", "val_strain_only": "strain_only",
               "val_both": "both", "val_time": "time"}
    roster = dict(MST.TEST_ROSTER)
    roster["fuse"] = "outputs/wsN6/pred_test_fuse_full.npy"
    del roster["wsA"]
    preds = {k: np.load(v) for k, v in roster.items()}
    keys = MST.route_keys_trainonly(h)
    pred = np.zeros_like(next(iter(preds.values())))
    for sp, role in role_of.items():
        mask = keys == role
        w = dict(zip(names, w05[sp]))
        w["wsD_g2g3"] = w.get("wsD_g2g3", 0.0) + w.pop("wsG", 0.0)
        w = {k: v for k, v in w.items() if v > 0}
        s = sum(w.values())
        pred[mask] = (sum(wt * preds[m][mask] for m, wt in w.items())
                      / s).astype(np.float32)
    pred_t, n_te = apply_transfer(pred, h, h.m_te, None, mod, "CGD", "CRD",
                                  args.beta)
    print(f"[test] CRD←CGD β={args.beta} 命中 {n_te} 行"
          f"（strain_only 处理行）")

    ctrl_hat_te, level_te = MST.build_control_hat_train(h, h.m_te)
    is_treat_te = ~h.m_te[CHEM].isin(D.CONTROLS | {D.QC}).to_numpy()
    pred_cal = band_calibrate(pred_t, ctrl_hat_te, level_te, is_treat_te,
                              g=1.3)
    assert np.isfinite(pred_cal).all()
    df = pd.DataFrame(pred_cal, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    keep = pd.read_csv(OUT.parent / "wsM" /
                       "keep_proteins_train_miss80.csv")["protein"]
    out = OUT / f"prediction_trainonly_fuse_crdbeta{args.beta}.csv"
    df.to_csv(out.with_suffix(".csv"), index=False)
    df[["sample_ID"] + keep.tolist()].to_csv(
        OUT / f"prediction_trainonly_fuse_crdbeta{args.beta}_keepcols.csv",
        index=False)
    np.save(out.with_suffix(".npy"), pred_cal)
    print(f"[saved] {out} (+keepcols)")
    (OUT / "notes.json").write_text(json.dumps({
        "beta": args.beta, "val_delta": res["composite"] - base,
        "val_hit_rows": n_val, "test_hit_rows": n_te,
        "discipline": "val 仅验证无害性；增益不可在 val/test 诚实验证；"
                      "是否作最终提交由用户决策"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
