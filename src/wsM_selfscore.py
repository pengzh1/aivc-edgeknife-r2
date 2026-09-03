"""wsM: test 真值一次性自评（官方口径）——仅报告，绝不回馈建模。

许可依据：2026-08-10 组委会书面答复（outref/replymail.txt Q4）——
"测试集蛋白质组真值确实随赛题数据一并发放，其成绩供参赛者自行参考"；
新版手册 P15/P17 红字确认"测试集成绩供自评参考，不作最终排名依据"。

红线约束（用户 2026-08-11 批准一次性自评 + 回退前后对比）：
- 本脚本只读 Y_te、只写报告 JSON；不得被任何训练/调参/模型选择代码 import；
- 结果仅用于记录与提交文档叙述；不据此调整任何模型/权重/口径
  （若未来需要据此调整，必须重新开会讨论并记录）。

官方口径要点（replymail Q4/Q5 + 手册 P17/P19）：
- 真值 = test 文件 raw intensity 的 log2，不做额外归一化；
- 评测位置 = 真值非缺失 ∩ train 保留蛋白列表（本地用 train miss<80% 的
  4,422 列近似官方 4,232 feature contract；官方列表到达后用 --keep 重跑）；
- y_control = 测试集真实对照（7 键匹配，重复取均值）；
- 残差参照 μ_ctx/μ_drug = train split 冻结（Harness 同款）；
- composite 权重与 metrics.W 一致（fid .20/FC .25/ctx .20/drug .20/
  both_time .10/DEP .05）。

用法:
    python -m src.wsM_selfscore --preds outputs/prediction.csv outputs/prediction_trainonly.csv
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .evaluate import Harness

OUT = Path("outputs/wsM")
TE_SPLITS = ["test_chem_only", "test_strain_only", "test_both", "test_time"]


def build_test_delta(h: Harness):
    """test 处理样本的 Δ_true / Δ_pred 对照基线（y_control = test 真实对照）。"""
    idx_te = {s: i for i, s in enumerate(h.m_te["sample_ID"])}
    delta_true = np.full_like(h.Y_te, np.nan)
    ctrl_mean = np.full_like(h.Y_te, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, sid in enumerate(h.m_te["sample_ID"]):
            ctrls = [idx_te[c] for c in h.ctrl_map.get(sid, []) if c in idx_te]
            if not ctrls:
                continue
            cmean = np.nanmean(h.Y_te[ctrls], axis=0)
            delta_true[i] = h.Y_te[i] - cmean
            ctrl_mean[i] = cmean
    is_treat = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    return delta_true, ctrl_mean, is_treat


def score_test(h: Harness, pred: np.ndarray, keep_mask: np.ndarray,
               delta_true: np.ndarray, ctrl_mean: np.ndarray,
               is_treat: np.ndarray) -> dict:
    """官方口径逐划分评分（仅在 真值非缺失 ∩ keep 列 上）。

    Δ 类指标（FC/resid/DEP）只在"匹配到 test 真实对照"的处理行上计算
    （无匹配对照的行 Δ_true 未定义，剔除而非计 0——test 对照仅 202 行，
    chem_only 匹配率 3%、time 6%，计 0 会造成稀释性伪坍塌）。
    """
    Yk = h.Y_te[:, keep_mask]
    pk = pred[:, keep_mask]
    has_ctrl = ~np.isnan(ctrl_mean).all(axis=1)
    per_split = {}
    for split in TE_SPLITS:
        rows = np.where((h.m_te["split_final"] == split).to_numpy())[0]
        yt, yp = Yk[rows], pk[rows]
        fid = M.fidelity_scores(yt, yp)
        s = {"sample_PCC": fid["sample_PCC"], "sample_R2": fid["sample_R2"],
             "protein_PCC": fid["protein_PCC"],
             "fidelity": float(np.mean([fid["sample_PCC"], fid["sample_R2"],
                                        fid["protein_PCC"]]))}
        trows_all = rows[is_treat[rows]]
        trows = trows_all[has_ctrl[trows_all]]
        s["n_treat"] = int(len(trows_all))
        s["n_ctrl_matched"] = int(len(trows))
        dt_full = delta_true[:, keep_mask]
        # Δ_pred = ŷ_treat − test 真实对照均值（keep 列）
        dpred_full = np.full((len(h.m_te), int(keep_mask.sum())), np.nan,
                             dtype=np.float32)
        dpred_full[is_treat] = (pred[is_treat][:, keep_mask]
                                - ctrl_mean[is_treat][:, keep_mask])
        if len(trows):
            s.update(M.fc_scores(dt_full, dpred_full, trows))
            if "chem_only" in split:
                mu = h.mu_ctx_for(h.m_te.iloc[trows])[:, keep_mask]
                s["resid_PCC"] = M.residual_scores(dt_full, dpred_full, mu,
                                                   trows)["resid_PCC"]
            if "strain_only" in split:
                mu = h.mu_drug_for(h.m_te.iloc[trows])[:, keep_mask]
                s["resid_PCC"] = M.residual_scores(dt_full, dpred_full, mu,
                                                   trows)["resid_PCC"]
            s.update(M.dep_scores(dt_full, dpred_full, trows))
        per_split[split] = s
    return {"per_split": per_split, "composite": M.composite(per_split)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="待评 prediction CSV 或 npy（与 m_te 行序一致）")
    ap.add_argument("--keep", default=None,
                    help="保留蛋白列表 CSV（列名 protein）；缺省用 train miss<80%")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    h = Harness()
    if args.keep:
        keep_names = pd.read_csv(args.keep)["protein"].to_numpy()
        keep_mask = np.isin(h.proteins, keep_names)
    else:
        keep_mask = np.isnan(h.Y_train).mean(axis=0) < 0.80
    print(f"[selfscore] keep proteins = {int(keep_mask.sum())} "
          f"({'官方列表' if args.keep else 'train miss<80% 本地复现'})")
    delta_true, ctrl_mean, is_treat = build_test_delta(h)
    n_ctrl_hit = int((is_treat & ~np.isnan(ctrl_mean).all(axis=1)).sum())
    print(f"[selfscore] test 处理样本 {int(is_treat.sum())} 行，"
          f"匹配到 test 真实对照 {n_ctrl_hit} 行")

    report = {"keep_n": int(keep_mask.sum()), "splits": TE_SPLITS,
              "note": "官方口径自评（log2、非缺失∩keep、test 真实对照）；"
                      "仅供自评参考，不作最终排名依据；本报告不回馈建模。",
              "results": {}}
    for p in args.preds:
        path = Path(p)
        if path.suffix == ".npy":
            pred = np.load(path)
            keep_eff = keep_mask
        else:
            # CSV：按列名回填到 5,243 全列；评分位置随之限定到提交列 ∩ keep
            # （官方口径：只在属于保留列表的位置计分；未提交列不计）
            df = pd.read_csv(path)
            full = pd.DataFrame(np.nan, index=range(len(df)),
                                columns=h.proteins)
            for c in df.columns[1:]:
                full[c] = df[c].to_numpy()
            pred = full.to_numpy(dtype=np.float32)
            keep_eff = keep_mask & np.isin(h.proteins, df.columns[1:])
            if len(df.columns) - 1 < len(h.proteins):
                print(f"[note] {path.name} 为裁剪版（{len(df.columns)-1} 列），"
                      f"评分位置 = 提交列 ∩ keep（{int(keep_eff.sum())} 列）")
        assert pred.shape == (len(h.m_te), len(h.proteins)), (p, pred.shape)
        t0 = __import__("time").time()
        res = score_test(h, pred, keep_eff, delta_true, ctrl_mean, is_treat)
        report["results"][path.name] = res
        print(f"\n===== {path.name} =====")
        Harness._print(res["per_split"], res["composite"])
        print(f"({__import__('time').time()-t0:.0f}s)")

    tag = f"_{args.tag}" if args.tag else ""
    out = OUT / f"selfscore{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
