"""wsT8：C7/C8 合并裁决表（第二轮唯一一次 val 看）。

预注册候选集（先于任何 val 计算）：
- C8  ：hgb_full（10 特征）P + wsT7 的 OOD τ 规则
        {chem←τ_loco, strain←τ_loso, both←τ_both, time←τ_oof}
- C7  ：hgb14（+Messner 4 特征）P + 全局 τ_c7（仅当 wsT6 train 闸过）
- C7+C8：hgb14 P + C8 τ 规则（τ 跨特征集迁移，记为零成本近似；仅当 C7 过闸）
采纳规则：max DEP_F1 均值，且 composite ≥ 0.5541−0.0005，且 F1 ≥ 0.2442+0.003
才替换 C1；全部不过 → 维持 C1，第二轮封盘。

合规：各组件均 train-only 拟合/调定；本表为唯一 val 使用；Y_te 零接触。

用法: python -m src.wsT8_arbitrate2
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .evaluate import Harness
from .wsN11_grandrouter import SPLITS
from .wsT1_depgate import CACHE as T1C, MIN_PUSH, PUSH_TO, _predict_val

OUT = Path("outputs/wsT8")
T0C = Path("outputs/wsT0/cache")
T6C = Path("outputs/wsT6/cache")
T7C = Path("outputs/wsT7/cache")


def apply_rule(h, routed, ctrl_hat, P_by_split, tau_by_split, tag):
    pred = routed.astype(np.float64)
    total = 0
    for sp in SPLITS:
        vrows = np.load(T1C / f"val_rows_{sp}.npy")
        P = P_by_split[sp]
        d_v = (routed[vrows] - ctrl_hat[vrows]).astype(np.float64)
        fl = (P >= tau_by_split[sp]) & (np.abs(d_v) >= MIN_PUSH)
        tgt = np.sign(d_v) * np.maximum(np.abs(d_v), PUSH_TO)
        pred[vrows] = pred[vrows] + np.where(fl, tgt - d_v, 0.0)
        total += int(fl.sum())
    print(f"  [{tag}] val flags={total:,}", flush=True)
    return pred.astype(np.float32)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load(T0C / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0C / "control_hat.npy")
    taus = json.loads((T7C / "taus.json").read_text())
    tau_rule = {"val_chem_only": taus["tau_chem"][0],
                "val_strain_only": taus["tau_strain"][0],
                "val_both": taus["tau_both"][0],
                "val_time": taus["tau_time"][0]}
    print(f"[arb] τ 规则: {tau_rule}", flush=True)

    clf10 = joblib.load(T1C / "hgb_full.joblib")
    P10 = {sp: _predict_val(h, clf10, sp)[1] for sp in SPLITS}

    table = {}
    cands = [("C8", P10, tau_rule)]

    t6_report = T6C / "wsT6_report.json"
    if t6_report.exists():
        rep6 = json.loads(t6_report.read_text())
        if rep6.get("verdict") == "GATE_PASSED":
            P14 = {sp: np.load(T6C / f"P14_{sp}.npy") for sp in SPLITS}
            cands.append(("C7", P14, {sp: rep6["tau"] for sp in SPLITS}))
            cands.append(("C7+C8", P14, tau_rule))
            print(f"[arb] C7 过闸（AUC14={rep6['auc14']:.4f}），并入裁决",
                  flush=True)
        else:
            print("[arb] C7 未过闸，不入表", flush=True)
    else:
        print("[arb] 无 C7（闸未过或数据未到）", flush=True)

    f1 = lambda r: float(np.mean([r["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    for tag, P, tau_by in cands:
        pred = apply_rule(h, routed, ctrl_hat, P, tau_by, tag)
        res = h.score_val(pred, verbose=False)
        table[tag] = res
        np.save(OUT / f"pred_trainval_{tag}.npy", pred)
        per = {sp: round(res["per_split"][sp]["DEP_F1"], 4) for sp in SPLITS}
        print(f"[arb] {tag:<6} composite={res['composite']:.4f} "
              f"F1={f1(res):.4f} per-split={per}", flush=True)
        (OUT / "arbitrate2.json").write_text(json.dumps(table, indent=1,
                                                      default=float))
    print(f"[done] ({time.time()-t0:.0f}s) 对照 C1: 0.5541/0.2442", flush=True)


if __name__ == "__main__":
    main()
