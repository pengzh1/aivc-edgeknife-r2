"""wsT9：DEP 对冲打包框架 —— composite↔F1 交换曲线定价（第三轮）。

框架思想（决策论打包，非调参）：隐藏评测 DEP 权重不明（本地 5%，出题人明牌
高效应为重点）。C1 的 τ 是一个连续的价格调节器：τ↓ ⇒ 旗标↑ ⇒ F1↑ 而
composite↓。wsT1 只评了预算内的 τ*=0.35；预算外的激进档（train F1 更高、
副作用超预算）从未定价。本模块把 {τ=0.25, τ=0.20} 两个预注册档位在 val 上
一次性定价（本轮唯一新增 val 看），产出第三条 test 备件，与 band/C1 构成
三候选对冲打包集：打包时若赛事方给出 DEP 权重信息，按
w·ΔF1 vs (1−w)·Δcomposite 线性定价选档；无信息则默认 C1。

预注册规则：
- 候选：C1@0.25、C1@0.20（与 C1 唯一差异是 τ，旗标/推送机制逐位同式）；
- 每个档位若 composite ≥ 0.548 且 F1 ≥ 0.25 → 生成 test 备件
  （wsT2 同管线，仅 τ 不同）；否则只记录数字不生备件；
- 记录逐 split F1 与副作用，供打包决策用。

合规：τ 为 wsT1 既有 train 网格上的既定档（非新拟合）；val 单次看；
Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsT9_hedge
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import data as D
from .evaluate import Harness
from .wsN11_grandrouter import SPLITS
from .wsT1_depgate import CACHE as T1C, MIN_PUSH, PUSH_TO, _predict_val

OUT = Path("outputs/wsT9")
T0C = Path("outputs/wsT0/cache")
HEDGE_TAUS = [0.25, 0.20]
ART_BAR = {"composite": 0.548, "f1": 0.25}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load(T0C / "routed_r07_trainval.npy")
    ctrl_hat = np.load(T0C / "control_hat.npy")
    clf = joblib.load(T1C / "hgb_full.joblib")

    report = {"hedge_taus": HEDGE_TAUS, "bar": ART_BAR}
    f1m = lambda r: float(np.mean([r["per_split"][sp]["DEP_F1"]
                                   for sp in SPLITS]))
    for tau in HEDGE_TAUS:
        pred = routed.astype(np.float64)
        n_flag = 0
        for sp in SPLITS:
            vrows, P = _predict_val(h, clf, sp)
            d_v = (routed[vrows] - ctrl_hat[vrows]).astype(np.float64)
            fl = (P >= tau) & (np.abs(d_v) >= MIN_PUSH)
            tgt = np.sign(d_v) * np.maximum(np.abs(d_v), PUSH_TO)
            pred[vrows] = pred[vrows] + np.where(fl, tgt - d_v, 0.0)
            n_flag += int(fl.sum())
        # train 行一致性
        rows = np.load(T1C / "train_rows.npy")
        d_est = np.load(T1C / "train_dest.npy").astype(np.float64)
        X = np.load(T1C / "train_X.npy")
        P = clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(
            X.shape[:2])
        fl = (P >= tau) & (np.abs(d_est) >= MIN_PUSH)
        tgt = np.sign(d_est) * np.maximum(np.abs(d_est), PUSH_TO)
        pred[rows] = pred[rows] + np.where(fl, tgt - d_est, 0.0)
        pred = pred.astype(np.float32)
        np.save(OUT / f"pred_trainval_tau{tau:.2f}.npy", pred)
        res = h.score_val(pred, verbose=False)
        per = {sp: round(res["per_split"][sp]["DEP_F1"], 4) for sp in SPLITS}
        make = (res["composite"] >= ART_BAR["composite"]
                and f1m(res) >= ART_BAR["f1"])
        print(f"[hedge] τ={tau:.2f} composite={res['composite']:.4f} "
              f"F1={f1m(res):.4f} flags={n_flag:,} per={per} "
              f"备件={'生成' if make else '不生成'}", flush=True)
        report[f"tau{tau:.2f}"] = {"res": res, "flags": n_flag,
                                   "artifact": bool(make)}
    (OUT / "hedge_pricing.json").write_text(json.dumps(report, indent=1,
                                                       default=float))
    print(f"[done] ({time.time()-t0:.0f}s) 对照 C1: 0.5541/0.2442",
          flush=True)


if __name__ == "__main__":
    main()
