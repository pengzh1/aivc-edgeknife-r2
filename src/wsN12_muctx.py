"""wsN12: both 角色专用退化模型（control_hat + μ_ctx 精确回退）。

both 划分（双未见）的理论内容上限 = 上下文共享药物响应 μ_ctx；
当前路由在该角色仍给 ridge 0.388。本模型精确实现理论退化形：
  ŷ = control_hat(train 对照组均值) + μ_ctx(同上下文 train 药物 Δ 均值)
若其在 val_both 上强于 ridge/现混权重，即应进入大阵容。

合规：全部统计 train-only（control 池/μ_ctx 均为 train 冻结参照）。

用法: python -m src.wsN12_muctx
"""
import json
from pathlib import Path

import numpy as np

from .evaluate import Harness
from .make_submission_trainonly import build_control_hat_train
from . import data as D

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN12"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    ctrl_hat, level = build_control_hat_train(h, h.m_tr)
    mu_ctx = h.mu_ctx_for(h.m_tr)
    is_treat = h.is_treat_tr
    pred = ctrl_hat.copy()
    pred[is_treat] = ctrl_hat[is_treat] + mu_ctx[is_treat]
    pred = h.stats.impute(pred).astype(np.float32)
    res = h.score_val(pred, verbose=False)
    for sp, s in res["per_split"].items():
        print(f"{sp:<18} fid={s['fidelity']:.4f} FC={s['FC_PCC']:.4f} "
              f"resid={s.get('resid_PCC', 0.0):.4f}")
    print(f"[wsN12] composite={res['composite']:.4f}")
    print("参考：ridge 0.4608 | both FC 现混成员最好 0.248")
    np.save(OUT / "pred_trainval.npy", pred)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"],
         "per_split": res["per_split"]}, default=float, indent=1))


if __name__ == "__main__":
    main()
