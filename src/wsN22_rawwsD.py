"""wsN22: raw 空间 wsD 变体（无 Z 标准化，直接回归 log2 丰度）。

动机：评分 R²/PCC 均在原始 log2 空间计算；wsD 系在 Z 标准化空间训练
（每蛋白方差≈1），模型容量在蛋白间均匀分配；而 raw 空间 R² 对大动态范围
蛋白更敏感。raw 空间训练让损失权重自动对齐评分的量纲结构。
wsB 的 Δ̂ 模型已实证"raw 保持方差权重"有益（strain 最强）——同一逻辑
尚未在绝对丰度主力模型上试过。

合规同 wsM（train-only；注意 FrozenStats 仍用于插补与对照，与训练目标
空间无关）。3 种子 val 对照 wsD 3 种子 0.5423。

用法: python -m src.wsN22_rawwsD
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsD_arch import Cfg, ProteoMLP2, get_tensors, seen_cats, predict
from .wsD_arch import masked_huber

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN22"
WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
SEEDS = [0, 1, 2]


def train_raw(h, cfg, seed, device="cuda"):
    """wsD train_one 的 raw 版：目标 = log2 丰度原值（不减均值不除方差）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Z_all, M_all = get_tensors(h)
    # 用 raw 目标重建 Z_all：Z_raw = Z*std + mean（即原始 log2 值）
    Z_raw = Z_all * std + mean
    M_raw = ~torch.isnan(torch.tensor(h.Y_tr, dtype=torch.float32))
    Z_raw = torch.nan_to_num(
        torch.tensor(h.Y_tr, dtype=torch.float32), nan=0.0)
    rows = h.tr_rows
    n_prot = h.Y_tr.shape[1]
    model = ProteoMLP2(enc.n_cats, n_prot, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    X_dev = X_all.to(device)
    Z_dev = Z_raw.to(device)
    M_dev = M_raw.to(device)
    rows_dev = torch.tensor(rows, device=device)
    tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
    tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]
    for ep in range(cfg.epochs):
        model.train()
        gs = int(np.random.choice(tr_s)); gc = int(np.random.choice(tr_c))
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            xb[xb[:, 0] == gs, 0] = 0
            xb[xb[:, 1] == gc, 1] = 0
            for col in [0, 1]:
                dm = torch.rand(len(r), device=device) < 0.15
                xb[dm, col] = 0
            pred = model(xb)
            loss = masked_huber(pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    # raw 空间输出直接是 log2 预测：predict 时 mean=0/std=1
    zero = torch.zeros_like(mean)
    one = torch.ones_like(std)
    return model, enc, zero, one


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    cfg = Cfg.from_dict(WSD_CFG)
    seen = seen_cats(h, h.tr_rows)
    pv, pt = [], []
    for s in SEEDS:
        t0 = time.time()
        model, enc, mean, std = train_raw(h, cfg, s)
        pv.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
        pt.append(predict(model, enc, mean, std, h.m_te, g3_seen=seen))
        print(f"[wsN22] seed={s} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(pv, axis=0).astype(np.float32)
    res = h.score_val(P, verbose=False)
    fid = {sp: round(res["per_split"][sp]["fidelity"], 4)
           for sp in res["per_split"]}
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsN22 raw] composite={res['composite']:.4f} fid={fid} FC={fc}")
    print("对照 wsD Z空间 3种子: 0.5423")
    np.save(OUT / "pred_trainval.npy", P)
    np.save(OUT / "pred_test.npy", np.mean(pt, axis=0).astype(np.float32))
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "fid": fid, "FC": fc,
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
