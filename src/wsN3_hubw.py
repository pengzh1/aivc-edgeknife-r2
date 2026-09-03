"""wsN3: 高变化蛋白 loss 加权（wsD 配方变体）。

动机（出题人 8.10 分享会）："单细胞做法一般会给基因排序，把变化高的基因在
loss 里赋予更高权重"；"5000 个 protein 约 3000 个变化量 <0.25，在噪声水平
波动；隐藏任务本质看模型能否把高变化量特点预测好"；"只用 MSE 会倾向预测均值"。

做法：wsD 最佳配方（5 层 300ep Huber G2 增强）+ per-protein 损失权重
  w_p = clip( (Δstd_p / median(Δstd))^pow , lo, hi )
  Δstd_p = train 处理样本 strict Δ 的逐蛋白标准差（train-only 冻结）
对照：同配方 w≡1（wsD 3 种子基线，val 复核）。

合规：train-only 训练与统计；val 仅模型选择；不改动 wsD 原文件
（本地复制 train_one 循环，仅插入权重项）。

用法: python -m src.wsN3_hubw            # 2 变体 × 3 种子 + 对照，val 评分
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsD_arch import (Cfg, ProteoMLP2, get_tensors, seen_cats, transform_g3,
                       predict)
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN3"

WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
SEEDS = [0, 1, 2]

# 变体：(pow, lo, hi)；identity 为对照
VARIANTS = {
    "identity": (0.0, 1.0, 1.0),          # w≡1 对照（= wsD 配方 3 种子复核）
    "sqrt_soft": (0.5, 0.7, 1.7),         # 软加权
    "linear": (1.0, 0.5, 3.0),            # 线性加权（截断）
}


def delta_std_train(h: Harness) -> np.ndarray:
    """train 处理样本 strict Δ 的逐蛋白标准差（NaN 忽略；全缺失→0）。"""
    delta, _, valid = strict_delta_train(h)
    d = delta[valid]
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(d, axis=0)
    return np.nan_to_num(sd, nan=0.0).astype(np.float32)


def weighted_huber(pred, target, mask, w, beta: float = 1.0):
    se = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none",
                                            beta=beta)
    return (se * mask * w).sum() / (mask * w).sum().clamp_min(1.0)


def train_one_w(h: Harness, cfg: Cfg, seed: int, w: torch.Tensor,
                device: str = "cuda"):
    """wsD train_one 的加权副本（逻辑一致，仅 loss 乘 per-protein 权重）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Z_all, M_all = get_tensors(h)
    rows = h.tr_rows
    n_prot = h.Y_tr.shape[1]
    model = ProteoMLP2(enc.n_cats, n_prot, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    X_dev, Z_dev = X_all.to(device), Z_all.to(device)
    M_dev = M_all.to(device)
    w_dev = w.to(device)
    rows_dev = torch.tensor(rows, device=device)
    if cfg.g2_aug:
        tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
        tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]
    for ep in range(cfg.epochs):
        model.train()
        if cfg.g2_aug:
            gs = int(np.random.choice(tr_s)); gc = int(np.random.choice(tr_c))
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            if cfg.g2_aug:
                xb[xb[:, 0] == gs, 0] = 0
                xb[xb[:, 1] == gc, 1] = 0
                for col in [0, 1]:
                    dm = torch.rand(len(r), device=device) < 0.15
                    xb[dm, col] = 0
            pred = model(xb)
            loss = weighted_huber(pred, Z_dev[r], M_dev[r].float(), w_dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model, enc, mean, std


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    cfg = Cfg.from_dict(WSD_CFG)
    sd = delta_std_train(h)
    med = float(np.median(sd[sd > 0]))
    print(f"[wsN3] Δstd median={med:.4f} | p90={np.quantile(sd,0.9):.4f} "
          f"| p99={np.quantile(sd,0.99):.4f}")
    seen = seen_cats(h, h.tr_rows)

    summary = {}
    for tag, (pow_, lo, hi) in VARIANTS.items():
        w_np = np.clip((sd / med) ** pow_, lo, hi).astype(np.float32)
        w = torch.tensor(w_np)
        preds = []
        for s in SEEDS:
            t0 = time.time()
            model, enc, mean, std = train_one_w(h, cfg, s, w)
            preds.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
            print(f"[{tag}] seed={s} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        P = np.mean(preds, axis=0).astype(np.float32)
        res = h.score_val(P, verbose=False)
        fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
              for sp in res["per_split"]}
        dep = {sp: round(res["per_split"][sp].get("DEP_F1", 0.0), 4)
               for sp in res["per_split"]}
        summary[tag] = {"composite": res["composite"], "FC": fc,
                        "DEP_F1": dep, "per_split": res["per_split"]}
        print(f"[{tag}] composite={res['composite']:.4f} FC={fc} DEP_F1={dep}",
              flush=True)
        np.save(OUT / f"pred_trainval_{tag}.npy", P)

    (OUT / "scores.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
