"""wsN4: wsD 配方 + tail-checkpoint 平均（免费方差缩减）。

动机：wsC 的配方含 tail_avg（后 30% epoch 每 10 步取快照平均），wsD 没有。
对 wsD 8 种子各取 tail 快照平均，等价于免费增加集成多样性。
合规同 wsM（train-only 训练，val 仅模型选择）。

用法: python -m src.wsN4_tailavg      # 8 种子 tail_avg → val 评分 + test 预测
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import Harness
from .wsD_arch import Cfg, get_tensors, seen_cats, predict

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN4"

WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
SEEDS = list(range(8))


def train_one_tail(h: Harness, cfg: Cfg, seed: int, device: str = "cuda"):
    """wsD train_one + tail 快照收集（ep≥70% 且每 10 epoch）。"""
    from .wsD_arch import ProteoMLP2, masked_huber, masked_mse
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
    rows_dev = torch.tensor(rows, device=device)
    tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
    tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]
    tail = []
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
            loss = (masked_huber if cfg.loss == "huber" else masked_mse)(
                pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if cfg.epochs >= 20 and ep + 1 >= int(cfg.epochs * 0.7) \
                and (ep + 1) % 10 == 0:
            tail.append({k: v.detach().clone()
                         for k, v in model.state_dict().items()})
    return model, enc, mean, std, tail


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    cfg = Cfg.from_dict(WSD_CFG)
    seen = seen_cats(h, h.tr_rows)
    pv, pt = [], []
    for s in SEEDS:
        t0 = time.time()
        model, enc, mean, std, tail = train_one_tail(h, cfg, s)
        states = [model.state_dict()] + tail
        spv, spt = [], []
        for st in states:
            model.load_state_dict(st)
            spv.append(predict(model, enc, mean, std, h.m_tr, g3_seen=seen))
            spt.append(predict(model, enc, mean, std, h.m_te, g3_seen=seen))
        pv.append(np.mean(spv, axis=0))
        pt.append(np.mean(spt, axis=0))
        print(f"[wsN4] seed={s} states={len(states)} ({time.time()-t0:.0f}s)",
              flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(pv, axis=0).astype(np.float32)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsN4] val composite={res['composite']:.4f} FC={fc} "
          f"(wsD 8种子基线 0.5431)")
    np.save(OUT / "pred_trainval_tailavg.npy", P)
    PT = np.mean(pt, axis=0).astype(np.float32)
    assert np.isfinite(PT).all()
    np.save(OUT / "pred_test_tailavg.npy", PT)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc,
         "per_split": res["per_split"], "seeds": SEEDS},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}/pred_trainval_tailavg.npy + pred_test_tailavg.npy")


if __name__ == "__main__":
    main()
