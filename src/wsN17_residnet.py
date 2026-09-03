"""wsN17: 残差直连模型（直接优化模块3：chem_only 的 μ_ctx 残差 PCC）。

动机：composite 偏导分析——chem 残差每 +0.01 = 总分 +0.002（FC 的 3.2 倍）。
现有模型都预测绝对丰度，残差是派生物；本模型直接把残差作为训练目标：
  target = Δ − μ_ctx（strict train Δ − train 冻结 μ_ctx，逐蛋白 raw 尺度）
  input  = fuse128 化合物特征 + 上下文嵌入（培养基/温度/时间/仪器/来源/板号）
           + 菌株嵌入（UNK 回退）
  ŷ      = control_hat(train 对照组均值) + μ_ctx(ctx) + resid̂
对未见化合物：fuse 特征提供特异响应方向；μ_ctx 已由评分扣除，模型容量全部
用于化合物特异分量。

合规：全部统计 train-only（strict Δ / μ_ctx / 对照池 / FrozenStats）；
val 仅模型选择；不改既有文件。

用法: python -m src.wsN17_residnet        # 5 种子，约 5 分钟
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness
from . import wsB_twostage as B
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN17"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
CTX_COLS = ["Medium", "Temperature", "pert_time", "instrument",
            "data_source", "Yeast_cell_plate"]
EMB = {"Strains": 8, "Medium": 2, "Temperature": 2, "pert_time": 4,
       "instrument": 6, "data_source": 3, "Yeast_cell_plate": 32}


class ResidMLP(nn.Module):
    def __init__(self, n_cats, chem_dim, n_out, hidden=(512, 1024),
                 p_drop=0.1):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB[c]) for n, c in
            zip(n_cats, ["Strains"] + CTX_COLS)])
        d_in = sum(EMB[c] for c in ["Strains"] + CTX_COLS) + chem_dim
        layers, d = [], d_in
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.GELU(), nn.LayerNorm(hdim),
                       nn.Dropout(p_drop)]
            d = hdim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)
        nn.init.zeros_(self.head.weight)  # 从零残差出发（先验=μ_ctx 基线）
        nn.init.zeros_(self.head.bias)

    def forward(self, x_cat, chem_feat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(e + [chem_feat], dim=1)
        return self.head(self.trunk(x))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    # ---- 目标：strict Δ − μ_ctx ----
    delta, treat_all, treat_valid = strict_delta_train(h)
    m_train = h.m_train
    mu_ctx_train = h.mu_ctx_for(m_train)  # train 冻结参照
    resid = delta - mu_ctx_train
    # ---- 特征 ----
    df = pd.read_csv("outputs/wsN6/chem_features_fuse.csv").set_index(
        "compound")
    pc = [c for c in df.columns if c.startswith("pc")]
    chem_map = {n: df.loc[n, pc].to_numpy(dtype=np.float32)
                for n in df.index}
    chem_dim = len(pc)
    mean_feat = np.stack(list(chem_map.values())).mean(0)

    enc = B.Encoder(["Strains"] + CTX_COLS).fit(m_train)
    n_prot = h.Y_tr.shape[1]

    def featurize(m):
        X = torch.tensor(enc.transform(m), dtype=torch.long)
        F = torch.tensor(np.stack([
            chem_map.get(c, mean_feat)
            for c in m["perturbation_no_concentration"]]), dtype=torch.float32)
        return X, F

    X_all, F_all = featurize(m_train)
    R = torch.tensor(np.nan_to_num(resid, nan=0.0), dtype=torch.float32)
    Mk = torch.tensor(~np.isnan(resid))

    X_dev, F_dev = X_all.cuda(), F_all.cuda()
    R_dev, Mk_dev = R.cuda(), Mk.float().cuda()
    rows_dev = torch.tensor(treat_valid).cuda()

    preds_val = []
    for sd in SEEDS:
        t0 = time.time()
        torch.manual_seed(sd)
        np.random.seed(sd)
        model = ResidMLP(enc.n_cats, chem_dim, n_prot).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        n_steps = EPOCHS * int(np.ceil(len(treat_valid) / 256))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps)
        for ep in range(EPOCHS):
            model.train()
            perm = rows_dev[torch.randperm(len(rows_dev), device="cuda")]
            for i in range(0, len(perm), 256):
                r = perm[i:i + 256]
                xb = X_dev[r].clone()
                # 菌株 UNK dropout（OOD 回退学习）
                dm = torch.rand(len(r), device="cuda") < 0.25
                xb[dm, 0] = 0
                fb = F_dev[r].clone()
                dm = torch.rand(len(r), device="cuda") < 0.25
                fb[dm] = torch.tensor(mean_feat).cuda()
                pred = model(xb, fb)
                se = (pred - R_dev[r]) ** 2 * Mk_dev[r]
                loss = se.sum() / Mk_dev[r].sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        # 推理全部 train_val 行
        model.eval()
        Xv, Fv = featurize(h.m_tr)
        outs = []
        with torch.no_grad():
            Xv_d = Xv.cuda()
            for i in range(0, len(Xv), 1024):
                # 推理期：train 未见证的菌株 → UNK(0)
                xb = Xv_d[i:i + 1024].clone()
                seen_strains = set(m_train["Strains"].unique())
                mask = ~h.m_tr["Strains"].iloc[i:i + 1024].isin(
                    seen_strains).to_numpy()
                xb[torch.tensor(mask).cuda(), 0] = 0
                outs.append(model(xb, Fv[i:i + 1024].cuda()).float().cpu())
        resid_pred = torch.cat(outs).numpy()
        preds_val.append(resid_pred)
        print(f"[wsN17] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    resid_hat = np.mean(preds_val, axis=0).astype(np.float32)
    # 合成：control_hat + μ_ctx + resid̂（仅处理行）
    from .make_submission_trainonly import build_control_hat_train
    ctrl_hat, _ = build_control_hat_train(h, h.m_tr)
    mu_ctx_all = h.mu_ctx_for(h.m_tr)
    is_treat = h.is_treat_tr
    pred = ctrl_hat.copy()
    pred[is_treat] = (ctrl_hat[is_treat] + mu_ctx_all[is_treat]
                      + resid_hat[is_treat])
    pred = h.stats.impute(pred).astype(np.float32)
    res = h.score_val(pred, verbose=False)
    for sp, s in res["per_split"].items():
        print(f"{sp:<18} fid={s['fidelity']:.4f} FC={s['FC_PCC']:.4f} "
              f"resid={s.get('resid_PCC', 0.0):.4f}")
    print(f"[wsN17] composite={res['composite']:.4f}")
    print("参考：fuse 单族 0.5246（chem resid 0.46-0.48 区间）")
    np.save(OUT / "pred_trainval.npy", pred)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"],
         "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}/scores.json")


if __name__ == "__main__":
    main()
