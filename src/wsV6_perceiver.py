"""wsV6：Perceiver 式合作解码族（框架级突破尝试 #1：打破独立线性头）。

审计发现的架构大陆：现行 21 族全部是"共享 trunk → 每蛋白独立线性头"，
样本内蛋白预测互不通信。本族首次引入**跨蛋白合作解码**：
  编码器：DLT 类别嵌入 → MLP → 条件潜变量 z（64 维 × 8 latent）
  解码器：5,243 个可学习蛋白 query 与 latent 做双向交叉注意力
          （Perceiver block ×2：latents  attend queries → queries attend
          latents），蛋白间经由潜瓶颈交换信息（响应模块结构），
          每蛋白输出头读最终 query 态。
机制假说：FC/DEP 的弱项在样本内排序（模块级共变未被利用）；合作解码
让"某蛋白的预测"吸收同样本其他蛋白的响应上下文。

预注册裁决：单族 composite ≥ 0.5400（fuse3e150 档）且边际扫描 Δ ≥ +0.0003
（chem 与 strain_both 两区 α∈{0.05,0.12,0.22}）入路由；否则关闭归档。
8 种子 × 100ep（GPU 约 40min）。val 仅本包一次看。

合规：train-only（h.tr_rows 全行含对照，masked MSE，Z-scored 目标——
与 wsA/wsN24 主族同口径）；新文件不改旧文件；Y_te 零接触。

用法: python -m src.wsV6_perceiver
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .evaluate import Harness
from .train_mlp import CAT_COLS, EMB_DIMS, Encoder, masked_mse

OUT = Path("outputs/wsV6")
SEEDS = list(range(8))
EPOCHS = 60
BS = 256
D_MODEL = 128
N_LATENT = 8
N_MODULES = 256        # 模块 query 数（蛋白经由模块共享，全分辨率解码）
N_BLOCKS = 2


class PerceiverProt(nn.Module):
    """条件潜变量 + 模块级合作解码：蛋白不再各自独立，而是经由 256 个
    可学习响应模块共享样本内上下文（打破 21 族的独立线性头限制）。"""

    def __init__(self, n_cats, n_prot):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[c]) for n, c in zip(n_cats, CAT_COLS)])
        d_in = sum(EMB_DIMS[c] for c in CAT_COLS)
        self.enc = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, D_MODEL), nn.GELU(), nn.LayerNorm(D_MODEL))
        self.latents = nn.Parameter(torch.randn(N_LATENT, D_MODEL) * 0.02)
        self.mod_tokens = nn.Parameter(
            torch.randn(N_MODULES, D_MODEL) * 0.02)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln_x = nn.LayerNorm(D_MODEL)
        self.ln_l = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(),
                                nn.Linear(256, D_MODEL))
        self.ln_f = nn.LayerNorm(D_MODEL)
        # 秩 N_MODULES 合作头：蛋白经模块混合系数共享响应程序
        self.W2 = nn.Parameter(torch.randn(n_prot, N_MODULES) * 0.02)
        self.v = nn.Parameter(torch.randn(D_MODEL) * 0.02)
        self.Wz = nn.Parameter(torch.randn(n_prot, D_MODEL) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_prot))

    def forward(self, x_cat):
        B = x_cat.shape[0]
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)],
                      dim=1)
        z = self.enc(e)                                  # (B, D)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        x = self.mod_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        for _ in range(N_BLOCKS):
            a = torch.einsum("bld,bmd->blm", self.ln_l(lat),
                             self.ln_x(x)) / np.sqrt(D_MODEL)
            lat = lat + self.o_proj(torch.einsum(
                "blm,bmd->bld", torch.softmax(a, dim=2), self.v_proj(x)))
            a2 = torch.einsum("bmd,bld->bml", self.ln_x(x),
                              self.ln_l(lat)) / np.sqrt(D_MODEL)
            x = x + self.o_proj(torch.einsum(
                "bml,bld->bmd", torch.softmax(a2, dim=2), self.v_proj(lat)))
            x = x + self.ff(self.ln_f(x))
        h = torch.einsum("bmd,d->bm", x, self.v)         # (B, M) 模块激活
        out = h @ self.W2.T + z @ self.Wz.T + self.bias  # (B, P)
        return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    m = h.m_tr
    enc = Encoder().fit(m.iloc[h.tr_rows])
    X_all = torch.tensor(enc.transform(m), dtype=torch.long)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)
    dev = "cuda"
    X_dev, Z_dev, M_dev = (X_all.to(dev), Z_all.to(dev),
                           M_all.to(dev).float())
    rows_dev = torch.tensor(h.tr_rows, device=dev)
    n_prot = h.Y_tr.shape[1]

    pv = []
    for sd in SEEDS:
        t0 = time.time()
        torch.manual_seed(sd)
        np.random.seed(sd)
        model = PerceiverProt(enc.n_cats, n_prot).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        n_steps = EPOCHS * int(np.ceil(len(rows_dev) / BS))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps)
        for ep in range(EPOCHS):
            model.train()
            perm = rows_dev[torch.randperm(len(rows_dev), device=dev)]
            tot, nb = 0.0, 0
            for i in range(0, len(perm), BS):
                r = perm[i:i + BS]
                xb = X_dev[r].clone()
                drop = torch.rand(len(r), device=dev) < 0.25
                xb[drop, 0] = 0                     # 菌株 UNK 增广
                pred = model(xb)
                loss = masked_mse(pred, Z_dev[r], M_dev[r])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
                nb += 1
            if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
                print(f"    [s{sd}] ep {ep+1:>3}/{EPOCHS} "
                      f"loss={tot/nb:.4f}", flush=True)
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, X_dev.shape[0], 256):
                outs.append((model(X_dev[i:i + 256]) * std.to(dev)
                             + mean.to(dev)).float().cpu())
        P = torch.cat(outs).numpy().astype(np.float32)
        bad = ~np.isfinite(P)
        if bad.any():
            r_, c_ = np.where(bad)
            P[r_, c_] = np.take(h.stats.protein_mean, c_)
        pv.append(P)
        del model
        torch.cuda.empty_cache()
        print(f"[wsV6] seed {sd} ({time.time()-t0:.0f}s)", flush=True)

    P = np.mean(pv, axis=0).astype(np.float32)
    np.save(OUT / "pred_trainval.npy", P)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"]
                        for sp in ["val_chem_only", "val_strain_only",
                                   "val_both", "val_time"]]))
    print(f"[wsV6] composite={res['composite']:.4f} FC={fc} F1={f1:.4f}",
          flush=True)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "F1": f1,
         "per_split": res["per_split"]}, default=float, indent=1))

    if res["composite"] >= 0.5400:
        h.prepare_fast_eval()
        routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
        base = h.score_val(routed, verbose=False)["composite"]
        print(f"[scan] routed 基线 {base:.4f}", flush=True)
        for tag, sps in [("strain_both", ["val_strain_only", "val_both"]),
                         ("chem", ["val_chem_only"])]:
            rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
            for a in (0.05, 0.12, 0.22):
                trial = routed.copy()
                trial[rows] = (1 - a) * routed[rows] + a * P[rows]
                c = h.score_val(trial, verbose=False)["composite"]
                print(f"  [{tag} α={a}] composite={c:.4f} (Δ{c-base:+.4f})",
                      flush=True)


if __name__ == "__main__":
    main()
