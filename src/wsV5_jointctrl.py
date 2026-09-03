"""wsV5：对照-处理联合训练族（机制：辅助监督信号）。

动机：59 实验的全部模型只用 5,078 个处理行训练；1,842 个对照/QC 行只用于
control_hat 估计。对照行携带大量"批次/上下文如何塑造蛋白组"的监督信号，
却从未进入任何主干。本族让共享 trunk 同时服务两个头：
  对照行（train 对照 751 有效）：输入=上下文嵌入（无化合物），目标=标准化丰度
  处理行：输入=上下文+化合物嵌入，目标=train Δ（wsB 同式）
trunk 从对照监督中学到的批次/上下文结构应迁移到 Δ 预测。

预注册裁决：单族 composite ≥ 0.5385（wsB_s16 基线）且边际扫描 Δ ≥ +0.0003；
8 种子时间盒。val 一次看；零增益按 T5 关闭。

用法: python -m src.wsV5_jointctrl
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .evaluate import Harness
from .wsB_twostage import DLT_COLS, CTRL_COLS, EMB_DIMS, Encoder, masked_mse

OUT = Path("outputs/wsV5")
SEEDS = list(range(8))
EPOCHS = 150
BS = 256


class JointMLP(torch.nn.Module):
    def __init__(self, n_cats_d, n_cats_c, n_prot):
        super().__init__()
        self.embs_d = torch.nn.ModuleList([
            torch.nn.Embedding(n, EMB_DIMS[c])
            for n, c in zip(n_cats_d, DLT_COLS)])
        self.embs_c = torch.nn.ModuleList([
            torch.nn.Embedding(n, EMB_DIMS[c])
            for n, c in zip(n_cats_c, CTRL_COLS)])
        d_d = sum(EMB_DIMS[c] for c in DLT_COLS)
        d_c = sum(EMB_DIMS[c] for c in CTRL_COLS)
        self.adapt_d = torch.nn.Linear(d_d, 64)
        self.adapt_c = torch.nn.Linear(d_c, 64)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(64, 512), torch.nn.GELU(),
            torch.nn.LayerNorm(512), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1024), torch.nn.GELU(),
            torch.nn.LayerNorm(1024), torch.nn.Dropout(0.1))
        self.head_d = torch.nn.Linear(1024, n_prot)   # Δ̂ 头
        self.head_c = torch.nn.Linear(1024, n_prot)   # 对照丰度头
        torch.nn.init.zeros_(self.head_d.bias)
        torch.nn.init.zeros_(self.head_c.bias)

    def forward_d(self, x):
        e = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embs_d)], 1)
        return self.head_d(self.trunk(self.adapt_d(e)))

    def forward_c(self, x):
        e = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embs_c)], 1)
        return self.head_c(self.trunk(self.adapt_c(e)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    m = h.m_tr
    pert = m["perturbation_no_concentration"]
    tr = h.tr_rows
    treat_rows = tr[h.is_treat_tr[tr]]
    ctrl_rows = tr[pert.isin(D.CONTROLS).to_numpy()[tr]]
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    enc_c = Encoder(CTRL_COLS).fit(m.iloc[ctrl_rows])
    Xd = torch.tensor(enc_d.transform(m), dtype=torch.long)
    Xc = torch.tensor(enc_c.transform(m), dtype=torch.long)
    Dmat = torch.tensor(np.nan_to_num(h.delta_tr_all, nan=0.0),
                        dtype=torch.float32)
    Md = (~torch.isnan(torch.tensor(h.delta_tr_all))).float()
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
    Zc = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    Mc = (~torch.isnan(torch.tensor(h.Y_tr))).float()
    Zc = torch.nan_to_num(Zc, nan=0.0)
    dev = "cuda"
    Xd, Xc = Xd.to(dev), Xc.to(dev)
    Dmat, Md = Dmat.to(dev), Md.to(dev)
    Zc, Mc = Zc.to(dev), Mc.to(dev)
    rd = torch.tensor(treat_rows, device=dev)
    rc = torch.tensor(ctrl_rows, device=dev)

    pv = []
    for sd in SEEDS:
        t0 = time.time()
        torch.manual_seed(sd)
        np.random.seed(sd)
        model = JointMLP(enc_d.n_cats, enc_c.n_cats, h.Y_tr.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_steps = EPOCHS * int(np.ceil((len(rd) + len(rc)) / BS))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        for ep in range(EPOCHS):
            model.train()
            tot, nb = 0.0, 0
            n_batches = int(np.ceil(len(rd) / BS))
            for _ in range(n_batches):
                if np.random.rand() < 0.62:      # 按行数比例混合
                    r = rd[torch.randint(0, len(rd), (BS,), device=dev)]
                    pred = model.forward_d(Xd[r])
                    loss = masked_mse(pred, Dmat[r], Md[r])
                else:
                    r = rc[torch.randint(0, len(rc), (BS,), device=dev)]
                    pred = model.forward_c(Xc[r])
                    loss = masked_mse(pred, Zc[r], Mc[r])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
                nb += 1
            if (ep + 1) % 50 == 0 or ep == EPOCHS - 1:
                print(f"    [s{sd}] ep {ep+1}/{EPOCHS} loss={tot/nb:.4f}",
                      flush=True)
        # 推理：处理行 Δ̂ + wsE control_hat 锚点（与 wsB 非锚定版同口径不行——
        # 本族只交 Δ̂；装配用 wsT0 的 control_hat，与 wsT3/wsU1 交付口径一致）
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(m), 2048):
                outs.append(model.forward_d(
                    Xd[i:i + 2048]).float().cpu())
        delta_hat = torch.cat(outs).numpy().astype(np.float32)
        pv.append(delta_hat)
        del model
        torch.cuda.empty_cache()
        print(f"[wsV5] seed {sd} ({time.time()-t0:.0f}s)", flush=True)

    ctrl_hat = np.load("outputs/wsT0/cache/control_hat.npy")
    P = np.mean(pv, axis=0)
    pred = ctrl_hat.copy()
    pred[h.is_treat_tr] += P[h.is_treat_tr]
    pred = h.stats.impute(pred).astype(np.float32)
    np.save(OUT / "pred_trainval.npy", pred)
    res = h.score_val(pred, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"]
                        for sp in res["per_split"]]))
    print(f"[wsV5] composite={res['composite']:.4f} FC={fc} F1={f1:.4f}",
          flush=True)
    (OUT / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "F1": f1,
         "per_split": res["per_split"]}, default=float, indent=1))

    if res["composite"] >= 0.5385:
        h.prepare_fast_eval()
        routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
        base = h.score_val(routed, verbose=False)["composite"]
        print(f"[scan] routed 基线 {base:.4f}", flush=True)
        for tag, sps in [("strain_both", ["val_strain_only", "val_both"]),
                         ("chem", ["val_chem_only"])]:
            rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
            for a in (0.05, 0.12, 0.22):
                trial = routed.copy()
                trial[rows] = (1 - a) * routed[rows] + a * pred[rows]
                c = h.score_val(trial, verbose=False)["composite"]
                print(f"  [{tag} α={a}] composite={c:.4f} (Δ{c-base:+.4f})",
                      flush=True)


if __name__ == "__main__":
    main()
