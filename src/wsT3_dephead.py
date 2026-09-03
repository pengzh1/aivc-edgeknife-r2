"""wsT3：DEP 专职族（handoff 8.15 §T1.2）——wsB 两阶段 × Δ̂ 分支 DEP 分类头。

与 wsB 的唯一差异：Δ̂ 模型的 trunk 上多一个并行的 DEP logit 头
（每蛋白 P(|Δ_true|>1)），联合训练：
  loss = masked_MSE(Δ̂, Δ) + λ·masked_BCE(logit, 1{|Δ|>1}; pos_weight=10)
回归输出不被重加权（wsN3 全局加权已证伪；本头是独立输出单元，不扭曲 Δ̂）。
其余（Encoder/UNK/G2 语义/control MLP w*=1.0/QC/合成）与 wsB 逐行一致。

16 种子均值：ŷ 走 family 评分；p_hi 均值存盘供 C4 交换测试。

预注册裁决包（单次 val 看，三问一表）：
  Q1 单族 composite（vs wsB_s16 基线 0.5385 档）
  Q2 wsN30 式边际扫描：routed r=0.7 基线上掺 α（strain+both 行组 / chem 行组，
     α∈{0.05,0.12,0.22}，Δcomposite ≥ +0.0003 才算有潜力）
  Q3 C4 交换：C1 选择器的 P 换成 wsT3 p_hi（τ 用 train 重定，in-sample p
     记为乐观偏差，val 这一眼定夺）——与 C1 的 0.5541/0.2442 比，
     双指标不劣才替换。
零潜力 → 关闭归档，不追加变体（T5 纪律）。

合规：train-only 训练与 τ 调定；val 仅本包一次；Y_te 零接触；新文件不改旧文件。

用法: python -m src.wsT3_dephead --stage train   # 16 种子（GPU ~1h）
      python -m src.wsT3_dephead --stage eval    # 预注册裁决包
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness
from .wsB_twostage import (EMB_DIMS, CTRL_COLS, DLT_COLS, Encoder, CondMLP,
                           masked_mse, GroupMeanControl, QCGroupMean,
                           predict_model, train_model)
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsT3")
CACHE = OUT / "cache"
SEEDS = list(range(16))
LAM_BCE = 1.0
POS_WEIGHT = 10.0
EPOCHS_CTRL, EPOCHS_DLT = 100, 100


class CondMLPDep(nn.Module):
    """wsB CondMLP trunk + 并行双头：Δ̂ 回归头 + DEP logit 头。"""

    def __init__(self, n_cats, cols, n_out, hidden=(512, 1024), p_drop=0.1):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[c]) for n, c in zip(n_cats, cols)])
        d_in = sum(EMB_DIMS[c] for c in cols)
        layers, d = [], d_in
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.GELU(), nn.LayerNorm(hdim),
                       nn.Dropout(p_drop)]
            d = hdim
        self.trunk = nn.Sequential(*layers)
        self.head_d = nn.Linear(d, n_out)
        self.head_p = nn.Linear(d, n_out)
        nn.init.zeros_(self.head_d.bias)
        nn.init.constant_(self.head_p.bias, -3.5)  # ≈ 基准率 2.9% 的先验

    def forward(self, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)],
                      dim=1)
        z = self.trunk(e)
        return self.head_d(z), self.head_p(z)


def train_dep_model(model, X_all, Dmat, Md, LBL, rows, epochs, seed,
                    drop_cols=(0, 1), emb_drop=0.25, lr=1e-3, bs=256,
                    device="cuda", tag=""):
    """wsB.train_model 的双头版：MSE(Δ̂) + λ·BCE(DEP logit, pos_weight)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * int(np.ceil(len(rows) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    rows_dev = torch.tensor(rows, device=device)
    pw = torch.tensor(POS_WEIGHT, device=device)
    for ep in range(epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            xb = X_all[r].clone()
            for col in drop_cols:
                drop = torch.rand(len(r), device=device) < emb_drop
                xb[drop, col] = 0
            pd_, pl_ = model(xb)
            m = Md[r]
            loss = masked_mse(pd_, Dmat[r], m)
            bce = nn.functional.binary_cross_entropy_with_logits(
                pl_, LBL[r], weight=m, pos_weight=pw,
                reduction="sum") / m.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)  # 必须先清零再 backward（踩过反序坑）
            (loss + LAM_BCE * bce).backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1:>3}/{epochs}  mse={tot/nb:.4f}",
                  flush=True)
    return model


@torch.no_grad()
def predict_dep(model, X_all, device="cuda", bs=1024):
    model.eval()
    ds, ps = [], []
    for i in range(0, X_all.shape[0], bs):
        pd_, pl_ = model(X_all[i:i + bs])
        ds.append(pd_.float().cpu())
        ps.append(torch.sigmoid(pl_).float().cpu())
    return torch.cat(ds).numpy(), torch.cat(ps).numpy()


def cmd_train(args):
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    h = Harness()
    m = h.m_tr
    n_prot = h.Y_tr.shape[1]
    pert = m["perturbation_no_concentration"]
    is_ctrl = pert.isin(D.CONTROLS).to_numpy()
    is_qc = (pert == D.QC).to_numpy()
    is_treat = h.is_treat_tr
    tr = h.tr_rows
    ctrl_rows = tr[is_ctrl[tr]]
    treat_rows = tr[is_treat[tr]]
    qc_rows = tr[is_qc[tr]]
    dev = args.device
    print(f"[wsT3] train controls={len(ctrl_rows)} treated={len(treat_rows)}",
          flush=True)

    # ---- Stage 1: control MLP（wsB 同式，3 种子）----
    t0 = time.time()
    enc_c = Encoder(CTRL_COLS).fit(m.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m), dtype=torch.long, device=dev)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32, device=dev)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32, device=dev)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32, device=dev) - mean) / std
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    ctrl_mlp = np.zeros((len(m), n_prot), np.float32)
    for sd in (0, 1, 2):
        model = CondMLP(enc_c.n_cats, CTRL_COLS, n_prot)
        model = train_model(model, Xc, Z, Mz.float(), ctrl_rows,
                            EPOCHS_CTRL, sd, drop_cols=(0,), emb_drop=0.25,
                            device=dev, tag=f"ctrl s{sd}")
        ctrl_mlp += predict_model(model, Xc, dev).astype(np.float32)
    ctrl_mlp = ctrl_mlp / 3 * h.stats.protein_std + h.stats.protein_mean
    np.save(CACHE / "ctrl_mlp.npy", ctrl_mlp)
    print(f"[wsT3] control MLP done {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: Δ̂+DEP 双头（16 种子）----
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    Xd = torch.tensor(enc_d.transform(m), dtype=torch.long, device=dev)
    Dmat = torch.tensor(h.delta_tr_all, dtype=torch.float32, device=dev)
    Md = ~torch.isnan(Dmat)
    LBL = (torch.abs(Dmat) > 1.0).float()
    Dmat = torch.nan_to_num(Dmat, nan=0.0)
    delta_hat = np.zeros((len(m), n_prot), np.float32)
    p_hat = np.zeros((len(m), n_prot), np.float32)
    for sd in SEEDS:
        t0 = time.time()
        model = CondMLPDep(enc_d.n_cats, DLT_COLS, n_prot)
        model = train_dep_model(model, Xd, Dmat, Md.float(), LBL, treat_rows,
                                EPOCHS_DLT, sd, device=dev, tag=f"dep s{sd}")
        d_sd, p_sd = predict_dep(model, Xd, dev)
        delta_hat += d_sd.astype(np.float32)
        p_hat += p_sd.astype(np.float32)
        print(f"[wsT3] seed {sd} done ({time.time()-t0:.0f}s)", flush=True)
    delta_hat /= len(SEEDS)
    p_hat /= len(SEEDS)
    np.save(CACHE / "delta_hat.npy", delta_hat)
    np.save(CACHE / "p_hat.npy", p_hat)

    # ---- QC + 合成（wsB 同式，w*=1.0 纯 MLP control）----
    qc_model = QCGroupMean().fit(m.iloc[qc_rows].reset_index(drop=True),
                                 h.Y_tr[qc_rows], h.stats.protein_mean)
    qc_pred = qc_model.predict(m)
    pred = ctrl_mlp.copy()
    pred[is_treat] += delta_hat[is_treat]
    pred[is_qc] = qc_pred[is_qc]
    pred = h.stats.impute(pred).astype(np.float32)
    np.save(CACHE / "pred_trainval.npy", pred)
    print(f"[wsT3] saved {CACHE/'pred_trainval.npy'}", flush=True)
    res = h.score_val(pred)
    (CACHE / "family_score.json").write_text(json.dumps(res, indent=1,
                                                      default=float))
    print(f"[wsT3] family composite={res['composite']:.4f}", flush=True)


def cmd_eval():
    """预注册裁决包：Q2 边际扫描 + Q3 C4 交换（Q1 已在 train 阶段评分）。"""
    h = Harness()
    h.prepare_fast_eval()
    routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
    pred_t3 = np.load(CACHE / "pred_trainval.npy")
    report = {}

    # ---- Q2: wsN30 式边际扫描 ----
    base = h.score_val(routed, verbose=False)["composite"]
    print(f"[Q2] routed 基线 composite={base:.4f}", flush=True)
    for tag, sps in [("strain_both", ["val_strain_only", "val_both"]),
                     ("chem", ["val_chem_only"])]:
        rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
        scan = {}
        for a in (0.05, 0.12, 0.22):
            trial = routed.copy()
            trial[rows] = (1 - a) * routed[rows] + a * pred_t3[rows]
            c = h.score_val(trial, verbose=False)["composite"]
            scan[str(a)] = c
            print(f"  [{tag} α={a}] composite={c:.4f} (Δ{c-base:+.4f})",
                  flush=True)
        report[f"blend_{tag}"] = scan

    # ---- Q3: C4 交换（C1 选择器 P → wsT3 p_hi）----
    from .wsE_depcal import fast_dep_f1, train_side_effects
    from .wsT1_depgate import _apply_push, MIN_PUSH, PUSH_TO
    p_hat = np.load(CACHE / "p_hat.npy")
    rows_tr = np.load("outputs/wsT1/cache/train_rows.npy")
    avail = np.load("outputs/wsT1/cache/train_avail.npy")
    dt = np.load("outputs/wsT1/cache/train_dt.npy").astype(np.float64)
    d_est = np.load("outputs/wsT1/cache/train_dest.npy").astype(np.float64)
    ctrl_hat = np.load("outputs/wsT0/cache/control_hat.npy")
    offset = (ctrl_hat[rows_tr].astype(np.float64)
              - h.Y_tr[rows_tr].astype(np.float64) + dt)
    dp0 = d_est + offset
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows_tr]).astype(np.float64)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows_tr]).astype(np.float64)
    eff0 = train_side_effects(dt, dp0, mu_ctx, mu_drug)
    f1_0 = fast_dep_f1(dt, dp0)
    P_tr = p_hat[rows_tr]
    absDE = np.abs(d_est)
    taus = np.round(np.arange(0.05, 0.951, 0.05), 2)
    best = None
    for tau in taus:
        fl = (P_tr >= tau) & (absDE >= MIN_PUSH) & avail
        dp = _apply_push(dp0, dp0, fl, "push")
        eff = train_side_effects(dt, dp, mu_ctx, mu_drug)
        dmg = sum(eff0[k] - eff[k] for k in ("FC", "ctx", "drug"))
        f1 = fast_dep_f1(dt, dp)
        if dmg <= 0.010 and (best is None or f1 > best[1]):
            best = (tau, f1, dmg, int(fl.sum()))
    tau_c4, f1_c4, dmg_c4, nfl_c4 = best
    print(f"[Q3] C4 train: τ*={tau_c4:.2f} F1={f1_c4:.4f} dmg={dmg_c4:.4f} "
          f"n_flag={nfl_c4:,}（base {f1_0:.4f}；in-sample p 乐观偏差在案）",
          flush=True)
    report["C4_train"] = {"tau": tau_c4, "F1": f1_c4, "dmg": dmg_c4}

    # val 单次应用 C4（无 band，同 C1 形态）
    out = routed.copy()
    for sp in SPLITS:
        rows = h._fast[sp]["rows"]
        trows = rows[h.is_treat_tr[rows]]
        d_v = (routed[trows] - ctrl_hat[trows]).astype(np.float64)
        P_v = p_hat[trows]
        fl = (P_v >= tau_c4) & (np.abs(d_v) >= MIN_PUSH)
        tgt = np.sign(d_v) * np.maximum(np.abs(d_v), PUSH_TO)
        out[trows] = (out[trows].astype(np.float64)
                      + np.where(fl, tgt - d_v, 0.0)).astype(np.float32)
    # train 行同法（in-sample，不参与评分）
    fl = (P_tr >= tau_c4) & (absDE >= MIN_PUSH)
    tgt = np.sign(d_est) * np.maximum(absDE, PUSH_TO)
    out[rows_tr] = (out[rows_tr].astype(np.float64)
                    + np.where(fl, tgt - d_est, 0.0)).astype(np.float32)
    res_c4 = h.score_val(out, verbose=False)
    f1 = lambda r: float(np.mean([r["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    report["C4_val"] = res_c4
    np.save(CACHE / "pred_trainval_C4.npy", out)
    print(f"[Q3] C4 val: composite={res_c4['composite']:.4f} F1={f1(res_c4):.4f} "
          f"（对照 C1: 0.5541 / 0.2442）", flush=True)

    (CACHE / "eval_report.json").write_text(json.dumps(report, indent=1,
                                                       default=float))
    print(f"[saved] {CACHE/'eval_report.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    t0 = time.time()
    {"train": lambda: cmd_train(args), "eval": cmd_eval}[args.stage]()
    print(f"[done] {args.stage} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
