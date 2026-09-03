"""wsI：Q3 诊断 —— "BAI 可用"上界（隔离诊断，不进入任何提交与集成）。

背景：组委会 Q3 —— "训练数据"指 train_val 全体还是仅 split_final=train 子集？
若答案是 train_val 全体，则最终 test 模型可合法使用 val 划分数据：BAI 菌株
（test_both 432/1129 行）与 6 个 val 化合物将从"未见"变"已见"。本模块在 val
划分内模拟"已见 vs 未见"，量化该口径变化对 test_both（BAI 部分）的提分上界，
为是否向组委会争取提供数字依据。

结构镜像：val_strain_only = BAI × 40 个 train 化合物（BAI 信息的来源）；
val_both = BAI × 6 个 val 新化合物，与 test_both 的 BAI 部分（BAI × 11 个 test
新化合物）同构（菌株/化合物双未见）。

协议（统一 wsD 简化配方 MLP：hidden (512,1024,2048)、150 epochs、emb_drop 0.35、
p_drop 0.3、Huber loss、bs 256、lr 1e-3、wd 1e-4、chem_emb 32、2 种子均值）：
  A 未见基线（合规复现）：训练 = h.tr_rows；
    评估 val_strain_only 全部 / vso 后半（与 B 同评估集）/ val_both。
  B 半已见上界：训练 = train + val_strain_only 随机一半（RandomState(0) 固定）；
    评估另一半 + val_both 全部。
  C 全已见乐观上界：训练 = train + val_strain_only 全部；评估 val_both。

口径控制：归一化统计（protein_mean/std）与 strain 侧 μ_drug 参照均按 train split
冻结，B/C 不重算；Δ_true 与对照匹配沿用官方 train_val 池；Y_te 零接触。

防 OOM：训练 batch=256（≤512）；GPU 仅驻留 1 个全量标签矩阵（Y_dev），
mask 按 batch 现算。

隔离：全部输出在 outputs/wsI/ 且文件名带 _DIAG；pred_DIAG.npy 禁止进入集成。

用法:
    python -m src.wsI_bai_diag
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics as M
from .evaluate import Harness
from .train_mlp import Encoder
from .wsD_arch import Cfg, ProteoMLP2, masked_huber

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsI"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# wsD 配方简化版（任务指定：3 层 (512,1024,2048)、150ep、emb_drop 0.35、2 种子）
CFG = Cfg(hidden=[512, 1024, 2048], epochs=150, lr=1e-3, wd=1e-4,
          emb_drop=0.35, p_drop=0.3, bs=256, chem_emb=32, loss="huber")
SEEDS = [0, 1]
HALF_SEED = 0  # val_strain_only 随机二分的固定种子

_TENSORS = {}


def get_tensors(h: Harness):
    """编码张量与原始标签（含 NaN）只构建一次，全部驻留 CPU。

    归一化统计固定用 h.stats（train split 冻结），三协议共用，保证口径一致。
    """
    if "t" not in _TENSORS:
        enc = Encoder().fit(h.m_tr)  # 类别覆盖 train_val 全集（可见性由训练行控制）
        X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
        mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
        std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
        Y_all = torch.tensor(h.Y_tr, dtype=torch.float32)  # 保留 NaN，mask 现算
        _TENSORS["t"] = (enc, X_all, mean, std, Y_all)
    return _TENSORS["t"]


def train_one(h: Harness, rows: np.ndarray, seed: int,
              device: str = "cuda") -> tuple:
    """单次训练。与 wsD 配方一致，唯一区别：GPU 只驻留 Y_all 一个全量标签矩阵，
    标准化与 mask 按 batch 现算（数学上与预计算 Z/M 完全等价）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Y_all = get_tensors(h)
    n_prot = h.Y_tr.shape[1]

    model = ProteoMLP2(enc.n_cats, n_prot, CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    n_steps = CFG.epochs * int(np.ceil(len(rows) / CFG.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev = X_all.to(device)
    Y_dev = Y_all.to(device)  # 唯一的全量标签矩阵（float32, ~188MB）
    mean_dev = mean.to(device)
    std_dev = std.to(device)
    rows_dev = torch.tensor(rows, device=device)

    for ep in range(CFG.epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), CFG.bs):
            r = perm[i:i + CFG.bs]
            xb = X_dev[r].clone()
            # embedding dropout：菌株/化合物随机替换为 UNK(0)，学 OOD 回退
            for col in [0, 1]:
                dm = torch.rand(len(r), device=device) < CFG.emb_drop
                xb[dm, col] = 0
            pred = model(xb)
            yb = Y_dev[r]
            msk = ~torch.isnan(yb)
            zb = torch.nan_to_num((yb - mean_dev) / std_dev, nan=0.0)
            loss = masked_huber(pred, zb, msk.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 50 == 0 or ep == CFG.epochs - 1:
            print(f"    epoch {ep+1:>3}/{CFG.epochs}  loss={tot/nb:.4f}",
                  flush=True)
    return model, enc, mean, std


@torch.no_grad()
def predict(model, enc, mean, std, m, device="cuda", bs: int = 512) -> np.ndarray:
    model.eval()
    X = torch.tensor(enc.transform(m), dtype=torch.long, device=device)
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(X[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def eval_rows(h: Harness, pred: np.ndarray, rows: np.ndarray,
              kind: str) -> dict:
    """在指定行集上评估 fidelity / FC_PCC（/ strain 侧 resid_PCC）。

    kind="strain" 时用 train split 冻结的 μ_drug 做残差参照（三协议口径一致）。
    Δ_pred = ŷ − 匹配对照真值（官方 train_val 池），仅处理样本计入 FC/resid。
    """
    yt, yp = h.Y_tr[rows], pred[rows]
    s = {"n_rows": int(len(rows))}
    s["sample_PCC"] = float(np.mean(M._masked_pcc_axis1(yt, yp)))
    s["sample_R2"] = float(np.mean(np.clip(M._masked_r2_axis1(yt, yp), -1, 1)))
    trows = rows[h.is_treat_tr[rows]]
    s["n_treat"] = int(len(trows))
    dpred = h._delta_pred(trows, pred)
    dt = h.delta_tr_all[trows]
    s["FC_PCC"] = float(np.nanmean(M._masked_pcc_axis1(dt, dpred)))
    if kind == "strain":
        mu = h.mu_drug_for(h.m_tr.iloc[trows])  # train split 冻结，不重算
        s["resid_PCC"] = float(np.nanmean(
            M._masked_pcc_axis1(dt - mu, dpred - mu)))
    return s


def run_protocol(h: Harness, name: str, rows: np.ndarray,
                 eval_sets: dict) -> tuple[dict, np.ndarray]:
    """2 种子训练 + 均值集成，逐评估集打分。返回 (结果, 集成预测)。"""
    print(f"[{name}] train rows={len(rows)}  seeds={SEEDS}", flush=True)
    preds, per_seed = [], []
    for s in SEEDS:
        t0 = time.time()
        model, enc, mean, std = train_one(h, rows, s)
        pred = predict(model, enc, mean, std, h.m_tr)
        per_seed.append({es: eval_rows(h, pred, r, k)
                         for es, (r, k) in eval_sets.items()})
        print(f"  seed={s} done ({time.time()-t0:.0f}s)", flush=True)
        preds.append(pred)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    ens = {es: eval_rows(h, P, r, k) for es, (r, k) in eval_sets.items()}
    return {"train_rows": int(len(rows)), "per_seed": per_seed,
            "ensemble": ens}, P


def main():
    h = Harness()
    sp = h.m_tr["split_final"].to_numpy()
    vso = np.where(sp == "val_strain_only")[0]  # BAI × train 化合物（含对照/QC）
    vb = np.where(sp == "val_both")[0]          # BAI × 6 个 val 新化合物

    # val_strain_only 按样本随机二分（种子固定）：前半入 B 训练，后半留评估
    rng = np.random.RandomState(HALF_SEED)
    perm = rng.permutation(len(vso))
    half = len(vso) // 2
    vso_tr_half = np.sort(vso[perm[:half]])
    vso_ev_half = np.sort(vso[perm[half:]])
    print(f"[split] vso={len(vso)} -> train_half={len(vso_tr_half)} "
          f"eval_half={len(vso_ev_half)} | val_both={len(vb)}", flush=True)

    results = {"cfg": {k: getattr(CFG, k) for k in
                       ["hidden", "epochs", "lr", "wd", "emb_drop",
                        "p_drop", "bs", "chem_emb", "loss"]},
               "seeds": SEEDS, "half_seed": HALF_SEED,
               "note": "DIAG ONLY - not for submission/ensemble"}

    # 协议 A：未见基线（合规复现）
    resA, _ = run_protocol(h, "A_unseen", h.tr_rows, {
        "vso_full": (vso, "strain"),
        "vso_half": (vso_ev_half, "strain"),
        "val_both": (vb, "both"),
    })
    results["A"] = resA

    # 协议 B：半已见上界（train + vso 前半；评估 vso 后半 + val_both）
    rows_B = np.concatenate([h.tr_rows, vso_tr_half])
    resB, PB = run_protocol(h, "B_half_seen", rows_B, {
        "vso_half": (vso_ev_half, "strain"),
        "val_both": (vb, "both"),
    })
    results["B"] = resB

    # 协议 C：全已见乐观上界（train + vso 全部；评估 val_both）
    rows_C = np.concatenate([h.tr_rows, vso])
    resC, _ = run_protocol(h, "C_full_seen", rows_C, {
        "val_both": (vb, "both"),
    })
    results["C"] = resC

    # 诊断预测存档（协议 B 集成；_DIAG 后缀，禁止进入集成）
    assert not np.isnan(PB).any() and not np.isinf(PB).any()
    np.save(OUT_DIR / "pred_DIAG.npy", PB)

    (OUT_DIR / "diag_results_DIAG.json").write_text(
        json.dumps(results, indent=1))
    print(f"[saved] {OUT_DIR/'pred_DIAG.npy'}  {OUT_DIR/'diag_results_DIAG.json'}")

    # 汇总打印
    print("\n===== 集成（2 种子均值）汇总 =====")
    hdr = f"{'protocol':<12}{'eval_set':<12}{'n_treat':>8}{'sample_PCC':>11}" \
          f"{'FC_PCC':>9}{'resid_PCC':>10}"
    print(hdr)
    print("-" * len(hdr))
    for proto in ["A", "B", "C"]:
        for es, s in results[proto]["ensemble"].items():
            print(f"{proto:<12}{es:<12}{s['n_treat']:>8}"
                  f"{s['sample_PCC']:>11.4f}{s['FC_PCC']:>9.4f}"
                  f"{s.get('resid_PCC', float('nan')):>10.4f}")


if __name__ == "__main__":
    main()
