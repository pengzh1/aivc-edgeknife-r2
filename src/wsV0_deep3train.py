"""wsV0：deep3-fuse3 自定义训练基座（机制实验轮：loss 形状 / 输入增广）。

供 wsV1（listwise 排序辅助）与 wsV2（嵌入 mixup）共用；数据管线与
wsN24_fuse3e150 完全一致（fuse3 三源 PCA64 化合物表征、deep3 容量、
Z-scored 丰度目标、emb_drop/chem_drop=0.25、150ep、AdamW+cosine），
唯一差异 = 训练目标/增广（机制变量）。

合规：train-only（h.tr_rows）；val 仅经调用方一次性评分。
"""
import numpy as np
import torch
import torch.nn as nn

from . import wsA_chemfeat as WSA
from .wsN6_chemberta import table_to_loader

HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2


def build_data(h, table_path, device):
    """返回 (model, X_dev, Z_dev, M_dev, rows_dev)。"""
    import pandas as pd
    orig = WSA.load_chem_table
    df = pd.read_csv(table_path)
    WSA.load_chem_table = table_to_loader(df)
    try:
        enc = WSA.Encoder().fit(h.m_tr)
        feat, mean_vec = WSA.load_chem_table(h, WSA.OUT_DIR, full=False)
    finally:
        WSA.load_chem_table = orig
    chem_mat = WSA.make_chem_mat(enc, feat, mean_vec)
    n_prot = h.Y_tr.shape[1]
    model = WSA.ProteoMLPChem(enc.n_cats, chem_mat, n_prot,
                              hidden=HIDDEN, p_drop=P_DROP).to(device)
    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
    Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)
    rows = torch.tensor(h.tr_rows, device=device)
    return (model, X_all.to(device), Z_all.to(device),
            M_all.to(device).float(), rows)


def _embed(model, x):
    parts = [model.embs[k](x[:, i]) for k, i in enumerate(model.emb_cols)]
    cv = model.chem_proj(model.chem_mat[x[:, model.chem_col]])
    return torch.cat([parts[0], cv] + parts[1:], dim=1)


def _listwise_kl(pred, target, mask, tau=1.0):
    """逐样本 listwise：KL(softmax(t/τ) ‖ softmax(p/τ))，仅可用条目。"""
    p = pred.masked_fill(mask <= 0, -1e9)
    t = target.masked_fill(mask <= 0, -1e9)
    log_p = torch.log_softmax(p / tau, dim=1)
    log_t = torch.log_softmax(t / tau, dim=1)
    t_dist = log_t.exp()
    kl = (t_dist * (log_t - log_p)).sum(1)
    n = mask.sum(1)
    ok = n >= 8
    return kl[ok].mean() if ok.any() else kl.mean() * 0.0


def train_one(h, seed, table_path, epochs=150, bs=256, lr=1e-3,
              emb_drop=0.25, chem_drop=0.25, aux_rank=0.0, mixup_alpha=0.0,
              device="cuda", log_every=999):
    """单一种子训练。aux_rank>0 启用排序辅助（权重）；mixup_alpha>0 启用嵌入 mixup。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, X_dev, Z_dev, M_dev, rows_dev = build_data(h, table_path, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * int(np.ceil(len(rows_dev) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    for ep in range(epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            xb = X_dev[r].clone()
            if emb_drop > 0:
                drop = torch.rand(len(r), device=device) < emb_drop
                xb[drop, 0] = 0
            if chem_drop > 0:
                drop = torch.rand(len(r), device=device) < chem_drop
                xb[drop, 1] = 0
            if mixup_alpha > 0:
                perm_b = torch.randperm(len(r), device=device)
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                e = _embed(model, xb)
                e = lam * e + (1 - lam) * _embed(model, xb[perm_b])
                pred = model.head(model.trunk(e))
                Zb = Z_dev[r]
                Zm = lam * Zb + (1 - lam) * Zb[perm_b]
                Mb = (M_dev[r] * M_dev[r[perm_b]])
            else:
                pred = model(xb)
                Zm, Mb = Z_dev[r], M_dev[r]
            loss = WSA.masked_mse(pred, Zm, Mb)
            if aux_rank > 0:
                loss = loss + aux_rank * _listwise_kl(pred, Zm, Mb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    [s{seed}] ep {ep+1:>3}/{epochs} loss={tot/nb:.4f}",
                  flush=True)
    return model


@torch.no_grad()
def predict_all(model, h, device="cuda", bs=2048):
    """对全部 train_val 行推理（train 词表；与 wsN24 val 版同口径）。"""
    model.eval()
    enc = WSA.Encoder().fit(h.m_tr)
    X = torch.tensor(enc.transform(h.m_tr), dtype=torch.long, device=device)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32,
                        device=device)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32,
                       device=device)
    outs = []
    for i in range(0, X.shape[0], bs):
        p = model(X[i:i + bs]) * std + mean
        outs.append(p.float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)


def family_eval(h, preds_by_seed, out_path, tag):
    """多种子平均 + NaN 保险 + 评分 + 落盘。返回 res。"""
    import json
    P = np.mean(preds_by_seed, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    np.save(out_path, P)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"]
                        for sp in res["per_split"]]))
    print(f"[{tag}] composite={res['composite']:.4f} FC={fc} F1={f1:.4f}",
          flush=True)
    (out_path.parent / "scores.json").write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "F1": f1,
         "per_split": res["per_split"]}, default=float, indent=1))
    return res
