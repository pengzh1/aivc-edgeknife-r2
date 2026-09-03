"""wsU1：Messner-as-samples 多任务族（创造性主攻，第三轮）。

判语驱动：覆盖问题是头号瓶颈（4 训练菌株，strain FC 0.35 天花板，11 路证据）。
Messner 2023（Cell，Mendeley w8jtmnszd9 v2，CC-BY，**须登记披露**）提供
4,699 个 KO 遗传背景的实测蛋白组响应——同物种、同数据类型、唯一大规模
同源数据集。C7 已证明"蛋白级先验"冗余（闸口关闭）；本族改打**响应结构**：
共享 trunk 同时学 KO 扰动 Δ（Messner）与化合物扰动 Δ（我方 train），
把 4,699 个背景的蛋白质响应流形蒸馏进 trunk。

结构（新文件，不碰旧文件）：
  z_ms   = Linear(E_orf[ko 基因]‖E_batch[plate])   ─┐
                                                     ├→ 共享 trunk(64→512→1024)
  z_ours = Linear(E_chem‖E_strain‖E_ctx[wsB 8 列])  ─┘
  头：H_ms(1024→1856 共享蛋白 Δ) / H_ours(1024→5243 我方 Δ)
  逐 batch 随机任务（p=0.5），masked MSE，1:1；wsB 同配方（GELU/LN/Drop、
  AdamW 1e-3、cosine、UNK 语义仅我方侧）。

交付与裁决（预注册单次 val 看）：
  装配 ŷ = wsT3 缓存 control MLP + Δ̂（处理行）/ QC 组均值（QC 行）
  → 单族 score_val（对照 wsB_s16 0.5385 / wsT3 0.5315）；
  ≥0.53 才做 wsN30 式边际扫描（chem / strain+both 区，α∈{0.05,0.12,0.22}，
  Δcomposite ≥ +0.0003 才算有潜力）；零潜力按 T5 关闭归档。

合规：我方侧拟合全部 train-only（Δ 目标/编码器/装配）；Messner 为公开外部
数据（登记）；val 仅本包一次；Y_te 零接触。

用法: python -m src.wsU1_msmultitask            # 全部：prep+train+eval（~30min）
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness
from .wsB_twostage import (EMB_DIMS, DLT_COLS, Encoder, QCGroupMean,
                           masked_mse)
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsU1")
MSDIR = Path("extref/messner2023")
T3C = Path("outputs/wsT3/cache")
SEEDS = list(range(8))
EPOCHS = 200
BS = 256
N_PROT_OURS = 5243


# ------------------------------------------------------------ 数据准备

def prep_ms(h: Harness, device):
    """Messner：log2 宽表 → KO Δ（qc+HIS3 中位基线）+ ORF/plate 索引 + 轴映射。"""
    wide = pd.read_csv(MSDIR / "yeast5k_noimpute_wide.csv")
    meta = pd.read_csv(MSDIR / "yeast5k_metadata.csv")
    assert (wide.columns[1:].to_numpy() == meta["Filename"].to_numpy()).all(), \
        "wide 列与 metadata 行未对齐"
    prots_ms = wide["Protein.Group"].to_numpy()
    W = wide.iloc[:, 1:].to_numpy(dtype=np.float32).T  # (样本, 蛋白)
    W = np.where(W > 0, W, np.nan)
    W = np.log2(W)
    is_ctrl = meta["sampletype"].isin(["qc", "HIS3"]).to_numpy()
    with np.errstate(invalid="ignore"):
        baseline = np.nanmedian(W[is_ctrl], axis=0)
    is_ko = (meta["sampletype"] == "ko").to_numpy()
    D_ms = (W[is_ko] - baseline[None, :]).astype(np.float32)  # (4699, 1850)
    M_ms = ~np.isnan(D_ms)
    D_ms = np.nan_to_num(D_ms, nan=0.0)
    orfs = meta.loc[meta["sampletype"] == "ko", "ORF"].to_numpy()
    plates = meta.loc[meta["sampletype"] == "ko",
                      "Plate (batch) nr"].to_numpy()
    orf_map = {v: i + 1 for i, v in enumerate(sorted(set(orfs)))}
    plate_map = {v: i + 1 for i, v in enumerate(sorted(set(plates)))}
    orf_idx = np.array([orf_map[o] for o in orfs])
    plate_idx = np.array([plate_map[p] for p in plates])
    print(f"[prep] Messner KO {D_ms.shape} | ORF {len(orf_map)} "
          f"plate {len(plate_map)} | 可用条目 {M_ms.mean():.1%}", flush=True)

    # 轴映射：我方蛋白 → Messner 行（UniProt accession）
    up = pd.read_csv(MSDIR / "uniprot_sgd_map.tsv", sep="\t")
    up_gene = {}
    for _, r in up.iterrows():
        names = set()
        for col in ("Gene Names", "Gene Names (ordered locus)"):
            v = r.get(col)
            if isinstance(v, str):
                names.update(v.split())
        for nm in names:
            up_gene.setdefault(nm, r["Entry"])
    g2l = json.loads(Path("extref/hop/gene2locus.json").read_text())
    ms_idx_of = {a: i for i, a in enumerate(prots_ms)}
    our_prots = h.proteins
    shared_our, shared_ms = [], []
    for j, p in enumerate(our_prots):
        acc = up_gene.get(p) or up_gene.get(g2l.get(p, ""))
        if acc in ms_idx_of:
            shared_our.append(j)
            shared_ms.append(ms_idx_of[acc])
    print(f"[prep] 共享蛋白轴 {len(shared_our)} 条", flush=True)
    np.savez(OUT / "ms_data.npz", D_ms=D_ms, M_ms=M_ms,
             orf_idx=orf_idx, plate_idx=plate_idx,
             shared_our=np.array(shared_our), shared_ms=np.array(shared_ms),
             n_orf=len(orf_map) + 1, n_plate=len(plate_map) + 1)
    t = lambda a: torch.tensor(a, device=device)
    return (t(D_ms), t(M_ms.astype(np.float32)), t(orf_idx), t(plate_idx),
            len(orf_map) + 1, len(plate_map) + 1,
            np.array(shared_our), np.array(shared_ms))


# ------------------------------------------------------------ 模型

class JointMLP(nn.Module):
    def __init__(self, n_orf, n_plate, n_cats_ours, n_ms, n_ours,
                 p_drop=0.1):
        super().__init__()
        self.E_orf = nn.Embedding(n_orf, 32)
        self.E_plate = nn.Embedding(n_plate, 8)
        self.adapter_ms = nn.Linear(40, 64)
        self.embs = nn.ModuleList([
            nn.Embedding(n, EMB_DIMS[c]) for n, c in zip(n_cats_ours, DLT_COLS)])
        d_ours = sum(EMB_DIMS[c] for c in DLT_COLS)
        self.adapter_ours = nn.Linear(d_ours, 64)
        self.trunk = nn.Sequential(
            nn.Linear(64, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(p_drop),
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024),
            nn.Dropout(p_drop))
        self.H_ms = nn.Linear(1024, n_ms)
        self.H_ours = nn.Linear(1024, n_ours)
        nn.init.zeros_(self.H_ms.bias)
        nn.init.zeros_(self.H_ours.bias)

    def _z_ms(self, xo, xp):
        return self.adapter_ms(torch.cat(
            [self.E_orf(xo), self.E_plate(xp)], dim=1))

    def _z_ours(self, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)],
                      dim=1)
        return self.adapter_ours(e)

    def forward_ms(self, xo, xp):
        return self.H_ms(self.trunk(self._z_ms(xo, xp)))

    def forward_ours(self, x_cat):
        return self.H_ours(self.trunk(self._z_ours(x_cat)))


# ------------------------------------------------------------ 训练

def train_joint(seed, ms, ours, n_cats_ours, n_ms, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D_ms, M_ms, orf_idx, plate_idx = ms
    X_ours, D_ours, M_ours, rows_ours = ours
    model = JointMLP(int(orf_idx.max()) + 1, int(plate_idx.max()) + 1,
                     n_cats_ours, n_ms, N_PROT_OURS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_ms_rows = len(orf_idx)
    n_ours_rows = len(rows_ours)
    steps_per_ep = int(np.ceil((n_ms_rows + n_ours_rows) / 2 / BS))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * steps_per_ep)
    for ep in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for _ in range(steps_per_ep):
            if np.random.rand() < 0.5:
                r = torch.randint(0, n_ms_rows, (BS,), device=device)
                pred = model.forward_ms(orf_idx[r], plate_idx[r])
                loss = masked_mse(pred, D_ms[r], M_ms[r])
            else:
                r = rows_ours[torch.randint(0, n_ours_rows, (BS,),
                                            device=device)]
                pred = model.forward_ours(X_ours[r])
                loss = masked_mse(pred, D_ours[r], M_ours[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 40 == 0 or ep == EPOCHS - 1:
            print(f"    [s{seed}] ep {ep+1:>3}/{EPOCHS} loss={tot/nb:.4f}",
                  flush=True)
    return model


@torch.no_grad()
def predict_ours(model, X_ours, device, bs=2048):
    model.eval()
    outs = []
    for i in range(0, X_ours.shape[0], bs):
        outs.append(model.forward_ours(X_ours[i:i + bs]).float().cpu())
    return torch.cat(outs).numpy()


# ------------------------------------------------------------ 主流程

def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = Harness()
    h.prepare_fast_eval()
    m = h.m_tr
    pert = m["perturbation_no_concentration"]
    is_ctrl = pert.isin(D.CONTROLS).to_numpy()
    is_qc = (pert == D.QC).to_numpy()
    is_treat = h.is_treat_tr
    tr = h.tr_rows
    treat_rows = tr[is_treat[tr]]
    qc_rows = tr[is_qc[tr]]

    # ---- Messner 数据 ----
    ms = prep_ms(h, device)
    (D_ms_t, M_ms_t, orf_t, plate_t, n_orf, n_plate,
     shared_our, shared_ms) = ms
    n_ms = len(shared_ms)
    # Messner 目标压缩到共享轴（H_ms 输出 = 共享蛋白的 Δ；顺序=shared_our）
    D_ms_s = D_ms_t[:, torch.tensor(shared_ms, device=device)]
    M_ms_s = M_ms_t[:, torch.tensor(shared_ms, device=device)]
    ms_pack = (D_ms_s, M_ms_s, orf_t, plate_t)

    # ---- 我方数据（wsB 同式 Δ 目标）----
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    X_ours = torch.tensor(enc_d.transform(m), dtype=torch.long, device=device)
    Dmat = torch.tensor(h.delta_tr_all, dtype=torch.float32, device=device)
    M_ours = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)
    ours_pack = (X_ours, Dmat, M_ours.float(),
                 torch.tensor(treat_rows, device=device))

    # ---- 8 种子联合训练 ----
    delta_hat = np.zeros((len(m), N_PROT_OURS), np.float32)
    for sd in SEEDS:
        t1 = time.time()
        model = train_joint(sd, ms_pack, ours_pack, enc_d.n_cats, n_ms, device)
        delta_hat += predict_ours(model, X_ours, device).astype(np.float32)
        print(f"  [wsU1] seed {sd} ({time.time()-t1:.0f}s)", flush=True)
    delta_hat /= len(SEEDS)
    np.save(OUT / "delta_hat.npy", delta_hat)

    # ---- 装配（wsB 同式：control MLP 复用 wsT3 缓存 + QC 组均值）----
    ctrl_mlp = np.load(T3C / "ctrl_mlp.npy")
    qc_model = QCGroupMean().fit(m.iloc[qc_rows].reset_index(drop=True),
                                 h.Y_tr[qc_rows], h.stats.protein_mean)
    qc_pred = qc_model.predict(m)
    pred = ctrl_mlp.copy()
    pred[is_treat] += delta_hat[is_treat]
    pred[is_qc] = qc_pred[is_qc]
    pred = h.stats.impute(pred).astype(np.float32)
    np.save(OUT / "pred_trainval.npy", pred)
    res = h.score_val(pred)
    f1 = float(np.mean([res["per_split"][sp]["DEP_F1"] for sp in SPLITS]))
    print(f"[wsU1] family composite={res['composite']:.4f} F1={f1:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    (OUT / "family_score.json").write_text(json.dumps(
        {"res": res, "seeds": SEEDS, "epochs": EPOCHS}, indent=1,
        default=float))

    # ---- 边际扫描（≥0.53 才做）----
    if res["composite"] >= 0.53:
        routed = np.load("outputs/wsT0/cache/routed_r07_trainval.npy")
        base = h.score_val(routed, verbose=False)["composite"]
        report = {"base": base}
        for tag, sps in [("strain_both", ["val_strain_only", "val_both"]),
                         ("chem", ["val_chem_only"])]:
            rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
            scan = {}
            for a in (0.05, 0.12, 0.22):
                trial = routed.copy()
                trial[rows] = (1 - a) * routed[rows] + a * pred[rows]
                c = h.score_val(trial, verbose=False)["composite"]
                scan[str(a)] = c
                print(f"  [{tag} α={a}] composite={c:.4f} (Δ{c-base:+.4f})",
                      flush=True)
            report[f"blend_{tag}"] = scan
        (OUT / "blendscan.json").write_text(json.dumps(report, indent=1,
                                                       default=float))
    else:
        print("[wsU1] 单族 <0.53，不做边际扫描（T5 关闭线）", flush=True)


if __name__ == "__main__":
    main()
