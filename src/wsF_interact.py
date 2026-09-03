"""wsF：显式 chem×ctx 交互 MLP + 低秩响应程序分解（封闭数据榜，主攻 strain_only/both）。

动机：strain_only 残差 = PCC(Δ−μ_drug)，对未见菌株（BAI）唯一可迁移信号是
"化合物响应如何被上下文（培养基/温度/时间）调制"。两个方向：

方向1（d1）显式交互 MLP：在 src/train_mlp.py 框架上加乘性交互通路
  - inter_ctx    : chem_emb ⊙ W_ctx·[medium,temp,time]（chem×ctx，可迁移到未见菌株）
  - inter_strain : chem_emb ⊙ W_s·strain_emb（chem×strain 低秩三向）
  交互向量与嵌入拼接进 trunk。训练配置沿用 wsD 最佳：Huber、300ep、dropout 0.3、
  emb_drop 0.35、bs 256、lr 1e-3、wd 1e-4。
  另加 unk_unseen：推理时把"train 行未出现的类别"（BAI、val 专属化合物等）映射到
  UNK(0)——这些私有嵌入从未得到梯度（保持随机初始化），而 UNK 经 emb_drop 显式训练过。

方向2（d2）低秩响应程序：
  - train split 处理样本 Δ（对照池严格限 train split 行）NaN→0，CPU TruncatedSVD → 响应基 P
  - w* = 带缺失掩膜的岭最小二乘投影（GPU 分块解 r×r 正规方程）
  - 小 MLP：条件嵌入 → ẑ（标准化权重），损失 = masked MSE(ẑ·diag(σ_w)·P, Δ) + λ·MSE(ẑ, z*)
  - 推理 ŷ = control_hat + ŵ·P（对照/QC 行 = control_hat，复用 wsE.build_control_hat）
  - use_strain True/False 两版；未见菌株行可路由到无 strain 版

防 OOM：batch ≤ 512；GPU 上至多 1 个 (8958×5243) f32（Z）+ 1 个 bool 掩膜；
SVD 在 CPU（float32）；投影分块 256。

合规：训练只用 h.tr_rows；Δ 训练目标的对照池只含 train split 行（比 harness 更严）；
val 划分仅经 h.score_val/score_fast 评分；不触碰 h.Y_te；随机种子固定。

用法：
  python -m src.wsF_interact d2prep
  python -m src.wsF_interact d1 --name d1_ctx3L --cfg '{"inter_ctx":true}' --seeds 0,1,2
  python -m src.wsF_interact d2 --name L32S --cfg '{"rank":32,"use_strain":true}' --seeds 0,1,2
  python -m src.wsF_interact d2route --s L32S --ns L32N
  python -m src.wsF_interact blend --a <predA.npy> --b <predB.npy>
  python -m src.wsF_interact deliver --src <pred.npy>
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from . import metrics as M
from .evaluate import Harness
from .train_mlp import CAT_COLS, EMB_DIMS, Encoder, masked_mse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "wsF"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.jsonl"

COL = {c: i for i, c in enumerate(CAT_COLS)}
CHEM = COL["perturbation_no_concentration"]
STRAIN = COL["Strains"]


def log_result(rec: dict):
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def masked_huber(pred, target, mask, beta: float = 1.0):
    se = F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * mask
    return se.sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------- 编码器

class SeenEncoder:
    """train_val 全类别编码 + 推理期"训练行未见类别 → UNK(0)"映射。

    Encoder 给 train_val 全部类别分配索引，但 val 专属类别（BAI、6 个 val 化合物）
    的嵌入在训练中从不获梯度（保持随机初始化）。emb_drop 训练的是 UNK(0) 回退，
    因此推理时应把未见类别显式映射到 0。seen 集合只来自 train split 元数据。
    """

    def __init__(self, enc: Encoder, m_train: pd.DataFrame):
        self.enc = enc
        self.seen = [set(m_train[c].unique()) for c in CAT_COLS]

    @property
    def n_cats(self):
        return self.enc.n_cats

    def transform(self, m: pd.DataFrame, unk_unseen: bool = True) -> np.ndarray:
        cols = []
        for c, seen in zip(CAT_COLS, self.seen):
            mp = self.enc.maps[c]
            s = m[c]
            idx = s.map(lambda v: mp.get(v, 0)).to_numpy()
            if unk_unseen:
                idx[~s.isin(seen).to_numpy()] = 0
            cols.append(idx.astype(np.int64))
        return np.stack(cols, axis=1)


def build_senc(h: Harness) -> SeenEncoder:
    return SeenEncoder(Encoder().fit(h.m_tr), h.m_train)


# ---------------------------------------------------------------- 方向 1：显式交互 MLP

@dataclass
class CfgD1:
    hidden: list = field(default_factory=lambda: [512, 1024, 2048])
    epochs: int = 300
    lr: float = 1e-3
    wd: float = 1e-4
    emb_drop: float = 0.35
    p_drop: float = 0.3
    bs: int = 256
    loss: str = "huber"          # huber | mse
    inter_ctx: bool = True       # chem ⊙ W_ctx·[medium,temp,time]
    inter_strain: bool = False   # chem ⊙ W_s·strain
    unk_unseen: bool = True      # 未见类别 → UNK（推理期）
    aux_delta: float = 0.0       # >0：Δ 模式辅助损失权重（见 aux_pack_d1）
    aux_terms: str = "fc,ctx,drug"   # 辅助损失项：fc / ctx(μ_ctx残差) / drug(μ_drug残差)
    strain_blind: bool = False   # True：训练时 strain 恒为 UNK（纯 chem×ctx 响应模型）

    @classmethod
    def from_dict(cls, d: dict) -> "CfgD1":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


class InteractMLP(nn.Module):
    def __init__(self, n_cats, n_prot, cfg: CfgD1):
        super().__init__()
        dims = dict(EMB_DIMS)
        self.cfg = cfg
        self.embs = nn.ModuleList(
            [nn.Embedding(n, dims[c]) for n, c in zip(n_cats, CAT_COLS)])
        d_emb = sum(dims.values())
        cdim = dims["perturbation_no_concentration"]
        d_in = d_emb
        if cfg.inter_ctx:
            d_ctx = dims["Medium"] + dims["Temperature"] + dims["pert_time"]
            self.ctx_proj = nn.Linear(d_ctx, cdim)
            d_in += cdim
        if cfg.inter_strain:
            self.strain_proj = nn.Linear(dims["Strains"], cdim)
            d_in += cdim
        blocks, d = [], d_in
        for hdim in cfg.hidden:
            blocks += [nn.Linear(d, hdim), nn.GELU(), nn.LayerNorm(hdim),
                       nn.Dropout(cfg.p_drop)]
            d = hdim
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        parts = [torch.cat(e, dim=1)]
        chem = e[CHEM]
        if self.cfg.inter_ctx:
            ctx = torch.cat([e[COL["Medium"]], e[COL["Temperature"]],
                             e[COL["pert_time"]]], dim=1)
            parts.append(chem * self.ctx_proj(ctx))
        if self.cfg.inter_strain:
            parts.append(chem * self.strain_proj(e[STRAIN]))
        h = torch.cat(parts, dim=1)
        return self.head(self.trunk(h))


def tensors_d1(h: Harness, senc: SeenEncoder, unk_unseen: bool):
    key = f"d1_{unk_unseen}"
    if not hasattr(tensors_d1, "_c"):
        tensors_d1._c = {}
    if key not in tensors_d1._c:
        X_all = torch.tensor(senc.transform(h.m_tr, unk_unseen=unk_unseen),
                             dtype=torch.long)
        mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
        std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
        Z = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
        Mk = ~torch.isnan(Z)
        Z = torch.nan_to_num(Z, nan=0.0)
        tensors_d1._c[key] = (X_all, mean, std, Z, Mk)
    return tensors_d1._c[key]


# ---- Δ 模式辅助损失（直接对齐 FC/ctx_resid/drug_resid 三个逐样本 PCC 指标）----

def _center_rows(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按掩膜逐行去均值（掩膜外置 0）。x/mask (B,P)。"""
    n = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu = (x * mask).sum(dim=1, keepdim=True) / n
    return (x - mu) * mask


def aux_pack_d1(h: Harness):
    """train 处理行的辅助损失张量（CPU 缓存 → GPU）。

    anchor  : control_hat[rows]（train 冻结对照查找，≈ 评分用匹配对照）
    T0      : centered(Δ_true)
    CM1/CM2 : centered(μ_ctx)/centered(μ_drug)（harness 冻结参照，逐行）
    掩膜统一为 Δ_true 观测掩膜。辅助损失：
      aux = MSE(cp, T0) + MSE(cp − CM1, T0 − CM1) + MSE(cp − CM2, T0 − CM2)
      cp  = centered(ŷ_raw − anchor)，即评分口径 Δ̂ 的逐样本中心化模式。
    """
    f = CACHE / "aux_d1.npz"
    if f.exists():
        z = np.load(f)
        pack = {k: torch.tensor(z[k]) for k in
                ["anchor", "T0", "CM1", "CM2", "Mδ", "row2t"]}
        pack["vΔ"] = float(z["vΔ"])
        return pack
    dt, rows = d2_delta_train(h)
    ctrl = d2_control_hat(h)
    anchor = ctrl[rows].astype(np.float32)
    mδ = ~np.isnan(dt)
    dt0 = np.nan_to_num(dt, nan=0.0).astype(np.float32)
    mu_ctx = h.mu_ctx_for(h.m_tr.iloc[rows]).astype(np.float32)
    mu_drug = h.mu_drug_for(h.m_tr.iloc[rows]).astype(np.float32)

    def center(a):
        t = torch.tensor(a)
        m = torch.tensor(mδ, dtype=torch.float32)
        return _center_rows(t, m).numpy().astype(np.float32)

    T0 = center(dt0)
    CM1 = center(mu_ctx)
    CM2 = center(mu_drug)
    vΔ = float((T0[mδ] ** 2).mean())
    row2t = np.full(len(h.m_tr), -1, dtype=np.int64)
    row2t[rows] = np.arange(len(rows))
    np.savez(f, anchor=anchor, T0=T0, CM1=CM1, CM2=CM2,
             Mδ=mδ.astype(np.bool_), row2t=row2t, vΔ=np.float64(vΔ))
    print(f"[aux] train treated={len(rows)} vΔ={vΔ:.4f}")
    pack = {k: torch.tensor(np.load(f)[k]) for k in
            ["anchor", "T0", "CM1", "CM2", "Mδ", "row2t"]}
    pack["vΔ"] = vΔ
    return pack


def aux_delta_loss(pred_z, r, mean, std, pk, terms: str):
    """pred_z (B,P) 标准化预测；r 绝对行号；pk 张量已在 device 上。"""
    t_idx = pk["row2t"][r]
    ok = t_idx >= 0
    if not ok.any():
        return pred_z.new_zeros(())
    ti = t_idx[ok]
    anchor = pk["anchor"][ti]
    Mδ = pk["Mδ"][ti].float()
    T0 = pk["T0"][ti]
    y_raw = pred_z[ok] * std + mean
    cp = _center_rows(y_raw - anchor, Mδ)
    loss = pred_z.new_zeros(())
    if "fc" in terms:
        loss = loss + masked_mse(cp, T0, Mδ)
    if "ctx" in terms:
        CM1 = pk["CM1"][ti]
        loss = loss + masked_mse(cp - CM1, T0 - CM1, Mδ)
    if "drug" in terms:
        CM2 = pk["CM2"][ti]
        loss = loss + masked_mse(cp - CM2, T0 - CM2, Mδ)
    return loss / pk["vΔ"]


def train_d1(h: Harness, cfg: CfgD1, seed: int, senc: SeenEncoder,
             device: str = "cuda", log_every: int = 100,
             rows: np.ndarray | None = None, tensors=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_all, mean, std, Z_all, M_all = (
        tensors if tensors is not None
        else tensors_d1(h, senc, cfg.unk_unseen))
    if rows is None:
        rows = h.tr_rows
    model = InteractMLP(senc.n_cats, h.Y_tr.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    X_dev = X_all.to(device)
    Z_dev = Z_all.to(device)       # 唯一 1 个 (8958×5243) f32
    M_dev = M_all.to(device)       # bool
    rows_dev = torch.tensor(rows, device=device)
    mean_d, std_d = mean.to(device), std.to(device)
    pk = None
    if cfg.aux_delta > 0:
        pk = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in aux_pack_d1(h).items()}
    for ep in range(cfg.epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            if cfg.strain_blind:
                xb[:, STRAIN] = 0
            if cfg.emb_drop > 0:
                for col in [STRAIN, CHEM]:
                    dm = torch.rand(len(r), device=device) < cfg.emb_drop
                    xb[dm, col] = 0
            pred = model(xb)
            if cfg.loss == "huber":
                loss = masked_huber(pred, Z_dev[r], M_dev[r].float())
            else:
                loss = masked_mse(pred, Z_dev[r], M_dev[r].float())
            if pk is not None:
                loss = loss + cfg.aux_delta * aux_delta_loss(
                    pred, r, mean_d, std_d, pk, cfg.aux_terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if log_every and ((ep + 1) % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  [d1 seed{seed}] epoch {ep+1:>3}/{cfg.epochs} "
                  f"loss={tot/nb:.4f}", flush=True)
    return model, mean, std


@torch.no_grad()
def predict_d1(model, X_all, mean, std, device="cuda", bs: int = 512):
    model.eval()
    X_dev = X_all.to(device)
    outs = []
    for i in range(0, len(X_dev), bs):
        outs.append(model(X_dev[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def cmd_d1(name: str, cfg: CfgD1, seeds: list):
    h = Harness()
    h.prepare_fast_eval()
    senc = build_senc(h)
    preds, seed_fast = [], []
    for s in seeds:
        t0 = time.time()
        model, mean, std = train_d1(h, cfg, s, senc)
        X_all = tensors_d1(h, senc, cfg.unk_unseen)[0]
        pred = predict_d1(model, X_all, mean, std)
        preds.append(pred)
        seed_fast.append(float(h.score_fast(pred)))
        print(f"[d1 {name}] seed={s} fast_comp={seed_fast[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert not np.isnan(P).any() and not np.isinf(P).any()
    print(f"[d1 {name}] {len(seeds)}种子均值预测，完整评分：")
    res = h.score_val(P)
    np.save(CACHE / f"d1_{name}_mean.npy", P)
    log_result({"dir": "d1", "name": name, "cfg": asdict(cfg), "seeds": seeds,
                "seed_fast": seed_fast, "composite": res["composite"],
                "per_split": res["per_split"]})
    return res


# ---------------------------------------------------------------- 方向 2：低秩响应程序

@dataclass
class CfgD2:
    rank: int = 32
    hidden: list = field(default_factory=lambda: [256, 512])
    epochs: int = 200
    lr: float = 1e-3
    wd: float = 1e-4
    emb_drop: float = 0.35
    p_drop: float = 0.2
    bs: int = 256
    use_strain: bool = True
    w_anchor: float = 0.1
    ridge_rel: float = 1e-2

    @classmethod
    def from_dict(cls, d: dict) -> "CfgD2":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


def d2_delta_train(h: Harness):
    """train split 处理样本 Δ；对照池严格限 train split 行（排除 val/test 对照）。"""
    f_d, f_r = CACHE / "delta_train.npy", CACHE / "delta_rows.npy"
    if f_d.exists() and f_r.exists():
        return np.load(f_d), np.load(f_r)
    tr_set = set(h.tr_rows.tolist())
    idx_pool = {sid: i for i, sid in enumerate(h.m_tr["sample_ID"])
                if i in tr_set}
    delta, _ = M.compute_delta(h.Y_tr, h.m_tr, h.ctrl_map, h.Y_tr, idx_pool)
    rows = h.tr_rows[h.is_treat_tr[h.tr_rows]]
    dt = delta[rows].astype(np.float32)
    keep = ~np.isnan(dt).all(axis=1)  # 对照全在 val 的行无法算 Δ，剔除
    rows, dt = rows[keep], dt[keep]
    np.save(f_d, dt)
    np.save(f_r, rows)
    print(f"[d2prep] train treated rows={len(rows)} "
          f"(dropped {int((~keep).sum())} w/o train-pool controls)")
    return dt, rows


def d2_svd(dt: np.ndarray, rank: int = 64) -> np.ndarray:
    f = CACHE / f"svd_P{rank}.npy"
    if f.exists():
        return np.load(f)
    from sklearn.decomposition import TruncatedSVD
    X = np.nan_to_num(dt, nan=0.0)  # (5078,5243) f32，CPU
    svd = TruncatedSVD(n_components=rank, random_state=0)
    t0 = time.time()
    svd.fit(X)
    P = svd.components_.astype(np.float32)  # (rank, 5243) 行正交
    np.save(f, P)
    evr = svd.explained_variance_ratio_
    print(f"[d2prep] SVD rank={rank} done ({time.time()-t0:.0f}s) "
          f"EVR top8={np.round(evr[:8], 4).tolist()} "
          f"cum={float(evr.sum()):.4f}")
    del X, svd
    return P


def d2_control_hat(h: Harness):
    f = CACHE / "control_hat.npy"
    if f.exists():
        return np.load(f)
    from .wsE_depcal import build_control_hat
    ctrl, _level = build_control_hat(h)
    np.save(f, ctrl)
    return ctrl


@torch.no_grad()
def project_weights(dt: np.ndarray, P: np.ndarray, ridge_rel: float = 1e-2,
                    device: str = "cuda", chunk: int = 256) -> np.ndarray:
    """w* = argmin ||m⊙(d − w·P)||² + λ||w||²，P 行正交；分块 GPU 解 r×r 正规方程。"""
    N, r = dt.shape[0], P.shape[0]
    W = np.zeros((N, r), np.float32)
    Pt = torch.tensor(P, device=device)
    eye = torch.eye(r, device=device)
    for s in range(0, N, chunk):
        d = torch.tensor(np.nan_to_num(dt[s:s + chunk], nan=0.0),
                         device=device)
        m = torch.tensor(~np.isnan(dt[s:s + chunk]), device=device,
                         dtype=torch.float32)
        Pm = Pt.unsqueeze(0) * m.unsqueeze(1)       # (B,r,p)
        G = Pm @ Pt.T                                # (B,r,r)
        b = torch.einsum("brp,bp->br", Pm, d)
        lam = ridge_rel * G.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)
        w = torch.linalg.solve(G + lam * eye, b.unsqueeze(-1)).squeeze(-1)
        W[s:s + chunk] = w.cpu().numpy()
    return W


class WeightMLP(nn.Module):
    """条件嵌入 → 标准化响应权重 ẑ（Δ̂ = (ẑ·σ_w)·P）。"""

    def __init__(self, n_cats, rank: int, cfg: CfgD2):
        super().__init__()
        dims = dict(EMB_DIMS)
        self.use_cols = [c for c in CAT_COLS
                         if cfg.use_strain or c != "Strains"]
        self.embs = nn.ModuleDict({
            c: nn.Embedding(n_cats[COL[c]], dims[c]) for c in self.use_cols})
        d_in = sum(dims[c] for c in self.use_cols)
        blocks, d = [], d_in
        for hdim in cfg.hidden:
            blocks += [nn.Linear(d, hdim), nn.GELU(), nn.LayerNorm(hdim),
                       nn.Dropout(cfg.p_drop)]
            d = hdim
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(d, rank)

    def forward(self, x_cat):
        e = torch.cat([self.embs[c](x_cat[:, COL[c]]) for c in self.use_cols],
                      dim=1)
        return self.head(self.trunk(e))


def build_d2_pack(h: Harness, senc: SeenEncoder, rank: int,
                  ridge_rel: float = 1e-2):
    """d2 训练数据包：treated 行 Δ/掩膜/P/z*/σ_w + 全行类别索引。"""
    dt, rows = d2_delta_train(h)
    P = d2_svd(dt, 64)[:rank].copy()  # TruncatedSVD 分量按 σ 排序且正交
    f_w = CACHE / f"w_star_r{rank}.npy"
    if f_w.exists():
        W = np.load(f_w)
    else:
        W = project_weights(dt, P, ridge_rel)
        np.save(f_w, W)
    w_std = W.std(axis=0) + 1e-6
    Z = (W / w_std).astype(np.float32)
    X_all = torch.tensor(senc.transform(h.m_tr, unk_unseen=True),
                         dtype=torch.long)
    pack = {
        "X_all": X_all,
        "rows": rows,
        "X_t": X_all[rows],
        "Dt": torch.tensor(np.nan_to_num(dt, nan=0.0), dtype=torch.float32),
        "Mt": torch.tensor(~np.isnan(dt)),
        "P": torch.tensor(P, dtype=torch.float32),
        "Z": torch.tensor(Z, dtype=torch.float32),
        "w_std": torch.tensor(w_std, dtype=torch.float32),
    }
    return pack


def train_d2(h: Harness, cfg: CfgD2, seed: int, senc: SeenEncoder,
             pack: dict, device: str = "cuda", log_every: int = 100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Dt = pack["Dt"].to(device)          # (5078,5243) f32
    Mt = pack["Mt"].to(device)          # bool
    Pt = pack["P"].to(device)           # (r,5243)
    Zt = pack["Z"].to(device)
    ws = pack["w_std"].to(device)
    Xt = pack["X_t"].to(device)
    N = len(pack["rows"])
    model = WeightMLP(senc.n_cats, cfg.rank, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(N / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    drop_cols = [STRAIN, CHEM] if cfg.use_strain else [CHEM]
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        tot, nb = 0.0, 0
        for i in range(0, N, cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = Xt[r].clone()
            if cfg.emb_drop > 0:
                for col in drop_cols:
                    dm = torch.rand(len(r), device=device) < cfg.emb_drop
                    xb[dm, col] = 0
            z = model(xb)
            dhat = (z * ws) @ Pt
            loss = masked_mse(dhat, Dt[r], Mt[r].float())
            if cfg.w_anchor > 0:
                loss = loss + cfg.w_anchor * F.mse_loss(z, Zt[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if log_every and ((ep + 1) % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  [d2 seed{seed}] epoch {ep+1:>3}/{cfg.epochs} "
                  f"loss={tot/nb:.5f}", flush=True)
    return model


@torch.no_grad()
def predict_w(model, X_all, device="cuda", bs: int = 512) -> np.ndarray:
    model.eval()
    X_dev = X_all.to(device)
    outs = []
    for i in range(0, len(X_dev), bs):
        outs.append(model(X_dev[i:i + bs]).float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)  # (8958, r) ẑ


def assemble_d2(h: Harness, control_hat: np.ndarray, z_all: np.ndarray,
                P: np.ndarray, w_std: np.ndarray) -> np.ndarray:
    """ŷ = control_hat + (ẑ·σ_w)·P（仅处理样本加 Δ̂）。"""
    pred = control_hat.copy()
    dhat = (z_all * w_std.astype(np.float32)) @ P.astype(np.float32)
    tr = h.is_treat_tr
    pred[tr] = (pred[tr] + dhat[tr]).astype(np.float32)
    return pred.astype(np.float32)


def cmd_d2(name: str, cfg: CfgD2, seeds: list):
    h = Harness()
    h.prepare_fast_eval()
    senc = build_senc(h)
    pack = build_d2_pack(h, senc, cfg.rank, cfg.ridge_rel)
    ctrl = d2_control_hat(h)
    P = pack["P"].numpy()
    w_std = pack["w_std"].numpy()
    zs, seed_fast = [], []
    for s in seeds:
        t0 = time.time()
        model = train_d2(h, cfg, s, senc, pack)
        z = predict_w(model, pack["X_all"])
        np.save(CACHE / f"d2_{name}_s{s}_z.npy", z)
        zs.append(z)
        pred = assemble_d2(h, ctrl, z, P, w_std)
        seed_fast.append(float(h.score_fast(pred)))
        print(f"[d2 {name}] seed={s} fast_comp={seed_fast[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    z_mean = np.mean(zs, axis=0).astype(np.float32)
    np.save(CACHE / f"d2_{name}_zmean.npy", z_mean)
    pred = assemble_d2(h, ctrl, z_mean, P, w_std)
    assert not np.isnan(pred).any() and not np.isinf(pred).any()
    print(f"[d2 {name}] {len(seeds)}种子均值预测，完整评分：")
    res = h.score_val(pred)
    log_result({"dir": "d2", "name": name, "cfg": asdict(cfg), "seeds": seeds,
                "seed_fast": seed_fast, "composite": res["composite"],
                "per_split": res["per_split"]})
    return res


def cmd_d2route(name_s: str, name_ns: str):
    """未见菌株行路由到无 strain 版（或与含 strain 版混合），对比评分。"""
    h = Harness()
    senc = build_senc(h)
    cfg = CfgD2()  # rank 从文件名约定取：L<rank>S/N
    rank = int(name_s.split("L")[1].split("S")[0].split("N")[0])
    pack = build_d2_pack(h, senc, rank)
    ctrl = d2_control_hat(h)
    P = pack["P"].numpy()
    w_std = pack["w_std"].numpy()
    z_s = np.load(CACHE / f"d2_{name_s}_zmean.npy")
    z_n = np.load(CACHE / f"d2_{name_ns}_zmean.npy")
    train_strains = set(h.m_train["Strains"].unique())
    unseen = ~h.m_tr["Strains"].isin(train_strains).to_numpy()
    print(f"[d2route] 未见菌株行 {int(unseen.sum())} / {len(unseen)}")
    variants = {
        "S_only": z_s,
        "N_only": z_n,
        "route": np.where(unseen[:, None], z_n, z_s),
        "mix05_unseen": np.where(unseen[:, None], 0.5 * (z_s + z_n), z_s),
    }
    for vn, z in variants.items():
        pred = assemble_d2(h, ctrl, z.astype(np.float32), P, w_std)
        assert not np.isnan(pred).any()
        print(f"[d2route {vn}] 完整评分：")
        res = h.score_val(pred)
        np.save(CACHE / f"d2_route_{vn}_r{rank}_pred.npy", pred)
        log_result({"dir": "d2route", "name": f"{vn}_r{rank}",
                    "s": name_s, "ns": name_ns,
                    "composite": res["composite"],
                    "per_split": res["per_split"]})


def cmd_blend(pa: str, pb: str):
    """两方向预测混合（仅分析用）：α 网格 score_fast → 最优 α 完整评分。"""
    h = Harness()
    h.prepare_fast_eval()
    A = np.load(pa)
    B = np.load(pb)
    best = (None, -1)
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        c = float(h.score_fast(a * A + (1 - a) * B))
        print(f"[blend] α={a:.1f} fast_comp={c:.4f}", flush=True)
        if c > best[1]:
            best = (a, c)
    a = best[0]
    pred = (a * A + (1 - a) * B).astype(np.float32)
    assert not np.isnan(pred).any()
    print(f"[blend] α*={a} 完整评分：")
    res = h.score_val(pred)
    np.save(CACHE / f"blend_a{int(a*100)}_pred.npy", pred)
    log_result({"dir": "blend", "a": a, "pa": pa, "pb": pb,
                "composite": res["composite"], "per_split": res["per_split"]})
    return res


def cmd_d1route(pa: str, pb: str):
    """按"菌株是否见于 train"路由两个 d1 预测（未见 → B），并测试混合。"""
    h = Harness()
    A = np.load(pa)
    B = np.load(pb)
    train_strains = set(h.m_train["Strains"].unique())
    unseen = ~h.m_tr["Strains"].isin(train_strains).to_numpy()
    print(f"[d1route] 未见菌株行 {int(unseen.sum())} / {len(unseen)} "
          f"(A={pa}, B={pb})")
    variants = {
        "route": np.where(unseen[:, None], B, A),
        "routeMix05": np.where(unseen[:, None], 0.5 * (A + B), A),
    }
    for vn, pred in variants.items():
        pred = pred.astype(np.float32)
        assert not np.isnan(pred).any()
        print(f"[d1route {vn}] 完整评分：")
        res = h.score_val(pred)
        np.save(CACHE / f"d1route_{vn}_pred.npy", pred)
        log_result({"dir": "d1route", "name": vn, "pa": pa, "pb": pb,
                    "composite": res["composite"],
                    "per_split": res["per_split"]})


def cmd_deliver(src: str):
    src_p = Path(src)
    pred = np.load(src_p)
    assert pred.shape == (8958, 5243) and pred.dtype == np.float32
    assert not np.isnan(pred).any() and not np.isinf(pred).any()
    np.save(OUT / "pred_trainval.npy", pred)
    h = Harness()
    print(f"[deliver] {src_p.name} → outputs/wsF/pred_trainval.npy，完整评分：")
    res = h.score_val(pred)
    (OUT / "final_score.json").write_text(json.dumps(
        {"src": str(src_p), "composite": res["composite"],
         "per_split": res["per_split"]}, indent=1, default=str))
    print("[deliver] composite =", res["composite"])


def cmd_d1test(name: str, cfg: CfgD1, seeds: list):
    """全量 train_val 重训（统计全量重估）→ 多种子均值 → outputs/wsF/pred_test.npy。

    严禁事项：不使用 h.Y_te 的任何数值；FrozenStats 用全部 train_val 行重估；
    seen 集合取全部 train_val（此时 train_val 内无"未见"实体；test 专属实体
    仍经 unk_unseen/UNK(0) 回退）。aux_delta 必须为 0（辅助目标仅 train split）。
    """
    assert cfg.aux_delta == 0.0, "全量重训不支持 aux_delta（目标只含 train split）"
    h = Harness()
    rows = np.arange(len(h.m_tr))
    stats_full = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)
    enc = Encoder().fit(h.m_tr)
    senc_full = SeenEncoder(enc, h.m_tr)  # seen = 全部 train_val 实体
    X_all = torch.tensor(senc_full.transform(h.m_tr, unk_unseen=True),
                         dtype=torch.long)
    mean = torch.tensor(stats_full.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats_full.protein_std, dtype=torch.float32)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    Mk = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    tensors = (X_all, mean, std, Z, Mk)

    preds = []
    for s in seeds:
        t0 = time.time()
        model, mean_t, std_t = train_d1(h, cfg, s, senc_full, rows=rows,
                                        tensors=tensors)
        X_te = torch.tensor(senc_full.transform(h.m_te, unk_unseen=True),
                            dtype=torch.long)
        pred = predict_d1(model, X_te, mean_t, std_t)
        preds.append(pred.astype(np.float32))
        print(f"[d1test] seed={s} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert P.shape == (len(h.m_te), h.Y_tr.shape[1]), P.shape
    assert not np.isnan(P).any() and not np.isinf(P).any(), "pred_test 含 NaN/Inf"
    np.save(OUT / "pred_test.npy", P)
    (OUT / "test_results.json").write_text(json.dumps(
        {"name": name, "cfg": asdict(cfg), "seeds": seeds,
         "shape": list(P.shape), "train_rows": int(len(rows))}, indent=1))
    print(f"[d1test] saved {OUT/'pred_test.npy'} shape={P.shape} dtype={P.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["d1", "d2prep", "d2", "d2route",
                                      "blend", "d1route", "deliver",
                                      "d1test"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--s", default=None)
    ap.add_argument("--ns", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--src", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    if args.stage == "d1":
        cmd_d1(args.name, CfgD1.from_dict(json.loads(args.cfg)), seeds)
    elif args.stage == "d2prep":
        h = Harness()
        dt, rows = d2_delta_train(h)
        d2_svd(dt, 64)
        d2_control_hat(h)
        print("[d2prep] done")
    elif args.stage == "d2":
        cmd_d2(args.name, CfgD2.from_dict(json.loads(args.cfg)), seeds)
    elif args.stage == "d2route":
        cmd_d2route(args.s, args.ns)
    elif args.stage == "blend":
        cmd_blend(args.a, args.b)
    elif args.stage == "d1route":
        cmd_d1route(args.a, args.b)
    elif args.stage == "deliver":
        cmd_deliver(args.src)
    elif args.stage == "d1test":
        cmd_d1test(args.name, CfgD1.from_dict(json.loads(args.cfg)), seeds)


if __name__ == "__main__":
    main()
