"""wsC：连续时间建模 + QC/批次校正的 MLP 变体（基于 src/train_mlp.py 框架）。

子方向 1（连续时间）：
- pert_time 的 6 档 one-hot 嵌入可替换/补充为连续时间基函数
  x = log2(t/15)，xn = x/2 - 1 ∈ [-1,1]；basis = [xn, RBF(5 中心)] 或 poly3
- 化合物/菌株嵌入 × 时间基的双线性外积作为 trunk 输入，
  让每种化合物学自己的平滑时间曲线，且对未见 (化合物,时间) 组合可分解外推

子方向 2（QC/批次）：
- 仅用 train split 的 QC 样本估计板效应 plate_effect = mean(QC_板) − mean(QC_全部train)
  （按蛋白缺失数收缩；缺失板回退到同 instrument 板效应均值，再回退 0）
- 用法 A（target）：训练目标先减去板效应，预测时加回 α·effect
- 用法 B（feature）：板效应 PCA 得分 + 统计量作为样本附加输入特征
- 用法 C（post）：训练后 ŷ + α·effect（已知与 plate embedding 冗余，仅作消融）

合规：一切 QC/归一化统计只来自 train split；val 仅用于 score_val 评分。

用法:
    python -m src.wsC_timebatch run --tag A0 --time_mode onehot --epochs 60
    python -m src.wsC_timebatch run --tag A3 --time_mode both --interact chemstrain \
        --qc feature --epochs 60 --seeds 0,1,2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness
from .train_mlp import Encoder, masked_mse, CAT_COLS, EMB_DIMS

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsC"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------- 连续时间基 ----------------

def time_basis(t_minutes: np.ndarray, kind: str = "rbf") -> np.ndarray:
    """t (n,) 分钟 → (n, k) 连续基。x=log2(t/15) ∈ [0,4]，xn ∈ [-1,1]。"""
    x = np.log2(np.asarray(t_minutes, dtype=np.float64) / 15.0)
    xn = x / 2.0 - 1.0
    if kind == "poly":
        feats = [xn, xn ** 2, xn ** 3]
    else:  # rbf
        centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
        bw = 0.55
        feats = [xn] + [np.exp(-0.5 * ((xn - c) / bw) ** 2) for c in centers]
    return np.stack(feats, axis=-1).astype(np.float32)


# ---------------- QC 板效应 ----------------

class QCCorrector:
    """用 QC 样本估计板效应向量（缺失板回退 instrument 均值 → 0）。

    rows=None：仅 train split 的 QC（本地验证合规）；
    rows=arange：全量 train_val 的 QC（仅用于最终 test 提交重训）。
    """

    def __init__(self, h: Harness, shrink_k: float = 2.0, inst_shrink: float = 0.5,
                 rows: np.ndarray | None = None):
        if rows is None:
            m, Y = h.m_train, h.Y_train
        else:
            m = h.m_tr.iloc[rows].reset_index(drop=True)
            Y = h.Y_tr[rows]
        qc = (m["perturbation_no_concentration"] == D.QC).to_numpy()
        Yq, mq = Y[qc], m[qc]
        self.mu_qc = np.nan_to_num(np.nanmean(Yq, axis=0))
        plates = mq["Yeast_cell_plate"].to_numpy()
        self.eff, cnt = {}, {}
        for p in np.unique(plates):
            sub = Yq[plates == p]
            n = np.sum(~np.isnan(sub), axis=0)
            e = np.nan_to_num(np.nanmean(sub, axis=0) - self.mu_qc)
            self.eff[p] = e * (n / (n + shrink_k))  # 按有效观测数收缩
            cnt[p] = len(sub)
        inst_map = mq.groupby("Yeast_cell_plate")["instrument"].first()
        self.inst_eff = {}
        for inst, pls in inst_map.groupby(inst_map).groups.items():
            es = [self.eff[p] * cnt[p] for p in pls]
            ns = sum(cnt[p] for p in pls)
            self.inst_eff[inst] = np.sum(es, axis=0) / ns * inst_shrink
        self.zero = np.zeros_like(self.mu_qc)
        self.n_qc_plates = len(self.eff)

    def vector(self, plate, inst):
        if plate in self.eff:
            return self.eff[plate]
        return self.inst_eff.get(inst, self.zero)

    def matrix_for(self, df: pd.DataFrame) -> np.ndarray:
        return np.stack([self.vector(p, i) for p, i in
                         zip(df["Yeast_cell_plate"], df["instrument"])]
                        ).astype(np.float32)

    def features_for(self, df: pd.DataFrame, n_pc: int = 8) -> np.ndarray:
        """板效应 PCA 得分 + 标量统计作为输入特征（全部来自 train QC 冻结统计）。"""
        if not hasattr(self, "_pca"):
            plates = sorted(self.eff)
            E = np.stack([self.eff[p] for p in plates])          # (n_qc_plate, P)
            mu = E.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(E - mu, full_matrices=False)
            k = min(n_pc, U.shape[1])
            self._pca = (mu.squeeze(0), Vt[:k], plates)          # 均值、载荷、板列表
            self._scale = 1.0 / (S[:k].mean() + 1e-8)
        mu, Vt, plates = self._pca
        out = np.zeros((len(df), Vt.shape[0] + 2), dtype=np.float32)
        has_qc_set = set(plates)
        for i, (p, inst) in enumerate(zip(df["Yeast_cell_plate"], df["instrument"])):
            v = self.vector(p, inst)
            out[i, :Vt.shape[0]] = ((v - mu) @ Vt.T) * self._scale
            out[i, -2] = float(p in has_qc_set)
            out[i, -1] = float(v.mean()) * 10.0
        return out


# ---------------- 模型 ----------------

class WsCMLP(nn.Module):
    def __init__(self, n_cats, n_prot, k_time, q_dim, time_onehot=True,
                 interact="none", hidden=(512, 1024), p_drop=0.1,
                 emb_scale=None):
        super().__init__()
        dims = dict(EMB_DIMS)
        if emb_scale:
            dims.update(emb_scale)
        self.k_time = k_time
        self.interact = interact
        self.col = {c: i for i, c in enumerate(CAT_COLS)}
        use_cols = [c for c in CAT_COLS if c != "pert_time" or time_onehot]
        self.embs = nn.ModuleDict({
            c: nn.Embedding(n, dims[c])
            for c, n in zip(CAT_COLS, n_cats) if c in use_cols})
        d_emb = sum(dims[c] for c in use_cols)
        d_in = d_emb + k_time + q_dim
        if interact in ("chem", "chemstrain"):
            d_in += dims["perturbation_no_concentration"] * k_time
        if interact == "chemstrain":
            d_in += dims["Strains"] * k_time
        layers, d = [], d_in
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.GELU(), nn.LayerNorm(hdim),
                       nn.Dropout(p_drop)]
            d = hdim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat, x_cont):
        es = [self.embs[c](x_cat[:, i]) for c, i in self.col.items()
              if c in self.embs]
        basis = x_cont[:, :self.k_time]
        feats = es + [x_cont]
        if self.interact in ("chem", "chemstrain"):
            chem = self.embs["perturbation_no_concentration"](
                x_cat[:, self.col["perturbation_no_concentration"]])
            feats.append((chem.unsqueeze(2) * basis.unsqueeze(1)).flatten(1))
        if self.interact == "chemstrain":
            st = self.embs["Strains"](x_cat[:, self.col["Strains"]])
            feats.append((st.unsqueeze(2) * basis.unsqueeze(1)).flatten(1))
        return self.head(self.trunk(torch.cat(feats, dim=1)))


# ---------------- 训练 / 预测 ----------------

def build_cont(h: Harness, corrector: QCCorrector | None, basis_kind: str,
               qc_mode: str, df: pd.DataFrame | None = None) -> np.ndarray:
    """给定行的连续特征 [时间基(k) + QC 特征(q)]。df=None 时用全部 train_val 行。"""
    df = h.m_tr if df is None else df
    tb = time_basis(df["pert_time"].to_numpy(), basis_kind)
    if corrector is not None and "feature" in qc_mode:
        qf = corrector.features_for(df)
        return np.concatenate([tb, qf], axis=1)
    return tb


def train_model(h, rows, epochs, seed=0, emb_drop=0.15, lr=1e-3, bs=256,
                device="cuda", log_every=50, time_onehot=True, interact="none",
                qc_mode="none", corrector=None, basis_kind="rbf",
                hidden=(512, 1024), emb_scale=None, p_drop=0.1,
                tail_states=None, stats=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = Encoder().fit(h.m_tr)
    n_prot = h.Y_tr.shape[1]

    X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
    C_all = torch.tensor(build_cont(h, corrector, basis_kind, qc_mode),
                         dtype=torch.float32)
    k_time = time_basis(h.m_tr["pert_time"].to_numpy()[:4], basis_kind).shape[1]
    q_dim = C_all.shape[1] - k_time

    stats = stats or h.stats
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(stats.protein_std, dtype=torch.float32)
    Y = h.Y_tr.astype(np.float32)
    if "target" in qc_mode and corrector is not None:
        Y = Y - corrector.matrix_for(h.m_tr)     # 目标去板效应
    Z_all = (torch.tensor(Y, dtype=torch.float32) - mean) / std
    M_all = ~torch.isnan(Z_all)
    Z_all = torch.nan_to_num(Z_all, nan=0.0)

    model = WsCMLP(enc.n_cats, n_prot, k_time, q_dim, time_onehot,
                   interact, hidden=hidden, p_drop=p_drop,
                   emb_scale=emb_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * int(np.ceil(len(rows) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev, C_dev = X_all.to(device), C_all.to(device)
    Z_dev, M_dev = Z_all.to(device), M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)

    for ep in range(epochs):
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            xb = X_dev[r].clone()
            if emb_drop > 0:
                for col in [0, 1]:
                    drop = torch.rand(len(r), device=device) < emb_drop
                    xb[drop, col] = 0
            pred = model(xb, C_dev[r])
            loss = masked_mse(pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={tot/nb:.4f}")
        if tail_states is not None and epochs >= 20 and \
                ep + 1 >= int(epochs * 0.7) and (ep + 1) % 10 == 0:
            tail_states.append({k: v.detach().clone()
                                for k, v in model.state_dict().items()})
    return model, enc, mean, std, corrector


def g3_remap(X: torch.Tensor, h: Harness, df: pd.DataFrame) -> torch.Tensor:
    """wsG 修复（纯推理）：把 train split 中未出现的实体索引强制改回 UNK(0)。

    Encoder 在 train_val 全集 fit，val-only 实体（BAI、6 个 val 化合物）分到的
    是未训练的随机初始化嵌入行；训练期 emb_drop 已让 UNK(0) 学到回退表示，
    因此 OOD 推理应走 UNK。仅影响本地 val 预测；test 的未见证实体本就映射 0。
    """
    X = X.clone()
    for j, c in enumerate(CAT_COLS):
        seen = set(h.m_train[c].unique())
        mask = torch.tensor(~df[c].isin(seen).to_numpy(), device=X.device)
        X[mask, j] = 0
    return X


def predict(model, enc, mean, std, h, corrector, qc_mode, alpha,
            basis_kind="rbf", device="cuda", bs=1024, df=None, g3=False):
    model.eval()
    df = h.m_tr if df is None else df
    X = torch.tensor(enc.transform(df), dtype=torch.long, device=device)
    if g3:
        X = g3_remap(X, h, df)
    C = torch.tensor(build_cont(h, corrector, basis_kind, qc_mode, df=df),
                     dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(X[i:i + bs], C[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    pred = (Z * std.numpy() + mean.numpy()).astype(np.float32)
    if "target" in qc_mode and corrector is not None and alpha != 0.0:
        pred = pred + alpha * corrector.matrix_for(df)
    return pred


# ---------------- 实验入口 ----------------

def run(args):
    h = Harness()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.full:   # 全量 train_val 重训 → 预测 test（最终提交；仍禁 Y_te）
        return run_full(h, args, seeds)
    corrector = QCCorrector(h) if args.qc != "none" else None
    emb_scale = None
    if args.chem_emb: emb_scale = emb_scale or {}; emb_scale["perturbation_no_concentration"] = args.chem_emb
    if args.plate_emb: emb_scale = emb_scale or {}; emb_scale["Yeast_cell_plate"] = args.plate_emb
    if corrector is not None:
        print(f"[qc] plates with train QC: {corrector.n_qc_plates}")
    preds, preds_g3 = [], []
    for sd in seeds:
        t0 = time.time()
        tail = [] if args.tail_avg else None
        model, enc, mean, std, _ = train_model(
            h, h.tr_rows, args.epochs, seed=sd, emb_drop=args.emb_drop,
            lr=args.lr, time_onehot=(args.time_mode != "cont"),
            interact=args.interact, qc_mode=args.qc, corrector=corrector,
            basis_kind=args.basis, hidden=tuple(args.hidden),
            emb_scale=emb_scale, p_drop=args.p_drop, tail_states=tail)
        print(f"[train seed={sd}] {time.time()-t0:.0f}s")
        states = [model.state_dict()] + (tail if args.tail_avg else [])
        sp, sp3 = [], []
        for st in states:
            model.load_state_dict(st)
            sp.append(predict(model, enc, mean, std, h, corrector, args.qc,
                              args.alpha, basis_kind=args.basis))
            if args.g3_infer:
                sp3.append(predict(model, enc, mean, std, h, corrector,
                                   args.qc, args.alpha, basis_kind=args.basis,
                                   g3=True))
        preds.append(np.mean(sp, axis=0))
        if args.g3_infer:
            preds_g3.append(np.mean(sp3, axis=0))
    pred = np.mean(preds, axis=0).astype(np.float32)
    if args.g3_infer:
        pred_g3 = np.mean(preds_g3, axis=0).astype(np.float32)
        old_path = OUT / f"pred_{args.tag}.npy"
        if old_path.exists():
            old = np.load(old_path)
            cc = np.corrcoef(old.ravel()[::997], pred.ravel()[::997])[0, 1]
            print(f"[repro] corr(retrained, saved {old_path.name}) = {cc:.5f}")
        np.save(OUT / f"pred_{args.tag}_g3.npy", pred_g3)
        print(f"[saved] outputs/wsC/pred_{args.tag}_g3.npy")
    if args.qc == "post" and args.alpha != 0.0:
        pred = pred + args.alpha * corrector.matrix_for(h.m_tr)
    # 交付安全：NaN/Inf 用 train 蛋白均值填
    bad = ~np.isfinite(pred)
    if bad.any():
        r, c = np.where(bad)
        pred[r, c] = np.take(h.stats.protein_mean, c)
    np.save(OUT / f"pred_{args.tag}.npy", pred)
    res = h.score_val(pred)
    rec = {"tag": args.tag, "time_mode": args.time_mode, "basis": args.basis,
           "interact": args.interact, "qc": args.qc, "alpha": args.alpha,
           "epochs": args.epochs, "emb_drop": args.emb_drop,
           "hidden": args.hidden, "chem_emb": args.chem_emb,
           "plate_emb": args.plate_emb, "p_drop": args.p_drop,
           "tail_avg": args.tail_avg,
           "seeds": seeds, "composite": res["composite"],
           "per_split": res["per_split"]}
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[saved] outputs/wsC/pred_{args.tag}.npy  composite={res['composite']:.4f}")


def run_full(h, args, seeds):
    rows_all = np.arange(len(h.m_tr))
    stats_full = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows_all)
    corrector = (QCCorrector(h, rows=rows_all) if args.qc != "none" else None)
    emb_scale = None
    if args.chem_emb: emb_scale = {"perturbation_no_concentration": args.chem_emb}
    if args.plate_emb: emb_scale = dict(emb_scale or {}, Yeast_cell_plate=args.plate_emb)
    preds = []
    for sd in seeds:
        t0 = time.time()
        tail = [] if args.tail_avg else None
        model, enc, mean, std, _ = train_model(
            h, rows_all, args.epochs, seed=sd, emb_drop=args.emb_drop,
            lr=args.lr, time_onehot=(args.time_mode != "cont"),
            interact=args.interact, qc_mode=args.qc, corrector=corrector,
            basis_kind=args.basis, hidden=tuple(args.hidden),
            emb_scale=emb_scale, p_drop=args.p_drop, tail_states=tail,
            stats=stats_full, log_every=999)
        states = [model.state_dict()] + (tail if args.tail_avg else [])
        sp = []
        for st in states:
            model.load_state_dict(st)
            sp.append(predict(model, enc, mean, std, h, corrector, args.qc,
                              args.alpha, basis_kind=args.basis, df=h.m_te))
        preds.append(np.mean(sp, axis=0))
        print(f"[full seed={sd}] {time.time()-t0:.0f}s")
    pred = np.mean(preds, axis=0).astype(np.float32)
    if args.qc == "post" and args.alpha != 0.0:
        pred = pred + args.alpha * corrector.matrix_for(h.m_te)
    bad = ~np.isfinite(pred)
    if bad.any():
        r, c = np.where(bad)
        pred[r, c] = np.take(stats_full.protein_mean, c)
    assert pred.shape == (len(h.m_te), h.Y_tr.shape[1])
    np.save(OUT / f"pred_test_{args.tag}.npy", pred)
    print(f"[saved] outputs/wsC/pred_test_{args.tag}.npy shape={pred.shape}")
    rec = {"tag": args.tag, "full": True, "time_mode": args.time_mode,
           "basis": args.basis, "interact": args.interact, "qc": args.qc,
           "alpha": args.alpha, "epochs": args.epochs,
           "emb_drop": args.emb_drop, "seeds": seeds,
           "tail_avg": args.tail_avg}
    with open(OUT / "results_full.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--time_mode", default="onehot",
                   choices=["onehot", "cont", "both"])
    r.add_argument("--basis", default="rbf", choices=["rbf", "poly"])
    r.add_argument("--interact", default="none",
                   choices=["none", "chem", "chemstrain"])
    r.add_argument("--qc", default="none",
                   choices=["none", "target", "feature", "target_feature",
                            "post"])
    r.add_argument("--alpha", type=float, default=1.0)
    r.add_argument("--epochs", type=int, default=60)
    r.add_argument("--emb_drop", type=float, default=0.15)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--seeds", default="0")
    r.add_argument("--hidden", type=int, nargs="+", default=[512, 1024])
    r.add_argument("--chem_emb", type=int, default=0)
    r.add_argument("--plate_emb", type=int, default=0)
    r.add_argument("--p_drop", type=float, default=0.1)
    r.add_argument("--tail_avg", action="store_true")
    r.add_argument("--full", action="store_true",
                   help="全量 train_val 重训并预测 test")
    r.add_argument("--g3_infer", action="store_true",
                   help="推理时将 train split 未见证实体重映射到 UNK(0)")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
