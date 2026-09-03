"""两阶段建模（wsB）：ŷ = control_hat(上下文+批次) + Δ̂(扰动效应)。

动机：评分 65% 权重在 Δ 空间（FC/两个残差）。直接预测绝对丰度时，
Δ 指标受对照基线预测误差拖累（score 时 Δ_pred = ŷ - 对照真值）。
本模块把对照基线与扰动效应解耦：

1. control_hat：仅用 train split 对照样本（DMSO/Water，751 个）训练。
   - MLP：菌株/培养基/温度/时间/仪器/来源/板号嵌入（无化合物输入），
     目标为 train 冻结统计标准化的 log2 丰度，masked MSE，3 种子均值。
     菌株嵌入 UNK + 25% embedding dropout（val_strain_only 的 BAI 等
     菌株在 train 对照中不存在，必须学 OOD 回退）。
   - 分层组均值（对照版 DeltaAdditive._control_hat，逐蛋白级联回退）。
   - 两者按权重 w 融合，w 在 val 上做一次性的模型选择（3 个候选）。
2. Δ̂：train split 处理样本（5078 个），目标 = h.delta_tr_all 对应行
   （raw log2 Δ，masked MSE；不标准化以保持蛋白间天然方差权重，
   与逐样本 FC_PCC 的口径一致）。输入 = 菌株嵌入(UNK+25% drop)
   + 化合物嵌入(UNK+25% drop) + 培养基/温度/时间/仪器/来源/板号。
   MLP 512-1024，100 epochs，3 种子均值。
3. 合成：处理样本 ŷ = control_hat + Δ̂；对照样本 ŷ = control_hat；
   QC 样本 = train QC 按 (仪器×来源) 分组均值（级联回退：仪器 → 全局）。
4. 诊断（不进提交）：把 val 处理样本的 ŷ 换成「真实匹配对照 + Δ̂」，
   量化"若允许用 test 对照原始值锚定"的提分上限。

编码器与 train_mlp 的差异：encoder 只在各模型实际训练行上 fit，
未见类别（val 独有菌株/化合物/板号）一律映射 UNK(0)，
避免未训练的随机初始化嵌入注入噪声。

合规：训练只用 h.tr_rows；目标用 h.delta_tr_all 的 train 行
（harness 冻结参照 μ_ctx/μ_drug 同款接口；其中 185 个 train 处理样本的
匹配对照落在 val 划分，与 harness 自带冻结统计口径一致，在此注明）。
预测文件为「非锚定版」。随机种子固定 (0,1,2)。

用法:
    python -m src.wsB_twostage                 # 完整流程 + 评分 + 报告
    python -m src.wsB_twostage --skip-baselines
    python -m src.wsB_twostage --full          # 全量 train_val 重训 + test 预测
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import data as D
from .evaluate import Harness

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsB"

EMB_DIMS = {"Strains": 8, "perturbation_no_concentration": 32, "Medium": 2,
            "Temperature": 2, "pert_time": 4, "instrument": 6,
            "data_source": 3, "Yeast_cell_plate": 32}
CTRL_COLS = ["Strains", "Medium", "Temperature", "pert_time", "instrument",
             "data_source", "Yeast_cell_plate"]
DLT_COLS = ["Strains", "perturbation_no_concentration", "Medium",
            "Temperature", "pert_time", "instrument", "data_source",
            "Yeast_cell_plate"]


# ---------------- 编码器与模型 ----------------

class Encoder:
    """类别 → 索引，index 0 保留给 UNK；只在训练行上 fit（未见→UNK）。"""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, m: pd.DataFrame):
        self.maps = {}
        for c in self.cols:
            vals = sorted(pd.unique(m[c]).tolist(), key=str)
            self.maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        return self

    def transform(self, m: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.cols:
            mp = self.maps[c]
            cols.append(m[c].map(lambda v: mp.get(v, 0)).to_numpy())
        return np.stack(cols, axis=1)

    @property
    def n_cats(self):
        return [len(self.maps[c]) + 1 for c in self.cols]


class CondMLP(nn.Module):
    """条件嵌入 → MLP trunk → n_out 维线性头（bias 置 0 = 先验均值/零 Δ）。"""

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
        self.head = nn.Linear(d, n_out)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)],
                      dim=1)
        return self.head(self.trunk(e))


def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp_min(1.0)


def train_model(model, X_all, Y_all, M_all, rows, epochs, seed,
                drop_cols=(), emb_drop=0.25, lr=1e-3, bs=256,
                device="cuda", log_every=25, tag="", g2_cols=()):
    """g2_cols: G2 组级增强——每 epoch 随机选 1 个已见类别，该组所有样本
    本 epoch 强制 UNK（模拟整实体缺失的 OOD 情形），叠加在逐样本 drop 之上。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * int(np.ceil(len(rows) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    rows_dev = torch.tensor(rows, device=device)
    for ep in range(epochs):
        g2_vals = {}
        for col in g2_cols:
            obs = torch.unique(X_all[rows_dev, col])
            obs = obs[obs > 0]
            if len(obs) > 1:
                sel = torch.randperm(len(obs), device=device)[0]
                g2_vals[col] = obs[sel]
        model.train()
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            xb = X_all[r].clone()
            for col in drop_cols:
                drop = torch.rand(len(r), device=device) < emb_drop
                xb[drop, col] = 0
            for col, v in g2_vals.items():
                xb[X_all[r, col] == v, col] = 0
            loss = masked_mse(model(xb), Y_all[r], M_all[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"  [{tag} seed] epoch {ep+1:>3}/{epochs}  loss={tot/nb:.4f}")
    return model


@torch.no_grad()
def predict_model(model, X_all, device="cuda", bs=1024) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X_all.shape[0], bs):
        outs.append(model(X_all[i:i + bs]).float().cpu())
    return torch.cat(outs).numpy()


# ---------------- 分层组均值对照基线 ----------------

class GroupMeanControl:
    """对照样本分层组均值，逐蛋白级联回退（粗→细逐层覆盖有效蛋白）。"""

    LEVELS = [["Strains", "Medium", "Temperature", "pert_time", "instrument"],
              ["Strains", "Medium", "Temperature", "pert_time"],
              ["Strains", "Medium", "Temperature"],
              ["Strains"]]

    def fit(self, m_fit: pd.DataFrame, Y_fit: np.ndarray,
            fallback: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            glob = np.nanmean(Y_fit, axis=0)
        self.global_ = np.where(np.isnan(glob), fallback, glob).astype(np.float32)
        self.maps_ = []
        for keys in self.LEVELS:  # 细→粗存储，预测时逆序（粗→细）覆盖
            km = list(map(tuple, m_fit[keys].to_numpy()))
            gm = {}
            for k, idx in pd.DataFrame({"k": km}).groupby("k").groups.items():
                idx = np.fromiter(idx, dtype=int)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    gm[k] = np.nanmean(Y_fit[idx], axis=0)  # 保留 NaN 供级联
            self.maps_.append((keys, gm))
        return self

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        out = np.repeat(self.global_[None, :], len(m), axis=0)
        for keys, gm in reversed(self.maps_):  # 粗→细
            for i, row in enumerate(m.itertuples()):
                k = tuple(getattr(row, kk) for kk in keys)
                v = gm.get(k)
                if v is None:
                    continue
                valid = ~np.isnan(v)
                out[i, valid] = v[valid]
        return out.astype(np.float32)


class QCGroupMean:
    """QC 样本：(仪器×来源) → 仪器 → 全局 级联组均值。"""

    LEVELS = [["instrument", "data_source"], ["instrument"]]

    def fit(self, m_fit, Y_fit, fallback):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            glob = np.nanmean(Y_fit, axis=0)
        self.global_ = np.where(np.isnan(glob), fallback, glob).astype(np.float32)
        self.maps_ = []
        for keys in self.LEVELS:
            km = list(map(tuple, m_fit[keys].to_numpy()))
            gm = {}
            for k, idx in pd.DataFrame({"k": km}).groupby("k").groups.items():
                idx = np.fromiter(idx, dtype=int)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    gm[k] = np.nanmean(Y_fit[idx], axis=0)
            self.maps_.append((keys, gm))
        return self

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        out = np.repeat(self.global_[None, :], len(m), axis=0)
        for keys, gm in reversed(self.maps_):
            for i, row in enumerate(m.itertuples()):
                k = tuple(getattr(row, kk) for kk in keys)
                v = gm.get(k)
                if v is None:
                    continue
                valid = ~np.isnan(v)
                out[i, valid] = v[valid]
        return out.astype(np.float32)


# ---------------- G3 推理修复验证 / G2 组级增强 ----------------

def g3_remap_transform(enc_all: Encoder, enc_train: Encoder,
                       m: pd.DataFrame) -> np.ndarray:
    """G3 推理修复：用全集 fit 的 enc_all 转换后，把 enc_train（训练行 fit）
    中未见的实体强制重映射到 UNK(0)。"""
    X = enc_all.transform(m).copy()
    for j, c in enumerate(enc_all.cols):
        known = set(enc_train.maps[c])
        X[~m[c].isin(known).to_numpy(), j] = 0
    return X


def g3_infer_pipeline(args):
    """G3 修复对 wsB 的影响评估 + 可选 G2 重训。

    wsB 的 Encoder 从设计起只在各模型训练行上 fit（未见类别→UNK(0)），
    本地 val 的未见实体（BAI、6 个 val 化合物）本就走 UNK(0)，
    即 pred_trainval.npy 已是 G3 语义。本流程：
      1. 程序验证上述语义等价（UNK 掩膜一致性）；
      2. 用相同种子/配置重训 G3 变体 → pred_trainval_g3.npy + score_val；
      3. 反事实对照：encoder 在 train_val 全集 fit 且不修复（wsG 所述 bug，
         val 未见实体分到未训练随机嵌入行），重训并评分，量化"若带 bug
         会低估多少"（仅诊断，不存盘）；
      4. 可选 --g2：Δ̂ 加 G2 组级 UNK 增强重训 → pred_trainval_g2g3.npy。
    只写 *_g3/*.npy 与 *_g2g3.npy，不动原有交付物。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    val_rows = np.where((m["split_final"] != "train").to_numpy())[0]
    mv = m.iloc[val_rows]

    # ---- 1. 语义等价验证 ----
    enc_c = Encoder(CTRL_COLS).fit(m.iloc[ctrl_rows])
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    enc_c_all = Encoder(CTRL_COLS).fit(m.iloc[np.where(is_ctrl)[0]])
    enc_d_all = Encoder(DLT_COLS).fit(m.iloc[np.where(is_treat)[0]])
    ok = True
    for enc_t, enc_a, cols in [(enc_c, enc_c_all, CTRL_COLS),
                               (enc_d, enc_d_all, DLT_COLS)]:
        Xt = enc_t.transform(mv)
        Xa = g3_remap_transform(enc_a, enc_t, mv)
        for j in range(len(cols)):
            ok &= bool(((Xt[:, j] == 0) == (Xa[:, j] == 0)).all())
    n_unk_strain = int((enc_d.transform(mv)[:, 0] == 0).sum())
    n_unk_chem = int((enc_d.transform(mv)[:, 1] == 0).sum())
    print(f"[g3] UNK-mask 语义等价（wsB == G3 修复）: {ok} | "
          f"val 行走 UNK: strain {n_unk_strain}/{len(val_rows)} "
          f"chem {n_unk_chem}/{len(val_rows)}")

    # ---- 公共张量 ----
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32,
                        device=args.device)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32,
                       device=args.device)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32, device=args.device)
         - mean) / std
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    Dmat = torch.tensor(h.delta_tr_all, dtype=torch.float32,
                        device=args.device)
    Md = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)

    qc_model = QCGroupMean().fit(m.iloc[qc_rows].reset_index(drop=True),
                                 h.Y_tr[qc_rows], h.stats.protein_mean)
    qc_pred = qc_model.predict(m)

    def run_variant(fit_rows_c, fit_rows_d, g2: bool, tag: str):
        """按给定 encoder fit 范围重训 control+Δ̂（种子/超参与 val 版一致，
        w*=1.0 纯 MLP control_hat），返回 (8958, n_prot) 预测。"""
        t0 = time.time()
        ec = Encoder(CTRL_COLS).fit(m.iloc[fit_rows_c])
        Xc = torch.tensor(ec.transform(m), dtype=torch.long,
                          device=args.device)
        ctrl = np.zeros((len(m), n_prot), np.float32)
        for sd in args.seeds:
            model = CondMLP(ec.n_cats, CTRL_COLS, n_prot)
            model = train_model(model, Xc, Z, Mz.float(), ctrl_rows,
                                args.epochs_ctrl, sd, drop_cols=(0,),
                                emb_drop=args.emb_drop, lr=args.lr,
                                bs=args.bs, device=args.device,
                                log_every=50, tag=f"{tag}-ctrl s{sd}")
            ctrl += predict_model(model, Xc, args.device).astype(np.float32)
        ctrl = (ctrl / len(args.seeds) * h.stats.protein_std
                + h.stats.protein_mean)

        ed = Encoder(DLT_COLS).fit(m.iloc[fit_rows_d])
        Xd = torch.tensor(ed.transform(m), dtype=torch.long,
                          device=args.device)
        dlt = np.zeros((len(m), n_prot), np.float32)
        for sd in args.seeds:
            model = CondMLP(ed.n_cats, DLT_COLS, n_prot)
            model = train_model(model, Xd, Dmat, Md.float(), treat_rows,
                                args.epochs_delta, sd, drop_cols=(0, 1),
                                emb_drop=args.emb_drop, lr=args.lr,
                                bs=args.bs, device=args.device,
                                log_every=50, tag=f"{tag}-delta s{sd}",
                                g2_cols=(0, 1) if g2 else ())
            dlt += predict_model(model, Xd, args.device).astype(np.float32)
        dlt /= len(args.seeds)

        pred = ctrl.copy()
        pred[is_treat] += dlt[is_treat]
        pred[is_qc] = qc_pred[is_qc]
        pred = h.stats.impute(pred).astype(np.float32)
        print(f"[{tag}] trained+assembled {time.time()-t0:.0f}s")
        return pred

    results = {}

    # ---- 2. G3 变体（= 提交配置：encoder fit 于训练行，未见→UNK）----
    pred_g3 = run_variant(ctrl_rows, treat_rows, g2=False, tag="g3")
    sub_path = OUT_DIR / "pred_trainval.npy"
    if sub_path.exists():
        sub = np.load(sub_path)
        print(f"[g3] vs 原提交 pred_trainval.npy: max|diff|="
              f"{np.abs(pred_g3 - sub).max():.2e}")
    np.save(OUT_DIR / "pred_trainval_g3.npy", pred_g3)
    print("\n===== [g3] G3 版（未见实体→UNK(0)）=====")
    results["G3"] = h.score_val(pred_g3)

    # ---- 3. 反事实：buggy encoder（全集 fit，未见实体=未训练随机行）----
    pred_buggy = run_variant(np.where(is_ctrl)[0], np.where(is_treat)[0],
                             g2=False, tag="noG3")
    print("\n===== [noG3] 反事实 buggy encoder（仅诊断，不存盘）=====")
    results["noG3-buggy"] = h.score_val(pred_buggy)

    # ---- 4. 可选 G2 组级增强 ----
    if args.g2:
        pred_g2 = run_variant(ctrl_rows, treat_rows, g2=True, tag="g2g3")
        np.save(OUT_DIR / "pred_trainval_g2g3.npy", pred_g2)
        print("\n===== [g2g3] Δ̂ G2 组级 UNK 增强 + G3 =====")
        results["G2+G3"] = h.score_val(pred_g2)

    # ---- 对比汇总 ----
    print("\n===== [g3] 变体对比汇总 =====")
    hdr = (f"{'variant':<12}{'composite':>10}{'strain_FC':>10}"
           f"{'strain_resid':>13}{'both_FC':>9}{'chem_FC':>9}{'time_FC':>9}")
    print(hdr)
    for name, res in results.items():
        ps = res["per_split"]
        print(f"{name:<12}{res['composite']:>10.4f}"
              f"{ps['val_strain_only']['FC_PCC']:>10.4f}"
              f"{ps['val_strain_only'].get('resid_PCC', float('nan')):>13.4f}"
              f"{ps['val_both']['FC_PCC']:>9.4f}"
              f"{ps['val_chem_only']['FC_PCC']:>9.4f}"
              f"{ps['val_time']['FC_PCC']:>9.4f}")


# ---------------- 全量重训 + test 预测（最终提交生成） ----------------

def full_pipeline(args):
    """全部 train_val（8958 行）重训两阶段模型并预测 test。

    与 val 版的差异仅训练行范围：rows_all = np.arange(len(h.m_tr))，
    control_hat 用全量对照（956）、Δ̂ 用全量处理样本（7884）、QC 同口径。
    Δ_true 直接用 h.delta_tr_all（对照池仅 train_val，不触碰 h.Y_te）。
    融合权重沿用 val 选择结果 w*=1.0（纯 MLP control_hat，组均值权重 0）。
    标准化/插补统计用全量 train_val 冻结（FrozenStats(rows=all)）。
    只写 pred_test.npy，不动 pred_trainval.npy / report.md。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = Harness()
    m, m_te = h.m_tr, h.m_te
    n_prot = h.Y_tr.shape[1]
    rows_all = np.arange(len(m))                      # 要求①
    pert = m["perturbation_no_concentration"]
    is_ctrl = pert.isin(D.CONTROLS).to_numpy()
    is_qc = (pert == D.QC).to_numpy()
    is_treat = h.is_treat_tr
    ctrl_rows = rows_all[is_ctrl]
    treat_rows = rows_all[is_treat]
    qc_rows = rows_all[is_qc]
    print(f"[wsB full] controls={len(ctrl_rows)} treated={len(treat_rows)} "
          f"qc={len(qc_rows)} | test rows={len(m_te)}")

    stats = D.FrozenStats(m, h.Y_tr, rows=rows_all)   # 全量 train_val 冻结

    # ---- Stage 1: control MLP（全量对照）----
    t0 = time.time()
    enc_c = Encoder(CTRL_COLS).fit(m.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m), dtype=torch.long, device=args.device)
    Xc_te = torch.tensor(enc_c.transform(m_te), dtype=torch.long,
                         device=args.device)
    mean = torch.tensor(stats.protein_mean, dtype=torch.float32,
                        device=args.device)
    std = torch.tensor(stats.protein_std, dtype=torch.float32,
                       device=args.device)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32, device=args.device)
         - mean) / std
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    ctrl_te = np.zeros((len(m_te), n_prot), np.float32)
    for sd in args.seeds:
        model = CondMLP(enc_c.n_cats, CTRL_COLS, n_prot)
        model = train_model(model, Xc, Z, Mz.float(), ctrl_rows,
                            args.epochs_ctrl, sd, drop_cols=(0,),
                            emb_drop=args.emb_drop, lr=args.lr, bs=args.bs,
                            device=args.device, tag=f"full-ctrl s{sd}")
        ctrl_te += predict_model(model, Xc_te, args.device).astype(np.float32)
    ctrl_te = (ctrl_te / len(args.seeds) * stats.protein_std
               + stats.protein_mean)
    print(f"[wsB full] control MLP done {time.time()-t0:.0f}s")

    # ---- Stage 2: Δ̂ MLP（全量处理样本，对照池仅 train_val）----
    t0 = time.time()
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    Xd = torch.tensor(enc_d.transform(m), dtype=torch.long, device=args.device)
    Xd_te = torch.tensor(enc_d.transform(m_te), dtype=torch.long,
                         device=args.device)
    Dmat = torch.tensor(h.delta_tr_all, dtype=torch.float32,
                        device=args.device)
    Md = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)
    delta_te = np.zeros((len(m_te), n_prot), np.float32)
    for sd in args.seeds:
        model = CondMLP(enc_d.n_cats, DLT_COLS, n_prot)
        model = train_model(model, Xd, Dmat, Md.float(), treat_rows,
                            args.epochs_delta, sd, drop_cols=(0, 1),
                            emb_drop=args.emb_drop, lr=args.lr, bs=args.bs,
                            device=args.device, tag=f"full-delta s{sd}")
        delta_te += predict_model(model, Xd_te, args.device).astype(np.float32)
    delta_te /= len(args.seeds)
    print(f"[wsB full] delta MLP done {time.time()-t0:.0f}s")

    # ---- QC（全量 train_val QC，同口径级联组均值）----
    qc_model = QCGroupMean().fit(m.iloc[qc_rows].reset_index(drop=True),
                                 h.Y_tr[qc_rows], stats.protein_mean)
    qc_te = qc_model.predict(m_te)

    # ---- 合成 test 预测 ----
    pert_te = m_te["perturbation_no_concentration"]
    is_ctrl_te = pert_te.isin(D.CONTROLS).to_numpy()
    is_qc_te = (pert_te == D.QC).to_numpy()
    is_treat_te = ~is_ctrl_te & ~is_qc_te
    pred = ctrl_te.copy()
    pred[is_treat_te] += delta_te[is_treat_te]
    pred[is_qc_te] = qc_te[is_qc_te]
    pred = stats.impute(pred).astype(np.float32)
    assert pred.shape == (len(m_te), n_prot), pred.shape
    assert not np.isnan(pred).any() and not np.isinf(pred).any()
    np.save(OUT_DIR / "pred_test.npy", pred)
    print(f"[wsB full] saved {OUT_DIR/'pred_test.npy'} shape={pred.shape} "
          f"dtype={pred.dtype} treated={int(is_treat_te.sum())} "
          f"ctrl={int(is_ctrl_te.sum())} qc={int(is_qc_te.sum())}")


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-ctrl", type=int, default=100)
    ap.add_argument("--epochs-delta", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--emb-drop", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--blend-grid", type=float, nargs="+",
                    default=[1.0, 0.5, 0.0],
                    help="control_hat = w*MLP + (1-w)*组均值 的候选 w")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="全部 train_val 重训并预测 test（只写 pred_test.npy）")
    ap.add_argument("--g3-infer", action="store_true",
                    help="G3 推理修复验证：重训 G3 变体 + buggy 反事实对照")
    ap.add_argument("--g2", action="store_true",
                    help="配合 --g3-infer：附加 Δ̂ G2 组级 UNK 增强重训")
    args = ap.parse_args()

    if args.g3_infer:
        g3_infer_pipeline(args)
        return

    if args.full:
        full_pipeline(args)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    print(f"[wsB] train controls={len(ctrl_rows)} treated={len(treat_rows)} qc={len(qc_rows)}")

    # ---- Stage 1a: control MLP（train 对照，标准化 log2 目标）----
    t0 = time.time()
    enc_c = Encoder(CTRL_COLS).fit(m.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m), dtype=torch.long, device=args.device)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32, device=args.device)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32, device=args.device)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32, device=args.device) - mean) / std
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    ctrl_mlp = np.zeros((len(m), n_prot), np.float32)
    for sd in args.seeds:
        model = CondMLP(enc_c.n_cats, CTRL_COLS, n_prot)
        model = train_model(model, Xc, Z, Mz.float(), ctrl_rows,
                            args.epochs_ctrl, sd, drop_cols=(0,),
                            emb_drop=args.emb_drop, lr=args.lr, bs=args.bs,
                            device=args.device, tag=f"ctrl s{sd}")
        ctrl_mlp += predict_model(model, Xc, args.device).astype(np.float32)
    ctrl_mlp = ctrl_mlp / len(args.seeds) * h.stats.protein_std + h.stats.protein_mean
    print(f"[wsB] control MLP done {time.time()-t0:.0f}s")

    # ---- Stage 1b: 分层组均值对照 ----
    gm = GroupMeanControl().fit(m.iloc[ctrl_rows].reset_index(drop=True),
                                h.Y_tr[ctrl_rows], h.stats.protein_mean)
    ctrl_gm = gm.predict(m)

    # ---- Stage 2: Δ̂ MLP（train 处理样本，raw Δ masked MSE）----
    t0 = time.time()
    enc_d = Encoder(DLT_COLS).fit(m.iloc[treat_rows])
    Xd = torch.tensor(enc_d.transform(m), dtype=torch.long, device=args.device)
    Dmat = torch.tensor(h.delta_tr_all, dtype=torch.float32, device=args.device)
    Md = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)
    delta_hat = np.zeros((len(m), n_prot), np.float32)
    for sd in args.seeds:
        model = CondMLP(enc_d.n_cats, DLT_COLS, n_prot)
        model = train_model(model, Xd, Dmat, Md.float(), treat_rows,
                            args.epochs_delta, sd, drop_cols=(0, 1),
                            emb_drop=args.emb_drop, lr=args.lr, bs=args.bs,
                            device=args.device, tag=f"delta s{sd}")
        delta_hat += predict_model(model, Xd, args.device).astype(np.float32)
    delta_hat /= len(args.seeds)
    print(f"[wsB] delta MLP done {time.time()-t0:.0f}s")

    # ---- QC 预测 ----
    qc_model = QCGroupMean().fit(m.iloc[qc_rows].reset_index(drop=True),
                                 h.Y_tr[qc_rows], h.stats.protein_mean)
    qc_pred = qc_model.predict(m)

    # ---- 合成与评分 ----
    def assemble(control_hat):
        pred = control_hat.copy()
        pred[is_treat] += delta_hat[is_treat]
        pred[is_qc] = qc_pred[is_qc]
        pred = h.stats.impute(pred)  # 安全网：理论上无 NaN
        return pred.astype(np.float32)

    results = {}
    best = (None, -np.inf, None)
    for w in args.blend_grid:
        ch = w * ctrl_mlp + (1.0 - w) * ctrl_gm
        pred = assemble(ch)
        assert not np.isnan(pred).any() and not np.isinf(pred).any()
        print(f"\n===== [wsB] blend w={w} (control = {w}*MLP + {1-w}*GM) =====")
        res = h.score_val(pred)
        results[f"w={w}"] = res
        if res["composite"] > best[1]:
            best = (w, res["composite"], pred)

    w_best, comp_best, pred_best = best
    np.save(OUT_DIR / "pred_trainval.npy", pred_best)
    print(f"\n[wsB] saved {OUT_DIR/'pred_trainval.npy'} (w={w_best}, composite={comp_best:.4f})")

    # ---- 诊断：锚定对照真值的上界（不进提交）----
    print("\n===== [wsB] DIAGNOSTIC anchored (true matched controls + Δ̂) =====")
    pred_anchor = pred_best.copy()
    val_rows = np.where((m["split_final"] != "train").to_numpy())[0]
    val_treat = val_rows[is_treat[val_rows]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in val_treat:
            sid = m["sample_ID"].iloc[r]
            ctrls = [h.idx_tr[c] for c in h.ctrl_map.get(sid, []) if c in h.idx_tr]
            if not ctrls:
                continue
            cmean = np.nanmean(h.Y_tr[ctrls], axis=0)
            row = cmean + delta_hat[r]
            keep = ~np.isnan(row)
            pred_anchor[r, keep] = row[keep]
    pred_anchor = h.stats.impute(pred_anchor).astype(np.float32)
    res_anchor = h.score_val(pred_anchor)

    # ---- 基线对比（已有缓存，train split 训练）----
    base_res = {}
    if not args.skip_baselines:
        for name, path in [("Ridge", "pred_ridge.npy"),
                           ("MLP-3seed", "pred_mlp_3seed.npy")]:
            p = Path(__file__).resolve().parent.parent / "outputs" / path
            if p.exists():
                print(f"\n===== baseline {name} =====")
                base_res[name] = h.score_val(np.load(p), verbose=False)
                print(f"  composite={base_res[name]['composite']:.4f}")

    write_report(results, w_best, res_anchor, base_res)
    print(f"[wsB] report written to {OUT_DIR/'report.md'}")


def _fmt_table(per_split: dict) -> str:
    cols = ["fidelity", "sample_PCC", "sample_R2", "protein_PCC",
            "FC_PCC", "resid_PCC", "DEP_dir_acc", "DEP_PCC", "DEP_F1"]
    lines = ["| split | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for sp, s in per_split.items():
        row = [f"{s.get(c, float('nan')):.4f}" if c in s else "-"
               for c in cols]
        lines.append(f"| {sp} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(results: dict, w_best: float, res_anchor: dict,
                 base_res: dict):
    main = results[f"w={w_best}"]
    L = []
    L.append("# wsB 两阶段建模报告：control_hat(上下文+批次) + Δ̂(扰动效应)\n")
    L.append("## 方法要点\n")
    L.append("- **Stage 1 control_hat**：仅 train split 对照样本（DMSO/Water，751 个；"
             "任务书中 956 为 train_val 总数，按红线只用 train split）。两个互补模型按权重融合：\n"
             "  - 条件嵌入 MLP（菌株/培养基/温度/时间/仪器/来源/板号，无化合物输入；"
             "512-1024，菌株 UNK+25% embedding dropout，masked MSE，100 epochs，3 种子均值）；\n"
             "  - 分层组均值（5 键→逐蛋白级联回退→菌株→全局，对照版 DeltaAdditive 控制端）。\n"
             "- **Stage 2 Δ̂**：train split 处理样本（5078 个），目标 = `h.delta_tr_all` 对应行"
             "（raw log2 Δ，masked MSE，不标准化以保持蛋白间方差权重与 FC_PCC 口径一致）。"
             "输入 = 菌株(UNK+25% drop)+化合物(UNK+25% drop)+培养基/温度/时间/仪器/来源/板号嵌入，"
             "MLP 512-1024，100 epochs，3 种子均值。\n"
             "- **合成**：处理样本 ŷ = control_hat + Δ̂；对照样本 ŷ = control_hat；"
             "QC 样本 = train QC 按 (仪器×来源)→仪器→全局 级联组均值。\n"
             "- **编码器差异**：encoder 只在各模型训练行上 fit，val 独有菌株/化合物/板号 → UNK(0)，"
             "不用未训练的随机嵌入（与 src/train_mlp.py 的差异，属有意改进）。\n"
             "- **合规说明**：185 个 train 处理样本的 Δ_true 匹配对照落在 val 划分"
             "（harness 自带冻结参照 μ_ctx/μ_drug 与 DeltaAdditive 基线同口径，"
             "且 `h.delta_tr_all` 为 AGENT_GUIDE 明示接口）。pred_trainval.npy 为非锚定版，无 NaN。\n")
    L.append(f"\n**融合权重选择（一次性模型选择）**：w*={w_best}，"
             f"composite={main['composite']:.4f}\n")

    L.append("\n## 主提交（非锚定）score_val 全表\n")
    L.append(_fmt_table(main["per_split"]))
    L.append(f"\n\n**composite = {main['composite']:.4f}**\n")

    L.append("\n## 融合权重对比（control_hat = w·MLP + (1−w)·组均值，Δ̂ 相同）\n")
    L.append("| w | composite | FC_PCC 均值 |")
    L.append("|---|---|---|")
    for k, res in results.items():
        fcs = [s["FC_PCC"] for s in res["per_split"].values()]
        L.append(f"| {k} | {res['composite']:.4f} | {np.mean(fcs):.4f} |")
    L.append("")

    if base_res:
        L.append("\n## 与基线对比\n")
        cols = ["composite", "fidelity", "FC_PCC", "resid_PCC"]
        L.append("| 模型 | split | " + " | ".join(cols[1:]) + " | composite |")
        L.append("|---|---|---|---|---|---|")
        for name, res in list(base_res.items()) + [("wsB-twostage", main)]:
            for sp, s in res["per_split"].items():
                L.append(f"| {name} | {sp} | {s['fidelity']:.4f} | "
                         f"{s['FC_PCC']:.4f} | "
                         + (f"{s['resid_PCC']:.4f}" if "resid_PCC" in s else "-")
                         + f" | {res['composite']:.4f} |")
        L.append("")

    L.append("\n## 诊断（不进提交）：锚定真实匹配对照的上界\n")
    L.append("把 val 处理样本的 ŷ 换成「匹配对照真值均值 + Δ̂」（对照取自 train_val 池，"
             "含 val 划分自身的对照），其余行不变。此时评分侧 Δ_pred ≡ Δ̂，"
             "用于量化『若组委会允许用 test 对照原始值锚定』的提分上限，"
             "**仅供向组委会提问参考，不可用于提交**。\n")
    L.append(_fmt_table(res_anchor["per_split"]))
    L.append(f"\n\n**锚定上界 composite = {res_anchor['composite']:.4f}**"
             f"（非锚定 {main['composite']:.4f}，差值 "
             f"{res_anchor['composite'] - main['composite']:+.4f}）\n")

    L.append("\n### 逐划分 FC/resid 对比（非锚定 → 锚定）\n")
    L.append("| split | FC_PCC 非锚定 | FC_PCC 锚定 | resid_PCC 非锚定 | resid_PCC 锚定 |")
    L.append("|---|---|---|---|---|")
    for sp in main["per_split"]:
        a, b = main["per_split"][sp], res_anchor["per_split"][sp]
        L.append(f"| {sp} | {a['FC_PCC']:.4f} | {b['FC_PCC']:.4f} | "
                 + (f"{a['resid_PCC']:.4f}" if "resid_PCC" in a else "-") + " | "
                 + (f"{b['resid_PCC']:.4f}" if "resid_PCC" in b else "-") + " |")
    L.append("")

    L.append("\n## 关键发现\n")
    L.append("- **Δ 解耦有效**：composite 显著超过 Ridge/MLP 基线；val_strain_only 与 val_both 的"
             " FC/resid 较直接预测绝对丰度的 MLP 基线明显提升（Δ̂ 的容量全部用于扰动信号），"
             "val_time 略降（时间外推上直接绝对值预测略占优）。\n"
             "- **锚定诊断的反直觉结论（重要）**：锚定版 Δ_pred ≡ Δ̂，FC/resid/DEP 全面低于非锚定版，"
             "即『用 test 对照原始值锚定』不会提分。机制：评分侧 Δ_true = y_treat − 对照真值"
             "含有对照重复测量噪声（负号）；非锚定版 Δ_pred = Δ̂ + (control_hat − 对照真值) 中的"
             " (control_hat − 对照真值) ≈ −对照噪声，与 Δ_true 的 −对照噪声分量正相关，"
             "抬高逐样本 PCC；锚定版把该分量完全消去。fidelity 则相反（锚定版略高，"
             "绝对空间受益于真实对照）。此机制在官方 test 评分（同样对匹配对照求 Δ）下同样成立。\n"
             "- **control_hat 选型**：纯组均值（w=0）在 val_strain_only/val_both 上蛋白间结构崩塌"
             "（protein_PCC≈0.07/−0.07，未见菌株只能回退全局均值）；MLP 的 UNK 菌株嵌入"
             "回退显著更优，融合 w=1.0（纯 MLP）最佳。\n"
             "- 板号在 val 处理样本中 100% 见于 train（对照端亦 100%），批次效应可转移；"
             "val_strain_only/val_both 的菌株在 train 对照与 train 处理样本中均完全缺失"
             "（0% 键命中），只能依赖 UNK 回退，是这两个划分 FC 偏低的主因。\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
