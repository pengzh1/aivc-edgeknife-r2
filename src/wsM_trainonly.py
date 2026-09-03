"""wsM: train-only 回退 —— 六模型族 split_final=train 重训 + test 预测。

背景（2026-08-10 组委会书面答复，outref/replymail.txt + 新版手册 P15/P17）：
- 最终模型不得使用 val 划分重训；保留蛋白列表、归一化均值/标准差、
  对照匹配规则均只能由 split_final=train 样本估计；
- 双榜合并为统一排名；外部公开资源（PubChem/RDKit 等）允许用于实体特征构建。

本模块按 train-only 口径为提交阵容六族生成 test 预测：
  ridge / wsD(g2) / wsB(strict-Δ) / wsC(3×8 时间集成) / wsF(d1) / wsA(描述符)
并额外产出 wsB 严格版 val 预测（供 wsH 路由权重重拟合）。

严格化差异（相对 8.6 提交链路）：
1. 全部训练行 = h.tr_rows（split_final=train），冻结统计 = h.stats（train split）；
2. wsB 的 Δ̂ 训练目标改用"仅 train 对照池"计算的 Δ（原版 h.delta_tr_all 中有
   185 个 train 处理样本的匹配对照落在 val 划分，其对照蛋白值会进入训练目标，
   按最严口径这些行不参与 Δ̂ 训练；encoder 词表仍按全部 train 处理行 fit）；
3. test 推理一律 G3/UNK 修复（seen 集合 = train split 实际出现类别）。

合规：h.Y_te 零接触；val 蛋白值零训练接触；种子固定；产物全部落 outputs/wsM/，
不改动任何既有文件。

用法:
    python -m src.wsM_trainonly                     # 全部族（缺啥补啥，幂等续跑）
    python -m src.wsM_trainonly --families ridge,wsB # 只跑指定族
    python -m src.wsM_trainonly --smoke             # 快速冒烟（小 epoch，写 smoke/）
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .evaluate import Harness

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsM"

# 提交配方（与 8.6 提交链路一致，仅训练行口径改为 train-only）
WSD_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "epochs": 300,
           "lr": 0.001, "wd": 0.0001, "emb_drop": 0.35, "p_drop": 0.3,
           "bs": 256, "chem_emb": 32, "residual": False, "lowrank": 0,
           "loss": "huber", "film": False, "g2_aug": True}
WSD_SEEDS = list(range(8))

WSC_RECIPES = [  # 3 时间配方 × 8 种子 × 150ep + tail_avg（=pred_trainval_g3 配方）
    {"time_onehot": True, "interact": "none"},        # E1/R1 onehot
    {"time_onehot": True, "interact": "chemstrain"},  # E2/R2 both+interact
    {"time_onehot": False, "interact": "none"},       # E3/R3 cont
]
WSC_SEEDS = list(range(8))
WSC_EPOCHS = 150

WSF_CFG = {"hidden": [1024, 2048, 2048, 2048, 2048], "inter_ctx": True,
           "inter_strain": False}  # 其余 CfgD1 默认（300ep/huber/emb_drop .35）
WSF_SEEDS = [0, 1, 2]

WSA_SEEDS = [0, 1, 2, 3, 4]
WSA_EPOCHS = 100

WSB_SEEDS = [0, 1, 2]
WSB_EPOCHS = 100


def _guard(pred: np.ndarray, h: Harness, tag: str) -> np.ndarray:
    """交付契约：NaN/Inf 用 train 冻结蛋白均值填补。"""
    bad = ~np.isfinite(pred)
    if bad.any():
        print(f"[warn] {tag}: {int(bad.sum())} non-finite -> train protein mean")
        r, c = np.where(bad)
        pred[r, c] = np.take(h.stats.protein_mean, c)
    return pred.astype(np.float32)


def _save(pred: np.ndarray, path: Path, h: Harness, tag: str):
    pred = _guard(pred, h, tag)
    assert pred.shape == (len(h.m_te) if "test" in path.name else len(h.m_tr),
                          h.Y_tr.shape[1]), (tag, pred.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred)
    print(f"[saved] {path} shape={pred.shape} dtype={pred.dtype}", flush=True)


# ---------------------------------------------------------------- ridge

def run_ridge(h: Harness, out: Path, smoke: bool):
    from . import baselines
    t0 = time.time()
    model = baselines.RidgeMulti(lam=10.0).fit(h)  # rows=tr_rows, stats=train
    pred = model.predict(h.m_te)
    _save(pred, out, h, "ridge")
    return {"seconds": time.time() - t0}


# ---------------------------------------------------------------- wsD

def run_wsD(h: Harness, out: Path, smoke: bool):
    from . import wsD_arch as W
    cfg = W.Cfg.from_dict({**WSD_CFG, "epochs": 2 if smoke else WSD_CFG["epochs"]})
    seeds = WSD_SEEDS[:1] if smoke else WSD_SEEDS
    seen = W.seen_cats(h, h.tr_rows)  # G3：train split 实见类别
    preds = []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std, _ = W.train_one(h, cfg, s)
        preds.append(W.predict(model, enc, mean, std, h.m_te, g3_seen=seen))
        print(f"[wsD] seed={s} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    _save(np.mean(preds, axis=0), out, h, "wsD")
    return {"seeds": seeds, "cfg": cfg.__dict__}


# ---------------------------------------------------------------- wsB（strict-Δ）

def strict_delta_train(h: Harness):
    """仅用 train split 对照池计算 train 处理样本的 Δ_true。

    返回 (delta, treat_rows_all, treat_rows_valid)：
    delta 形状 (5920, P)，无 train 对照的处理行全 NaN；
    treat_rows_valid 为剔除全 NaN 后的训练行（相对 h.delta_tr_all 口径少 185 行）。
    """
    m_train, Y_train = h.m_train, h.Y_train
    pert = m_train["perturbation_no_concentration"]
    is_treat = ~pert.isin(D.CONTROLS | {D.QC}).to_numpy()
    treat_rows_all = np.where(is_treat)[0]
    ctrl_map = D.build_control_map(m_train)
    idx = {s: i for i, s in enumerate(m_train["sample_ID"])}
    delta = np.full((len(m_train), Y_train.shape[1]), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, sid in enumerate(m_train["sample_ID"]):
            if not is_treat[i]:
                continue
            ctrls = [idx[c] for c in ctrl_map.get(sid, []) if c in idx]
            if ctrls:
                delta[i] = Y_train[i] - np.nanmean(Y_train[ctrls], axis=0)
    has = ~np.isnan(delta[treat_rows_all]).all(axis=1)
    treat_rows_valid = treat_rows_all[has]
    print(f"[wsB strict] Δ 目标：train 处理 {len(treat_rows_all)} 行，"
          f"其中 {int((~has).sum())} 行无 train 对照被剔除 "
          f"-> 训练行 {len(treat_rows_valid)}")
    return delta, treat_rows_all, treat_rows_valid


def run_wsB(h: Harness, out_test: Path, out_val: Path, smoke: bool):
    """两阶段（control_hat + Δ̂）train-only 严格版，同时产出 val 与 test 预测。"""
    from . import wsB_twostage as B
    m_train, Y_train = h.m_train, h.Y_train
    n_prot = h.Y_tr.shape[1]
    pert = m_train["perturbation_no_concentration"]
    is_ctrl = pert.isin(D.CONTROLS).to_numpy()
    is_qc = (pert == D.QC).to_numpy()
    ctrl_rows = np.where(is_ctrl)[0]
    qc_rows = np.where(is_qc)[0]
    epochs_c = 2 if smoke else WSB_EPOCHS
    epochs_d = 2 if smoke else WSB_EPOCHS
    seeds = WSB_SEEDS[:1] if smoke else WSB_SEEDS

    delta, treat_rows_all, treat_rows = strict_delta_train(h)

    stats = h.stats  # train split 冻结
    mean_t = torch.tensor(stats.protein_mean, dtype=torch.float32, device="cuda")
    std_t = torch.tensor(stats.protein_std, dtype=torch.float32, device="cuda")
    Z = (torch.tensor(Y_train, dtype=torch.float32, device="cuda") - mean_t) / std_t
    Mz = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    Dmat = torch.tensor(delta, dtype=torch.float32, device="cuda")
    Md = ~torch.isnan(Dmat)
    Dmat = torch.nan_to_num(Dmat, nan=0.0)

    # QC 组均值（train QC 行）
    qc_model = B.QCGroupMean().fit(
        m_train.iloc[qc_rows].reset_index(drop=True), Y_train[qc_rows],
        stats.protein_mean)

    views = {"val": h.m_tr, "test": h.m_te}

    # ---- Stage 1: control MLP（train 对照 751 行）----
    t0 = time.time()
    enc_c = B.Encoder(B.CTRL_COLS).fit(m_train.iloc[ctrl_rows])
    Xc = torch.tensor(enc_c.transform(m_train), dtype=torch.long, device="cuda")
    Xc_v = {k: torch.tensor(enc_c.transform(mv), dtype=torch.long, device="cuda")
            for k, mv in views.items()}
    ctrl = {k: np.zeros((len(mv), n_prot), np.float32)
            for k, mv in views.items()}
    for sd in seeds:
        model = B.CondMLP(enc_c.n_cats, B.CTRL_COLS, n_prot)
        model = B.train_model(model, Xc, Z, Mz.float(), ctrl_rows, epochs_c,
                              sd, drop_cols=(0,), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=50,
                              tag=f"wsM-ctrl s{sd}")
        for k in views:
            ctrl[k] += B.predict_model(model, Xc_v[k], "cuda").astype(np.float32)
        del model
        torch.cuda.empty_cache()
    for k in views:
        ctrl[k] = ctrl[k] / len(seeds) * stats.protein_std + stats.protein_mean
    print(f"[wsB] control MLP done ({time.time()-t0:.0f}s)", flush=True)

    # ---- Stage 2: Δ̂ MLP（strict Δ 目标；encoder 词表 = 全部 train 处理行）----
    t0 = time.time()
    enc_d = B.Encoder(B.DLT_COLS).fit(m_train.iloc[treat_rows_all])
    Xd = torch.tensor(enc_d.transform(m_train), dtype=torch.long, device="cuda")
    Xd_v = {k: torch.tensor(enc_d.transform(mv), dtype=torch.long, device="cuda")
            for k, mv in views.items()}
    dlt = {k: np.zeros((len(mv), n_prot), np.float32)
           for k, mv in views.items()}
    for sd in seeds:
        model = B.CondMLP(enc_d.n_cats, B.DLT_COLS, n_prot)
        model = B.train_model(model, Xd, Dmat, Md.float(), treat_rows, epochs_d,
                              sd, drop_cols=(0, 1), emb_drop=0.25, lr=1e-3,
                              bs=256, device="cuda", log_every=50,
                              tag=f"wsM-delta s{sd}")
        for k in views:
            dlt[k] += B.predict_model(model, Xd_v[k], "cuda").astype(np.float32)
        del model
        torch.cuda.empty_cache()
    for k in views:
        dlt[k] /= len(seeds)
    print(f"[wsB] delta MLP done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 合成（w*=1.0 纯 MLP control_hat；QC 行用 QC 组均值）----
    outs = {}
    for k, mv in views.items():
        pert_v = mv["perturbation_no_concentration"]
        is_treat_v = ~pert_v.isin(D.CONTROLS | {D.QC}).to_numpy()
        is_qc_v = (pert_v == D.QC).to_numpy()
        pred = ctrl[k].copy()
        pred[is_treat_v] += dlt[k][is_treat_v]
        if is_qc_v.any():
            pred[is_qc_v] = qc_model.predict(mv)[is_qc_v]
        pred = stats.impute(pred)  # train 冻结均值补缺
        outs[k] = pred.astype(np.float32)
    _save(outs["test"], out_test, h, "wsB-test")
    _save(outs["val"], out_val, h, "wsB-val")
    return {"seeds": seeds, "delta_rows": int(len(treat_rows)),
            "dropped_no_train_control": int(len(treat_rows_all)
                                            - len(treat_rows))}


# ---------------------------------------------------------------- wsC

def run_wsC(h: Harness, out: Path, smoke: bool):
    from . import wsC_timebatch as W
    epochs = 2 if smoke else WSC_EPOCHS
    seeds = WSC_SEEDS[:1] if smoke else WSC_SEEDS
    recipes = WSC_RECIPES[:1] if smoke else WSC_RECIPES
    seed_means = []
    for ri, rec in enumerate(recipes):
        for sd in seeds:
            t0 = time.time()
            tail = []
            model, enc, mean, std, _ = W.train_model(
                h, h.tr_rows, epochs, seed=sd, emb_drop=0.25, lr=1e-3,
                time_onehot=rec["time_onehot"], interact=rec["interact"],
                qc_mode="none", corrector=None, basis_kind="rbf",
                hidden=(512, 1024), p_drop=0.1, tail_states=tail,
                log_every=999)
            states = [model.state_dict()] + tail
            sp = []
            for st in states:
                model.load_state_dict(st)
                sp.append(W.predict(model, enc, mean, std, h, None, "none",
                                    1.0, basis_kind="rbf", df=h.m_te, g3=True))
            seed_means.append(np.mean(sp, axis=0))
            print(f"[wsC] recipe{ri} seed={sd} states={len(states)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    _save(np.mean(seed_means, axis=0), out, h, "wsC")
    return {"recipes": recipes, "seeds": seeds, "epochs": epochs,
            "n_models": len(seed_means)}


# ---------------------------------------------------------------- wsF

def _senc_transform(senc, m: pd.DataFrame, unk_unseen: bool = True) -> np.ndarray:
    """wsF.SeenEncoder.transform 的可写副本。

    本机 pandas 版本下 `Series.map(...).to_numpy()` 返回只读视图，
    原实现的就地置 0 会报 "assignment destination is read-only"。
    语义完全一致（未见类别 → UNK(0)），仅显式复制数组。
    """
    from .wsF_interact import CAT_COLS
    cols = []
    for c, seen in zip(CAT_COLS, senc.seen):
        mp = senc.enc.maps[c]
        s = m[c]
        idx = np.array(s.map(lambda v: mp.get(v, 0)).to_numpy(),
                       dtype=np.int64)  # 显式复制，保证可写
        if unk_unseen:
            idx[~s.isin(seen).to_numpy()] = 0
        cols.append(idx)
    return np.stack(cols, axis=1)


def run_wsF(h: Harness, out: Path, smoke: bool):
    from . import wsF_interact as W
    cfg = W.CfgD1.from_dict(
        {**WSF_CFG, "epochs": 2 if smoke else W.CfgD1().epochs})
    seeds = WSF_SEEDS[:1] if smoke else WSF_SEEDS
    senc = W.build_senc(h)  # seen = train split（合规关键）
    # 复刻 tensors_d1（规避只读数组问题；统计 = h.stats train split）
    X_all = torch.tensor(_senc_transform(senc, h.m_tr, True), dtype=torch.long)
    mean = torch.tensor(h.stats.protein_mean, dtype=torch.float32)
    std = torch.tensor(h.stats.protein_std, dtype=torch.float32)
    Z = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
    Mk = ~torch.isnan(Z)
    Z = torch.nan_to_num(Z, nan=0.0)
    tensors = (X_all, mean, std, Z, Mk)
    X_te = torch.tensor(_senc_transform(senc, h.m_te, True), dtype=torch.long)
    preds = []
    for sd in seeds:
        t0 = time.time()
        model, mean_t, std_t = W.train_d1(h, cfg, sd, senc, log_every=100,
                                          tensors=tensors)
        preds.append(W.predict_d1(model, X_te, mean_t, std_t))
        print(f"[wsF] seed={sd} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    _save(np.mean(preds, axis=0), out, h, "wsF")
    return {"seeds": seeds, "cfg": cfg.__dict__}


# ---------------------------------------------------------------- wsA

def run_wsA(h: Harness, out: Path, smoke: bool):
    from . import wsA_chemfeat as W
    epochs = 2 if smoke else WSA_EPOCHS
    seeds = WSA_SEEDS[:1] if smoke else WSA_SEEDS
    preds = []
    for sd in seeds:
        t0 = time.time()
        # full=True：仅扩展 test 化合物词表并使用 43 化合物结构描述符
        # （外部 PubChem/RDKit 特征，无标签泄漏；训练行仍 = tr_rows）
        model, enc, mean, std = W.train_model(
            h, h.tr_rows, epochs, seed=sd, emb_drop=0.25, chem_drop=0.25,
            lr=1e-3, bs=256, device="cuda", log_every=50, full=True)
        preds.append(W.predict(model, enc, mean, std, h.m_te, device="cuda"))
        print(f"[wsA] seed={sd} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    _save(np.mean(preds, axis=0), out, h, "wsA")
    return {"seeds": seeds, "epochs": epochs}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="ridge,wsB,wsF,wsA,wsD,wsC")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = OUT / "smoke" if args.smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    h = Harness()

    jobs = {
        "ridge": (run_ridge, [out_dir / "pred_test_ridge_trainonly.npy"]),
        "wsB": (run_wsB, [out_dir / "pred_test_wsB_trainonly.npy",
                          out_dir / "pred_trainval_wsB_strict.npy"]),
        "wsF": (run_wsF, [out_dir / "pred_test_wsF_trainonly.npy"]),
        "wsA": (run_wsA, [out_dir / "pred_test_wsA_trainonly.npy"]),
        "wsD": (run_wsD, [out_dir / "pred_test_wsD_trainonly.npy"]),
        "wsC": (run_wsC, [out_dir / "pred_test_wsC_trainonly.npy"]),
    }
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) \
        if manifest_path.exists() else {}
    for fam in args.families.split(","):
        fam = fam.strip()
        fn, outs = jobs[fam]
        if all(p.exists() for p in outs):
            print(f"[skip] {fam} 产物已存在", flush=True)
            continue
        t0 = time.time()
        if fam == "wsB":
            info = fn(h, outs[0], outs[1], args.smoke)
        else:
            info = fn(h, outs[0], args.smoke)
        info["total_seconds"] = time.time() - t0
        info["train_rows"] = int(len(h.tr_rows))
        info["compliance"] = "train-only fit; FrozenStats=train split; Y_te untouched"
        manifest[fam] = info
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                            indent=1, default=str))
        print(f"[done] {fam} ({info['total_seconds']:.0f}s)", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
