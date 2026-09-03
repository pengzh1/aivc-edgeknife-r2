"""wsW1：官方口径修正重评分（分数口径核对，零新拟合、零新 val 信息）。

动机：本地 composite 是官方口径的近似，逐字核对《参赛指南》+Datawhale 进阶教程
后发现三处实质偏差：
  B1 保真度缺 protein_R²（官方=逐样本 corr/R² + 逐蛋白 corr/R² 四件套）；
  B2 μ_ctx 缺"冻结批次"（官方上下文=菌株/培养基/温度/时间/冻结批次 五键；
     本模块以孔板为冻结批次代理、仪器为敏感性对照）；
  B3 DEP 为指标族（方向准确率/高效应PCC/F1 等，聚合未写明 → 变体扫描）。
聚合的不可消解歧义（保真度内部配比、DEP 指标选择、both/time 组合）以
变体扫描覆盖，看候选排序对口径的稳健性。

候选（全部已落盘预测，train-only 参照）：裸路由 / C1@τ0.35 / C1@τ0.25 /
r=1.0 两角 / band。若修正口径下排序变化 → 打包决策改价。

合规：纯重算（预测与参照全部既有/train-only）；Y_te 零接触；新文件。
用法: python -m src.wsW1_official_score
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from .evaluate import Harness
from .wsN11_grandrouter import SPLITS

OUT = Path("outputs/wsW1")
W = {"fidelity": 0.20, "FC": 0.25, "ctx_resid": 0.20, "drug_resid": 0.20,
     "both_time": 0.10, "DEP": 0.05}

CANDS = {
    "routed": "outputs/wsT0/cache/routed_r07_trainval.npy",
    "band13": "outputs/wsT0/cache/routed_r07_band13_trainval.npy",
    "C1_t35": "outputs/wsT1/pred_trainval.npy",
    "C1_t25": "outputs/wsT9/pred_trainval_tau0.25.npy",
    "r100_t35": "outputs/wsV_round/pred_trainval_r100_c1t35.npy",
    "r100_t25": "outputs/wsV_round/pred_trainval_r100_c1t25.npy",
}


def protein_r2(yt: np.ndarray, yp: np.ndarray) -> float:
    """逐蛋白 R²（跨样本，逐列 masked），聚合取均值，clip [-1,1]。"""
    r2s = []
    for j in range(yt.shape[1]):
        t, p = yt[:, j], yp[:, j]
        mk = ~np.isnan(t)
        if mk.sum() > 3 and np.var(t[mk]) > 1e-9:
            ss_res = np.sum((t[mk] - p[mk]) ** 2)
            ss_tot = np.sum((t[mk] - t[mk].mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(np.clip(r2s, -1, 1))) if r2s else 0.0


def build_mu_ctx(h: Harness, keys: list[str]) -> tuple[dict, dict]:
    """train-only μ_ctx：keys 分组均值 + 逐键回退链（官方'同上下文下训练药物'）。"""
    m_train = h.m_train
    is_tr = ~m_train["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    dlt = h.delta_train[is_tr]
    df = m_train[is_tr]
    maps = {}
    for klen in range(len(keys), 0, -1):
        ks = keys[:klen] if klen < len(keys) else keys
        km = list(map(tuple, df[ks].to_numpy()))
        pool = {}
        for k, idx in pd.DataFrame({"k": km}).groupby("k").groups.items():
            idx = np.fromiter(idx, dtype=int)
            with np.errstate(invalid="ignore"):
                pool[k] = np.nanmean(dlt[idx], axis=0)
        maps[klen] = (ks, pool)
    # 行查找：先全键，再逐级放宽（优先级=键长）
    def mu_for(m_target: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(m_target), h.Y_tr.shape[1]), np.float32)
        for i, row in enumerate(m_target.itertuples()):
            for klen in range(len(keys), 0, -1):
                ks, pool = maps[klen]
                k = tuple(getattr(row, kk) for kk in ks)
                v = pool.get(k)
                if v is not None:
                    out[i] = np.nan_to_num(v, nan=0.0)
                    break
        return out
    return mu_for


def score_candidate(h: Harness, pred: np.ndarray,
                    mu_ctx_for, mu_ctx_keys) -> dict:
    """官方口径全套组件 + 变体就绪的 per-split 组件字典。"""
    comp = {}
    for sp in SPLITS:
        rows = np.where((h.m_tr["split_final"] == sp).to_numpy())[0]
        trows = rows[h.is_treat_tr[rows]]
        yt, yp = h.Y_tr[rows], pred[rows]
        s_pcc = float(np.nanmean(M._masked_pcc_axis1(yt, yp)))
        s_r2 = float(np.nanmean(np.clip(M._masked_r2_axis1(yt, yp), -1, 1)))
        p_pcc = float(np.nanmean(M._masked_pcc_axis1(yt.T, yp.T)))
        p_r2 = protein_r2(yt, yp)
        dpred = np.full_like(h.delta_tr_all, np.nan)
        dpred[rows] = h._delta_pred(rows, pred)
        fc = float(np.nanmean(M._masked_pcc_axis1(
            h.delta_tr_all[trows], dpred[trows])))
        dep = M.dep_scores(h.delta_tr_all, dpred, trows)
        entry = {"s_pcc": s_pcc, "s_r2": s_r2, "p_pcc": p_pcc, "p_r2": p_r2,
                 "FC": fc, **dep}
        if "chem_only" in sp:
            mu = mu_ctx_for(h.m_tr.iloc[trows])
            entry["ctx_resid"] = float(np.nanmean(M._masked_pcc_axis1(
                h.delta_tr_all[trows] - mu, dpred[trows] - mu)))
            entry["ctx_hit_keys"] = mu_ctx_keys
        if "strain_only" in sp:
            mu = h.mu_drug_for(h.m_tr.iloc[trows])
            entry["drug_resid"] = float(np.nanmean(M._masked_pcc_axis1(
                h.delta_tr_all[trows] - mu, dpred[trows] - mu)))
        comp[sp] = entry
    return comp


def synthesize(comp: dict, fid_mode: str, dep_mode: str) -> float:
    """口径变体合成。fid_mode: 3(现行)/4(官方四件套)/corr4；dep_mode: f1/dep3。"""
    def fid_of(e):
        if fid_mode == "3":
            return np.mean([e["s_pcc"], e["s_r2"], e["p_pcc"]])
        if fid_mode == "4":
            return np.mean([e["s_pcc"], e["s_r2"], e["p_pcc"], e["p_r2"]])
        return np.mean([e["s_pcc"], e["p_pcc"]])
    def dep_of(e):
        if dep_mode == "f1":
            return e["DEP_F1"]
        return np.mean([e["DEP_dir_acc"], e["DEP_PCC"], e["DEP_F1"]])
    fid = np.mean([fid_of(comp[sp]) for sp in SPLITS])
    fc = np.mean([comp[sp]["FC"] for sp in SPLITS])
    dep = np.mean([dep_of(comp[sp]) for sp in SPLITS])
    ctx = comp["val_chem_only"]["ctx_resid"]
    drug = comp["val_strain_only"]["drug_resid"]
    bt = np.mean([0.5 * (comp[sp]["FC"] + fid_of(comp[sp]))
                  for sp in SPLITS if "both" in sp or "time" in sp])
    return (W["fidelity"] * fid + W["FC"] * fc + W["DEP"] * dep
            + W["ctx_resid"] * ctx + W["drug_resid"] * drug
            + W["both_time"] * bt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = Harness()
    print("[wsW1] 官方口径重评分（train-only 参照）", flush=True)
    mu4 = build_mu_ctx(h, D.CTX_KEYS)
    mu5_plate = build_mu_ctx(h, D.CTX_KEYS + ["Yeast_cell_plate"])
    mu5_instr = build_mu_ctx(h, D.CTX_KEYS + ["instrument"])

    table = {}
    for name, path in CANDS.items():
        pred = np.load(path)
        comp4 = score_candidate(h, pred, mu4, "4键（现行）")
        comp5p = score_candidate(h, pred, mu5_plate, "5键·板")
        comp5i = score_candidate(h, pred, mu5_instr, "5键·仪器")
        row = {}
        for label, comp in [("ctx4", comp4), ("ctx5p", comp5p),
                            ("ctx5i", comp5i)]:
            for fid_mode in ("3", "4"):
                for dep_mode in ("f1", "dep3"):
                    key = f"{label}|fid{fid_mode}|dep{dep_mode}"
                    row[key] = round(synthesize(comp, fid_mode, dep_mode), 4)
        table[name] = row
        print(f"[{name}] " + " ".join(
            f"{k}={v:.4f}" for k, v in list(row.items())[:4]), flush=True)
        (OUT / f"comp_{name}.json").write_text(json.dumps(
            {"ctx4": comp4, "ctx5p": comp5p, "ctx5i": comp5i}, indent=1,
            default=float))
    (OUT / "official_rescore.json").write_text(json.dumps(
        table, indent=1, default=float))
    # 排序稳健性摘要
    keys = list(next(iter(table.values())).keys())
    print("\n[排序稳健性]（各口径变体下的第一名）", flush=True)
    for k in keys:
        rank = sorted(table, key=lambda n: -table[n][k])
        print(f"  {k:<24} {' > '.join(rank[:3])}", flush=True)
    print(f"[done] ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
