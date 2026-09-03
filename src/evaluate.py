"""本地验证流水线：载入数据 → 构建对照/参照统计 → 模型预测 → 分划分评分。

用法:
    python -m src.evaluate --model global_mean
    python -m src.evaluate --model ridge --lam 10
"""
import argparse
import time
import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M


class Harness:
    """封装数据、对照匹配、冻结参照统计与评分。"""

    def __init__(self):
        t0 = time.time()
        al = D.load_aligned()
        self.m_tr, self.m_te = al["m_tr"], al["m_te"]
        self.Y_tr, self.Y_te = al["Y_tr"], al["Y_te"]
        self.proteins = al["proteins"]
        self.idx_tr, self.idx_te = al["idx_tr"], al["idx_te"]
        self.stats = D.FrozenStats(self.m_tr, self.Y_tr)

        m_all = pd.concat([self.m_tr, self.m_te], ignore_index=True)
        self.ctrl_map = D.build_control_map(m_all)

        # train split 视图
        tr_mask = (self.m_tr["split_final"] == "train").to_numpy()
        self.tr_rows = np.where(tr_mask)[0]
        self.m_train = self.m_tr.iloc[self.tr_rows].reset_index(drop=True)
        self.Y_train = self.Y_tr[self.tr_rows]

        # 全部 train_val 处理样本的 Δ_true（对照取自 train_val 池，本地评信用）
        self.delta_tr_all, self.is_treat_tr = M.compute_delta(
            self.Y_tr, self.m_tr, self.ctrl_map, self.Y_tr, self.idx_tr)
        # 仅用 train split 的 Δ_true 做冻结参照
        self.delta_train = self.delta_tr_all[self.tr_rows]

        # μ_ctx：同上下文（菌株×培养基×温度×时间）train 药物 Δ 均值
        self.mu_ctx_map, self.mu_ctx_fallback = self._build_mu_ctx()
        # μ_drug：同药物跨上下文 train Δ 均值
        self.mu_drug_map, self.mu_drug_global = self._build_mu_drug()
        print(f"[harness] loaded in {time.time()-t0:.1f}s | train rows={len(self.tr_rows)}")

    # ---- 冻结参照统计 ----

    def _build_mu_ctx(self):
        df = self.m_train.copy()
        is_tr = ~df["perturbation_no_concentration"].isin(D.CONTROLS | {D.QC})
        dlt = self.delta_train[is_tr.to_numpy()]
        df = df[is_tr].reset_index(drop=True)
        df["__k"] = df.apply(D.ctx_key, axis=1)
        mu, fb = {}, {}
        for k, sub in df.groupby("__k"):
            mu[k] = np.nanmean(dlt[sub.index.to_numpy()], axis=0)
        # 回退：培养基×温度×时间（跨菌株）
        df["__k2"] = list(zip(df["Medium"], df["Temperature"], df["pert_time"]))
        for k, sub in df.groupby("__k2"):
            fb[k] = np.nanmean(dlt[sub.index.to_numpy()], axis=0)
        return mu, fb

    def _build_mu_drug(self):
        df = self.m_train
        is_tr = ~df["perturbation_no_concentration"].isin(D.CONTROLS | {D.QC})
        dlt = self.delta_train[is_tr.to_numpy()]
        chems = df.loc[is_tr, "perturbation_no_concentration"].to_numpy()
        mu = {}
        for c in pd.unique(chems):
            mu[c] = np.nanmean(dlt[chems == c], axis=0)
        return mu, np.nanmean(dlt, axis=0)

    def mu_ctx_for(self, m: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(m), self.Y_tr.shape[1]), dtype=np.float32)
        for i, row in enumerate(m.itertuples()):
            k = (row.Strains, row.Medium, row.Temperature, row.pert_time)
            v = self.mu_ctx_map.get(k)
            if v is None:
                v = self.mu_ctx_fallback.get(
                    (row.Medium, row.Temperature, row.pert_time))
            if v is None:
                v = self.mu_drug_global * 0.0
            out[i] = np.nan_to_num(v, nan=0.0)
        return out

    def mu_drug_for(self, m: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(m), self.Y_tr.shape[1]), dtype=np.float32)
        for i, c in enumerate(m["perturbation_no_concentration"]):
            v = self.mu_drug_map.get(c, self.mu_drug_global)
            out[i] = np.nan_to_num(v, nan=0.0)
        return out

    # ---- 评分 ----

    def score_val(self, pred_tr: np.ndarray, verbose: bool = True,
                  fast: bool = False) -> dict:
        """对 train_val 中的四个 val 划分评分。pred_tr 与 Y_tr 同形。

        fast=True 时跳过逐蛋白 PCC 与 DEP（慢），fidelity 仅用样本轴，
        composite 用近似口径（仅供集成搜索内相对比较）。
        """
        per_split = {}
        for split in ["val_chem_only", "val_strain_only", "val_both", "val_time"]:
            rows = np.where((self.m_tr["split_final"] == split).to_numpy())[0]
            yt, yp = self.Y_tr[rows], pred_tr[rows]
            s = {}
            if fast:
                s_pcc = M._masked_pcc_axis1(yt, yp)
                s_r2 = M._masked_r2_axis1(yt, yp)
                s["sample_PCC"] = float(np.mean(s_pcc))
                s["sample_R2"] = float(np.mean(np.clip(s_r2, -1, 1)))
                s["fidelity"] = float(np.mean([s["sample_PCC"], s["sample_R2"]]))
            else:
                fid = M.fidelity_scores(yt, yp)
                s.update(fid)
                s["fidelity"] = float(np.mean([fid["sample_PCC"], fid["sample_R2"],
                                               fid["protein_PCC"]]))
            # Δ 指标仅处理样本
            trows = rows[self.is_treat_tr[rows]]
            # Δ_pred = ŷ_treat - 匹配对照真值
            dpred = np.full_like(self.delta_tr_all, np.nan)
            dpred[rows] = self._delta_pred(rows, pred_tr)
            s.update(M.fc_scores(self.delta_tr_all, dpred, trows))
            if "chem_only" in split:
                mu = self.mu_ctx_for(self.m_tr.iloc[trows])
                r = M.residual_scores(self.delta_tr_all, dpred, mu, trows)
                s["resid_PCC"] = r["resid_PCC"]
            if "strain_only" in split:
                mu = self.mu_drug_for(self.m_tr.iloc[trows])
                r = M.residual_scores(self.delta_tr_all, dpred, mu, trows)
                s["resid_PCC"] = r["resid_PCC"]
            if not fast:
                s.update(M.dep_scores(self.delta_tr_all, dpred, trows))
            per_split[split] = s
        total = M.composite(per_split)
        if verbose:
            self._print(per_split, total)
        return {"per_split": per_split, "composite": total}

    def _delta_pred(self, rows: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Δ_pred = ŷ_treat - 匹配对照真值均值（对照从 train_val 池取）。"""
        import warnings
        out = np.full((len(rows), pred.shape[1]), np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, r in enumerate(rows):
                sid = self.m_tr["sample_ID"].iloc[r]
                ctrls = [self.idx_tr[c] for c in self.ctrl_map.get(sid, [])
                         if c in self.idx_tr]
                if not ctrls:
                    continue
                cmean = np.nanmean(self.Y_tr[ctrls], axis=0)
                out[i] = pred[r] - cmean
        return out

    # ---- 快速评分（集成搜索用）：预计算全部静态量，每次评估全向量化 ----

    def prepare_fast_eval(self):
        """预计算四个 val 划分的静态参照（对照均值/Δ_true/残差参照）。"""
        import warnings
        self._fast = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 每个 train_val 样本的匹配对照均值（行对齐全 train_val）
            C = np.full_like(self.Y_tr, np.nan)
            for i, sid in enumerate(self.m_tr["sample_ID"]):
                ctrls = [self.idx_tr[c] for c in self.ctrl_map.get(sid, [])
                         if c in self.idx_tr]
                if ctrls:
                    C[i] = np.nanmean(self.Y_tr[ctrls], axis=0)
            for split in ["val_chem_only", "val_strain_only", "val_both", "val_time"]:
                rows = np.where((self.m_tr["split_final"] == split).to_numpy())[0]
                trows = rows[self.is_treat_tr[rows]]
                entry = {
                    "rows": rows, "trows": trows,
                    "yt": self.Y_tr[rows],
                    "C_t": C[trows],
                    "dt": self.delta_tr_all[trows],
                }
                if "chem_only" in split:
                    entry["mu"] = self.mu_ctx_for(self.m_tr.iloc[trows])
                if "strain_only" in split:
                    entry["mu"] = self.mu_drug_for(self.m_tr.iloc[trows])
                self._fast[split] = entry

    def score_fast(self, pred_tr: np.ndarray) -> float:
        """全向量化近似 composite（跳过逐蛋白PCC/DEP），供集成搜索。"""
        if not hasattr(self, "_fast"):
            self.prepare_fast_eval()
        per_split = {}
        for split, e in self._fast.items():
            s = {}
            yp = pred_tr[e["rows"]]
            s_pcc = M._masked_pcc_axis1(e["yt"], yp)
            s_r2 = M._masked_r2_axis1(e["yt"], yp)
            s["fidelity"] = float(np.mean([np.mean(s_pcc),
                                           np.mean(np.clip(s_r2, -1, 1))]))
            dp_t = pred_tr[e["trows"]] - e["C_t"]
            s["FC_PCC"] = float(np.nanmean(M._masked_pcc_axis1(e["dt"], dp_t)))
            if "mu" in e:
                rp = M._masked_pcc_axis1(e["dt"] - e["mu"], dp_t - e["mu"])
                s["resid_PCC"] = float(np.nanmean(rp))
            per_split[split] = s
        return M.composite(per_split)

    @staticmethod
    def _print(per_split: dict, total: float):
        cols = ["fidelity", "sample_PCC", "sample_R2", "protein_PCC",
                "FC_PCC", "resid_PCC", "DEP_dir_acc", "DEP_PCC", "DEP_F1"]
        hdr = f"{'split':<18}" + "".join(f"{c:>12}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for sp, s in per_split.items():
            print(f"{sp:<18}" + "".join(
                f"{s.get(c, float('nan')):>12.4f}" if c in s else f"{'-':>12}"
                for c in cols))
        print(f"\n[composite] {total:.4f}")


def get_model(name: str, **kw):
    from . import baselines
    return {
        "global_mean": baselines.GlobalMean,
        "delta_additive": baselines.DeltaAdditive,
        "ridge": baselines.RidgeMulti,
    }[name](**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="global_mean")
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    h = Harness()
    if args.model == "global_mean":
        model = get_model(args.model)
        model.fit(h)
    elif args.model == "ridge":
        model = get_model(args.model, lam=args.lam)
        model.fit(h)
    else:
        model = get_model(args.model)
        model.fit(h)
    pred = model.predict(h.m_tr)
    h.score_val(pred)


if __name__ == "__main__":
    main()
