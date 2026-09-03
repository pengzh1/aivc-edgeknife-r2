"""基线模型。

- GlobalMean: 每蛋白 train 均值（官方均值基线的本地版）
- DeltaAdditive: ŷ = control_hat(上下文+批次) + Δ̂，Δ̂ 按
  Δ = a[菌株] + b[化合物] + c[培养基×温度×时间] + d[菌株×化合物]
  在 train Δ_true 上 backfitting（均值平滑），天然支持未见实体的回退
- RidgeMulti: 条件 one-hot + 交互特征的多输出岭回归（闭式解）
"""
import numpy as np
import pandas as pd

from . import data as D


class GlobalMean:
    def fit(self, h):
        self.mu = h.stats.protein_mean
        return self

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mu[None, :], len(m), axis=0).astype(np.float32)


# ---------------- DeltaAdditive ----------------


def _group_mean_vecs(keys, R, shrink: float = 0.0):
    """keys: array of tuples; R: (n, p) 残差矩阵。返回 {key: mean_vec}。"""
    df = pd.DataFrame({"k": keys})
    out = {}
    for k, idx in df.groupby("k").groups.items():
        idx = np.fromiter(idx, dtype=int)
        v = np.nanmean(R[idx], axis=0)
        if shrink > 0:
            v = v * len(idx) / (len(idx) + shrink)
        out[k] = np.nan_to_num(v, nan=0.0)
    return out


class DeltaAdditive:
    """对照基线 + 加性 Δ 分解。"""

    N_ITER = 10
    SHRINK_PAIR = 4.0   # d[菌株×化合物] 交互的收缩强度

    def fit(self, h):
        m_train, Y_train = h.m_train, h.Y_train
        is_ctrl = m_train["perturbation_no_concentration"].isin(D.CONTROLS).to_numpy()
        is_tr = ~m_train["perturbation_no_concentration"].isin(
            D.CONTROLS | {D.QC}).to_numpy()

        # ---- 对照基线（分层组均值，含批次键） ----
        mc = m_train[is_ctrl]
        Yc = Y_train[is_ctrl]
        self.ctrl_levels = []
        for keys in [
            ["Strains", "Medium", "Temperature", "pert_time", "instrument"],
            ["Strains", "Medium", "Temperature", "pert_time"],
            ["Strains", "Medium", "Temperature"],
            ["Strains"],
        ]:
            km = list(map(tuple, mc[keys].to_numpy()))
            self.ctrl_levels.append((keys, _group_mean_vecs(km, Yc)))
        self.ctrl_global = np.nan_to_num(np.nanmean(Yc, axis=0), nan=0.0)

        # ---- Δ backfitting ----
        mt = m_train[is_tr].reset_index(drop=True)
        Dt = np.nan_to_num(h.delta_train[is_tr], nan=0.0)
        # 屏蔽全 NaN 行（无对照的，实际没有）
        strains = mt["Strains"].to_numpy()
        chems = mt["perturbation_no_concentration"].to_numpy()
        ctxs = list(zip(mt["Medium"], mt["Temperature"], mt["pert_time"]))
        pairs = list(zip(mt["Strains"], mt["perturbation_no_concentration"]))

        a = {s: np.zeros(Dt.shape[1], np.float32) for s in set(strains)}
        b = {c: np.zeros(Dt.shape[1], np.float32) for c in set(chems)}
        c = {k: np.zeros(Dt.shape[1], np.float32) for k in set(ctxs)}
        d = {p: np.zeros(Dt.shape[1], np.float32) for p in set(pairs)}

        def current(i):
            return a[strains[i]] + b[chems[i]] + c[ctxs[i]] + d[pairs[i]]

        for it in range(self.N_ITER):
            for comp in ["a", "b", "c", "d"]:
                R = Dt.copy()
                for i in range(len(mt)):
                    if comp == "a":
                        R[i] -= b[chems[i]] + c[ctxs[i]] + d[pairs[i]]
                    elif comp == "b":
                        R[i] -= a[strains[i]] + c[ctxs[i]] + d[pairs[i]]
                    elif comp == "c":
                        R[i] -= a[strains[i]] + b[chems[i]] + d[pairs[i]]
                    else:
                        R[i] -= a[strains[i]] + b[chems[i]] + c[ctxs[i]]
                if comp == "a":
                    a = _group_mean_vecs(strains, R)
                elif comp == "b":
                    b = _group_mean_vecs(chems, R)
                elif comp == "c":
                    c = _group_mean_vecs(ctxs, R)
                else:
                    d = _group_mean_vecs(pairs, R, shrink=self.SHRINK_PAIR)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.zero = np.zeros(Y_train.shape[1], np.float32)
        return self

    def _control_hat(self, m: pd.DataFrame) -> np.ndarray:
        out = np.repeat(self.ctrl_global[None, :], len(m), axis=0)
        hit = np.zeros(len(m), dtype=bool)
        for keys, gm in self.ctrl_levels:
            for i, row in enumerate(m.itertuples()):
                if hit[i]:
                    continue
                k = tuple(getattr(row, kk) for kk in keys)
                if k in gm:
                    out[i] = gm[k]
                    hit[i] = True
        return out.astype(np.float32)

    def delta_hat(self, m: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(m), len(self.zero)), dtype=np.float32)
        for i, row in enumerate(m.itertuples()):
            s, ch = row.Strains, row.perturbation_no_concentration
            ctx = (row.Medium, row.Temperature, row.pert_time)
            out[i] = (self.a.get(s, self.zero) + self.b.get(ch, self.zero)
                      + self.c.get(ctx, self.zero)
                      + self.d.get((s, ch), self.zero))
        return out

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        base = self._control_hat(m)
        is_tr = ~m["perturbation_no_concentration"].isin(
            D.CONTROLS | {D.QC}).to_numpy()
        out = base.copy()
        out[is_tr] += self.delta_hat(m[is_tr])
        return out.astype(np.float32)


# ---------------- RidgeMulti ----------------


class RidgeMulti:
    """one-hot 条件特征 + 交互，闭式多输出岭回归。"""

    def __init__(self, lam: float = 10.0):
        self.lam = lam

    def _fit_encoders(self, m: pd.DataFrame):
        self.cats = {}
        for col in ["Strains", "perturbation_no_concentration", "Medium",
                    "Temperature", "pert_time", "instrument", "data_source",
                    "Yeast_cell_plate"]:
            self.cats[col] = {v: i for i, v in
                              enumerate(pd.unique(m[col]))}
        self.chems = self.cats["perturbation_no_concentration"]
        self.times = self.cats["pert_time"]
        self.strains = self.cats["Strains"]

    def featurize(self, m: pd.DataFrame) -> np.ndarray:
        n = len(m)
        blocks = []
        for col in ["Strains", "perturbation_no_concentration", "Medium",
                    "Temperature", "pert_time", "instrument", "data_source",
                    "Yeast_cell_plate"]:
            cmap = self.cats[col]
            X = np.zeros((n, len(cmap)), dtype=np.float32)
            for i, v in enumerate(m[col]):
                j = cmap.get(v)
                if j is not None:
                    X[i, j] = 1.0
            blocks.append(X)
        # chem × time 交互
        Xct = np.zeros((n, len(self.chems) * len(self.times)), np.float32)
        nt = len(self.times)
        for i, (ch, t) in enumerate(zip(m["perturbation_no_concentration"],
                                        m["pert_time"])):
            jc, jt = self.chems.get(ch), self.times.get(t)
            if jc is not None and jt is not None:
                Xct[i, jc * nt + jt] = 1.0
        blocks.append(Xct)
        # strain × chem 交互
        Xsc = np.zeros((n, len(self.strains) * len(self.chems)), np.float32)
        nc = len(self.chems)
        for i, (s, ch) in enumerate(zip(m["Strains"],
                                        m["perturbation_no_concentration"])):
            js, jc = self.strains.get(s), self.chems.get(ch)
            if js is not None and jc is not None:
                Xsc[i, js * nc + jc] = 1.0
        blocks.append(Xsc)
        blocks.append(np.ones((n, 1), np.float32))
        return np.concatenate(blocks, axis=1)

    def fit(self, h, rows: np.ndarray | None = None, stats=None):
        if rows is None:
            rows = h.tr_rows
        stats = stats or h.stats
        m_fit = h.m_tr.iloc[rows].reset_index(drop=True)
        self._fit_encoders(h.m_tr)  # 编码器覆盖全部 train_val 类别
        X = self.featurize(m_fit)
        Y = stats.impute(h.Y_tr[rows])
        XtX = X.T @ X
        reg = self.lam * np.eye(XtX.shape[0], dtype=np.float32)
        reg[-1, -1] = 1e-6  # 截距不正则
        self.W = np.linalg.solve(XtX + reg, X.T @ Y).astype(np.float32)
        return self

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        return (self.featurize(m) @ self.W).astype(np.float32)
