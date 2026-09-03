"""wsN: 近邻响应迁移（化合物 Tanimoto kNN + 菌株 SNP 近邻）。

动机（出题人 8.10 分享会，outref/meeting_结论总结.md）：
"来一个新药，找到训练集最像的药，拿这个训练级的药去作为它的预测结果——
本质上也是统计方法；很多深度学习模型其实没打过这种统计模型。"
wsK 已证伪"基因组统计特征进 MLP"，但**直接的近邻 Δ 迁移**这一形态从未试过；
CRD(test 菌株) 与 CGD(train) SNP 距离仅 0.398%，近邻迁移对 strain 侧可能真实有效。

模型（纯统计，无训练；全部统计仅 train split 冻结）：
  ŷ(row) = control_hat(ctx) + Δ̂(strain, chem)
  Δ̂(s, c) = Σ_k w_k · [Δ̄_{c_k} + m_{ŝ,c_k}]
    - c 已见：w 为 onehot；c 未见：w_k ∝ Tanimoto(c, c_k)^α（train 化合物 kNN）
    - ŝ = s（已见菌株）或最近邻 train 菌株（未见菌株，按 1011 SNP 距离）
    - Δ̄_c = train 中化合物 c 的平均 Δ（strict：仅 train 对照池）
    - m_{s,c} = (s,c) 对均值 − Δ̄_c，收缩 m *= n/(n+8)（pair 样本量约 34）
  control_hat：wsB GroupMeanControl（train 对照 751 行，级联回退）

外部数据披露：PubChem SMILES（outputs/wsA/smiles.csv，RDKit 指纹）+
1011 酵母基因组计划距离矩阵（outputs/wsK/genomes/）——均已登记，单轨允许。

合规：h.Y_te 零接触；val 仅用于模型选择（α/K/收缩等超参）；train-only 统计。

用法:
    python -m src.wsN_transfer            # 全量：val 评分 + test 预测 + 报告
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .evaluate import Harness
from .wsB_twostage import GroupMeanControl
from .wsM_trainonly import strict_delta_train

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN"

CHEM = "perturbation_no_concentration"


# ---------------------------------------------------------------- 相似度

def chem_tanimoto(smiles_csv: Path) -> pd.DataFrame:
    """全部化合物（含 test）两两 Morgan 指纹 Tanimoto 相似度矩阵。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    df = pd.read_csv(smiles_csv)
    names = df["compound"].tolist()
    fps = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smi).strip())
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                   if mol else None)
    n = len(fps)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i + 1, n):
            if fps[j] is None:
                continue
            S[i, j] = S[j, i] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    np.fill_diagonal(S, 1.0)
    return pd.DataFrame(S, index=names, columns=names)


def strain_snp_dist() -> pd.DataFrame:
    """6 个赛题菌株间的 1011 SNP 距离子矩阵（原始比例单位）。"""
    import gzip
    p = next((Path(__file__).resolve().parent.parent
              / "outputs" / "wsK" / "genomes").glob(
                  "1011DistanceMatrixBasedOnSNPs.tab.gz"))
    with gzip.open(p, "rt") as f:
        mat = pd.read_csv(f, sep="\t", index_col=0)
    strains = ["BAH", "BAI", "CEK", "CGD", "CRD", "DHY210"]
    strains = [s for s in strains if s in mat.index]
    return mat.loc[strains, strains].astype(float)


# ---------------------------------------------------------------- 核心模型

class TransferModel:
    """近邻 Δ 迁移统计模型。fit 仅用 train split。"""

    def __init__(self, alpha: float = 4.0, shrink: float = 8.0,
                 strain_mode: str = "nn1"):
        self.alpha = alpha          # Tanimoto 锐化指数
        self.shrink = shrink        # pair 项收缩强度
        self.strain_mode = strain_mode  # nn1 | soft | none(未见菌株无调制)

    def fit(self, h: Harness, sim: pd.DataFrame, sdist: pd.DataFrame):
        m_train = h.m_train
        delta, treat_all, treat_valid = strict_delta_train(h)
        m_t = m_train.iloc[treat_valid]
        d_t = delta[treat_valid]
        self.train_chems = sorted(m_t[CHEM].unique().tolist())
        self.train_strains = sorted(m_t["Strains"].unique().tolist())

        # Δ̄_c（每化合物均值）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.delta_chem = {c: np.nanmean(d_t[(m_t[CHEM] == c).to_numpy()],
                                           axis=0)
                               for c in self.train_chems}
            # (s,c) pair 均值与样本量
            self.pair_mu, self.pair_n = {}, {}
            for (s, c), sub in m_t.groupby(["Strains", CHEM]):
                idx = sub.index.to_numpy()
                rel = np.where(np.isin(treat_valid, idx))[0]
                self.pair_mu[(s, c)] = np.nanmean(d_t[rel], axis=0)
                self.pair_n[(s, c)] = len(rel)
        self.delta_glob = np.nan_to_num(
            np.nanmean(d_t, axis=0), nan=0.0).astype(np.float32)
        for c in self.train_chems:  # 全 NaN 防御
            self.delta_chem[c] = np.nan_to_num(self.delta_chem[c], nan=0.0) \
                if not np.isnan(self.delta_chem[c]).all() else self.delta_glob

        # control_hat（train 对照）
        is_ctrl = m_train[CHEM].isin(D.CONTROLS).to_numpy()
        self.ctrl = GroupMeanControl().fit(
            m_train[is_ctrl].reset_index(drop=True), h.Y_train[is_ctrl],
            h.stats.protein_mean)
        # QC 行：train QC 全局均值
        is_qc = (m_train[CHEM] == D.QC).to_numpy()
        self.qc_mean = np.nan_to_num(np.nanmean(h.Y_train[is_qc], axis=0),
                                     nan=0.0) if is_qc.any() else None

        # 相似度资源
        self.sim = sim
        self.sdist = sdist
        self.strain_nn = {}
        snp_strains = [t for t in self.train_strains if t in sdist.columns]
        for s in sdist.index:
            if s in self.train_strains:
                continue
            d = sdist.loc[s, snp_strains]
            self.strain_nn[s] = d.idxmin()
        print(f"[wsN] strain 最近邻: {self.strain_nn}")
        return self

    # ---- 内部 ----

    def _chem_weights(self, c: str) -> tuple[list, np.ndarray]:
        if c in self.train_chems:
            return [c], np.array([1.0])
        if c not in self.sim.index:
            return self.train_chems, np.full(len(self.train_chems),
                                             1.0 / len(self.train_chems))
        t = self.sim.loc[c, self.train_chems].to_numpy(dtype=np.float64)
        t = np.clip(t, 0, 1) ** self.alpha
        if t.sum() < 1e-9:
            t = np.ones_like(t)
        return self.train_chems, t / t.sum()

    def _strain_of(self, s: str) -> str | None:
        if s in self.train_strains:
            return s
        if self.strain_mode == "none":
            return None
        return self.strain_nn.get(s)

    def delta_hat(self, s: str, c: str) -> np.ndarray:
        chems, w = self._chem_weights(c)
        shat = self._strain_of(s)
        out = np.zeros_like(self.delta_glob)
        for ck, wk in zip(chems, w):
            base = self.delta_chem[ck]
            if shat is not None and (shat, ck) in self.pair_mu:
                pmu = self.pair_mu[(shat, ck)]
                ok = np.isfinite(pmu)  # pair 内全缺失的蛋白不参与调制
                adj = base.copy()
                adj[ok] = base[ok] + (self.pair_n[(shat, ck)]
                                      / (self.pair_n[(shat, ck)] + self.shrink)
                                      ) * (pmu[ok] - base[ok])
                base = adj
            out = out + wk * base
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def predict(self, m: pd.DataFrame) -> np.ndarray:
        base = self.ctrl.predict(m)
        out = base.copy()
        pert = m[CHEM]
        is_treat = ~pert.isin(D.CONTROLS | {D.QC}).to_numpy()
        for i, row in enumerate(m.itertuples()):
            if not is_treat[i]:
                continue
            out[i] = base[i] + self.delta_hat(row.Strains,
                                              row.perturbation_no_concentration)
        if self.qc_mean is not None:
            out[(pert == D.QC).to_numpy()] = self.qc_mean
        # 交付契约：非有限值回退 train 蛋白均值
        bad = ~np.isfinite(out)
        if bad.any():
            fb = np.broadcast_to(self.ctrl.global_, out.shape)
            out = np.where(bad, fb, out)
        return out.astype(np.float32)


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    sim = chem_tanimoto(Path("outputs/wsA/smiles.csv"))
    sdist = strain_snp_dist()
    print(f"[wsN] chem sim {sim.shape} | strain dist {sdist.shape} "
          f"({time.time()-t0:.0f}s)")

    results = {}
    preds_val, preds_test = {}, {}
    # 超参网格（val 模型选择）：α × 菌株模式
    for alpha in (2.0, 4.0, 8.0):
        for smode in ("nn1", "none"):
            tag = f"a{alpha:g}_{smode}"
            model = TransferModel(alpha=alpha, strain_mode=smode).fit(
                h, sim, sdist)
            pv = model.predict(h.m_tr)
            res = h.score_val(pv, verbose=False)
            results[tag] = res
            fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
                  for sp in res["per_split"]}
            rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
                  for sp in res["per_split"]}
            print(f"[{tag}] composite={res['composite']:.4f} FC={fc} resid={rz}",
                  flush=True)
            preds_val[tag] = pv
            if smode == "nn1":  # 每个 α 档存一份 test 预测备用
                preds_test[tag] = model.predict(h.m_te)

    # 选最优（按 composite，val 模型选择）
    best = max(results, key=lambda k: results[k]["composite"])
    print(f"\n[wsN] best={best} composite={results[best]['composite']:.4f}")
    np.save(OUT / "pred_trainval.npy", preds_val[best].astype(np.float32))
    # 最优 α 的 nn1 版 test 预测
    tkey = best if best in preds_test else best.replace("none", "nn1")
    if tkey not in preds_test:
        model = TransferModel(
            alpha=float(best.split("_")[0][1:]), strain_mode="nn1").fit(
            h, sim, sdist)
        preds_test[tkey] = model.predict(h.m_te)
    np.save(OUT / "pred_test.npy", preds_test[tkey].astype(np.float32))
    (OUT / "scores.json").write_text(json.dumps(
        {"best": best,
         "results": {k: {"composite": v["composite"],
                         "per_split": v["per_split"]}
                     for k, v in results.items()}},
        ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT/'pred_trainval.npy'} + pred_test.npy + scores.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
