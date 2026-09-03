"""wsN6: ChemBERTa 预训练分子表征 → wsA 管线变体（E2）。

动机：wsA 的 RDKit 手工描述符已证明有效（shuffle 消融 −32% FC），预训练分子
表征（ChemBERTa-77M-MLM，77M SMILES 自监督）携带与手工描述符互补的
药效团/骨架信息；进阶教程 5.3.2 表征梯度终点即"预训练分子表示"。

协议与 wsA 完全对齐（公平对照）：
- 特征：SMILES → ChemBERTa last-hidden mean-pool（384 维）→ 标准化 →
  PCA 64 维（仅 train 37 个化合物拟合，val 版）/ 43 个（full 版）
- 训练：wsA.train_model 原配方（100ep × 5 种子，train-only 行）
- 对照：wsA RDKit 版 composite 0.5102 / chem FC 0.490
- 变体：bert（仅 ChemBERTa）、fuse（RDKit64 + ChemBERTa64 拼接 128 维）

外部数据披露：DeepChem/ChemBERTa-77M-MLM（HuggingFace，2026-08-11 经
hf-mirror 下载，缓存 cache/chemberta/）；SMILES 来源同 wsA（PubChem）。

合规：train-only 训练；val 仅模型选择；monkey-patch 进程内生效，不改原文件。

用法: python -m src.wsN6_chemberta            # val 对照实验
      python -m src.wsN6_chemberta --test      # 若胜出：full 版 test 预测
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import Harness
from . import data as D
from . import wsA_chemfeat as WSA

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN6"
CKPT = Path(__file__).resolve().parent.parent / "cache" / "chemberta"
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100


@torch.no_grad()
def chemberta_embed(smiles: list[str], device="cuda") -> np.ndarray:
    """SMILES → ChemBERTa mean-pool 表征（384 维）。"""
    from transformers import AutoTokenizer, RobertaForMaskedLM
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = RobertaForMaskedLM.from_pretrained(CKPT).to(device).eval()
    outs = []
    for i in range(0, len(smiles), 16):
        batch = tok([str(s) for s in smiles[i:i + 16]], padding=True,
                    truncation=True, max_length=256, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        h = model.roberta(**batch).last_hidden_state  # (B, L, 384)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        outs.append(emb.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def build_bert_table(fit_names: set, cache: Path,
                     rdkit_csv: Path | None = None) -> pd.DataFrame:
    """ChemBERTa 特征表（pc0..63）；rdkit_csv 给定时拼接为 128 维 fuse 表。"""
    from sklearn.decomposition import PCA
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    E = chemberta_embed(smi["smiles"].tolist())
    train_idx = np.array([i for i, c in enumerate(names) if c in fit_names])
    if rdkit_csv is not None:  # fuse：拼接 RDKit 64 维
        R = pd.read_csv(rdkit_csv).set_index("compound").loc[names]
        E = np.concatenate([E, R.to_numpy(dtype=np.float32)], axis=1)
    mu, sd = E[train_idx].mean(0), E[train_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (E - mu) / sd
    n_comp = min(64, len(train_idx) - 1, Z.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(Z[train_idx])
    P = pca.transform(Z)
    pm, ps = P[train_idx].mean(0), P[train_idx].std(0)
    ps[ps < 1e-8] = 1.0
    P = ((P - pm) / ps).astype(np.float32)
    if n_comp < 64:
        P = np.concatenate([P, np.zeros((len(P), 64 - n_comp), np.float32)],
                           axis=1)
    df = pd.DataFrame(P, columns=[f"pc{i}" for i in range(64)])
    df.insert(0, "compound", names)
    df.to_csv(cache, index=False)
    return df


def table_to_loader(df: pd.DataFrame):
    """把特征表包装成 wsA.load_chem_table 的替换函数。"""
    pc_cols = [f"pc{i}" for i in range(64)]
    indexed = df.set_index("compound")
    feat = {name: indexed.loc[name, pc_cols].to_numpy(dtype=np.float32)
            for name in indexed.index}
    mean_vec = np.stack(list(feat.values())).mean(0)

    def loader(h_, out_dir, full=False):
        return feat, mean_vec
    return loader


def run_variant(h: Harness, tag: str, df: pd.DataFrame, test: bool):
    orig = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    try:
        preds_val, preds_test = [], []
        for s in SEEDS:
            t0 = time.time()
            # val 对照用 full=False（与 wsA 0.5102 完全同协议）；
            # --test 时用 full=True（扩展 test 化合物词表+特征）
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, EPOCHS, seed=s, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999, full=test)
            preds_val.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                         device="cuda"))
            if test:
                preds_test.append(WSA.predict(model, enc, mean, std, h.m_te,
                                              device="cuda"))
            print(f"[{tag}] seed={s} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.load_chem_table = orig
    P = np.mean(preds_val, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[{tag}] composite={res['composite']:.4f} FC={fc} resid={rz}",
          flush=True)
    np.save(OUT / f"pred_trainval_{tag}.npy", P)
    out = {"composite": res["composite"], "FC": fc, "resid": rz,
           "per_split": res["per_split"]}
    if test:
        PT = np.mean(preds_test, axis=0).astype(np.float32)
        bad = ~np.isfinite(PT)
        if bad.any():
            r, c = np.where(bad)
            PT[r, c] = np.take(h.stats.protein_mean, c)
        np.save(OUT / f"pred_test_{tag}.npy", PT)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    m_tr = h.m_tr
    if args.test:
        # full 协议：PCA 拟合 train_val 全部 43 化合物（外部结构数据，合规）
        fit_names = set(m_tr.loc[
            m_tr["perturbation_no_concentration"].notna(),
            "perturbation_no_concentration"].unique())
        fit_names -= D.CONTROLS | {D.QC}
        tag_suffix, fu = "fuse", OUT / "chem_features_fuse_full.csv"
        df = pd.read_csv(fu) if fu.exists() else build_bert_table(
            fit_names, fu, rdkit_csv="outputs/wsA/chem_features_full.csv")
        summary = {"fuse_full": run_variant(h, "fuse_full", df, True)}
    else:
        fit_names = set(m_tr.loc[m_tr["chemical_role"] == "train",
                                 "perturbation_no_concentration"].unique())
        tables = {}
        cb = OUT / "chem_features_bert.csv"
        tables["bert"] = pd.read_csv(cb) if cb.exists() else \
            build_bert_table(fit_names, cb)
        fu = OUT / "chem_features_fuse.csv"
        tables["fuse"] = pd.read_csv(fu) if fu.exists() else \
            build_bert_table(fit_names, fu,
                             rdkit_csv="outputs/wsA/chem_features.csv")
        summary = {}
        for tag, df in tables.items():
            summary[tag] = run_variant(h, tag, df, False)
    (OUT / "scores.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT}/scores.json | 对照 wsA(RDKit): 0.5102")


if __name__ == "__main__":
    main()
