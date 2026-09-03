"""wsN23: fuse3 = RDKit ⊕ ChemBERTa ⊕ MolFormer-XL 三源融合 + deep3 训练。

MolFormer-XL（768 维，ibm/MoLFormer-XL-both-10pct，2026-08-12 经 hf-mirror
下载，cache/molformer/）为第三表征源；三源拼接后按 wsA 协议标准化+PCA64
（val 版 fit 37 train 化合物），deep3 架构（1024,2048,2048, p_drop 0.2）
16 种子 val 对照 deep3_s16（fuse 双源）0.5390。

用法: python -m src.wsN23_fuse3            # val 对照
      python -m src.wsN23_fuse3 --test     # full 协议 test 预测
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
from .wsN6_chemberta import table_to_loader

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN23"
SEEDS = list(range(16))
HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2


@torch.no_grad()
def molformer_embed(smiles, device="cuda"):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("cache/molformer",
                                        trust_remote_code=True)
    model = AutoModel.from_pretrained("cache/molformer",
                                      trust_remote_code=True).to(device).eval()
    outs = []
    for i in range(0, len(smiles), 16):
        enc = tok([str(s) for s in smiles[i:i + 16]], padding=True,
                  truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        outs.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def build_fuse3(fit_names, cache):
    from sklearn.decomposition import PCA
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    # 三源：RDKit(2048+16) + ChemBERTa(384) + MolFormer(768)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    fps = []
    for smi_s in smi["smiles"]:
        mol = Chem.MolFromSmiles(str(smi_s).strip())
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        bits = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, bits)
        fps.append(bits)
    E_rdkit = np.stack(fps)
    from .wsN6_chemberta import chemberta_embed
    E_cb = chemberta_embed(smi["smiles"].tolist())
    E_mf = molformer_embed(smi["smiles"].tolist())
    E = np.concatenate([E_rdkit, E_cb, E_mf], axis=1)
    train_idx = np.array([i for i, c in enumerate(names) if c in fit_names])
    mu, sd = E[train_idx].mean(0), E[train_idx].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (E - mu) / sd
    n_comp = min(64, len(train_idx) - 1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    if args.test:
        fit_names = set(h.m_tr.loc[
            h.m_tr["perturbation_no_concentration"].notna(),
            "perturbation_no_concentration"].unique()) - D.CONTROLS - {D.QC}
        cache = OUT / "chem_features_fuse3_full.csv"
    else:
        fit_names = set(h.m_tr.loc[h.m_tr["chemical_role"] == "train",
                                   "perturbation_no_concentration"].unique())
        cache = OUT / "chem_features_fuse3.csv"
    df = pd.read_csv(cache) if cache.exists() else build_fuse3(fit_names,
                                                               cache)

    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot, hidden=HIDDEN,
                  p_drop=P_DROP)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    try:
        pv, pt = [], []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, 100, seed=sd, emb_drop=0.25, chem_drop=0.25,
                lr=1e-3, bs=256, device="cuda", log_every=999,
                full=args.test)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            if args.test:
                pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                      device="cuda"))
            print(f"[fuse3] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    for arr in [P]:
        bad = ~np.isfinite(arr)
        if bad.any():
            r, c = np.where(bad)
            arr[r, c] = np.take(h.stats.protein_mean, c)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[fuse3-deep3-s16 {'full' if args.test else 'val'}] "
          f"composite={res['composite']:.4f} FC={fc} resid={rz}")
    np.save(OUT / ("pred_trainval.npy" if not args.test
                   else "pred_trainval_fullproto.npy"), P)
    if args.test:
        PT = np.mean(pt, axis=0).astype(np.float32)
        bad = ~np.isfinite(PT)
        if bad.any():
            r, c = np.where(bad)
            PT[r, c] = np.take(h.stats.protein_mean, c)
        np.save(OUT / "pred_test.npy", PT)
    (OUT / ("scores_test.json" if args.test else "scores.json")).write_text(
        json.dumps({"composite": res["composite"], "FC": fc, "resid": rz,
                    "per_split": res["per_split"]}, default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
