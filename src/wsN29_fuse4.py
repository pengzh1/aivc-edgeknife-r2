"""wsN29: fuse4 = RDKit ⊕ ChemBERTa ⊕ MolFormer-XL ⊕ ChemBERTa-MTR 四源融合。

第四表征源：DeepChem/ChemBERTa-77M-MTR（多任务回归版，199 维分子性质回归头，
hidden 384 / 3 层；"77M"指预训练分子数非参数量）。2026-08-13 经 hf-mirror
下载，cache/chemberta_mtr/（pytorch_model.bin 14,036,589 B 已校验完整，
legacy pickle 格式）。MTR 与 MLM 预训练任务互补（性质回归 vs 掩码重建），
预期携带更偏理化的表征。

协议与 wsN23 完全对齐（公平对照）：四源拼接 → train 化合物拟合标准化 →
PCA64 → deep3 (1024,2048,2048) p_drop 0.2 → 16 种子。
对照基线：fuse3-100ep 0.5397 / fuse3-e150 0.5412（当前最强单族）。

用法: python -m src.wsN29_fuse4              # val 对照（100ep）
      python -m src.wsN29_fuse4 --epochs 150 # e150 对照
      python -m src.wsN29_fuse4 --test       # full 协议 test 预测（胜出入路由时）
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
from .wsN6_chemberta import table_to_loader, chemberta_embed
from .wsN23_fuse3 import molformer_embed

OUT = Path(__file__).resolve().parent.parent / "outputs" / "wsN29"
CKPT_MTR = Path(__file__).resolve().parent.parent / "cache" / "chemberta_mtr"
SEEDS = list(range(16))
HIDDEN = (1024, 2048, 2048)
P_DROP = 0.2


@torch.no_grad()
def mtr_embed(smiles, device="cuda"):
    """SMILES → ChemBERTa-77M-MTR mean-pool 表征（384 维）。

    DeepChem 的 RobertaForRegression 非 transformers 标准类（键为
    roberta.* + regression.*），手动剥离前缀载入 RobertaModel 编码器。
    """
    from transformers import AutoConfig, AutoTokenizer, RobertaModel
    tok = AutoTokenizer.from_pretrained(CKPT_MTR)
    cfg = AutoConfig.from_pretrained(CKPT_MTR)
    try:
        sd = torch.load(CKPT_MTR / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
    except Exception:
        sd = torch.load(CKPT_MTR / "pytorch_model.bin", map_location="cpu",
                        weights_only=False)
    enc_sd = {k[len("roberta."):]: v for k, v in sd.items()
              if k.startswith("roberta.")}
    model = RobertaModel(cfg)
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    real_missing = [k for k in missing
                    if not k.endswith("position_ids") and "pooler" not in k]
    assert not real_missing, f"MTR 编码器权重缺失: {real_missing}"
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(smiles), 16):
        batch = tok([str(s) for s in smiles[i:i + 16]], padding=True,
                    truncation=True, max_length=256, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        h = model(**batch).last_hidden_state  # (B, L, 384)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        outs.append(emb.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def build_fuse4(fit_names, cache):
    from sklearn.decomposition import PCA
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    smi = pd.read_csv("outputs/wsA/smiles.csv")
    names = smi["compound"].tolist()
    # 四源：RDKit(2048+16) + ChemBERTa-MLM(384) + MolFormer(768) + MTR(384)
    fps = []
    for smi_s in smi["smiles"]:
        mol = Chem.MolFromSmiles(str(smi_s).strip())
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        bits = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, bits)
        fps.append(bits)
    E_rdkit = np.stack(fps)
    E_cb = chemberta_embed(smi["smiles"].tolist())
    E_mf = molformer_embed(smi["smiles"].tolist())
    E_mtr = mtr_embed(smi["smiles"].tolist())
    E = np.concatenate([E_rdkit, E_cb, E_mf, E_mtr], axis=1)
    print(f"[fuse4] 四源拼接 {E.shape}（RDKit {E_rdkit.shape[1]} + "
          f"MLM {E_cb.shape[1]} + MolFormer {E_mf.shape[1]} + "
          f"MTR {E_mtr.shape[1]}）", flush=True)
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
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    if args.test:
        fit_names = set(h.m_tr.loc[
            h.m_tr["perturbation_no_concentration"].notna(),
            "perturbation_no_concentration"].unique()) - D.CONTROLS - {D.QC}
        cache = OUT / "chem_features_fuse4_full.csv"
    else:
        fit_names = set(h.m_tr.loc[h.m_tr["chemical_role"] == "train",
                                   "perturbation_no_concentration"].unique())
        cache = OUT / "chem_features_fuse4.csv"
    df = pd.read_csv(cache) if cache.exists() else build_fuse4(fit_names,
                                                               cache)

    orig_init = WSA.ProteoMLPChem.__init__

    def new_init(self, n_cats, chem_mat, n_prot):
        orig_init(self, n_cats, chem_mat, n_prot, hidden=HIDDEN,
                  p_drop=P_DROP)
    WSA.ProteoMLPChem.__init__ = new_init
    orig_loader = WSA.load_chem_table
    WSA.load_chem_table = table_to_loader(df)
    tag = f"fuse4-e{args.epochs}"
    try:
        pv, pt = [], []
        for sd in SEEDS:
            t0 = time.time()
            model, enc, mean, std = WSA.train_model(
                h, h.tr_rows, args.epochs, seed=sd, emb_drop=0.25,
                chem_drop=0.25, lr=1e-3, bs=256, device="cuda",
                log_every=999, full=args.test)
            pv.append(WSA.predict(model, enc, mean, std, h.m_tr,
                                  device="cuda"))
            if args.test:
                pt.append(WSA.predict(model, enc, mean, std, h.m_te,
                                      device="cuda"))
            print(f"[{tag}] seed={sd} ({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        WSA.ProteoMLPChem.__init__ = orig_init
        WSA.load_chem_table = orig_loader
    P = np.mean(pv, axis=0).astype(np.float32)
    bad = ~np.isfinite(P)
    if bad.any():
        r, c = np.where(bad)
        P[r, c] = np.take(h.stats.protein_mean, c)
    res = h.score_val(P, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    rz = {sp: round(res["per_split"][sp].get("resid_PCC", 0.0), 4)
          for sp in res["per_split"]}
    print(f"[{tag}-s16 {'full' if args.test else 'val'}] "
          f"composite={res['composite']:.4f} FC={fc} resid={rz}")
    suffix = f"e{args.epochs}" if args.epochs != 100 else ""
    np.save(OUT / (f"pred_trainval{suffix}.npy" if not args.test
                   else f"pred_trainval_fullproto{suffix}.npy"), P)
    if args.test:
        PT = np.mean(pt, axis=0).astype(np.float32)
        bad = ~np.isfinite(PT)
        if bad.any():
            r, c = np.where(bad)
            PT[r, c] = np.take(h.stats.protein_mean, c)
        np.save(OUT / f"pred_test{suffix}.npy", PT)
    (OUT / (f"scores_test{suffix}.json" if args.test
            else f"scores{suffix}.json")).write_text(json.dumps(
        {"composite": res["composite"], "FC": fc, "resid": rz,
         "per_split": res["per_split"],
         "baselines": {"fuse3_100ep": 0.5397, "fuse3_e150": 0.5412}},
        default=float, indent=1))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
