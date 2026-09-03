"""wsQ: DNABERT-2 基因组窗口 embedding → 菌株特征（S2 上限第九路检验）。

动机：wsO 证明 k-mer 组成特征单族强（0.5434/strain FC 0.3345）但集成零边际
——组成统计已达上限。DNABERT-2（zhihan1996/DNABERT-2-117M，2026-08-14 经
hf-mirror 下载 cache/dnabert2/，447MB 校验完整）是**上下文序列表征**：
窗口 embedding 携带基因内容/调控结构信息，与 k-mer 属不同表征族。
检验问题：该表征族在 strain 侧是否仍有 wsB 统计路径未覆盖的信息。

管线：6 株基因组 → 500bp 非重叠窗口（N>10% 剔除）→ 每株种子固定抽 3000
窗口 → DNABERT-2 mean-pool(768) → 株级 mean+std(1536) → 株间标准化 →
PCA5 → wsK 同款 ProteoMLPStrain（wsD 配方 300ep Huber G2）8 种子 val。

兼容修复（已写入 cache/dnabert2/dl.log）：transformers≥5 下 alibi 预计算
撞 ambient meta-device，源文件 bert_layers.py 打 device='cpu' 补丁；
einops 依赖已装。

对照基线：wsK 0.5330（strain FC 0.3102）/ wsO 0.5434（0.3345）/
路由 strain FC 0.351。判定同前：val 无显著增益即关闭归档复赛素材；
有增益再过 wsN30 式边际扫描定入列。

用法: python -m src.wsQ_dnastrain                # 建特征 + 8 种子 val
      python -m src.wsQ_dnastrain --build-only   # 只建特征表
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
from .wsK_strainfeat import DEFAULT_CFG, load_strain_mat, predict_g3, train_one, _finalize
from .wsD_arch import seen_cats
from .wsO_genomefeat import (FASTA_DIR, STRAIN_FASTA, TRAIN_STRAINS, N_PCA,
                             _read_fasta)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsQ"
CKPT = Path(__file__).resolve().parent.parent / "cache" / "dnabert2"
WIN = 500
N_WIN = 3000
MAX_N_FRAC = 0.10


def genome_windows(seqs, win=WIN):
    """非重叠窗口切片（剔除 N 比例过高窗口）。"""
    out = []
    for s in seqs:
        su = s.upper()
        for i in range(0, len(su) - win + 1, win):
            w = su[i:i + win]
            if (w.count("N") + len(w) - sum(w.count(b) for b in "ACGT")) \
                    <= win * MAX_N_FRAC:
                out.append(w)
    return out


@torch.no_grad()
def dnabert_embed(seqs, device="cuda", bs=64):
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    cfg.pad_token_id = tok.pad_token_id
    model = AutoModel.from_pretrained(
        CKPT, config=cfg, trust_remote_code=True).to(device).eval()
    outs = []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], return_tensors="pt", padding=True).to(device)
        h = model(**enc)[0]  # (B, L, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        outs.append(emb.float().cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"  embed {i}/{len(seqs)}", flush=True)
    return np.concatenate(outs).astype(np.float32)


def build_features():
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(20260814)
    rows = {}
    for strain, fn in STRAIN_FASTA.items():
        t0 = time.time()
        wins = genome_windows(_read_fasta(FASTA_DIR / fn))
        idx = rng.permutation(len(wins))[:N_WIN]
        sel = [wins[i] for i in sorted(idx)]
        E = dnabert_embed(sel)
        rows[strain] = np.concatenate([E.mean(0), E.std(0)])  # 1536
        print(f"[wsQ] {strain}: {len(wins)} 窗口→抽{len(sel)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    names = list(rows)
    X = np.stack([rows[n] for n in names])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    pca = PCA(n_components=N_PCA, svd_solver="full", random_state=0)
    P = pca.fit_transform(Z)
    pm, ps = P.mean(0), P.std(0)
    ps[ps < 1e-12] = 1.0
    P = (P - pm) / ps
    print(f"[wsQ] PCA 解释率: "
          f"{np.round(pca.explained_variance_ratio_, 3).tolist()}")
    df = pd.DataFrame(P, index=names,
                      columns=[f"dnabert_pc{i}" for i in range(N_PCA)])
    df.loc["UNK_MEAN"] = df.loc[TRAIN_STRAINS].mean(0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache = OUT_DIR / "strain_features_dnabert.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
    else:
        df = build_features()
        df.to_csv(cache)
        print(f"[saved] {cache} {df.shape}")
    print(f"[wsQ] 特征表 {df.shape}")
    if args.build_only:
        return

    cfg = dict(DEFAULT_CFG, epochs=args.epochs)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = Harness()
    strain_mat, row_of = load_strain_mat(None, cache)
    seen = seen_cats(h, h.tr_rows)
    preds = []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std = train_one(
            h, h.tr_rows, cfg, s, strain_mat, row_of, device=device,
            log_every=999)
        print(f"[wsQ seed {s}] {time.time()-t0:.0f}s", flush=True)
        preds.append(predict_g3(model, enc, mean, std, h.m_tr, seen,
                                strain_mat, row_of, device=device))
    pred = _finalize(np.mean(preds, axis=0), h.stats.protein_mean)
    np.save(OUT_DIR / "pred_trainval.npy", pred)
    res = h.score_val(pred, verbose=False)
    fc = {sp: round(res["per_split"][sp]["FC_PCC"], 4)
          for sp in res["per_split"]}
    print(f"[wsQ dnabert] composite={res['composite']:.4f} FC={fc}")
    print("[wsQ] 基线: wsK 0.5330(0.3102) / wsO 0.5434(0.3345) / "
          "路由 strain 0.351")
    scores_path = OUT_DIR / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    scores["wsQ_dnabert"] = {
        "composite": res["composite"], "FC": fc,
        "per_split": res["per_split"],
        "baselines": {"wsK": 0.5330, "wsO": 0.5434, "router_strain_FC": 0.351}}
    scores_path.write_text(json.dumps(scores, ensure_ascii=False,
                                      indent=1, default=float))
    print(f"[saved] {scores_path}")


if __name__ == "__main__":
    main()
