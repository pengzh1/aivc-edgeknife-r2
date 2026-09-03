"""wsN30: 候选族边际贡献扫描（wsO / fuse4 → 当前 21 族路由之上）。

不重跑全路由（逐划分稠密搜索 ~1.5h/轮），直接在新 grand_router.json 的
r=0.7 路由 val 预测上做 1 维权重扫描：
- wsO（k-mer 菌株特征族，单族 0.5434 / strain FC 0.3345）：扫
  val_strain_only + val_both 行的混入权重 α（两划分同权，其余行不变）；
- fuse4（四源族，单族 e150 0.5416）：扫 val_chem_only 行混入权重 α。
α 为对既有路由混合的额外掺入比例（混入后该行组预测 = (1-α)·routed + α·cand）。

判定：composite 提升 ≥ +0.0003（约种子噪声量级）才考虑入路由；
否则关闭归档。该扫描与路由同用 val，属模型选择口径；test 零接触。

用法: python -m src.wsN30_blendscan
"""
import json
from pathlib import Path

import numpy as np

from .evaluate import Harness
from . import wsH_router as H
from .wsN11_grandrouter import CANDIDATES, SPLITS

CANDS = {
    "wsO": ("outputs/wsO/pred_trainval_kmer.npy",
            ["val_strain_only", "val_both"]),
    "fuse4": ("outputs/wsN29/pred_trainvale150.npy", ["val_chem_only"]),
}
ALPHAS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.22, 0.30]


def main():
    h = Harness()
    h.prepare_fast_eval()
    gr = json.loads(Path("outputs/wsN11/grand_router.json").read_text())
    names = gr["names"]
    r = 0.7
    w_glob = np.array(gr["w_global"])
    best_w = {sp: np.array(gr["best_w_split"][sp]) for sp in SPLITS}
    w_use = {sp: r * best_w[sp] + (1 - r) * w_glob for sp in SPLITS}

    files = [(n, p) for n, p in CANDIDATES if n in names]
    preds = H.load_preds(files, h)

    routed = np.empty(preds.shape[1:], dtype=np.float32)
    routed[h.tr_rows] = H.blend_rows(preds, w_glob, h.tr_rows)
    for sp in SPLITS:
        rows = h._fast[sp]["rows"]
        routed[rows] = H.blend_rows(preds, w_use[sp], rows)
    base = h.score_val(routed, verbose=False)["composite"]
    print(f"[scan] 21族路由 r=0.7 重建基线 composite={base:.4f}", flush=True)

    report = {"base_composite": base}
    for tag, (path, sps) in CANDS.items():
        if not Path(path).exists():
            print(f"[scan] {tag} 缺 {path}，跳过")
            continue
        pc = np.load(path)
        rows = np.concatenate([h._fast[sp]["rows"] for sp in sps])
        results = {}
        for a in ALPHAS:
            trial = routed.copy()
            trial[rows] = ((1 - a) * routed[rows] + a * pc[rows])
            c = h.score_val(trial, verbose=False)["composite"]
            results[a] = c
            print(f"  [{tag} α={a:.2f} @{ '+'.join(sps) }] "
                  f"composite={c:.4f} (Δ{c-base:+.4f})", flush=True)
        best_a = max(results, key=lambda a: results[a])
        report[tag] = {"splits": sps, "scan": results,
                       "best_alpha": best_a, "best": results[best_a],
                       "delta": results[best_a] - base}
        print(f"[scan] {tag} 最优 α={best_a} Δ={results[best_a]-base:+.4f} "
              f"{'→ 候选入路由' if results[best_a]-base >= 3e-4 else '→ 关闭'}",
              flush=True)
    (Path("outputs/wsN30") ).mkdir(exist_ok=True)
    Path("outputs/wsN30/blendscan.json").write_text(json.dumps(
        report, indent=1, default=float))
    print("[saved] outputs/wsN30/blendscan.json")


if __name__ == "__main__":
    main()
