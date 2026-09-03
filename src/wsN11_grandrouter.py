"""wsN11: 大阵容稠密路由（突破 0.56 攻关）。

相对 wsH v2 / make_submission_trainonly 的升级：
1. 候选矩阵从 306 → 4600+（Dirichlet(1)×2000 + Dirichlet(0.3)×1000 +
   Dirichlet(0.5)×500 + 顶点 + top-50 两两平均 + 最优点坐标精修）；
2. 阵容扩展：既有 7 族 + fuse + wsN7（特征条件）+ wsN9（wsB-G2）+
   wsN10 架构变体（deep6/wide/mse/ep500/drop02）；
3. 收缩档细扫 r ∈ {0.3..0.8}，全部 full score_val 定夺；
4. 防过拟合纪律：锚点沿用 wsH v2 给定点规则；权重拟合仅用 val 预测。

用法:
    python -m src.wsN11_grandrouter                # 全阵容稠密重拟合
    python -m src.wsN11_grandrouter --quick        # 仅既有成员（不等 wsN10）
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .evaluate import Harness
from . import wsH_router as H
from .wsH_router import SplitScorer, blend_rows

OUT = Path("outputs/wsN11")
SEED = 20260806
SPLITS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]

# 全阵容（存在的文件才会纳入；优先 s16/s8 扩容版）
CANDIDATES = [
    ("wsD_s16", "outputs/wsN14/pred_trainval_wsD_s16.npy"),
    ("wsD_g2g3", "outputs/wsD/pred_trainval_g2g3.npy"),
    ("wsG", "outputs/wsG/pred_trainval.npy"),
    ("wsF", "outputs/wsN15/pred_trainval_wsF_s8.npy"),
    ("wsC_g3", "outputs/wsC/pred_trainval_g3.npy"),
    ("wsB", "outputs/wsN16/pred_trainval_wsB_s16.npy"),
    ("wsBg2", "outputs/wsN9/pred_trainval.npy"),
    ("ridge", "outputs/pred_ridge.npy"),
    ("wsA", "outputs/wsA/pred_trainval.npy"),
    ("fuse", "outputs/wsN16/pred_trainval_fuse_s16.npy"),
    ("fuseDeep3", "outputs/wsN20/pred_trainval_deep3_s16.npy"),
    ("fuse3e150", "outputs/wsN24/pred_trainval.npy"),
    ("wsN7", "outputs/wsN7/pred_trainval.npy"),
    ("wsN1", "outputs/wsN/pred_trainval.npy"),
    ("wsD_bag0", "outputs/wsN16/pred_trainval_bag0.npy"),
    ("wsD_bag1", "outputs/wsN16/pred_trainval_bag1.npy"),
    ("wsD_deep6", "outputs/wsN10/pred_trainval_deep6.npy"),
    ("wsD_wide", "outputs/wsN10/pred_trainval_wide.npy"),
    ("wsD_mse", "outputs/wsN10/pred_trainval_mse.npy"),
    ("wsD_ep500", "outputs/wsN10/pred_trainval_ep500.npy"),
    ("wsD_drop02", "outputs/wsN10/pred_trainval_drop02.npy"),
]


def dense_candidates(k: int, rng, n1=700, n03=300, n05=200):
    cands = [np.eye(k)[i] for i in range(k)]
    cands += [rng.dirichlet(np.ones(k)) for _ in range(n1)]
    cands += [rng.dirichlet(np.ones(k) * 0.3) for _ in range(n03)]
    cands += [rng.dirichlet(np.ones(k) * 0.5) for _ in range(n05)]
    return cands


def refine_coordinate(scorer, P_s, w0, iters=2, step=0.05, obj_w=None):
    """从最优点做坐标微调（向顶点方向 ±step 后归一）。"""
    w_fc, w_resid, w_fid = obj_w or (0.45, 0.45, 0.10)

    def obj_of(m):
        return (w_fc * m["FC_PCC"] + w_fid * m["fidelity"]
                + (w_resid * m["resid_PCC"] if "resid_PCC" in m else 0.0))

    def ev(w):
        return obj_of(scorer.evaluate(
            (w @ P_s.reshape(P_s.shape[0], -1)).reshape(
                P_s.shape[1], P_s.shape[2])))

    best_w, best_obj = w0.copy(), ev(w0)
    k = len(w0)
    for _ in range(iters):
        improved = False
        for j in range(k):
            for sgn in (1, -1):
                w = best_w.copy()
                w[j] = max(0.0, w[j] + sgn * step)
                if w.sum() < 1e-9:
                    continue
                w = w / w.sum()
                obj = ev(w)
                if obj > best_obj + 1e-7:
                    best_obj, best_w, improved = obj, w, True
        if not improved:
            break
    return best_w, best_obj


def search_split_dense(scorer, P_s, cand, block=16, obj_w=None):
    """分块批量评估候选权重。

    obj_w: (w_fc, w_resid, w_fid) 目标权重；None 用 wsH 默认 (0.45,0.45,0.10)。
    对齐 composite 偏导（推荐）：chem/strain = (0.0625, 0.20, 0.05)，
    both/time = (0.0875, 0, 0.075)。
    """
    k, n_r, n_p = P_s.shape
    flat = P_s.reshape(k, -1)
    w_fc, w_resid, w_fid = obj_w or (0.45, 0.45, 0.10)

    def obj_of(m):
        return (w_fc * m["FC_PCC"] + w_fid * m["fidelity"]
                + (w_resid * m["resid_PCC"] if "resid_PCC" in m else 0.0))

    def eval_block(ws):
        out = np.empty(len(ws))
        for i in range(0, len(ws), block):
            chunk = ws[i:i + block]
            yp = chunk @ flat
            for j, y in enumerate(yp):
                out[i + j] = obj_of(scorer.evaluate(y.reshape(n_r, n_p)))
        return out

    objs = eval_block(np.array(cand))
    order = np.argsort(-objs)
    best_w, best_obj = cand[order[0]].copy(), float(objs[order[0]])
    tops = [cand[i] for i in order[:30]]
    pairs = [(tops[i] + tops[j]) / 2
             for i in range(len(tops)) for j in range(i + 1, len(tops))]
    objs2 = eval_block(np.array(pairs))
    if len(objs2) and objs2.max() > best_obj:
        best_obj = float(objs2.max())
        best_w = pairs[int(objs2.argmax())].copy()
    best_w, best_obj = refine_coordinate(scorer, P_s, best_w, obj_w=obj_w)
    return best_w, best_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = Harness()
    h.prepare_fast_eval()

    files = [(n, p) for n, p in CANDIDATES if Path(p).exists()]
    if args.quick:
        files = files[:11]
    names = [n for n, _ in files]
    print(f"[grand] 阵容 {len(names)}: {names}", flush=True)
    preds = H.load_preds(files, h)
    k = len(names)

    scorers = {sp: SplitScorer(h, sp) for sp in SPLITS}
    P_splits = {sp: np.ascontiguousarray(preds[:, h._fast[sp]["rows"], :])
                for sp in SPLITS}
    rng = np.random.default_rng(SEED)
    cand = dense_candidates(k, rng)

    # 全局锚点：20 族稠密重搜版（outputs/wsN11/anchor_new.json，full 0.5465）
    _anchor_file = Path("outputs/wsN11/anchor_new.json")
    if _anchor_file.exists():
        _an = json.loads(_anchor_file.read_text())
        _wmap = dict(zip(_an["names"], _an["w_new"]))
        w_glob = np.array([_wmap.get(n, 0.0) for n in names])
        w_glob = w_glob / w_glob.sum()
    else:
        anchor7 = {"wsD_s16": 0.765, "wsB": 0.21, "ridge": 0.024}
        w_glob = np.array([anchor7.get(n, 0.0) for n in names])
        w_glob = w_glob / w_glob.sum()
    all_rows = np.arange(len(h.m_tr))
    full_glob = h.score_val(
        blend_rows(preds, w_glob, all_rows).astype(np.float32),
        verbose=False)["composite"]
    print(f"[grand] 全局锚点(给定) full={full_glob:.4f} "
          f"w={np.round(w_glob, 3)}", flush=True)

    # 逐划分稠密搜索（composite 偏导对齐目标：chem/strain 残差权重 3.2×）
    OBJW = {"val_chem_only": (0.0625, 0.20, 0.05),
            "val_strain_only": (0.0625, 0.20, 0.05),
            "val_both": (0.0875, 0.0, 0.075),
            "val_time": (0.0875, 0.0, 0.075)}
    best_w = {}
    for sp in SPLITS:
        t1 = time.time()
        w_star, obj_star = search_split_dense(
            scorers[sp], P_splits[sp], cand, obj_w=OBJW[sp])
        best_w[sp] = w_star
        print(f"  [{sp}] dense obj={obj_star:.4f} "
              f"w*={ {n: round(float(x),3) for n,x in zip(names,w_star) if x>0.01} }"
              f" ({time.time()-t1:.0f}s)", flush=True)

    # 收缩细扫（full score_val 定夺；r 上限扩到 1.0 供诚实对照）
    results = {}
    for r in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        w_shr = {sp: r * best_w[sp] + (1 - r) * w_glob for sp in SPLITS}
        routed = np.empty(preds.shape[1:], dtype=np.float32)
        routed[h.tr_rows] = blend_rows(preds, w_glob, h.tr_rows)
        for sp in SPLITS:
            rows = h._fast[sp]["rows"]
            routed[rows] = blend_rows(preds, w_shr[sp], rows)
        res = h.score_val(routed, verbose=False)
        results[r] = res["composite"]
        print(f"  [r={r}] full composite = {res['composite']:.4f}", flush=True)
        if r == 0.5:
            np.save(OUT / "pred_trainval_routed_r05.npy", routed)

    best_r = max(results, key=lambda r: results[r])
    print(f"\n[grand] best r={best_r} composite={results[best_r]:.4f} "
          f"(基线 0.5502)", flush=True)
    payload = {
        "names": names, "seed": SEED,
        "w_global": w_glob.tolist(), "full_global": full_glob,
        "best_w_split": {sp: best_w[sp].tolist() for sp in SPLITS},
        "r_scan": {str(r): c for r, c in results.items()},
        "best_r": best_r, "best_composite": results[best_r],
        "seconds": time.time() - t0,
    }
    (OUT / "grand_router.json").write_text(json.dumps(
        payload, ensure_ascii=False, indent=1, default=float))
    print(f"[saved] {OUT/'grand_router.json'} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
