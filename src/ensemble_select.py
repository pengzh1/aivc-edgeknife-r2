"""集成选择：对多个 pred_trainval.npy 做权重搜索，最大化 val composite。

策略：fast 评分（跳过逐蛋白PCC/DEP）做 Dirichlet 随机搜索 + 坐标精调，
再对 top-5 用完整评分复核，避免在近似口径上过拟合。

用法:
    python -m src.ensemble_select outputs/wsD/pred_trainval.npy outputs/wsB/pred_trainval.npy ...
"""
import sys
from pathlib import Path

import numpy as np

from .evaluate import Harness


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    assert len(paths) >= 2, "至少两个预测文件"
    h = Harness()
    preds = np.stack([np.load(p) for p in paths])  # (k, n, p)
    k = len(paths)
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        assert arr.shape == (len(h.m_tr), len(h.proteins)), (p, arr.shape)
    assert np.isfinite(preds).all()

    print("== 单模型 composite（完整评分）==")
    singles = []
    for p, arr in zip(paths, preds):
        s = h.score_val(arr, verbose=False)["composite"]
        singles.append(s)
        print(f"  {s:.4f}  {p}")

    rng = np.random.default_rng(0)
    cand = [np.eye(k)[i] for i in range(k)]  # 顶点
    cand += [rng.dirichlet(np.ones(k)) for _ in range(300)]
    cand += [rng.dirichlet(np.ones(k) * 0.3) for _ in range(200)]  # 偏向顶点

    scored = []
    for w in cand:
        blend = np.tensordot(w, preds, axes=1).astype(np.float32)
        s = h.score_fast(blend)
        scored.append((s, w))
    scored.sort(key=lambda x: -x[0])

    # 坐标精调 top1
    w = scored[0][1].copy()
    best_fast = scored[0][0]
    for _ in range(3):
        improved = False
        for i in range(k):
            for delta in (-0.1, 0.1, -0.05, 0.05):
                w2 = w.copy()
                w2[i] += delta
                if (w2 < 0).any():
                    continue
                w2 /= w2.sum()
                blend = np.tensordot(w2, preds, axes=1).astype(np.float32)
                s = h.score_fast(blend)
                if s > best_fast:
                    best_fast, w, improved = s, w2, True
        if not improved:
            break

    # 完整评分复核：top5 随机点 + 精调点 + 最佳单模
    print("\n== 完整评分复核 ==")
    finals = {}
    seen = set()
    for s_fast, w_ in scored[:5] + [(best_fast, w)]:
        key = tuple(np.round(w_, 3))
        if key in seen:
            continue
        seen.add(key)
        blend = np.tensordot(w_, preds, axes=1).astype(np.float32)
        s = h.score_val(blend, verbose=False)["composite"]
        finals[key] = s
        print(f"  {s:.4f}  w={key}")
    best_w = max(finals, key=finals.get)
    print(f"\n[best] composite={finals[best_w]:.4f} weights={best_w}")
    print("单模型最佳:", f"{max(singles):.4f}")

    w_arr = np.array(best_w)
    blend = np.tensordot(w_arr, preds, axes=1).astype(np.float32)
    out = Path("outputs/ensemble_best_pred_trainval.npy")
    np.save(out, blend)
    np.save(Path("outputs/ensemble_best_weights.npy"), w_arr)
    print(f"[saved] {out}")
    h.score_val(blend, verbose=True)


if __name__ == "__main__":
    main()
