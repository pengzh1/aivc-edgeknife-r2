"""wsT2：DEP 选择器的 test 侧交付（wsN13_final 同流程 + 候选推送）。

仅在 wsT1 arbitrate 裁决为绿后使用。管线顺序：
  21 族 test 预测 r=0.7 路由（wsN13 同式，含 MERGE_INTO 归并）
  → band γ=1.3（若胜者为 C2；C1/C3 则跳过 band）
  → DEP 选择器推送（全 test 处理行，含 strain/both 角色——选择器突破 7 键限制）
  → CRD←CGD β=0.35 有界迁移（wsN13 同序）
  → 列裁剪 + 5.2.4 认证。

test 侧特征与 train 侧同构（10 维）：跨族 std 用 MERGE_INTO 复制填满 21 槽
（wsG/wsN7→wsD_s16，wsBg2→wsB），与 wsT0 train 侧分布口径一致；
control_hat 用 MST.build_control_hat_train（train 冻结对照池）；
prot 统计/τ/分类器全部来自 wsT1 cache（train-only 拟合）。

合规：Y_te 零接触（本脚本只写 ŷ_te，不读任何 test 真值）；不改既有文件。

用法: python -m src.wsT2_finaltest --name C2 --tag r07_c2
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data as D
from .evaluate import Harness
from . import make_submission_trainonly as MST
from .make_submission import band_calibrate
from .wsN8_crdtransfer import strain_modulation, apply_transfer
from .wsN13_final import TEST_NPY, MERGE_INTO, ROLE_OF
from .wsN11_grandrouter import SPLITS
from .wsT1_depgate import (CACHE as T1C, MIN_PUSH, PULL_BAND, PULL_TO, PUSH_TO,
                           _pert_time_num)

OUT = Path("outputs/wsT2")


def load_test_stack(h: Harness, names):
    """载入各族 test 预测并按 names 顺序（21 槽，MERGE_INTO 复制填满）。"""
    preds = {}
    for n in names:
        p = TEST_NPY.get(n)
        if p and Path(p).exists():
            preds[n] = np.load(p)
    stack = []
    for n in names:
        if n in preds:
            stack.append(preds[n])
        elif n in MERGE_INTO and MERGE_INTO[n] in preds:
            stack.append(preds[MERGE_INTO[n]])
        else:
            raise FileNotFoundError(f"{n} 无 test 预测且无法归并")
    return np.stack([s.astype(np.float32) for s in stack]), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="wsT1 candidates.json 中的胜者名")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tau", type=float, default=None,
                    help="覆盖候选 τ（对冲档位定价用，如 0.25）")
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--no-crd", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cands = {c["name"]: c for c in
             json.loads((T1C / "candidates.json").read_text())}
    cand = cands[args.name]
    kind, tau_low = cand["kind"], cand.get("tau_low", 0.10)
    tau = args.tau if args.tau is not None else cand["tau"]
    band_first = kind == "C2"
    print(f"[wsT2] candidate={args.name} kind={kind} τ={tau} τ_low={tau_low} "
          f"band_first={band_first}", flush=True)

    gr = json.loads(Path("outputs/wsN11/grand_router.json").read_text())
    names = gr["names"]
    r = 0.7
    w_glob = np.array(gr["w_global"])
    best_w = {sp: np.array(gr["best_w_split"][sp]) for sp in ROLE_OF}
    w_use = {sp: r * best_w[sp] + (1 - r) * w_glob for sp in ROLE_OF}

    h = Harness()
    stack, preds = load_test_stack(h, names)
    # 跨族 std（21 槽复制版，与 train 侧口径一致；累加器法避免 f64 全量拷贝）
    k = stack.shape[0]
    m_acc = np.zeros(stack.shape[1:], np.float64)
    s_acc = np.zeros_like(m_acc)
    mw_acc = np.zeros_like(m_acc)
    sw_acc = np.zeros_like(m_acc)
    w_gl = w_glob / w_glob.sum()
    for i in range(k):
        x = stack[i].astype(np.float64)
        m_acc += x
        s_acc += x * x
        mw_acc += w_gl[i] * x
        sw_acc += w_gl[i] * x * x
    m_acc /= k
    x_std = np.sqrt(np.clip(s_acc / k - m_acc * m_acc, 0, None)).astype(np.float32)
    x_std_w = np.sqrt(np.clip(sw_acc - mw_acc * mw_acc, 0, None)).astype(np.float32)
    del m_acc, s_acc, mw_acc, sw_acc

    # 路由（wsN13 同式）
    keys = MST.route_keys_trainonly(h)
    pred = np.zeros_like(stack[0])
    for sp, role in ROLE_OF.items():
        mask = keys == role
        if not mask.any():
            continue
        w = dict(zip(names, w_use[sp]))
        for dead, target in MERGE_INTO.items():
            if dead in w:
                w[target] = w.get(target, 0.0) + w.pop(dead)
        w = {k2: v for k2, v in w.items() if k2 in preds and v > 0}
        s = sum(w.values())
        pred[mask] = (sum(wt * preds[m][mask] for m, wt in w.items())
                      / s).astype(np.float32)
        print(f"  {role:<12} n={int(mask.sum()):>5}", flush=True)
    del stack

    # band（若 C2）
    ctrl_hat_te, level_te = MST.build_control_hat_train(h, h.m_te)
    is_treat_te = ~h.m_te["perturbation_no_concentration"].isin(
        D.CONTROLS | {D.QC}).to_numpy()
    out = band_calibrate(pred, ctrl_hat_te, level_te, is_treat_te, g=1.3) \
        if band_first else pred.copy()

    # ---- 选择器推送 ----
    z = np.load(T1C / "prot_stats.npz")
    prot_dep_rate, prot_absq90 = z["prot_dep_rate"], z["prot_absq90"]
    pt_med = json.loads((T1C / "pt_med.json").read_text())["pt_med"]
    clf = joblib.load(T1C / "hgb_full.joblib")

    rows = np.where(is_treat_te)[0]
    m_rows = h.m_te.iloc[rows]
    d_est = (pred[rows] - ctrl_hat_te[rows]).astype(np.float64)
    absDE = np.abs(d_est)
    d_cur = (out[rows] - ctrl_hat_te[rows]).astype(np.float64)
    mu_ctx = h.mu_ctx_for(m_rows).astype(np.float64)
    mu_drug = h.mu_drug_for(m_rows).astype(np.float64)
    lv = np.broadcast_to(level_te[rows][:, None].astype(np.float64), d_est.shape)
    pt = np.broadcast_to(_pert_time_num(m_rows, pt_med)[:, None],
                         d_est.shape).astype(np.float64)
    X = np.stack([absDE, x_std[rows], x_std_w[rows],
                  np.broadcast_to(prot_dep_rate, d_est.shape),
                  np.broadcast_to(prot_absq90, d_est.shape),
                  absDE / (np.broadcast_to(prot_absq90, d_est.shape) + 0.1),
                  np.abs(mu_ctx), np.abs(mu_drug), lv, pt],
                 axis=-1).astype(np.float32)
    P = clf.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(X.shape[:2])
    print(f"[wsT2] test treated rows={len(rows)} P mean={P.mean():.4f}", flush=True)

    add = np.zeros_like(d_cur)
    fl = (P >= tau) & (absDE >= MIN_PUSH)
    if kind == "C2":
        fl &= np.abs(d_cur) <= 1.0
    tgt = np.sign(d_cur) * np.maximum(np.abs(d_cur), PUSH_TO)
    add += np.where(fl, tgt - d_cur, 0.0)
    if kind == "C3":
        fl3 = (P <= tau_low) & (np.abs(d_cur) > PULL_BAND[0]) \
            & (np.abs(d_cur) <= PULL_BAND[1])
        tgt3 = np.sign(d_cur) * np.minimum(np.abs(d_cur), PULL_TO)
        add += np.where(fl3, tgt3 - d_cur, 0.0)
    out[rows] = (out[rows].astype(np.float64) + add).astype(np.float32)
    print(f"[wsT2] pushed entries={int((add != 0).sum()):,} "
          f"({(add != 0).sum() / add.size:.4%})", flush=True)

    # CRD 有界迁移（wsN13 同序）
    if not args.no_crd:
        mod = strain_modulation(h)
        out, n_hit = apply_transfer(out, h, h.m_te, None, mod, "CGD", "CRD",
                                    args.beta)
        print(f"[crd] β={args.beta} 命中 {n_hit} 行", flush=True)
    assert np.isfinite(out).all()

    # 输出 + 认证（wsN13 同式）
    tag = args.tag or f"{args.name.lower()}_r07"
    df = pd.DataFrame(out, columns=h.proteins)
    df.insert(0, "sample_ID", h.m_te["sample_ID"].to_numpy())
    keep = pd.read_csv("outputs/wsM/keep_proteins_train_miss80.csv")["protein"]
    out_full = OUT / f"prediction_wst2_{tag}.csv"
    df.to_csv(out_full, index=False)
    df[["sample_ID"] + keep.tolist()].to_csv(
        OUT / f"prediction_wst2_{tag}_keepcols.csv", index=False)
    np.save(out_full.with_suffix(".npy"), out)

    mt = pd.read_csv("input/WAYB_WAYC_metadata_test(1).csv")
    ok = {"rows": len(df) == 4454,
          "id_order": bool((df["sample_ID"].to_numpy()
                            == mt["sample_ID"].to_numpy()).all()),
          "finite": bool(np.isfinite(out).all()),
          "cols_keep": list(df.columns[1:]) == h.proteins.tolist(),
          "range": [float(out.min()), float(out.max())],
          "candidate": args.name, "tau": tau, "tau_low": tau_low,
          "band_first": band_first, "beta": args.beta,
          "n_pushed": int((add != 0).sum())}
    print("[certify]", ok, flush=True)
    (OUT / f"certify_{tag}.json").write_text(json.dumps(ok, indent=1))
    print(f"[saved] {out_full}", flush=True)


if __name__ == "__main__":
    main()
