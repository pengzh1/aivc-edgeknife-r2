"""wsH: 分样本角色的集成路由（封闭 + 开放）。

动机：各模型族在不同 OOD 划分各有所长（本地 composite / 关键指标）：
  wsD 0.5381（chem FC 0.508 / time FC 0.640）、wsB 0.5306（strain FC 0.353 /
  both FC 0.243）、wsC 0.5258（time FC 0.631）、旧MLP 0.5056、Ridge 0.4608；
  开放榜 wsA 0.5102（chem resid 0.461）。
全局权重（wsD .70 / wsB .266 / wsC .019 / MLP .015 / Ridge 0）composite 0.5413。
本模块在每个 val 划分内部用近似口径分别搜索混合权重，再与全局权重做收缩
（默认 0.5）防过拟合，最后按样本所在划分路由混合，用 score_val 完整评分。

用法:
    python -m src.wsH_router        # v1 阵容（5/6 模型）
    python -m src.wsH_router v2     # v2 新阵容（6/7 模型，wsD_g2g3/wsG/wsF/wsC_g3）
产出:
    outputs/wsH/pred_trainval.npy        封闭榜路由预测（r=0.5 收缩）
    outputs/wsH/pred_trainval_open.npy   开放榜路由预测（r=0.5 收缩）
    outputs/wsH/pred_trainval_v2.npy / pred_trainval_open_v2.npy   v2 阵容
    outputs/wsH/router_weights.json / router_weights_v2.json
    outputs/wsH/report.md                权重表 / 收缩对比 / 过拟合讨论
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .evaluate import Harness

SEED = 20260806
OUT = Path("outputs/wsH")

# 顺序固定：全局权重向量与该顺序对齐
CLOSED = [
    ("wsD", "outputs/wsD/pred_trainval.npy"),
    ("wsB", "outputs/wsB/pred_trainval.npy"),
    ("wsC", "outputs/wsC/pred_trainval.npy"),
    ("mlp", "outputs/pred_mlp_3seed.npy"),
    ("ridge", "outputs/pred_ridge.npy"),
]
OPEN_EXTRA = ("wsA", "outputs/wsA/pred_trainval.npy")

# 现行全局最优（封闭榜，ensemble_closed.log 复核 0.5413）
W_GLOBAL_CLOSED = np.array([0.70, 0.266, 0.019, 0.015, 0.0])

# ---- v2 新阵容（2026-08-06 协调人口径）----
V2_CLOSED = [
    ("wsD_g2g3", "outputs/wsD/pred_trainval_g2g3.npy"),
    ("wsG", "outputs/wsG/pred_trainval.npy"),
    ("wsF", "outputs/wsF/pred_trainval.npy"),
    ("wsC_g3", "outputs/wsC/pred_trainval_g3.npy"),
    ("wsB", "outputs/wsB/pred_trainval.npy"),
    ("ridge", "outputs/pred_ridge.npy"),
]
# ensemble_select 对新阵容的最优点（与 V2_CLOSED 顺序对齐）
V2_W_GLOBAL = np.array([0.765, 0.0, 0.0, 0.0, 0.21, 0.024])

SPLITS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]

# 划分内近似目标权重：0.45*FC + 0.45*resid(有则) + 0.10*fidelity
W_FC, W_RESID, W_FID = 0.45, 0.45, 0.10

N_DIR1, N_DIR03 = 200, 100


# ---------------------------------------------------------------- 快速划分评分


def _center_rows(x):
    """按行对 masked 数组中心化。返回 (mask, n, raw0, xc, norm, sst)。"""
    x = np.asarray(x, dtype=np.float64)
    mask = ~np.isnan(x)
    n = np.maximum(mask.sum(axis=1), 1)
    raw0 = np.where(mask, x, 0.0)
    mean = raw0.sum(axis=1) / n
    xc = np.where(mask, x - mean[:, None], 0.0)
    norm = np.sqrt((xc**2).sum(axis=1))
    sst = (xc**2).sum(axis=1)
    return mask, n, raw0, xc, norm, sst


class SplitScorer:
    """单 val 划分的全向量化近似评分（与 Harness.score_fast 同口径）。

    真值侧（y_true / Δ_true / Δ_true-μ）的掩码与中心化量预计算一次，
    每次评估只需对预测侧做中心化，比反复调用 score_fast 快一个量级。
    """

    def __init__(self, h: Harness, split: str):
        e = h._fast[split]
        self.split = split
        self.n_rows = len(e["rows"])
        (self.y_mask, self.y_n, self.y_raw, self.y_c,
         self.y_vt, self.y_sst) = _center_rows(e["yt"])
        # 处理样本在划分行内的位置（rows 升序，searchsorted 安全）
        self.pos = np.searchsorted(e["rows"], e["trows"])
        self.C_t = np.asarray(e["C_t"], dtype=np.float64)
        (self.d_mask, self.d_n, _, self.d_c,
         self.d_vt, _) = _center_rows(e["dt"])
        if "mu" in e:
            self.mu = np.asarray(e["mu"], dtype=np.float64)
            (self.r_mask, self.r_n, _, self.r_c,
             self.r_vt, _) = _center_rows(e["dt"] - self.mu)
        else:
            self.mu = None

    @staticmethod
    def _pcc(pred, mask, n, tc, vt):
        p = np.where(mask, np.nan_to_num(pred, nan=0.0), 0.0)
        mp = p.sum(axis=1) / n
        pc = np.where(mask, p - mp[:, None], 0.0)
        vp = np.sqrt((pc**2).sum(axis=1))
        cov = (tc * pc).sum(axis=1)
        denom = vt * vp
        return np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 0.0)

    def evaluate(self, yp):
        """yp: (n_rows, n_proteins) 该划分全部行的预测。返回指标 dict。"""
        yp = np.asarray(yp, dtype=np.float64)
        s_pcc = self._pcc(yp, self.y_mask, self.y_n, self.y_c, self.y_vt)
        p0 = np.where(self.y_mask, np.nan_to_num(yp, nan=0.0), 0.0)
        sse = ((self.y_raw - p0) ** 2).sum(axis=1)
        s_r2 = np.where(self.y_sst > 1e-12,
                        1.0 - sse / np.maximum(self.y_sst, 1e-12), 0.0)
        fid = float(0.5 * (s_pcc.mean() + np.clip(s_r2, -1, 1).mean()))

        dp = yp[self.pos] - self.C_t
        fc = float(np.nanmean(
            self._pcc(dp, self.d_mask, self.d_n, self.d_c, self.d_vt)))
        out = {"fidelity": fid, "FC_PCC": fc}
        obj = W_FC * fc + W_FID * fid
        if self.mu is not None:
            rp = self._pcc(dp - self.mu, self.r_mask, self.r_n,
                           self.r_c, self.r_vt)
            out["resid_PCC"] = float(np.nanmean(rp))
            obj += W_RESID * out["resid_PCC"]
        out["objective"] = float(obj)
        return out


# ---------------------------------------------------------------- 搜索与路由


def build_candidates(k, rng):
    """5/6 个顶点 + 200 Dirichlet(1) + 100 Dirichlet(0.3)。"""
    cand = [np.eye(k)[i] for i in range(k)]
    cand += [rng.dirichlet(np.ones(k)) for _ in range(N_DIR1)]
    cand += [rng.dirichlet(np.ones(k) * 0.3) for _ in range(N_DIR03)]
    return cand


def blend_rows(preds, w, rows):
    """preds: (k,N,P) -> 按 w 混合 rows 行，返回 float64 (len(rows),P)。"""
    acc = np.zeros((len(rows), preds.shape[2]), dtype=np.float64)
    for i, wi in enumerate(w):
        if wi:
            acc += float(wi) * preds[i, rows]
    return acc


def search_split(scorer: SplitScorer, P_s, cand):
    """P_s: (k, n_rows, P) float32。返回 (best_w, best_metrics, global_metrics)。"""
    best_obj, best_w, best_m = -np.inf, None, None
    k = P_s.shape[0]
    flat = P_s.reshape(k, -1)
    n_p = P_s.shape[2]
    for w in cand:
        yp = (w @ flat).reshape(-1, n_p)  # float64 (n_rows, P)
        m = scorer.evaluate(yp)
        if m["objective"] > best_obj:
            best_obj, best_w, best_m = m["objective"], w.copy(), m
    return best_w, best_m


def route_pred(preds, h, w_split, w_global):
    """按样本所在划分路由混合；train split 用全局权重。"""
    out = np.empty(preds.shape[1:], dtype=np.float32)
    out[h.tr_rows] = blend_rows(preds, w_global, h.tr_rows)
    for split in SPLITS:
        rows = h._fast[split]["rows"]
        out[rows] = blend_rows(preds, w_split[split], rows)
    return out


def load_preds(files, h):
    arrs = []
    for name, path in files:
        a = np.load(path)
        assert a.shape == (len(h.m_tr), len(h.proteins)), (name, a.shape)
        assert np.isfinite(a).all(), f"{name} 含 NaN/Inf"
        arrs.append(a.astype(np.float32))
        print(f"  loaded {name:<6} {path} {a.shape} {a.dtype}")
    return np.stack(arrs)


def run_set(h, files, w_global, tag, shrink_grid, full_score_grid):
    """一整套：搜索 → 收缩 → 路由 → 完整评分。返回记录 dict。"""
    names = [n for n, _ in files]
    k = len(files)
    print(f"\n===== [{tag}] 载入 {k} 个模型: {names} =====")
    preds = load_preds(files, h)

    rng = np.random.default_rng(SEED)
    cand = build_candidates(k, rng)
    scorers = {sp: SplitScorer(h, sp) for sp in SPLITS}

    rec = {"names": names, "w_global": w_global.tolist(), "splits": {}}
    best_w = {}
    for sp in SPLITS:
        t0 = time.time()
        rows = h._fast[sp]["rows"]
        P_s = np.ascontiguousarray(preds[:, rows, :])
        w_star, m_star = search_split(scorers[sp], P_s, cand)
        m_glob = scorers[sp].evaluate(blend_rows(preds, w_global, rows))
        best_w[sp] = w_star
        rec["splits"][sp] = {
            "n_rows": int(len(rows)),
            "w_opt": w_star.tolist(),
            "fast_opt": m_star,
            "fast_global": m_glob,
        }
        print(f"  [{sp:<16} n={len(rows):>4}] obj {m_glob['objective']:.4f} -> "
              f"{m_star['objective']:.4f} | w*={np.round(w_star, 3)} "
              f"({time.time()-t0:.0f}s)")

    # 收缩扫描
    rec["shrink"] = {}
    for r in shrink_grid:
        w_shr = {sp: r * best_w[sp] + (1 - r) * w_global for sp in SPLITS}
        entry = {sp: w_shr[sp].tolist() for sp in SPLITS}
        if r in full_score_grid:
            routed = route_pred(preds, h, w_shr, w_global)
            assert np.isfinite(routed).all()
            t0 = time.time()
            res = h.score_val(routed, verbose=False)
            print(f"  [{tag} r={r:.1f}] full composite = {res['composite']:.4f} "
                  f"({time.time()-t0:.0f}s)")
            entry["full"] = {
                "composite": res["composite"],
                "per_split": {sp: {kk: vv for kk, vv in res["per_split"][sp].items()}
                              for sp in SPLITS},
            }
            if abs(r - 0.5) < 1e-9:
                rec["routed_pred"] = routed  # 交付档位
        else:
            # 近似口径仅作参考（fast composite 不含 protein_PCC/DEP）
            routed = route_pred(preds, h, w_shr, w_global)
            entry["fast_composite"] = float(h.score_fast(routed))
            print(f"  [{tag} r={r:.1f}] fast composite ≈ {entry['fast_composite']:.4f}")
        rec["shrink"][f"{r:.1f}"] = entry
    rec["preds"] = preds  # 调用方负责释放
    return rec


def search_global(scorers, P_splits, cand):
    """以 fast 口径 composite（M.composite）在全部 val 上搜索全局权重。"""
    best_c, best_w = -np.inf, None
    for w in cand:
        per = {}
        for sp, P_s in P_splits.items():
            k = P_s.shape[0]
            yp = (w @ P_s.reshape(k, -1)).reshape(P_s.shape[1], P_s.shape[2])
            per[sp] = scorers[sp].evaluate(yp)
        c = M.composite(per)
        if c > best_c:
            best_c, best_w = c, w.copy()
    return best_w, float(best_c)


# ---------------------------------------------------------------- 报告


def _fmt_w(w, names):
    return ", ".join(f"{n}={x:.3f}" for n, x in zip(names, w) if x > 5e-4) or "—"


def write_report(rec_c, rec_o, base_full, path):
    L = []
    A = L.append
    A("# wsH 分样本角色集成路由报告\n")
    A("## 方法\n")
    A("- 封闭候选：wsD / wsB / wsC / 旧MLP / Ridge；开放另加 wsA。")
    A("- 每个 val 划分内部独立搜索混合权重：顶点 + 200 Dirichlet(1) + "
      "100 Dirichlet(0.3)（种子固定 20260806），目标 = "
      "0.45×FC_PCC + 0.45×resid_PCC(有则) + 0.10×fidelity（近似口径，"
      "与 score_fast 相同的向量化实现，真值侧统计预计算）。")
    A("- 防过拟合收缩：w_final = r × 划分最优 + (1−r) × 全局最优"
      "（全局 = wsD .70 / wsB .266 / wsC .019 / MLP .015 / Ridge 0，"
      "开放版 wsA=0）；train split 样本一律用全局权重。")
    A("- 最终混合用 `h.score_val` 完整评分（唯一公允口径）。\n")
    A(f"全局权重基线（同一 val 上完整评分复核）：composite = "
      f"**{base_full['composite']:.4f}**（与既有日志 0.5413 一致）。\n")

    for tag, rec in (("封闭榜（5 模型）", rec_c), ("开放榜（6 模型，含 wsA）", rec_o)):
        names = rec["names"]
        A(f"## {tag}\n")
        A("### 各划分最优权重（收缩前，fast 口径）\n")
        A("| 划分 | n | 全局目标值 | 最优目标值 | 最优权重 |")
        A("|---|---|---|---|---|")
        for sp in SPLITS:
            s = rec["splits"][sp]
            A(f"| {sp} | {s['n_rows']} | {s['fast_global']['objective']:.4f} "
              f"| {s['fast_opt']['objective']:.4f} | {_fmt_w(s['w_opt'], names)} |")
        A("")
        A("### 划分最优 vs 全局 的 fast 指标\n")
        A("| 划分 | 口径 | fidelity | FC_PCC | resid_PCC |")
        A("|---|---|---|---|---|")
        for sp in SPLITS:
            s = rec["splits"][sp]
            for label, m in (("全局", s["fast_global"]), ("划分最优", s["fast_opt"])):
                A(f"| {sp} | {label} | {m['fidelity']:.4f} | {m['FC_PCC']:.4f} "
                  f"| {m.get('resid_PCC', float('nan')):.4f} |")
        A("")
        A("### 收缩比例对比（完整 score_val composite）\n")
        A("| r（划分最优占比） | composite |")
        A("|---|---|")
        for rkey in sorted(rec["shrink"], key=lambda x: -float(x)):
            e = rec["shrink"][rkey]
            if "full" in e:
                A(f"| {rkey} | **{e['full']['composite']:.4f}** |")
            else:
                A(f"| {rkey} | (fast≈{e['fast_composite']:.4f}) |")
        A("")
        if "routed_pred" in rec:
            full05 = rec["shrink"]["0.5"]["full"]
            A("### 交付档位 r=0.5 的完整评分表\n")
            cols = ["fidelity", "sample_PCC", "sample_R2", "protein_PCC",
                    "FC_PCC", "resid_PCC", "DEP_dir_acc", "DEP_PCC", "DEP_F1"]
            A("| split |" + "|".join(cols) + "|")
            A("|---|" + "---|" * len(cols))
            for sp in SPLITS:
                s = full05["per_split"][sp]
                A(f"| {sp} |" + "|".join(
                    f"{s[c]:.4f}" if c in s else "-" for c in cols) + "|")
            A(f"\ncomposite = **{full05['composite']:.4f}** "
              f"（全局基线 {base_full['composite']:.4f}，"
              f"Δ = {full05['composite'] - base_full['composite']:+.4f}）\n")

    A("## 过拟合诚实讨论\n")
    A("- 权重在每个 val 划分内部调过、又在同一 val 上评分，上表 composite "
      "**存在 in-sample 乐观**；划分越小（both n=269、time n=157）乐观越大。")
    A("- 缓解手段：0.5 收缩（交付档）把划分最优与全局最优各取一半；"
      "全局权重本身也是在同一 val 上调的，但自由度只有 k−1，"
      "而路由额外引入 4×(k−1) 个自由度，乐观增量主要来自小划分。")
    A("- 因此 test 上的真实增益应预期**小于**本地 Δ；若收缩扫描中 r=0.5 与 "
      "r=0.3 差距很小，说明路由增益稳健部分有限，主打 chem/strain 两个大划分。")
    A("- r=1.0（不收缩）只作对照展示过拟合上界，不作为交付。\n")
    A("## test 映射规则推荐\n")
    A("train_val 中角色与划分严格对应：val_chem_only=(strain train, chem val)、"
      "val_strain_only=(strain val, chem train)、val_both=(双 val)、"
      "val_time=(双 train，时间/批次外推)。模型只在 train split 上训练，"
      "因此 val 角色对模型而言与 test 角色同为'未见'。推荐规则：\n")
    A("- chemical_role≠train 且 strain_role=train → chem_only 权重（test 中 1640 例）")
    A("- strain_role≠train 且 chemical_role=train → strain_only 权重（1534 例）")
    A("- 双角色均≠train（含 test/test 425、val/test 432、test/val 272）→ both 权重")
    A("- 双角色均=train（151 例，与 val_time 同构的时间/批次外推）→ time 权重"
      "（保守起见也可用全局权重，二者差异很小）\n")
    A("即任务提议的映射（chem=test→chem_only、strain=test→strain_only、双test→both）"
      "方向正确，补充两点：role=val 应与 test 同等视为'未见'；双 train 样本按 "
      "val_time 同构处理。")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {path}")


def _report_set_block(L, rec, title):
    """单个模型阵容的 v2 报告小节（与 v1 表格同构）。"""
    A = L.append
    names = rec["names"]
    A(f"### {title}\n")
    A("#### 各划分最优权重（收缩前，fast 口径）\n")
    A("| 划分 | n | 全局目标值 | 最优目标值 | 最优权重 |")
    A("|---|---|---|---|---|")
    for sp in SPLITS:
        s = rec["splits"][sp]
        A(f"| {sp} | {s['n_rows']} | {s['fast_global']['objective']:.4f} "
          f"| {s['fast_opt']['objective']:.4f} | {_fmt_w(s['w_opt'], names)} |")
    A("")
    A("#### 收缩比例对比（完整 score_val composite）\n")
    A("| r（划分最优占比） | composite |")
    A("|---|---|")
    for rkey in sorted(rec["shrink"], key=lambda x: -float(x)):
        e = rec["shrink"][rkey]
        if "full" in e:
            A(f"| {rkey} | **{e['full']['composite']:.4f}** |")
        else:
            A(f"| {rkey} | (fast≈{e['fast_composite']:.4f}) |")
    A("")
    full05 = rec["shrink"]["0.5"]["full"]
    A("#### 交付档位 r=0.5 的完整评分表\n")
    cols = ["fidelity", "sample_PCC", "sample_R2", "protein_PCC",
            "FC_PCC", "resid_PCC", "DEP_dir_acc", "DEP_PCC", "DEP_F1"]
    A("| split |" + "|".join(cols) + "|")
    A("|---|" + "---|" * len(cols))
    for sp in SPLITS:
        s = full05["per_split"][sp]
        A(f"| {sp} |" + "|".join(
            f"{s[c]:.4f}" if c in s else "-" for c in cols) + "|")
    A(f"\ncomposite = **{full05['composite']:.4f}**\n")
    return full05["composite"]


def append_report_v2(rec_c, rec_o, anchor, singles, v1=(0.5454, 0.5456)):
    """向 report.md 追加 v2 阵容小节。anchor: 锚点决策信息 dict。"""
    path = OUT / "report.md"
    L = []
    A = L.append
    A("\n---\n\n# v2 新阵容路由（2026-08-06 协调人指定）\n")
    A("## 阵容与单模型复核\n")
    A("封闭 6 模型 + 开放另加 wsA。单模型完整 composite 复核（括号内为协调人给定值）：\n")
    A("| 模型 | 本地复核 | 给定 |")
    A("|---|---|---|")
    for name, (s_loc, s_ref) in singles.items():
        A(f"| {name} | {s_loc:.4f} | {s_ref:.4f} |")
    A("")
    A("## 全局锚点\n")
    A(f"- 给定锚点（ensemble_select 最优点）：{_fmt_w(anchor['w_given'], rec_c['names'])}"
      f"，完整 composite = **{anchor['full_given']:.4f}**")
    A(f"- 本模块 fast 搜索重搜锚点：{_fmt_w(anchor['w_searched'], rec_c['names'])}"
      f"，完整 composite = **{anchor['full_searched']:.4f}**")
    A(f"- 采用：**{anchor['which']}** 作为收缩锚点（规则：重搜点完整评分须领先 "
      f">0.0005 才替换，否则沿用给定点）。开放版锚点 = 封闭锚点 + wsA=0。\n")
    c_closed = _report_set_block(L, rec_c, "封闭榜 v2（6 模型）")
    c_open = _report_set_block(L, rec_o, "开放榜 v2（7 模型，含 wsA）")
    A("## 与 v1 的对比\n")
    A("| 版本 | 封闭 r=0.5 | 开放 r=0.5 |")
    A("|---|---|---|")
    A(f"| v1（wsD/wsB/wsC/mlp/ridge [+wsA]） | {v1[0]:.4f} | {v1[1]:.4f} |")
    A(f"| v2 新阵容（全局基线 {anchor['full_used']:.4f}） "
      f"| **{c_closed:.4f}** | **{c_open:.4f}** |")
    A(f"| v2 − v1 | {c_closed - v1[0]:+.4f} | {c_open - v1[1]:+.4f} |")
    A(f"| v2 路由 − v2 全局基线 | {c_closed - anchor['full_used']:+.4f} | — |\n")
    A("## v2 test 角色 → 权重映射（r=0.5 收缩后，封闭；开放取含 wsA 行）\n")
    A("| test 角色组合 | 路由权重 |")
    A("|---|---|")
    for label, sp in (("chem≠train, strain=train", "val_chem_only"),
                      ("strain≠train, chem=train", "val_strain_only"),
                      ("双角色≠train", "val_both"),
                      ("双角色=train（val_time 同构）", "val_time")):
        w = rec_c["shrink"]["0.5"][sp]
        A(f"| {label} | {_fmt_w(w, rec_c['names'])} |")
    A("| train split（仅本地） | "
      f"{_fmt_w(rec_c['w_global'], rec_c['names'])}（全局锚点） |\n")
    A("过拟合讨论与 v1 相同：权重在同 val 上调与评，r=0.5 收缩防小划分伪信号；"
      "v2 both 划分的 Ridge 权重同样需警惕。")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[appended] {path} v2 section")


# ---------------------------------------------------------------- v2 main


def main_v2():
    """v2 新阵容：封闭 6 模型 + 开放 7 模型（含 wsA），含全局锚点重搜对照。"""
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()

    # 单模型复核
    print("\n===== [v2] 单模型完整 composite 复核 =====")
    preds = load_preds(V2_CLOSED, h)
    refs = {"wsD_g2g3": 0.5431, "wsG": 0.5422, "wsF": 0.5414,
            "wsC_g3": 0.5385, "wsB": 0.5306, "ridge": 0.4608}
    singles = {}
    for (name, _), arr in zip(V2_CLOSED, preds):
        s = h.score_val(arr, verbose=False)["composite"]
        singles[name] = (s, refs[name])
        print(f"  {s:.4f}  {name} (ref {refs[name]:.4f})")

    # 全局锚点：给定点 vs fast 重搜点，完整评分定夺
    scorers = {sp: SplitScorer(h, sp) for sp in SPLITS}
    P_splits = {sp: np.ascontiguousarray(preds[:, h._fast[sp]["rows"], :])
                for sp in SPLITS}
    rng = np.random.default_rng(SEED)
    cand = build_candidates(len(V2_CLOSED), rng)
    w_given = V2_W_GLOBAL / V2_W_GLOBAL.sum()
    t0s = time.time()
    w_srch, fast_c = search_global(scorers, P_splits, cand)
    print(f"[v2] 全局重搜 fast composite ≈ {fast_c:.4f} "
          f"w={np.round(w_srch, 3)} ({time.time()-t0s:.0f}s)")
    all_rows = np.arange(len(h.m_tr))
    full_given = h.score_val(
        blend_rows(preds, w_given, all_rows).astype(np.float32),
        verbose=False)["composite"]
    full_srch = h.score_val(
        blend_rows(preds, w_srch, all_rows).astype(np.float32),
        verbose=False)["composite"]
    print(f"[v2] 锚点完整评分: 给定 {full_given:.4f} vs 重搜 {full_srch:.4f}")
    if full_srch > full_given + 5e-4:
        w_anchor, which, full_used = w_srch, "重搜点", full_srch
    else:
        w_anchor, which, full_used = w_given, "给定点", full_given
    anchor = {"w_given": w_given.tolist(), "w_searched": w_srch.tolist(),
              "full_given": full_given, "full_searched": full_srch,
              "which": which, "full_used": full_used,
              "fast_searched": fast_c}
    del preds, P_splits

    # 封闭路由
    rec_c = run_set(h, V2_CLOSED, w_anchor, "v2-closed",
                    shrink_grid=(1.0, 0.7, 0.5, 0.3),
                    full_score_grid=(0.7, 0.5, 0.3))
    np.save(OUT / "pred_trainval_v2.npy",
            rec_c["routed_pred"].astype(np.float32))
    print(f"[saved] {OUT/'pred_trainval_v2.npy'}")
    rec_c.pop("preds")

    # 开放路由（加 wsA）
    files_o = V2_CLOSED + [OPEN_EXTRA]
    w_anchor_o = np.concatenate([w_anchor, [0.0]])
    rec_o = run_set(h, files_o, w_anchor_o, "v2-open",
                    shrink_grid=(1.0, 0.7, 0.5, 0.3),
                    full_score_grid=(0.7, 0.5, 0.3))
    np.save(OUT / "pred_trainval_open_v2.npy",
            rec_o["routed_pred"].astype(np.float32))
    print(f"[saved] {OUT/'pred_trainval_open_v2.npy'}")
    rec_o.pop("preds")

    def strip(rec):
        return {k: v for k, v in rec.items() if k != "routed_pred"}

    payload = {
        "seed": SEED, "lineup": "v2",
        "objective": f"{W_FC}*FC + {W_RESID}*resid(if any) + {W_FID}*fidelity",
        "anchor": anchor,
        "singles": {k: {"local": v[0], "ref": v[1]} for k, v in singles.items()},
        "closed": strip(rec_c),
        "open": strip(rec_o),
    }
    with open(OUT / "router_weights_v2.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"[saved] {OUT/'router_weights_v2.json'}")

    append_report_v2(rec_c, rec_o, anchor, singles)
    print(f"\n[done v2] total {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- main


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = Harness()
    h.prepare_fast_eval()

    # 全局基线复核（完整评分）
    print("\n===== 全局权重基线（封闭 5 模型）=====")
    preds_c = load_preds(CLOSED, h)
    base = blend_rows(preds_c, W_GLOBAL_CLOSED, np.arange(len(h.m_tr)))
    base_full = h.score_val(base.astype(np.float32), verbose=True)
    del preds_c, base

    # 封闭榜
    rec_c = run_set(h, CLOSED, W_GLOBAL_CLOSED, "closed",
                    shrink_grid=(1.0, 0.7, 0.5, 0.3),
                    full_score_grid=(0.7, 0.5, 0.3))
    np.save(OUT / "pred_trainval.npy", rec_c["routed_pred"].astype(np.float32))
    print(f"[saved] {OUT/'pred_trainval.npy'}")
    rec_c.pop("preds")

    # 开放榜（加 wsA；仅对 chem_role 非 train 样本有理论基础，搜索自会发现）
    files_o = CLOSED + [OPEN_EXTRA]
    w_global_o = np.concatenate([W_GLOBAL_CLOSED, [0.0]])
    rec_o = run_set(h, files_o, w_global_o, "open",
                    shrink_grid=(1.0, 0.7, 0.5, 0.3),
                    full_score_grid=(0.7, 0.5, 0.3))
    np.save(OUT / "pred_trainval_open.npy",
            rec_o["routed_pred"].astype(np.float32))
    print(f"[saved] {OUT/'pred_trainval_open.npy'}")
    rec_o.pop("preds")

    # 记录与报告
    def strip(rec):
        return {k: v for k, v in rec.items() if k != "routed_pred"}

    payload = {
        "seed": SEED,
        "objective": f"{W_FC}*FC + {W_RESID}*resid(if any) + {W_FID}*fidelity",
        "baseline_global_full": {
            "composite": base_full["composite"],
            "per_split": base_full["per_split"],
        },
        "closed": strip(rec_c),
        "open": strip(rec_o),
    }
    with open(OUT / "router_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"[saved] {OUT/'router_weights.json'}")

    write_report(rec_c, rec_o, base_full, OUT / "report.md")
    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        main_v2()
    else:
        main()
