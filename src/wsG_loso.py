"""wsG：组级 OOD 增强训练——直接优化"未见菌株"目标（封闭数据榜，主攻 val_strain_only）。

动机：逐样本 embedding dropout（emb_drop 0.35）下模型仍能在 65% 样本上依赖真实
菌株嵌入，UNK 通路欠训练；而真实评测是整组未见（BAI 的全部样本同时不可用）。
本模块在 wsD 最佳配置（5 层 (1024,2048,2048,2048,2048)、masked Huber、300ep、
dropout 0.3、lr 1e-3、wd 1e-4、bs 256、chem_emb 32）上对比组级增强策略：

- G0 基线：逐样本 emb_drop 0.35（复现 wsD T1，期望 composite ≈ 0.52）
- G1 组级菌株 dropout：每 epoch 随机选 1 个 train 菌株，其所有样本该 epoch 强制
  UNK；其余样本仍走逐样本 dropout 0.15
- G2 双组级：每 epoch 随机 1 菌株 + 随机 1 化合物同时整组 UNK
- G3 UNK 先验：推理时未见实体嵌入 = (1-α)·train 同类实体嵌入均值 + α·学到的 UNK，
  α ∈ {0, 0.5, 1}（对 G0/G1 各试一次，3 种子均值预测后评分）
- G4 留出轮换微调：G0 训练后依次 4 轮"留出第 k 菌株（全 UNK）微调 20 epoch"

合规：仅用 h.tr_rows 训练；val 划分只用于评分与 α 粗调（模型选择）；不触碰 h.Y_te；
种子固定（0-2）。显存 < 4GB：bs 256、宽度 ≤2048、GPU 上仅驻留 1 个全量 float
标签矩阵（mask 为 bool）。

用法:
    python -m src.wsG_loso --run --augs G0,G1,G2,G4 --seeds 0,1,2
    python -m src.wsG_loso --g3
    python -m src.wsG_loso --finalize
    python -m src.wsG_loso            # 依次执行以上三步
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import Harness
from .train_mlp import CAT_COLS, EMB_DIMS, Encoder, masked_mse

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsG"
CACHE = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    hidden: list = field(default_factory=lambda: [1024, 2048, 2048, 2048, 2048])
    epochs: int = 300
    lr: float = 1e-3
    wd: float = 1e-4
    emb_drop: float = 0.35      # G0 逐样本 dropout 率（G4 微调轮也用它）
    p_drop: float = 0.3
    bs: int = 256
    chem_emb: int = 32
    loss: str = "huber"
    aug: str = "G0"             # G0 | G1 | G2 | G4
    g_emb_drop: float = 0.15    # G1/G2 中非整组样本的逐样本 dropout 率
    ft_epochs: int = 20         # G4 每轮微调 epoch 数
    ft_lr: float = 2e-4         # G4 微调 lr


class Block(nn.Module):
    """Linear → GELU → LayerNorm → Dropout（与 wsD 最佳配置一致，无残差/FiLM）。"""

    def __init__(self, d_in, d_out, p_drop):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return self.drop(self.norm(self.act(self.fc(x))))


class ProteoMLP_G(nn.Module):
    """实体嵌入 MLP（参数初始化顺序与 wsD ProteoMLP2 一致，保证 G0 可复现 T1）。"""

    def __init__(self, n_cats, n_prot, cfg: Cfg):
        super().__init__()
        dims = dict(EMB_DIMS)
        dims["perturbation_no_concentration"] = cfg.chem_emb
        self.embs = nn.ModuleList([
            nn.Embedding(n, dims[c]) for n, c in zip(n_cats, CAT_COLS)])
        d_in = sum(dims[c] for c in CAT_COLS)
        self.blocks = nn.ModuleList()
        d = d_in
        for hdim in cfg.hidden:
            self.blocks.append(Block(d, hdim, cfg.p_drop))
            d = hdim
        self.head = nn.Linear(d, n_prot)

    def embed(self, x_cat):
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

    def forward_emb(self, e):
        h = e
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)

    def forward(self, x_cat):
        return self.forward_emb(self.embed(x_cat))


def masked_huber(pred, target, mask, beta: float = 1.0):
    se = F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * mask
    return se.sum() / mask.sum().clamp_min(1.0)


_TENSORS = {}


def get_tensors(h: Harness):
    """编码/标准化张量只构建一次（与 wsD 相同的冻结统计口径）。"""
    if "t" not in _TENSORS:
        enc = Encoder().fit(h.m_tr)  # 类别覆盖 train_val 全集（同基线/wsD 实现）
        stats = h.stats
        X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
        mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
        std = torch.tensor(stats.protein_std, dtype=torch.float32)
        Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
        M_all = ~torch.isnan(Z_all)
        Z_all = torch.nan_to_num(Z_all, nan=0.0)
        _TENSORS["t"] = (enc, X_all, mean, std, Z_all, M_all)
    return _TENSORS["t"]


def _epoch_loop(model, opt, sched, tens_dev, rows_dev, cfg: Cfg, device,
                epochs: int, aug: str, strain_ids, chem_ids,
                fixed_strain=None, log_every: int = 0) -> float:
    """训练 epochs 轮。aug: G0 逐样本 / G1 组级菌株 / G2 双组级 / holdout 固定留出菌株。

    每个 epoch 开头决定本 epoch 的整组 UNK 对象；组内样本的对应列强制为 0(UNK)，
    其余样本按 rate 逐样本 dropout（先逐样本、后整组覆盖）。
    """
    X_dev, Z_dev, M_dev = tens_dev
    final_loss = float("nan")
    for ep in range(epochs):
        model.train()
        gs = gc = None
        if aug in ("G1", "G2"):
            gs = strain_ids[np.random.randint(len(strain_ids))]
            if aug == "G2":
                gc = chem_ids[np.random.randint(len(chem_ids))]
        elif aug == "holdout":
            gs = fixed_strain
        rate = cfg.emb_drop if aug in ("G0", "holdout") else cfg.g_emb_drop
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            for col in (0, 1):
                dm = torch.rand(len(r), device=device) < rate
                xb[dm, col] = 0
            if gs is not None:
                xb[X_dev[r, 0] == gs, 0] = 0
            if gc is not None:
                xb[X_dev[r, 1] == gc, 1] = 0
            pred = model(xb)
            if cfg.loss == "huber":
                loss = masked_huber(pred, Z_dev[r], M_dev[r].float())
            else:
                loss = masked_mse(pred, Z_dev[r], M_dev[r].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        final_loss = tot / nb
        if log_every and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep+1:>3}/{epochs}  loss={final_loss:.4f}",
                  flush=True)
    return final_loss


def train_one(h: Harness, cfg: Cfg, seed: int, device: str = "cuda",
              log_every: int = 0):
    """主训练阶段（G4 的主阶段 = G0；留出微调由 finetune_holdout 另行执行）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Z_all, M_all = get_tensors(h)
    rows = h.tr_rows
    n_prot = h.Y_tr.shape[1]

    model = ProteoMLP_G(enc.n_cats, n_prot, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    tens_dev = (X_all.to(device), Z_all.to(device), M_all.to(device))
    rows_dev = torch.tensor(rows, device=device)
    strain_ids = torch.unique(tens_dev[0][rows_dev, 0]).tolist()
    chem_ids = torch.unique(tens_dev[0][rows_dev, 1]).tolist()

    main_aug = cfg.aug if cfg.aug in ("G1", "G2") else "G0"
    loss = _epoch_loop(model, opt, sched, tens_dev, rows_dev, cfg, device,
                       cfg.epochs, main_aug, strain_ids, chem_ids,
                       log_every=log_every)
    return model, enc, mean, std, strain_ids, chem_ids, loss


def finetune_holdout(h: Harness, cfg: Cfg, seed: int, model, strain_ids,
                     device: str = "cuda", log_every: int = 0):
    """G4：依次 4 轮"留出第 k 菌株（该菌株样本全 UNK）微调 ft_epochs epoch"。"""
    torch.manual_seed(seed + 1000)
    rng = np.random.RandomState(seed + 1000)
    enc, X_all, mean, std, Z_all, M_all = get_tensors(h)
    rows = h.tr_rows
    tens_dev = (X_all.to(device), Z_all.to(device), M_all.to(device))
    rows_dev = torch.tensor(rows, device=device)
    chem_ids = torch.unique(tens_dev[0][rows_dev, 1]).tolist()
    steps = int(np.ceil(len(rows) / cfg.bs))

    order = list(strain_ids)
    rng.shuffle(order)
    for k, gs in enumerate(order):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.ft_lr,
                                weight_decay=cfg.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.ft_epochs * steps)
        loss = _epoch_loop(model, opt, sched, tens_dev, rows_dev, cfg, device,
                           cfg.ft_epochs, "holdout", strain_ids, chem_ids,
                           fixed_strain=gs, log_every=log_every)
        print(f"    [G4] round {k+1}/4 holdout strain_id={gs} loss={loss:.4f}",
              flush=True)
    return model


@torch.no_grad()
def predict(model, enc, mean, std, m, device="cuda", bs: int = 1024,
            override: dict | None = None) -> np.ndarray:
    """override=None：与基线一致（未见实体用其未训练的随机初始化嵌入）。
    override 由 build_override 构建：未见实体嵌入替换为凸组合向量。"""
    model.eval()
    X = torch.tensor(enc.transform(m), dtype=torch.long, device=device)
    outs = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        if override is None:
            outs.append(model(xb).float().cpu())
        else:
            es = [emb(xb[:, j]) for j, emb in enumerate(model.embs)]
            for key, col in (("strain", 0), ("chem", 1)):
                v = override.get(f"vec_{key}")
                if v is not None:
                    mk = override[f"mask_{key}"][i:i + bs]
                    if bool(mk.any()):
                        es[col][mk] = v
            outs.append(model.forward_emb(torch.cat(es, dim=1)).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def build_override(model, enc, m, train_strains, train_chems, alpha: float,
                   device="cuda") -> dict:
    """G3：未见实体嵌入 ← (1-α)·train 同类实体嵌入均值 + α·学到的 UNK(0)。"""
    ov = {}
    w_s = model.embs[0].weight
    idx_s = torch.tensor([enc.maps["Strains"][s] for s in train_strains],
                         device=device)
    ov["vec_strain"] = ((1 - alpha) * w_s[idx_s].mean(dim=0)
                        + alpha * w_s[0]).detach()
    ov["mask_strain"] = torch.tensor(
        ~m["Strains"].isin(train_strains).to_numpy(), device=device)
    w_c = model.embs[1].weight
    ccol = "perturbation_no_concentration"
    idx_c = torch.tensor([enc.maps[ccol][c] for c in train_chems], device=device)
    ov["vec_chem"] = ((1 - alpha) * w_c[idx_c].mean(dim=0)
                      + alpha * w_c[0]).detach()
    ov["mask_chem"] = torch.tensor(
        ~m[ccol].isin(train_chems).to_numpy(), device=device)
    return ov


def train_groups(h: Harness):
    m = h.m_train
    return (sorted(m["Strains"].unique().tolist()),
            sorted(m["perturbation_no_concentration"].unique().tolist()))


# ---------------------------------------------------------------- 实验流程

def _load_sweep(path: Path):
    return json.loads(path.read_text()) if path.exists() else []


def run(h: Harness, augs: list, seeds: list, device: str = "cuda"):
    """G0/G1/G2/G4 训练矩阵：逐 (aug, seed) 训练 → 预测 → score_val → 缓存。"""
    sweep_path = OUT_DIR / "sweep.json"
    results = _load_sweep(sweep_path)
    done = {(r["aug"], r["seed"]) for r in results}
    for aug in augs:
        for seed in seeds:
            if (aug, seed) in done:
                print(f"[skip] {aug} seed={seed} 已完成", flush=True)
                continue
            cfg = Cfg(aug=aug)
            t0 = time.time()
            print(f"[run] {aug} seed={seed} cfg={asdict(cfg)}", flush=True)
            model, enc, mean, std, strain_ids, chem_ids, loss = train_one(
                h, cfg, seed, device, log_every=50)
            if aug == "G4":
                finetune_holdout(h, cfg, seed, model, strain_ids, device,
                                 log_every=10)
            pred = predict(model, enc, mean, std, h.m_tr, device)
            np.save(CACHE / f"pred_{aug}_s{seed}.npy", pred)
            torch.save(model.state_dict(), CACHE / f"model_{aug}_s{seed}.pt")
            res = h.score_val(pred, verbose=False)
            results.append({
                "aug": aug, "seed": seed, "cfg": asdict(cfg),
                "composite": res["composite"], "per_split": res["per_split"],
                "final_loss": loss, "seconds": time.time() - t0})
            sweep_path.write_text(json.dumps(results, indent=1))
            print(f"[done] {aug} seed={seed} composite={res['composite']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    return results


def run_g3(h: Harness, seeds: list, device: str = "cuda",
           alphas=(0.0, 0.5, 1.0), bases=("G0", "G1")):
    """G3 UNK 先验：加载缓存模型，未见实体嵌入替换后 3 种子均值评分。"""
    g3_path = OUT_DIR / "g3.json"
    results = _load_sweep(g3_path)
    done = {(r["base"], r["alpha"]) for r in results}
    train_strains, train_chems = train_groups(h)
    n_prot = h.Y_tr.shape[1]
    cfg = Cfg()
    enc, _, mean, std, _, _ = get_tensors(h)
    for base in bases:
        for alpha in alphas:
            if (base, alpha) in done:
                print(f"[skip] G3 {base} alpha={alpha} 已完成", flush=True)
                continue
            t0 = time.time()
            preds = []
            for seed in seeds:
                model = ProteoMLP_G(enc.n_cats, n_prot, cfg).to(device)
                model.load_state_dict(torch.load(
                    CACHE / f"model_{base}_s{seed}.pt", map_location=device))
                ov = build_override(model, enc, h.m_tr, train_strains,
                                    train_chems, alpha, device)
                preds.append(predict(model, enc, mean, std, h.m_tr, device,
                                     override=ov))
                del model
            P = np.mean(preds, axis=0).astype(np.float32)
            np.save(CACHE / f"pred_G3{base}_a{alpha}.npy", P)
            res = h.score_val(P, verbose=False)
            results.append({"base": base, "alpha": alpha,
                            "composite": res["composite"],
                            "per_split": res["per_split"],
                            "seconds": time.time() - t0})
            g3_path.write_text(json.dumps(results, indent=1))
            print(f"[G3] {base} alpha={alpha} composite={res['composite']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    return results


def _key_metrics(per_split: dict) -> dict:
    """抽取报告用关键指标。"""
    def g(split, key):
        return per_split.get(split, {}).get(key, float("nan"))
    return {
        "strain_only_FC": g("val_strain_only", "FC_PCC"),
        "strain_only_resid": g("val_strain_only", "resid_PCC"),
        "strain_only_fid": g("val_strain_only", "fidelity"),
        "both_FC": g("val_both", "FC_PCC"),
        "both_fid": g("val_both", "fidelity"),
        "chem_only_FC": g("val_chem_only", "FC_PCC"),
        "chem_only_resid": g("val_chem_only", "resid_PCC"),
        "chem_only_fid": g("val_chem_only", "fidelity"),
        "time_FC": g("val_time", "FC_PCC"),
        "time_fid": g("val_time", "fidelity"),
    }


def finalize(h: Harness, augs: list, seeds: list, device: str = "cuda"):
    """集成各 aug 的多种子均值 → 评分 → 与 G3 结果一起选最佳 → 交付文件。"""
    final_path = OUT_DIR / "final_results.json"
    out = {"ensembles": {}, "g3": _load_sweep(OUT_DIR / "g3.json")}

    candidates = {}  # name -> (composite, pred_path)
    for aug in augs:
        paths = [CACHE / f"pred_{aug}_s{s}.npy" for s in seeds]
        if not all(p.exists() for p in paths):
            print(f"[finalize] 跳过 {aug}（缓存不全）", flush=True)
            continue
        P = np.mean([np.load(p) for p in paths], axis=0).astype(np.float32)
        np.save(CACHE / f"pred_{aug}_ens.npy", P)
        print(f"[finalize] {aug} {len(paths)} 种子均值集成评分：", flush=True)
        res = h.score_val(P, verbose=True)
        out["ensembles"][aug] = {"composite": res["composite"],
                                 "per_split": res["per_split"],
                                 "key": _key_metrics(res["per_split"])}
        candidates[aug] = (res["composite"], CACHE / f"pred_{aug}_ens.npy")

    for r in out["g3"]:
        name = f"G3({r['base']},alpha={r['alpha']})"
        candidates[name] = (r["composite"],
                            CACHE / f"pred_G3{r['base']}_a{r['alpha']}.npy")
        out["ensembles"][name] = {"composite": r["composite"],
                                  "per_split": r["per_split"],
                                  "key": _key_metrics(r["per_split"])}

    assert candidates, "没有可用候选"
    best = max(candidates, key=lambda k: candidates[k][0])
    best_comp, best_path = candidates[best]
    P = np.load(best_path).astype(np.float32)
    assert P.shape == (len(h.m_tr), h.Y_tr.shape[1]), P.shape
    assert not np.isnan(P).any() and not np.isinf(P).any(), "预测含 NaN/Inf"
    np.save(OUT_DIR / "pred_trainval.npy", P)
    out["best"] = {"name": best, "composite": best_comp}
    final_path.write_text(json.dumps(out, indent=1))
    print(f"\n[finalize] 最佳配置 = {best}  composite={best_comp:.4f}", flush=True)
    print(f"[finalize] pred_trainval.npy 已保存 {P.shape}", flush=True)

    # 汇总对比表（供报告）
    hdr = (f"{'config':<24}{'composite':>10}{'sFC':>8}{'sResid':>8}"
           f"{'bFC':>8}{'cFC':>8}{'cResid':>8}{'tFC':>8}")
    print(hdr)
    for name, (comp, _) in sorted(candidates.items(), key=lambda kv: -kv[1][0]):
        k = out["ensembles"][name]["key"]
        print(f"{name:<24}{comp:>10.4f}{k['strain_only_FC']:>8.4f}"
              f"{k['strain_only_resid']:>8.4f}{k['both_FC']:>8.4f}"
              f"{k['chem_only_FC']:>8.4f}{k['chem_only_resid']:>8.4f}"
              f"{k['time_FC']:>8.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--g3", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--augs", default="G0,G1,G2,G4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    augs = args.augs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    do_all = not (args.run or args.g3 or args.finalize)

    h = Harness()
    if args.run or do_all:
        run(h, augs, seeds, args.device)
    if args.g3 or do_all:
        run_g3(h, seeds, args.device)
    if args.finalize or do_all:
        finalize(h, augs, seeds, args.device)


if __name__ == "__main__":
    main()
