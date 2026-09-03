"""wsD：MLP 架构/超参冲刺 + 多种子集成（封闭数据榜）。

在 src/train_mlp.py 的实体嵌入 MLP 基础上做系统调优：
- 容量 / 训练长度 / lr / weight_decay / embedding dropout / dropout / batch / 化合物嵌入维度
- 结构增强：残差连接、低秩输出头、Huber loss、FiLM 特征门控（scale/shift 由条件嵌入生成）

流程：粗扫（每配置 2 种子 × 100 epochs 比 composite）→ top3 精调（更长 epochs + 3 种子）
→ 最佳配置 8 种子（0-7）均值集成 → outputs/wsD/pred_trainval.npy。

合规：仅用 h.tr_rows 训练；val 划分只用于评分；不触碰 h.Y_te；随机种子固定。

用法:
    python -m src.wsD_arch --plan plan.json --seeds 0,1 --tag stageA
    python -m src.wsD_arch --final '{"hidden":[1024,2048],"epochs":200}'
"""
from __future__ import annotations

import argparse
import copy
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

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "wsD"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Cfg:
    hidden: list = field(default_factory=lambda: [512, 1024])
    epochs: int = 100
    lr: float = 1e-3
    wd: float = 1e-4
    emb_drop: float = 0.25
    p_drop: float = 0.1
    bs: int = 256
    chem_emb: int = 32
    residual: bool = False
    lowrank: int = 0          # 0 = 全连接头；>0 = 低秩头秩 r
    loss: str = "mse"         # mse | huber
    film: bool = False
    g2_aug: bool = False      # G2 组级增强：每 epoch 随机 1 菌株+1 化合物整组 UNK，
                              # 其余逐样本 0.15（启用时覆盖 emb_drop 逻辑）

    @classmethod
    def from_dict(cls, d: dict) -> "Cfg":
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


class Block(nn.Module):
    """Linear → GELU → LayerNorm → Dropout；同维时可残差；可 FiLM 门控。"""

    def __init__(self, d_in, d_out, p_drop, residual=False, film_dim=0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p_drop)
        self.residual = residual and d_in == d_out
        if film_dim:
            self.film = nn.Linear(film_dim, 2 * d_out)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)  # 初始为恒等（scale=1, shift=0）
        else:
            self.film = None

    def forward(self, x, cond=None):
        h = self.drop(self.norm(self.act(self.fc(x))))
        if self.film is not None:
            g, b = self.film(cond).chunk(2, dim=-1)
            h = h * (1.0 + g) + b
        if self.residual:
            h = h + x
        return h


class ProteoMLP2(nn.Module):
    def __init__(self, n_cats, n_prot, cfg: Cfg):
        super().__init__()
        dims = dict(EMB_DIMS)
        dims["perturbation_no_concentration"] = cfg.chem_emb
        self.embs = nn.ModuleList([
            nn.Embedding(n, dims[c]) for n, c in zip(n_cats, CAT_COLS)])
        d_in = sum(dims[c] for c in CAT_COLS)
        cond_dim = d_in if cfg.film else 0
        self.blocks = nn.ModuleList()
        d = d_in
        for hdim in cfg.hidden:
            self.blocks.append(Block(d, hdim, cfg.p_drop, cfg.residual, cond_dim))
            d = hdim
        if cfg.lowrank and cfg.lowrank > 0:
            self.head = nn.Sequential(nn.Linear(d, cfg.lowrank, bias=False),
                                      nn.Linear(cfg.lowrank, n_prot))
        else:
            self.head = nn.Linear(d, n_prot)

    def forward(self, x_cat):
        e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        h = e
        for blk in self.blocks:
            h = blk(h, e)
        return self.head(h)


def masked_huber(pred, target, mask, beta: float = 1.0):
    se = F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * mask
    return se.sum() / mask.sum().clamp_min(1.0)


_TENSORS = {}


def get_tensors(h: Harness, stats=None):
    """编码/标准化张量只构建一次（Encoder 由排序后的类别决定，与种子无关）。

    stats=None 用 train split 冻结统计（本地验证）；传入全量 train_val 重估的
    FrozenStats 用于最终 test 提交重训（仍不涉及 Y_te）。
    """
    key = "t" if stats is None else "t_full"
    if key not in _TENSORS:
        enc = Encoder().fit(h.m_tr)  # 类别覆盖 train_val 全集（同基线实现）
        stats = stats or h.stats
        X_all = torch.tensor(enc.transform(h.m_tr), dtype=torch.long)
        mean = torch.tensor(stats.protein_mean, dtype=torch.float32)
        std = torch.tensor(stats.protein_std, dtype=torch.float32)
        Z_all = (torch.tensor(h.Y_tr, dtype=torch.float32) - mean) / std
        M_all = ~torch.isnan(Z_all)
        Z_all = torch.nan_to_num(Z_all, nan=0.0)
        _TENSORS[key] = (enc, X_all, mean, std, Z_all, M_all)
    return _TENSORS[key]


def train_one(h: Harness, cfg: Cfg, seed: int, device: str = "cuda",
              log_every: int = 0, track: bool = False, rows=None,
              tensors=None):
    """单次训练。track=True 时记录 train loss 与 val 划分 masked-MSE（仅用于
    报告中的过拟合观察，不参与训练与模型选择）。
    rows=None 默认 h.tr_rows；tensors 可传入全量统计版本（test 重训用）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, X_all, mean, std, Z_all, M_all = (
        tensors if tensors is not None else get_tensors(h))
    if rows is None:
        rows = h.tr_rows
    n_prot = h.Y_tr.shape[1]

    model = ProteoMLP2(enc.n_cats, n_prot, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_steps = cfg.epochs * int(np.ceil(len(rows) / cfg.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    X_dev = X_all.to(device)
    Z_dev = Z_all.to(device)
    M_dev = M_all.to(device)
    rows_dev = torch.tensor(rows, device=device)
    if track:
        vr = np.where((h.m_tr["split_final"] != "train").to_numpy())[0]
        vr_dev = torch.tensor(vr, device=device)
    if cfg.g2_aug:
        # G2 候选组：训练行中实际出现过的菌株/化合物编码（排除 UNK=0）
        tr_s = np.unique(X_all[rows, 0].numpy()); tr_s = tr_s[tr_s > 0]
        tr_c = np.unique(X_all[rows, 1].numpy()); tr_c = tr_c[tr_c > 0]

    hist = []
    for ep in range(cfg.epochs):
        model.train()
        if cfg.g2_aug:
            gs = int(np.random.choice(tr_s))  # 本 epoch 整组 UNK 的菌株
            gc = int(np.random.choice(tr_c))  # 本 epoch 整组 UNK 的化合物
        perm = rows_dev[torch.randperm(len(rows_dev), device=device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg.bs):
            r = perm[i:i + cfg.bs]
            xb = X_dev[r].clone()
            if cfg.g2_aug:
                # 组级 UNK：模拟整个实体未见（与 G3 推理修复配套）
                xb[xb[:, 0] == gs, 0] = 0
                xb[xb[:, 1] == gc, 1] = 0
                # 其余逐样本 0.15
                for col in [0, 1]:
                    dm = torch.rand(len(r), device=device) < 0.15
                    xb[dm, col] = 0
            elif cfg.emb_drop > 0:
                # embedding dropout：菌株/化合物随机替换为 UNK(0)，学 OOD 回退
                for col in [0, 1]:
                    dm = torch.rand(len(r), device=device) < cfg.emb_drop
                    xb[dm, col] = 0
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
        if track and ((ep + 1) % 10 == 0 or ep == cfg.epochs - 1):
            model.eval()
            with torch.no_grad():
                vm = masked_mse(model(X_dev[vr_dev]), Z_dev[vr_dev],
                                M_dev[vr_dev].float()).item()
            hist.append({"epoch": ep + 1, "train_loss": tot / nb, "val_mse": vm})
        if log_every and ((ep + 1) % log_every == 0 or ep == cfg.epochs - 1):
            print(f"  epoch {ep+1:>3}/{cfg.epochs}  loss={tot/nb:.4f}", flush=True)
    return model, enc, mean, std, hist


def seen_cats(h: Harness, rows: np.ndarray) -> dict:
    """训练行中实际出现过的类别值集合（G3 推理修复用）。"""
    mt = h.m_tr.iloc[rows]
    return {c: set(mt[c].unique()) for c in CAT_COLS}


def transform_g3(enc: Encoder, m, seen: dict) -> np.ndarray:
    """G3：不在训练行类别集合中的实体强制映射为 UNK(0)。

    修复 Encoder 在 train_val 全集 fit 时，val 未见实体（BAI、val 化合物等）
    分到未训练随机嵌入行的问题。训练行的类别全部在 seen 中，不受影响。
    """
    cols = []
    for c in CAT_COLS:
        mp = enc.maps[c]
        keep = seen[c]
        cols.append(m[c].map(lambda v: mp[v] if v in keep else 0).to_numpy())
    return np.stack(cols, axis=1)


@torch.no_grad()
def predict(model, enc, mean, std, m, device="cuda", bs: int = 1024,
            g3_seen: dict | None = None) -> np.ndarray:
    model.eval()
    idx = transform_g3(enc, m, g3_seen) if g3_seen is not None else enc.transform(m)
    X = torch.tensor(idx, dtype=torch.long, device=device)
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(X[i:i + bs]).float().cpu())
    Z = torch.cat(outs).numpy()
    return (Z * std.numpy() + mean.numpy()).astype(np.float32)


def run_plan(h: Harness, plan: list, seeds: list, tag: str):
    """批量跑配置，结果增量写入 outputs/wsD/sweep_<tag>.json。"""
    out_json = OUT_DIR / f"sweep_{tag}.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else []
    done = {(r["name"], tuple(r["seeds"])) for r in results}
    for item in plan:
        name = item["name"]
        if (name, tuple(seeds)) in done:
            print(f"[skip] {name} 已完成", flush=True)
            continue
        cfg = Cfg.from_dict(item.get("cfg", {}))
        comps, per_seed = [], []
        t0 = time.time()
        for s in seeds:
            model, enc, mean, std, _ = train_one(h, cfg, s, log_every=0)
            pred = predict(model, enc, mean, std, h.m_tr)
            res = h.score_val(pred, verbose=False)
            comps.append(res["composite"])
            per_seed.append({"seed": s, "composite": res["composite"],
                             "per_split": res["per_split"]})
            del model
            torch.cuda.empty_cache()
        rec = {"name": name, "cfg": asdict(cfg), "seeds": list(seeds),
               "composites": comps, "mean": float(np.mean(comps)),
               "per_seed": per_seed, "seconds": time.time() - t0}
        results.append(rec)
        out_json.write_text(json.dumps(results, indent=1))
        print(f"{name:<38} mean={np.mean(comps):.4f}  "
              f"{['%.4f' % c for c in comps]}  ({time.time()-t0:.0f}s)", flush=True)
    return results


def run_final(h: Harness, cfg: Cfg, seeds=range(8)):
    """最佳配置多种子集成：均值预测 → pred_trainval.npy + final_results.json。"""
    preds, per_seed, hists = [], [], []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std, hist = train_one(h, cfg, s, track=(s == 0))
        pred = predict(model, enc, mean, std, h.m_tr)
        res = h.score_val(pred, verbose=False)
        per_seed.append({"seed": s, "composite": res["composite"],
                         "per_split": res["per_split"]})
        hists.append({"seed": s, "hist": hist})
        preds.append(pred)
        print(f"[final] seed={s} composite={res['composite']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert not np.isnan(P).any() and not np.isinf(P).any(), "预测含 NaN/Inf"
    np.save(OUT_DIR / "pred_trainval.npy", P)
    print("[final] 集成预测已保存，评分：")
    res = h.score_val(P, verbose=True)
    out = {"cfg": asdict(cfg), "seeds": list(seeds), "per_seed": per_seed,
           "ensemble": res, "history": hists}
    (OUT_DIR / "final_results.json").write_text(json.dumps(out, indent=1))
    return res


def run_final_test(h: Harness, cfg: Cfg, seeds=range(8)):
    """最终 test 提交：全部 train_val 重训（统计全量重估）→ 8 种子均值 → pred_test.npy。

    严禁事项：不使用 h.Y_te 的任何数值；统计仅用 train_val 全量行。
    """
    from . import data as D
    rows = np.arange(len(h.m_tr))
    stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)  # 全量 train_val 重估
    tensors = get_tensors(h, stats=stats)
    preds = []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std, _ = train_one(h, cfg, s, rows=rows,
                                             tensors=tensors)
        pred = predict(model, enc, mean, std, h.m_te)
        preds.append(pred.astype(np.float32))
        print(f"[test] seed={s} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert P.shape == (len(h.m_te), h.Y_tr.shape[1]), P.shape
    assert not np.isnan(P).any() and not np.isinf(P).any(), "预测含 NaN/Inf"
    np.save(OUT_DIR / "pred_test.npy", P)
    (OUT_DIR / "test_results.json").write_text(json.dumps(
        {"cfg": asdict(cfg), "seeds": list(seeds), "shape": list(P.shape),
         "train_rows": int(len(rows))}, indent=1))
    print(f"[test] saved {OUT_DIR/'pred_test.npy'} shape={P.shape} dtype={P.dtype}")
    return P


def _ensemble_val(h: Harness, cfg: Cfg, seeds, g3: bool, ckpt_dir: str,
                  out_name: str, results_name: str):
    """tr_rows 训练多种子 →（可选 G3 修复）推理 → 均值集成 → 保存 + score_val。"""
    seen = seen_cats(h, h.tr_rows) if g3 else None
    ck = OUT_DIR / ckpt_dir
    ck.mkdir(parents=True, exist_ok=True)
    preds, per_seed = [], []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std, _ = train_one(h, cfg, s)
        pred = predict(model, enc, mean, std, h.m_tr, g3_seen=seen)
        res = h.score_val(pred, verbose=False)
        per_seed.append({"seed": s, "composite": res["composite"],
                         "per_split": res["per_split"]})
        preds.append(pred)
        torch.save(model.state_dict(), ck / f"seed{s}.pt")
        print(f"[{out_name}] seed={s} composite={res['composite']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert not np.isnan(P).any() and not np.isinf(P).any(), "预测含 NaN/Inf"
    np.save(OUT_DIR / out_name, P)
    print(f"[{out_name}] 集成评分：")
    res = h.score_val(P, verbose=True)
    (OUT_DIR / results_name).write_text(json.dumps(
        {"cfg": asdict(cfg), "g3": g3, "seeds": list(seeds),
         "per_seed": per_seed, "ensemble": res}, indent=1))
    return res


def run_g3(h: Harness, cfg: Cfg, seeds):
    """交付 A：原配方 8 种子 val 模型（同代码同种子复现，权重落盘）+ 仅 G3 推理。"""
    return _ensemble_val(h, cfg, seeds, g3=True, ckpt_dir="ckpt_val",
                         out_name="pred_trainval_g3.npy",
                         results_name="g3_results.json")


def run_g2g3(h: Harness, cfg: Cfg, seeds):
    """交付 B：G2 组级增强重训 8 种子 + G3 推理。"""
    cfg2 = Cfg.from_dict({**asdict(cfg), "g2_aug": True})
    print(f"[g2g3] G2 增强生效：组级 UNK + 逐样本 0.15（覆盖 emb_drop={cfg.emb_drop}）")
    return _ensemble_val(h, cfg2, seeds, g3=True, ckpt_dir="ckpt_val_g2",
                         out_name="pred_trainval_g2g3.npy",
                         results_name="g2g3_results.json")


def run_test_g2(h: Harness, cfg: Cfg, seeds):
    """交付 C：G2 增强 + 全量 train_val 重训 8 种子 → pred_test_g2.npy。
    G3 对 test 无影响（全量重训已覆盖 train_val 全部类别；test 新实体本就
    经 Encoder 映射 UNK(0)），故用普通推理。"""
    from . import data as D
    cfg2 = Cfg.from_dict({**asdict(cfg), "g2_aug": True})
    rows = np.arange(len(h.m_tr))
    stats = D.FrozenStats(h.m_tr, h.Y_tr, rows=rows)  # 全量 train_val 重估
    tensors = get_tensors(h, stats=stats)
    preds = []
    for s in seeds:
        t0 = time.time()
        model, enc, mean, std, _ = train_one(h, cfg2, s, rows=rows,
                                             tensors=tensors)
        pred = predict(model, enc, mean, std, h.m_te)
        preds.append(pred.astype(np.float32))
        print(f"[test_g2] seed={s} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    P = np.mean(preds, axis=0).astype(np.float32)
    assert P.shape == (len(h.m_te), h.Y_tr.shape[1]), P.shape
    assert not np.isnan(P).any() and not np.isinf(P).any(), "预测含 NaN/Inf"
    np.save(OUT_DIR / "pred_test_g2.npy", P)
    (OUT_DIR / "test_g2_results.json").write_text(json.dumps(
        {"cfg": asdict(cfg2), "seeds": list(seeds), "shape": list(P.shape),
         "train_rows": int(len(rows))}, indent=1))
    print(f"[test_g2] saved {OUT_DIR/'pred_test_g2.npy'} "
          f"shape={P.shape} dtype={P.dtype}")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="JSON 文件: [{name, cfg}, ...]")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--final", default=None, help="最佳配置 JSON 字符串")
    ap.add_argument("--final_test", default=None,
                    help="全量 train_val 重训 + test 预测，配置 JSON 字符串")
    ap.add_argument("--g3", default=None,
                    help="原配方 val 模型 + 仅 G3 推理修复，配置 JSON 字符串")
    ap.add_argument("--g2g3", default=None,
                    help="G2 增强重训 + G3 推理，配置 JSON 字符串")
    ap.add_argument("--test_g2", default=None,
                    help="G2 增强 + 全量重训 + test 预测，配置 JSON 字符串")
    ap.add_argument("--final_seeds", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    h = Harness()
    if args.g3 is not None:
        cfg = Cfg.from_dict(json.loads(args.g3))
        seeds = [int(s) for s in args.final_seeds.split(",")]
        print(f"[g3] cfg={asdict(cfg)} seeds={seeds}")
        run_g3(h, cfg, seeds)
        return
    if args.g2g3 is not None:
        cfg = Cfg.from_dict(json.loads(args.g2g3))
        seeds = [int(s) for s in args.final_seeds.split(",")]
        print(f"[g2g3] cfg={asdict(cfg)} seeds={seeds}")
        run_g2g3(h, cfg, seeds)
        return
    if args.test_g2 is not None:
        cfg = Cfg.from_dict(json.loads(args.test_g2))
        seeds = [int(s) for s in args.final_seeds.split(",")]
        print(f"[test_g2] cfg={asdict(cfg)} seeds={seeds}")
        run_test_g2(h, cfg, seeds)
        return
    if args.final_test is not None:
        cfg = Cfg.from_dict(json.loads(args.final_test))
        seeds = [int(s) for s in args.final_seeds.split(",")]
        print(f"[test] cfg={asdict(cfg)} seeds={seeds}")
        run_final_test(h, cfg, seeds)
        return
    if args.final is not None:
        cfg = Cfg.from_dict(json.loads(args.final))
        seeds = [int(s) for s in args.final_seeds.split(",")]
        print(f"[final] cfg={asdict(cfg)} seeds={seeds}")
        run_final(h, cfg, seeds)
        return

    plan = json.loads(Path(args.plan).read_text())
    seeds = [int(s) for s in args.seeds.split(",")]
    run_plan(h, plan, seeds, args.tag)


if __name__ == "__main__":
    main()
