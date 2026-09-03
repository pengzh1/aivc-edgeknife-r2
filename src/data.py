"""数据加载与预处理：GOAI 方向一 虚拟酵母扰动响应预测.

- log2 变换原始蛋白强度（全部为正值，无需 pseudo-count）
- 缺失值掩膜（~24% 缺失，训练时用 masked loss / 冻结统计插补）
- 官方规则对照匹配：同 data_source/Strains/Medium/Temperature/pert_time/
  instrument/Yeast_cell_plate 组内匹配 DMSO/Water 对照
- 所有归一化统计仅用 train split 冻结

合规守卫（复现包用）：环境变量 GOAI_NO_TEST_GT=1 时，本模块对 test 蛋白
矩阵只做"全 NaN 占位壳"——不读取、不加载、不缓存任何 test 蛋白数值
（含 test 对照真值）；test 行序取自官方 test metadata。此时
WAYB_WAYC_proteome_raw_test.csv 即使不存在，全部流程也照常运行
（对应主办方"推理侧仅挂载 test metadata"的复现方式）。
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "input"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

META_TR = DATA_DIR / "WAYB_WAYC_metadata_train_val(1).csv"
META_TE = DATA_DIR / "WAYB_WAYC_metadata_test(1).csv"
PROT_TR = DATA_DIR / "WAYB_WAYC_proteome_raw_train_val.csv"
PROT_TE = DATA_DIR / "WAYB_WAYC_proteome_raw_test.csv"

CONTROLS = {"DMSO", "Water"}
QC = "Quality Control"

# 对照匹配键（官方规则）
MATCH_KEYS = ["data_source", "Strains", "Medium", "Temperature",
              "pert_time", "instrument", "Yeast_cell_plate"]
# 生物上下文键（用于 μ_ctx / μ_drug 等参照统计）
CTX_KEYS = ["Strains", "Medium", "Temperature", "pert_time"]

SPLITS_TR = ["train", "val_chem_only", "val_strain_only", "val_both", "val_time"]
SPLITS_TE = ["test_chem_only", "test_strain_only", "test_both", "test_time"]

# 复现包合规守卫：置 1 后 test 蛋白数值零接触（见模块 docstring）
NO_TEST_GT = os.environ.get("GOAI_NO_TEST_GT") == "1"


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    m_tr = pd.read_csv(META_TR)
    m_te = pd.read_csv(META_TE)
    return m_tr, m_te


def load_proteome_log2(force: bool = False):
    """加载蛋白组矩阵并做 log2 变换，返回 (Y_tr, Y_te, proteins)。

    Y_* 为 float32 矩阵，NaN 表示缺失。带 npz 缓存。
    GOAI_NO_TEST_GT=1 时：不读 test 蛋白文件（可缺席），Y_te 返回全 NaN
    占位壳（行序 = 官方 test metadata 顺序），且不读写共享 npz 缓存。
    """
    if NO_TEST_GT:
        cache = CACHE_DIR / "proteome_log2.npz"
        if cache.exists() and not force:
            z = np.load(cache, allow_pickle=False)
            Y_tr, proteins = z["Y_tr"], z["proteins"].astype(str)
        else:
            df_tr = pd.read_csv(PROT_TR)
            proteins = df_tr.columns[1:].to_numpy().astype(str)
            Y_tr = np.log2(df_tr.iloc[:, 1:].to_numpy(dtype=np.float32))
        n_te = len(pd.read_csv(META_TE, usecols=["sample_ID"]))
        Y_te = np.full((n_te, len(proteins)), np.nan, dtype=np.float32)
        return Y_tr, Y_te, proteins

    cache = CACHE_DIR / "proteome_log2.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        return z["Y_tr"], z["Y_te"], z["proteins"].astype(str)

    df_tr = pd.read_csv(PROT_TR)
    df_te = pd.read_csv(PROT_TE)
    proteins = df_tr.columns[1:].to_numpy().astype(str)
    assert (df_te.columns[1:].to_numpy().astype(str) == proteins).all()
    Y_tr = np.log2(df_tr.iloc[:, 1:].to_numpy(dtype=np.float32))
    Y_te = np.log2(df_te.iloc[:, 1:].to_numpy(dtype=np.float32))
    np.savez_compressed(cache, Y_tr=Y_tr, Y_te=Y_te, proteins=proteins)
    return Y_tr, Y_te, proteins


def load_aligned():
    """返回 metadata 与蛋白矩阵按 sample_ID 对齐后的视图。

    返回 dict，包含:
      m_tr, m_te: metadata（顺序与 Y 行一致）
      Y_tr, Y_te: log2 蛋白矩阵
      proteins: 蛋白名数组
      idx_tr/idx_te: sample_ID -> 行号
    """
    m_tr, m_te = load_metadata()
    Y_tr, Y_te, proteins = load_proteome_log2()
    # metadata 与 proteome 行顺序一致性检查（已验证集合相等，这里按 ID 对齐保险）
    prot_tr_ids = pd.read_csv(PROT_TR, usecols=["sample_ID"])["sample_ID"]
    if NO_TEST_GT:
        # 守卫模式：test 蛋白文件不读（可缺席）；metadata 顺序即官方顺序
        # （与蛋白文件行序的一致性已在冻结认证中核验，见 certify_final.json）
        prot_te_ids = m_te["sample_ID"]
    else:
        prot_te_ids = pd.read_csv(PROT_TE, usecols=["sample_ID"])["sample_ID"]
    m_tr = m_tr.set_index("sample_ID").loc[prot_tr_ids].reset_index()
    m_te = m_te.set_index("sample_ID").loc[prot_te_ids].reset_index()
    return {
        "m_tr": m_tr, "m_te": m_te, "Y_tr": Y_tr, "Y_te": Y_te,
        "proteins": proteins,
        "idx_tr": {s: i for i, s in enumerate(m_tr["sample_ID"])},
        "idx_te": {s: i for i, s in enumerate(m_te["sample_ID"])},
    }


def build_control_map(m_all: pd.DataFrame) -> dict:
    """每个处理样本 -> 精确匹配对照样本 ID 列表（跨 train/test 文件搜索）。

    返回 {sample_ID: [ctrl_ids...]}。覆盖率经验证为 100%。
    """
    ctrl = m_all[m_all["perturbation_no_concentration"].isin(CONTROLS)]
    groups = ctrl.groupby(MATCH_KEYS)["sample_ID"].apply(list).to_dict()
    out = {}
    is_treat = ~m_all["perturbation_no_concentration"].isin(CONTROLS | {QC})
    for row in m_all[is_treat].itertuples():
        key = tuple(getattr(row, k) for k in MATCH_KEYS)
        out[row.sample_ID] = groups.get(key, [])
    return out


def ctx_key(row) -> tuple:
    return tuple(row[k] for k in CTX_KEYS)


class FrozenStats:
    """冻结统计量。rows=None 时用 train split；也可指定行（如全量 train_val）。"""

    def __init__(self, m_tr: pd.DataFrame, Y_tr: np.ndarray,
                 rows: np.ndarray | None = None):
        if rows is None:
            rows = np.where((m_tr["split_final"] == "train").to_numpy())[0]
        self.m_train = m_tr.iloc[rows].reset_index(drop=True)
        self.Y_train = Y_tr[rows]
        # 每蛋白均值/标准差（log2 空间，忽略缺失）；全缺失蛋白回退到全局均值
        self.protein_mean = np.nanmean(self.Y_train, axis=0)
        global_mean = np.nanmean(self.Y_train)
        self.protein_mean = np.where(np.isnan(self.protein_mean),
                                     global_mean, self.protein_mean)
        self.protein_std = np.nanstd(self.Y_train, axis=0)
        self.protein_std = np.where(np.isnan(self.protein_std) | (self.protein_std < 1e-6),
                                    1.0, self.protein_std)
        # 每蛋白缺失率
        self.protein_miss = np.isnan(self.Y_train).mean(axis=0)

    def impute(self, Y: np.ndarray) -> np.ndarray:
        """用冻结的 train 蛋白均值填补缺失。"""
        out = Y.copy()
        r, c = np.where(np.isnan(out))
        out[r, c] = np.take(self.protein_mean, c)
        return out
