"""HIST_models step1：构建三概念 + 未来特征每日截面数据（供 HIST_models/models.py 的 HIST_FactorGCL 使用）【duckdb 版】

三概念 (GCN + GCN + 对比学习) 需要在同一股票集合上同时提供三种预定义概念，并附加未来特征：
    concept 1 : v0    —— 一级行业（IndustryCodeNew，来自 data/raw_data/stock_edge.pqt）【固定】
    concept 2 : v{n1} —— 可选边版本（data/edges/edges_v{n1}/{day_int}.npz）的连通分量【可选】
    concept 3 : v{n2} —— 可选边版本（data/edges/edges_v{n2}/{day_int}.npz）的连通分量【可选】
    x_fut     : 当日特征之后第 FUT_SHIFT 个交易日的特征（随 --horizon 变化：5d→5，20d→20），仅训练/验证集提供

支持的边版本：2, 3, 7, 9, 10, 12, 14

对齐方式：
    训练/验证集节点 = 特征 ∩ 标签 ∩ (有行业且有市值) ∩ edges_v{n1} 涉及股票 ∩ edges_v{n2} 涉及股票
    测试集节点 = 特征 ∩ (有行业且有市值) ∩ edges_v{n1} 涉及股票 ∩ edges_v{n2} 涉及股票（不依赖标签）
    训练/验证集额外要求：节点 ∈ 未来日特征（保证 x_fut 对齐）
    每日独立编码：
        industry_v0   : 行业整数编码（训练时 one-hot 成概念矩阵）
        industry_v{n1}: 第二边连通分量标签（训练时 one-hot 成概念矩阵）
        industry_v{n2}: 第三边连通分量标签（训练时 one-hot 成概念矩阵）

注意：
    1. 本脚本直接从 edges_v{n1} / edges_v{n2} 原始边文件在线计算连通分量，
       不依赖其它 hist 数据集的产出。
    2. 训练集/验证集：若某日之后不足 FUT_SHIFT 个交易日或未来特征缺失，该日被剔除。
    3. 测试集不包含 x_fut（模型推理时使用 output=True 模式，无需未来数据）。

样本格式（与 v0v{n1}v{n2} 三概念兼容，训练/验证集多出 x_fut）：
    训练/验证集：
        x              [N, F] 特征（CAT_{freq}_LAG{lags}）
        x_fut          [N, F] 未来 FUT_SHIFT 交易日的特征（与 x 同一股票集合）
        industry_v0    [N]    行业整数编码（concept 1，固定）
        industry_v{n1} [N]    第二边连通分量标签（concept 2，可选）
        industry_v{n2} [N]    第三边连通分量标签（concept 3，可选）
        market_value   [N]    流通市值
        y              [N]    标签（Return5d 或 Return20d）
        date           [3]    年/月/日

    测试集（无 x_fut）：
        x              [N, F] 特征
        industry_v0    [N]    行业整数编码
        industry_v{n1} [N]    第二边连通分量标签
        industry_v{n2} [N]    第三边连通分量标签
        market_value   [N]    流通市值
        y              [N]    股票代码（方便导出预测值）
        date           [3]    年/月/日

运行：python DataBuilder/hist_data_builder/build_FactorGCL_data_duckdb.py --edge-version-2 2 --edge-version-3 3 --freq 1D --lags 20 60 120 --label data/label/stock_ret.pqt --horizon 5d
特征文件：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt
标签文件：data/label/*.pqt（parquet 格式，通过 --label 参数指定，默认 data/label/stock_ret.pqt）
标签长度：通过 --horizon 参数指定，可选 5d（Return5d）或 20d（Return20d），默认 5d
保存：data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{n1}v{n2}_{label_name}_{horizon}_fut{fut_shift}/{train,valid,test}.pt
      其中 label_name 为标签文件名（不含扩展名），如 stock_ret、spec_ret；horizon 为标签长度，如 5d、20d
      fut_shift 为未来特征偏移（随 horizon 变化：5d→fut5，20d→fut20）
"""
import argparse
import os
from time import time

import pandas as pd
import numpy as np
import torch
import duckdb
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

# v0 行业边固定
EDGE_PATH = "data/raw_data/stock_edge.pqt"

# 支持的边版本
SUPPORTED_EDGE_VERSIONS = [2, 3, 7, 9, 10, 12, 14]

# 支持的特征频率
SUPPORTED_FREQS = ["1D", "five_minute"]

# 默认标签路径
DEFAULT_LABEL_PATH = "data/label/stock_ret.pqt"

# 支持的标签长度（horizon）
SUPPORTED_HORIZONS = ["5d", "20d"]
HORIZON_TO_LABEL_MARK = {"5d": "Return5d", "20d": "Return20d"}
DEFAULT_HORIZON = "5d"

# 未来特征偏移（交易日数）随标签长度变化：5d→5，20d→20
HORIZON_TO_FUT_SHIFT = {"5d": 5, "20d": 20}

# 全局路径与参数（由 parse_args 设置）
FEATURE_PATH = None
SAVE_DIR = None
EDGE_DIR_2 = None
EDGE_DIR_3 = None
EDGE_VERSION_2 = None
EDGE_VERSION_3 = None
LABEL_PATH = None
LABEL_MARK = None
FUT_SHIFT = HORIZON_TO_FUT_SHIFT[DEFAULT_HORIZON]


def build_feature_path(freq: str, lags: list) -> str:
    """按命名标准拼接特征文件路径：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/feature/CAT_{freq}_LAG{lag_str}.pqt"


def build_save_dir(freq: str, lags: list, edge_version_2: int, edge_version_3: int, label_name: str, horizon: str, fut_shift: int) -> str:
    """按命名标准拼接保存目录：data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{n1}v{n2}_{label_name}_{horizon}_fut{fut_shift}"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/hist_{freq}_lag{lag_str}_v0v{edge_version_2}v{edge_version_3}_{label_name}_{horizon}_fut{fut_shift}"


def parse_args():
    """解析命令行参数，设置全局路径。"""
    global FEATURE_PATH, SAVE_DIR, EDGE_DIR_2, EDGE_DIR_3, EDGE_VERSION_2, EDGE_VERSION_3, LABEL_PATH, LABEL_MARK, FUT_SHIFT  # pylint:disable=W0603

    parser = argparse.ArgumentParser(
        description="构建三概念 + 未来特征每日截面数据（v0 + v{{n1}} + v{{n2}} + fut{{FUT_SHIFT}}）【duckdb 版】")
    parser.add_argument(
        "--edge-version-2", "-v2",
        type=int,
        required=True,
        choices=SUPPORTED_EDGE_VERSIONS,
        help=f"第二边的版本号，可选：{SUPPORTED_EDGE_VERSIONS}"
    )
    parser.add_argument(
        "--edge-version-3", "-v3",
        type=int,
        required=True,
        choices=SUPPORTED_EDGE_VERSIONS,
        help=f"第三边的版本号，可选：{SUPPORTED_EDGE_VERSIONS}"
    )
    parser.add_argument(
        "--freq", "-f",
        type=str,
        default="1D",
        choices=SUPPORTED_FREQS,
        help=f"特征频率，可选：{SUPPORTED_FREQS}，默认 1D"
    )
    parser.add_argument(
        "--lags", "-l",
        type=int,
        nargs="+",
        default=[20, 60, 120],
        help="回看周期列表，如 --lags 20 60 120，默认 20 60 120"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_LABEL_PATH,
        help=f"标签文件路径（.pqt 格式），默认 {DEFAULT_LABEL_PATH}"
    )
    parser.add_argument(
        "--horizon", "-hz",
        type=str,
        default=DEFAULT_HORIZON,
        choices=SUPPORTED_HORIZONS,
        help=f"标签长度，可选：{SUPPORTED_HORIZONS}（分别对应 Return5d / Return20d），默认 {DEFAULT_HORIZON}"
    )
    _args = parser.parse_args()

    if _args.edge_version_2 == _args.edge_version_3:
        raise ValueError(f"第二边和第三边版本不能相同: v{_args.edge_version_2}")

    # 设置标签路径并校验
    label_path = _args.label
    if not label_path.endswith(".pqt"):
        raise ValueError(f"标签文件必须为 .pqt 格式: {label_path}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")
    LABEL_PATH = label_path

    # 提取标签文件名（不含扩展名）用于保存目录命名
    label_name = os.path.splitext(os.path.basename(label_path))[0]

    # 设置标签列名（Return5d 或 Return20d）
    LABEL_MARK = HORIZON_TO_LABEL_MARK[_args.horizon]

    # 设置未来特征偏移（随标签长度变化：5d→5，20d→20）
    FUT_SHIFT = HORIZON_TO_FUT_SHIFT[_args.horizon]

    EDGE_VERSION_2 = _args.edge_version_2
    EDGE_VERSION_3 = _args.edge_version_3
    EDGE_DIR_2 = f"data/edges/edges_v{EDGE_VERSION_2}"
    EDGE_DIR_3 = f"data/edges/edges_v{EDGE_VERSION_3}"
    FEATURE_PATH = build_feature_path(_args.freq, _args.lags)
    SAVE_DIR = build_save_dir(_args.freq, _args.lags, _args.edge_version_2, _args.edge_version_3, label_name, _args.horizon, FUT_SHIFT)

    # 检查边目录是否存在
    if not os.path.isdir(EDGE_DIR_2):
        raise FileNotFoundError(f"第二边目录不存在: {EDGE_DIR_2}")
    if not os.path.isdir(EDGE_DIR_3):
        raise FileNotFoundError(f"第三边目录不存在: {EDGE_DIR_3}")

    # 检查特征文件是否存在
    if not os.path.isfile(FEATURE_PATH):
        raise FileNotFoundError(f"特征文件不存在: {FEATURE_PATH}")

    return _args


def find_connected_components(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Union-Find 求连通分量标签（从 0 连续编号）。孤立节点自成一分量。"""
    parent = np.arange(num_nodes)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            union(edge_index[0, i], edge_index[1, i])

    labels = np.array([find(i) for i in range(num_nodes)])
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def load_edge_day(edge_dir: str, day_int: int):
    """读取单日边文件，返回 (edge_index(2,E), codes_str)；文件缺失返回 None。

    codes_str 统一为 6 位股票代码字符串：
        v12/v14 原始 codes 即字符串（'U6'）；v2/v3/v7/v9/v10 为 int64 需 zfill。
    """
    edge_path = os.path.join(edge_dir, f"{day_int}.npz")
    if not os.path.exists(edge_path):
        return None

    npz = np.load(edge_path, allow_pickle=True)
    ei_np = npz['edge_index']                      # (2, E) int32，codes 位置空间
    codes = npz['codes']                           # int64（前导零丢失）或 str（v12/v14）
    if codes.dtype.kind in ('U', 'S', 'O'):        # str/bytes/object -> 直接使用（v12/v14）
        codes_str = codes.astype('U6')
    else:                                          # int64 -> zfill 6 位
        codes_str = np.char.zfill(codes.astype('U16'), 6)
    return ei_np, codes_str


def remap_and_components(ei_np: np.ndarray, codes_str: np.ndarray, nodes: pd.Index) -> np.ndarray:
    """把边端点重映射到 nodes 位置空间（无效端点的边丢弃），返回连通分量标签 [N]。"""
    src_c = codes_str[ei_np[0]]
    dst_c = codes_str[ei_np[1]]
    src = nodes.get_indexer(src_c)
    dst = nodes.get_indexer(dst_c)
    valid = (src >= 0) & (dst >= 0)
    src, dst = src[valid].astype(np.int64), dst[valid].astype(np.int64)

    edge_index_np = np.stack([src, dst], axis=0) if len(src) > 0 else np.zeros((2, 0), dtype=np.int64)
    return find_connected_components(edge_index_np, len(nodes))


class ParquetDayReader:
    """使用 DuckDB 进行 parquet 文件按日读取，性能优于 pandas。"""

    def __init__(self):
        self.con = duckdb.connect()
        # 并行只交给 joblib，避免每个 worker 内 DuckDB 再开很多线程
        self.con.execute("PRAGMA threads=1")

    def read_day(
        self,
        parquet_path: str,
        day: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        cols = "*" if columns is None else ", ".join(
            f'"{col}"' for col in columns
        )

        return self.con.execute(
            f"""
            SELECT {cols}
            FROM read_parquet(?)
            WHERE DataDate = ?
            """,
            [parquet_path, pd.Timestamp(day)],
        ).df()

    def get_unique_dates(self, parquet_path: str) -> pd.DatetimeIndex:
        """获取 parquet 文件中所有唯一的 DataDate。"""
        result = self.con.execute(
            """
            SELECT DISTINCT DataDate
            FROM read_parquet(?)
            ORDER BY DataDate
            """,
            [parquet_path],
        ).df()
        return pd.DatetimeIndex(result["DataDate"])


# 进程级缓存：loky 的每个 worker 都有各自独立的全局变量
_WORKER_READER = None


def get_worker_reader() -> ParquetDayReader:
    global _WORKER_READER  # pylint:disable=W0603

    if _WORKER_READER is None:
        _WORKER_READER = ParquetDayReader()

    return _WORKER_READER


def get_day_data(
    day: str,
    fut_day: str = None,
    label_mark: str | None = None,
    test_set: bool = False
) -> dict:

    if label_mark is None:
        label_mark = LABEL_MARK if LABEL_MARK is not None else HORIZON_TO_LABEL_MARK[DEFAULT_HORIZON]

    reader = get_worker_reader()

    # 读取特征
    ft_day = reader.read_day(FEATURE_PATH, day)
    if ft_day.empty:
        return None

    ft_day = (
        ft_day.drop(columns=["DataDate"])
        .set_index("SecuCode")
    )

    # 测试集不读取标签，不用标签对齐
    if not test_set:
        lb_day = reader.read_day(
            LABEL_PATH,
            day,
            columns=["DataDate", "SecuCode", label_mark],
        )
        if lb_day.empty:
            return None

        lb_day = (
            lb_day.drop(columns=["DataDate"])
            .set_index("SecuCode")
        )
    else:
        lb_day = None

    # v0：行业 + 市值（使用 duckdb 读取）
    edge_day = reader.read_day(
        EDGE_PATH,
        day,
        columns=["SecuCode", "DataDate", "IndustryCodeNew", "FloatMarketCap"],
    )
    if edge_day.empty:
        return None

    edge_day = edge_day.drop(columns=["DataDate"]).set_index("SecuCode")

    d = pd.to_datetime(day)
    day_int = d.year * 10000 + d.month * 100 + d.day

    # v{n1}：第二边
    v2 = load_edge_day(EDGE_DIR_2, day_int)
    if v2 is None:
        return None
    ei_v2, codes_v2 = v2

    # v{n2}：第三边
    v3 = load_edge_day(EDGE_DIR_3, day_int)
    if v3 is None:
        return None
    ei_v3, codes_v3 = v3

    edge_day = edge_day[edge_day["IndustryCodeNew"].notna() & edge_day["FloatMarketCap"].notna()]

    # 对齐：
    #   训练/验证集：特征 ∩ 标签 ∩ (行业+市值) ∩ v{n1} 节点 ∩ v{n2} 节点
    #   测试集：特征 ∩ (行业+市值) ∩ v{n1} 节点 ∩ v{n2} 节点（不用标签对齐）
    if not test_set:
        avail_idx = ft_day.index.intersection(lb_day.index)
    else:
        avail_idx = ft_day.index

    base_nodes = (avail_idx
                  .intersection(edge_day.index)
                  .intersection(pd.Index(codes_v2))
                  .intersection(pd.Index(codes_v3)))
    if len(base_nodes) < 2 or ei_v2.shape[1] == 0 or ei_v3.shape[1] == 0:
        return None

    # 训练/验证集：读取未来特征并进一步对齐节点
    ft_fut_day = None
    if not test_set and fut_day is not None:
        ft_fut_day = reader.read_day(FEATURE_PATH, fut_day)

        if ft_fut_day.empty:
            return None  # 未来特征文件缺失该日，跳过

        ft_fut_day = (
            ft_fut_day.drop(columns=["DataDate"])
            .set_index("SecuCode")
        )

        # 最终节点 = 基础节点 ∩ 未来特征节点（保证 x_fut 对齐）
        nodes = base_nodes.intersection(ft_fut_day.index)
        if len(nodes) < 2:
            return None
    else:
        # 测试集：无需未来特征
        nodes = base_nodes

    nodes = nodes.sort_values()
    ft_day = ft_day.loc[nodes]
    if not test_set:
        lb_day = lb_day.loc[nodes]
    edge_day = edge_day.loc[nodes]

    # v0 概念：行业整数编码（每日独立）
    industry_v0, _ = pd.factorize(edge_day["IndustryCodeNew"].values)

    # v{n1} / v{n2} 概念：连通分量标签
    industry_v2 = remap_and_components(ei_v2, codes_v2, nodes)
    industry_v3 = remap_and_components(ei_v3, codes_v3, nodes)

    x = torch.tensor(ft_day.values, dtype=torch.float)
    industry_v0_t = torch.tensor(industry_v0, dtype=torch.long)
    industry_v2_t = torch.tensor(industry_v2, dtype=torch.long)
    industry_v3_t = torch.tensor(industry_v3, dtype=torch.long)
    market_value = torch.tensor(edge_day["FloatMarketCap"].values, dtype=torch.float)
    if not test_set:
        y = torch.tensor(lb_day[label_mark].values, dtype=torch.float)
    else:
        y = torch.tensor(nodes.values.astype(int), dtype=torch.float)

    result = {
        "x": x,
        "industry_v0": industry_v0_t,
        f"industry_v{EDGE_VERSION_2}": industry_v2_t,
        f"industry_v{EDGE_VERSION_3}": industry_v3_t,
        "market_value": market_value,
        "y": y,
        "date": torch.tensor([d.year, d.month, d.day], dtype=torch.long),
    }

    # 训练/验证集加入未来特征；测试集不包含 x_fut
    if ft_fut_day is not None:
        result["x_fut"] = torch.tensor(ft_fut_day.loc[nodes].values, dtype=torch.float)

    return result


def get_avail_days_and_split(
    train_start: str, train_end: str,
    valid_start: str, valid_end: str,
    test_start: str
):
    """获取可用交易日并划分训练/验证/测试集。

    训练/验证集日期：特征 ∩ 边 ∩ 标签（需要标签）
    测试集日期：特征 ∩ 边（不依赖标签）
    """
    reader = ParquetDayReader()

    ft_days = reader.get_unique_dates(FEATURE_PATH)
    edge_days = reader.get_unique_dates(EDGE_PATH)
    lb_days = reader.get_unique_dates(LABEL_PATH)

    # 训练/验证集日期：需要与标签对齐
    days_with_label = ft_days.intersection(edge_days).intersection(lb_days)
    # 测试集日期：不依赖标签
    days_all = ft_days.intersection(edge_days)

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)
    test_start = pd.Timestamp(test_start)

    splits = {
        "train": days_with_label[(days_with_label >= train_start) & (days_with_label <= train_end)],
        "valid": days_with_label[(days_with_label >= valid_start) & (days_with_label <= valid_end)],
        "test": days_all[days_all >= test_start]
    }

    return splits, days_all.sort_values()


def build_and_save_split(split_name, days, all_days, label_mark: str | None = None, n_jobs=32):
    """并行生成每个交易日的三概念 + 未来特征截面数据，按日期顺序保存为单个 .pt 文件。

    保存格式：{"days": [int, ...], "data_list": [dict, ...]}
    训练/验证集样本包含 x_fut；测试集不包含（y 为股票代码，方便导出预测值）。
    """
    if label_mark is None:
        label_mark = LABEL_MARK if LABEL_MARK is not None else HORIZON_TO_LABEL_MARK[DEFAULT_HORIZON]
    days = pd.DatetimeIndex(days).sort_values()
    day_strs = days.strftime("%Y-%m-%d").tolist()
    day_ints = (days.year * 10000 + days.month * 100 + days.day).tolist()

    test_set = (split_name == "test")

    # 构建未来日映射：排序后的全部交易日 -> 索引
    all_day_strs = all_days.strftime("%Y-%m-%d").tolist()
    day_to_idx = {ds: i for i, ds in enumerate(all_day_strs)}

    # 为每个交易日配对未来交易日
    day_fut_pairs = []
    for d in day_strs:
        if test_set:
            fut_ds = None
        else:
            idx = day_to_idx.get(d)
            if idx is not None and idx + FUT_SHIFT < len(all_day_strs):
                fut_ds = all_day_strs[idx + FUT_SHIFT]
            else:
                fut_ds = None  # 之后不足 FUT_SHIFT 个交易日，该日将被剔除
        day_fut_pairs.append((d, fut_ds))

    n_train_fut_missing = sum(1 for _, fd in day_fut_pairs if fd is None and not test_set)
    if n_train_fut_missing > 0:
        print(f"[{split_name}] 警告: {n_train_fut_missing} 天因缺少未来 {FUT_SHIFT} 交易日数据将被剔除")

    print(f"[{split_name}] 交易日数量: {len(day_strs)}，标签列: {label_mark}，并行生成中 ...")
    data_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(get_day_data)(d, fd, label_mark, test_set) for d, fd in day_fut_pairs
    )

    kept = [(di, dl) for di, dl in zip(day_ints, data_list) if dl is not None]
    if len(kept) < len(day_ints):
        print(f"[{split_name}] 警告: {len(day_ints) - len(kept)} 天被剔除")
    day_ints, data_list = zip(*kept) if kept else ([], [])

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{split_name}.pt")
    torch.save({"days": list(day_ints), "data_list": list(data_list)}, save_path)
    print(f"[{split_name}] 已保存 -> {save_path}")
    return save_path


class DailyTripleDataset(Dataset):
    """加载 build_and_save_split 保存的 .pt 文件，按交易日提供三概念 + 未来特征截面。

    训练/验证集样本包含 "x_fut" 键；测试集样本不包含该键。
    """

    def __init__(self, split: str, root: str = None):
        if root is None:
            root = SAVE_DIR
        path = os.path.join(root, f"{split}.pt")
        payload = torch.load(path, weights_only=False)
        self.days = payload["days"]
        self.data_list = payload["data_list"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloader(
    split: str,
    root: str = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """daily batch 训练用 batch_size=1；collate 直接返回单个样本。"""
    if root is None:
        root = SAVE_DIR
    return DataLoader(
        DailyTripleDataset(split, root=root),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=lambda batch: batch[0])


if __name__ == "__main__":
    args = parse_args()
    print(f"固定边: v0(行业) | 第二边: v{args.edge_version_2} | 第三边: v{args.edge_version_3}")
    print(f"EDGE_DIR_2={EDGE_DIR_2}, EDGE_DIR_3={EDGE_DIR_3}")
    print(f"特征: freq={args.freq} lags={args.lags} | FEATURE_PATH={FEATURE_PATH}")
    print(f"标签: LABEL_PATH={LABEL_PATH} | 标签列: {LABEL_MARK} (horizon={args.horizon})")
    print(f"未来特征偏移: {FUT_SHIFT} 个交易日")
    print(f"保存目录: {SAVE_DIR}")

    start = time()

    _splits, _all_days = get_avail_days_and_split(
        "2010-01-01", "2019-12-31",
        "2020-01-31", "2021-12-31",
        "2022-01-31"
    )
    for name, ds in _splits.items():
        print(f"{name}: {len(ds)} 天, {ds.min().date()} ~ {ds.max().date()}")

    for name, ds in _splits.items():
        build_and_save_split(name, ds, _all_days)

    print(f"总耗时: {time() - start:.1f}s")