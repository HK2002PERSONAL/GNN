"""HIST step1（V2）：构建双概念每日截面数据 —— 第一条边不锁死 V0

与 build_GCN_GCN_data.py 的唯一区别：
    原版 concept 1 固定锁死为 v0（一级行业 IndustryCodeNew）；
    V2 版 concept 1 改为来自可选边版本 edges_v{a} 的连通分量，不再锁死行业。

HIST_a6 (GCN + GCN) 需要在同一股票集合上同时提供两种预定义概念：
    concept 1 : v{a} —— 第一条边（data/edges/edges_v{a}/{day_int}.npz）的连通分量【可选】
    concept 2 : v{b} —— 第二条边（data/edges/edges_v{b}/{day_int}.npz）的连通分量【可选】

支持的边版本：2, 3, 7, 9, 10, 12

对齐方式：
    节点 = 特征 ∩ 标签 ∩ (有行业且有市值) ∩ edges_v{a} 涉及的股票 ∩ edges_v{b} 涉及的股票
    （保留行业/市值过滤，仅用于节点筛选与 market_value 字段，不作为概念）
    每日独立编码：
        industry_v{a} : 第一条边连通分量标签（训练时 one-hot 成概念矩阵）
        industry_v{b} : 第二条边连通分量标签（训练时 one-hot 成概念矩阵）

注意：本脚本直接从 edges_v{a} / edges_v{b} 原始边文件在线计算连通分量，
      不依赖其它 hist 数据集的产出。

样本格式（与 v0v{n} 双概念兼容，industry_v0 替换为 industry_v{a}）：
    x             [N, F] 特征（CAT_{freq}_LAG{lags}）
    industry_v{a} [N]    第一条边连通分量标签（concept 1）
    industry_v{b} [N]    第二条边连通分量标签（concept 2）
    market_value  [N]    流通市值
    y             [N]    标签（test 集为股票代码）
    date          [3]    年/月/日

运行：python DataBuilder/hist_data_builder/build_GCN_GCN_dataV2.py \
         --edge-version-1 2 --edge-version-2 3 --freq 1D --lags 20 60 120
特征文件：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt
保存：data/hist_{freq}_lag{lag1}_{lag2}_..._v{a}v{b}/{train,valid,test}.pt
"""
import argparse
import os
from time import time

import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

# 行业 + 市值（仅用于节点过滤与 market_value 字段，不再作为概念）
EDGE_PATH = "data/raw_data/stock_edge.pqt"

# 支持的边版本
SUPPORTED_EDGE_VERSIONS = [2, 3, 7, 9, 10, 12, 13, 14]

# 支持的特征频率
SUPPORTED_FREQS = ["1D", "five_minute"]

# 全局路径（由 parse_args 设置）
FEATURE_PATH = None
SAVE_DIR = None
EDGE_DIR_1 = None
EDGE_DIR_2 = None
EDGE_VERSION_1 = None
EDGE_VERSION_2 = None


def build_feature_path(freq: str, lags: list) -> str:
    """按命名标准拼接特征文件路径：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/feature/CAT_{freq}_LAG{lag_str}.pqt"


def build_save_dir(freq: str, lags: list, edge_version_1: int, edge_version_2: int) -> str:
    """按命名标准拼接保存目录：data/hist_{freq}_lag{lag1}_{lag2}_..._v{a}v{b}"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/hist_{freq}_lag{lag_str}_v{edge_version_1}v{edge_version_2}"


def parse_args():
    """解析命令行参数，设置全局路径。"""
    global FEATURE_PATH, SAVE_DIR, EDGE_DIR_1, EDGE_DIR_2  # pylint:disable=W0603
    global EDGE_VERSION_1, EDGE_VERSION_2  # pylint:disable=W0603

    parser = argparse.ArgumentParser(description="构建双概念每日截面数据（v{a} + v{b}），第一条边不锁死 v0")
    parser.add_argument(
        "--edge-version-1", "-v1",
        type=int,
        default=2,
        choices=SUPPORTED_EDGE_VERSIONS,
        help=f"第一条边的版本号，可选：{SUPPORTED_EDGE_VERSIONS}，默认 2"
    )
    parser.add_argument(
        "--edge-version-2", "-v2",
        type=int,
        default=3,
        choices=SUPPORTED_EDGE_VERSIONS,
        help=f"第二条边的版本号，可选：{SUPPORTED_EDGE_VERSIONS}，默认 3"
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
    _args = parser.parse_args()

    if _args.edge_version_1 == _args.edge_version_2:
        raise ValueError(
            f"两条边版本不能相同: edge_version_1 == edge_version_2 == {_args.edge_version_1}")

    EDGE_VERSION_1 = _args.edge_version_1
    EDGE_VERSION_2 = _args.edge_version_2
    EDGE_DIR_1 = f"data/edges/edges_v{EDGE_VERSION_1}"
    EDGE_DIR_2 = f"data/edges/edges_v{EDGE_VERSION_2}"
    FEATURE_PATH = build_feature_path(_args.freq, _args.lags)
    SAVE_DIR = build_save_dir(_args.freq, _args.lags, EDGE_VERSION_1, EDGE_VERSION_2)

    # 检查边目录是否存在
    if not os.path.isdir(EDGE_DIR_1):
        raise FileNotFoundError(f"第一条边目录不存在: {EDGE_DIR_1}")
    if not os.path.isdir(EDGE_DIR_2):
        raise FileNotFoundError(f"第二条边目录不存在: {EDGE_DIR_2}")

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
        v12 原始 codes 即字符串（'U6'）；v2/v3/v7/v9/v10 为 int64 需 zfill。
    """
    edge_path = os.path.join(edge_dir, f"{day_int}.npz")
    if not os.path.exists(edge_path):
        return None

    npz = np.load(edge_path, allow_pickle=True)
    ei_np = npz['edge_index']                      # (2, E) int32，codes 位置空间
    codes = npz['codes']                           # int64（前导零丢失）或 str（v12）
    if codes.dtype.kind in ('U', 'S', 'O'):        # str/bytes/object -> 直接使用（v12）
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


def get_day_data(
    day: str,
    label_mark: str = 'Return5d',
    test_set: bool = False
) -> dict:

    ft_day = pd.read_parquet(
        path=FEATURE_PATH,
        filters=[("DataDate", "==", pd.to_datetime(day))]
    ).droplevel(0)

    lb_day = pd.read_parquet(
        path="data/label/stock_ret.pqt",
        columns=[label_mark],
        filters=[("DataDate", "==", pd.to_datetime(day))]
    ).droplevel(0)

    # 行业 + 市值（仅用于节点过滤与 market_value，不再作为概念）
    edge_day = pd.read_parquet(
        path=EDGE_PATH,
        columns=["SecuCode", "DataDate", "IndustryCodeNew", "FloatMarketCap"],
        filters=[("DataDate", "==", pd.to_datetime(day))]
    ).set_index("SecuCode")

    d = pd.to_datetime(day)
    day_int = d.year * 10000 + d.month * 100 + d.day

    # 第一条边 v{a}（不再锁死 v0 行业）
    e1 = load_edge_day(EDGE_DIR_1, day_int)
    if e1 is None:
        return None
    ei_v1, codes_v1 = e1

    # 第二条边 v{b}
    e2 = load_edge_day(EDGE_DIR_2, day_int)
    if e2 is None:
        return None
    ei_v2, codes_v2 = e2

    avail_idx = ft_day.index.intersection(lb_day.index)
    edge_day = edge_day[edge_day["IndustryCodeNew"].notna() & edge_day["FloatMarketCap"].notna()]

    # 对齐：特征 ∩ 标签 ∩ (行业+市值) ∩ v{a} 节点 ∩ v{b} 节点
    nodes = (avail_idx
             .intersection(edge_day.index)
             .intersection(pd.Index(codes_v1))
             .intersection(pd.Index(codes_v2)))
    if len(nodes) < 2 or ei_v1.shape[1] == 0 or ei_v2.shape[1] == 0:
        return None

    nodes = nodes.sort_values()
    ft_day = ft_day.loc[nodes]
    lb_day = lb_day.loc[nodes]
    edge_day = edge_day.loc[nodes]

    # 概念 1 / 概念 2：两条边的连通分量标签（均不锁死行业）
    industry_v1 = remap_and_components(ei_v1, codes_v1, nodes)
    industry_v2 = remap_and_components(ei_v2, codes_v2, nodes)

    x = torch.tensor(ft_day.values, dtype=torch.float)
    industry_v1_t = torch.tensor(industry_v1, dtype=torch.long)
    industry_v2_t = torch.tensor(industry_v2, dtype=torch.long)
    market_value = torch.tensor(edge_day["FloatMarketCap"].values, dtype=torch.float)
    y = torch.tensor(
        lb_day[label_mark].values if not test_set else nodes.values.astype(int),
        dtype=torch.float)

    return {
        "x": x,
        "industry_v0": industry_v1_t,
        "industry_v2": industry_v2_t,
        "market_value": market_value,
        "y": y,
        "date": torch.tensor([d.year, d.month, d.day], dtype=torch.long),
    }


def get_avail_days_and_split(
    train_start: str, train_end: str,
    valid_start: str, valid_end: str,
    test_start: str
):
    ft_days = pd.read_parquet(path=FEATURE_PATH, columns=[]).index.get_level_values('DataDate').unique()
    edge_days = pd.read_parquet(path=EDGE_PATH, columns=["DataDate"])["DataDate"].unique()
    lb_days = pd.read_parquet(path="data/label/stock_ret.pqt", columns=[]).index.get_level_values('DataDate').unique()

    days = ft_days.intersection(pd.DatetimeIndex(edge_days)).intersection(lb_days)

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)
    test_start = pd.Timestamp(test_start)

    splits = {
        "train": days[(days >= train_start) & (days <= train_end)],
        "valid": days[(days >= valid_start) & (days <= valid_end)],
        "test": days[days >= test_start]
    }

    return splits


def build_and_save_split(split_name, days, label_mark="Return5d", n_jobs=32):
    """并行生成每个交易日的双概念截面数据，按日期顺序保存为单个 .pt 文件。

    保存格式：{"days": [int, ...], "data_list": [dict, ...]}
    test 集使用 test_set=True（y 为股票代码，方便导出预测值）。
    """
    days = pd.DatetimeIndex(days).sort_values()
    day_strs = days.strftime("%Y-%m-%d").tolist()
    day_ints = (days.year * 10000 + days.month * 100 + days.day).tolist()

    test_set = (split_name == "test")
    print(f"[{split_name}] 交易日数量: {len(day_strs)}，并行生成中 ...")
    data_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(get_day_data)(d, label_mark, test_set) for d in day_strs
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


class DailyDualDataset(Dataset):
    """加载 build_and_save_split 保存的 .pt 文件，按交易日提供双概念截面。"""

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
        DailyDualDataset(split, root=root),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=lambda batch: batch[0])


if __name__ == "__main__":
    args = parse_args()
    print(f"第一条边: v{EDGE_VERSION_1} | EDGE_DIR_1={EDGE_DIR_1}")
    print(f"第二条边: v{EDGE_VERSION_2} | EDGE_DIR_2={EDGE_DIR_2}")
    print(f"特征: freq={args.freq} lags={args.lags} | FEATURE_PATH={FEATURE_PATH}")
    print(f"保存目录: {SAVE_DIR}")

    start = time()

    _splits = get_avail_days_and_split(
        "2010-01-01", "2019-12-31",
        "2020-01-31", "2021-12-31",
        "2022-01-31"
    )
    for name, ds in _splits.items():
        print(f"{name}: {len(ds)} 天, {ds.min().date()} ~ {ds.max().date()}")

    for name, ds in _splits.items():
        build_and_save_split(name, ds)

    print(f"总耗时: {time() - start:.1f}s")