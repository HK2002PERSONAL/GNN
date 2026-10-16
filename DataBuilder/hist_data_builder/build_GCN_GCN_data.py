"""HIST_V_ablation step1：构建双概念每日截面数据（供 run_HIST_GCN_GCN.py 的 HIST_a6 使用）

HIST_a6 (GCN + GCN) 需要在同一股票集合上同时提供两种预定义概念：
    concept 1 : v0 —— 一级行业（IndustryCodeNew，来自 data/raw_data/stock_edge.pqt）【固定】
    concept 2 : v{n} —— 可选边版本（data/edges_v{n}/{day_int}.npz）的连通分量

支持的边版本：2, 3, 7, 9, 10, 12

对齐方式：
    节点 = 特征 ∩ 标签 ∩ (有行业且有市值) ∩ edges_v{n} 涉及的股票
    每日独立编码：
        industry_v0 : 行业整数编码（训练时 one-hot 成概念矩阵）
        industry_v2 : 连通分量标签（训练时 one-hot 成概念矩阵）

注意：本脚本不依赖 HIST_V2/step1_build_data.py 的产出（hist_daily_v2），
      直接从 edges_v{n} 原始边文件在线计算连通分量，因此可在 v2 数据集尚未构建时运行。

样本格式（与 v0/v1 兼容，多出 industry_v2）：
    x            [N, F] 特征（CAT_{freq}_LAG{lags}）
    industry_v0  [N]    行业整数编码
    industry_v2  [N]    连通分量标签
    market_value [N]    流通市值
    y            [N]    标签（test 集为股票代码）
    date         [3]    年/月/日

运行：python HIST_V_ablation/build_GCN_GCN_data.py --edge-version 2 --freq 1D --lags 20 60 120
特征文件：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt
保存：data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{n}/{train,valid,test}.pt
"""
import argparse
import os
from time import time

import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

# v0 行业边固定
EDGE_PATH = "data/raw_data/stock_edge.pqt"

# 支持的边版本
SUPPORTED_EDGE_VERSIONS = [2, 3, 7, 9, 10, 12]

# 支持的特征频率
SUPPORTED_FREQS = ["1D", "five_minute"]

# 全局路径（由 parse_args 设置）
FEATURE_PATH = None
SAVE_DIR = None
EDGE_DIR = None


def build_feature_path(freq: str, lags: list) -> str:
    """按命名标准拼接特征文件路径：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/feature/CAT_{freq}_LAG{lag_str}.pqt"


def build_save_dir(freq: str, lags: list, edge_version: int) -> str:
    """按命名标准拼接保存目录：data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{edge_version}"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/hist_{freq}_lag{lag_str}_v0v{edge_version}"


def parse_args():
    """解析命令行参数，设置全局路径。"""
    global FEATURE_PATH, SAVE_DIR, EDGE_DIR #pylint:disable=W0603

    parser = argparse.ArgumentParser(description="构建双概念每日截面数据（v0 + v{n}）")
    parser.add_argument(
        "--edge-version", "-v",
        type=int,
        required=True,
        choices=SUPPORTED_EDGE_VERSIONS,
        help=f"第二种边的版本号，可选：{SUPPORTED_EDGE_VERSIONS}"
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

    EDGE_DIR = f"data/edges/edges_v{_args.edge_version}"
    FEATURE_PATH = build_feature_path(_args.freq, _args.lags)
    SAVE_DIR = build_save_dir(_args.freq, _args.lags, _args.edge_version)

    # 检查边目录是否存在
    if not os.path.isdir(EDGE_DIR):
        raise FileNotFoundError(f"边目录不存在: {EDGE_DIR}")

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

    # v0：行业 + 市值
    edge_day = pd.read_parquet(
        path=EDGE_PATH,
        columns=["SecuCode", "DataDate", "IndustryCodeNew", "FloatMarketCap"],
        filters=[("DataDate", "==", pd.to_datetime(day))]
    ).set_index("SecuCode")

    # v2：相关性边
    d = pd.to_datetime(day)
    day_int = d.year * 10000 + d.month * 100 + d.day
    edge_path = os.path.join(EDGE_DIR, f"{day_int}.npz")
    if not os.path.exists(edge_path):
        return None  # 边文件缺失则跳过

    npz = np.load(edge_path, allow_pickle=True)
    ei_np = npz['edge_index']                      # (2, E) int32，codes 位置空间
    codes = npz['codes']                           # int64（前导零丢失）或 str（v12）
    if codes.dtype.kind in ('U', 'S', 'O'):        # str/bytes/object -> 直接使用（v12）
        codes_str = codes.astype('U6')
    else:                                          # int64 -> zfill 6 位（v2/v7/v9/v10）
        codes_str = np.char.zfill(codes.astype('U16'), 6)

    avail_idx = ft_day.index.intersection(lb_day.index)
    edge_day = edge_day[edge_day["IndustryCodeNew"].notna() & edge_day["FloatMarketCap"].notna()]

    # 对齐：特征 ∩ 标签 ∩ (行业+市值) ∩ 边节点
    nodes = avail_idx.intersection(edge_day.index).intersection(pd.Index(codes_str))
    if len(nodes) < 2 or ei_np.shape[1] == 0:
        return None

    nodes = nodes.sort_values()
    ft_day = ft_day.loc[nodes]
    lb_day = lb_day.loc[nodes]
    edge_day = edge_day.loc[nodes]

    # v0 概念：行业整数编码（每日独立）
    industry_v0, _ = pd.factorize(edge_day["IndustryCodeNew"].values)

    # v2 概念：边端点重映射到 nodes 位置空间，无效端点的边丢弃
    src_c = codes_str[ei_np[0]]
    dst_c = codes_str[ei_np[1]]
    src = nodes.get_indexer(src_c)
    dst = nodes.get_indexer(dst_c)
    valid = (src >= 0) & (dst >= 0)
    src, dst = src[valid].astype(np.int64), dst[valid].astype(np.int64)

    edge_index_np = np.stack([src, dst], axis=0) if len(src) > 0 else np.zeros((2, 0), dtype=np.int64)
    industry_v2 = find_connected_components(edge_index_np, len(nodes))

    x = torch.tensor(ft_day.values, dtype=torch.float)
    industry_v0_t = torch.tensor(industry_v0, dtype=torch.long)
    industry_v2_t = torch.tensor(industry_v2, dtype=torch.long)
    market_value = torch.tensor(edge_day["FloatMarketCap"].values, dtype=torch.float)
    y = torch.tensor(
        lb_day[label_mark].values if not test_set else nodes.values.astype(int),
        dtype=torch.float)

    return {
        "x": x,
        "industry_v0": industry_v0_t,
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
    print(f"边版本: v{args.edge_version} | EDGE_DIR={EDGE_DIR}")
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