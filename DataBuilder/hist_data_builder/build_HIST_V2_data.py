"""HIST_V2 step1：构建每日截面数据（HIST 概念-市值模型，基于 GCN_V2 的相关性边）

与 HIST_V0 的差异：
- V0 使用一级行业（IndustryCodeNew）作为预定义概念
- V2 使用 GCN_V2 的滚动相关性边（data/edges_v2/{day_int}.npz）构建连通分量作为概念
- 节点 = 特征 ∩ 标签 ∩ 边涉及股票（与 GCN_V2 一致，无边的股票不入图）

概念生成方式：
- 对当日边图做连通分量分解，每个连通分量作为一个"概念"
- 市值来源：data/raw_data/stock_edge.pqt（FloatMarketCap）

运行：python DataBuilder/hist_data_builder/build_HIST_V2_data.py --freq 1D --lags 20 60 120
特征文件：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt
保存：data/hist_{freq}_lag{lag1}_{lag2}_..._v2/{train,valid,test}.pt
"""
import argparse
import os
from time import time

import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader

# 支持的特征频率
SUPPORTED_FREQS = ["1D", "five_minute"]

# 全局路径（由 parse_args 设置）
FEATURE_PATH = None
SAVE_DIR = None

EDGE_DIR = "data/edges_v2"
STOCK_EDGE_PATH = "data/raw_data/stock_edge.pqt"


def build_feature_path(freq: str, lags: list) -> str:
    """按命名标准拼接特征文件路径：data/feature/CAT_{freq}_LAG{lag1}_{lag2}_..._.pqt"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/feature/CAT_{freq}_LAG{lag_str}.pqt"


def build_save_dir(freq: str, lags: list) -> str:
    """按命名标准拼接保存目录：data/hist_{freq}_lag{lag1}_{lag2}_..._v2"""
    lag_str = "_".join(str(x) for x in lags)
    return f"data/hist_{freq}_lag{lag_str}_v2"


def parse_args():
    """解析命令行参数，设置全局路径。"""
    global FEATURE_PATH, SAVE_DIR #pylint:disable=W0603

    parser = argparse.ArgumentParser(description="构建 HIST_V2 每日截面数据（基于 GCN_V2 相关性边的连通分量概念）")
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

    FEATURE_PATH = build_feature_path(_args.freq, _args.lags)
    SAVE_DIR = build_save_dir(_args.freq, _args.lags)

    # 检查特征文件是否存在
    if not os.path.isfile(FEATURE_PATH):
        raise FileNotFoundError(f"特征文件不存在: {FEATURE_PATH}")

    return _args


def find_connected_components(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """使用 Union-Find 算法找到图中所有连通分量。
    
    Args:
        edge_index: (2, E) 边索引
        num_nodes: 节点总数
    
    Returns:
        labels: (num_nodes,) 每个节点的连通分量标签（从 0 开始）
    """
    parent = np.arange(num_nodes)
    
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        # 路径压缩
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    
    # 合并所有边的端点
    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            union(edge_index[0, i], edge_index[1, i])
    
    # 收集连通分量标签
    labels = np.array([find(i) for i in range(num_nodes)])
    # 重新编号：从 0 开始连续编号
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

    # 读取 GCN_V2 step0 生成的边
    d = pd.to_datetime(day)
    day_int = d.year * 10000 + d.month * 100 + d.day
    edge_path = os.path.join(EDGE_DIR, f"{day_int}.npz")
    if not os.path.exists(edge_path):
        return None  # 边文件缺失则跳过

    npz = np.load(edge_path)
    ei_np = npz['edge_index']                      # (2, E) int32，codes 位置空间
    codes = npz['codes']                           # int64（前导零丢失）
    codes_str = np.char.zfill(codes.astype('U16'), 6)

    avail_idx = ft_day.index.intersection(lb_day.index)
    nodes = avail_idx.intersection(pd.Index(codes_str))   # 只保留有边的股票
    if len(nodes) < 2 or ei_np.shape[1] == 0:
        return None

    nodes = nodes.sort_values()                  # 排序，保证节点顺序确定
    ft_day = ft_day.loc[nodes]
    lb_day = lb_day.loc[nodes]

    # 端点重映射：codes 位置空间 -> nodes 位置空间
    src_c = codes_str[ei_np[0]]
    dst_c = codes_str[ei_np[1]]
    src = nodes.get_indexer(src_c)
    dst = nodes.get_indexer(dst_c)
    valid = (src >= 0) & (dst >= 0)
    src, dst = src[valid].astype(np.int64), dst[valid].astype(np.int64)

    if len(src) == 0:
        return None

    # 使用连通分量作为概念
    edge_index_np = np.stack([src, dst], axis=0)
    concept_labels = find_connected_components(edge_index_np, len(nodes))

    # 获取市值
    edge_day = pd.read_parquet(
        path=STOCK_EDGE_PATH,
        columns=["SecuCode", "DataDate", "FloatMarketCap"],
        filters=[("DataDate", "==", pd.to_datetime(day))]
    ).set_index("SecuCode")
    edge_day = edge_day.loc[nodes]
    
    # 剔除无市值的股票
    valid_mv = edge_day["FloatMarketCap"].notna()
    if valid_mv.sum() < 2:
        return None
    
    # 过滤节点和概念标签
    valid_mv_np = valid_mv.values
    ft_final = ft_day.loc[valid_mv]
    lb_final = lb_day.loc[valid_mv]
    nodes_final = nodes[valid_mv_np]
    concept_final = concept_labels[valid_mv_np]
    
    # 重新编号概念（过滤后可能不连续）
    _, concept_final = np.unique(concept_final, return_inverse=True)

    x = torch.tensor(ft_final.values, dtype=torch.float)
    concept_t = torch.tensor(concept_final, dtype=torch.long)
    market_value = torch.tensor(edge_day.loc[valid_mv, "FloatMarketCap"].values, dtype=torch.float)
    y = torch.tensor(
        lb_final[label_mark].values if not test_set else nodes_final.values.astype(int),
        dtype=torch.float)

    return {
        "x": x,
        "industry": concept_t,  # 概念标签（连通分量），命名保持与 V0 兼容
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
    g_days = pd.read_parquet(path="data/raw_data/stock_data.pqt", columns=[]).index.get_level_values('DataDate').unique()
    lb_days = pd.read_parquet(path="data/label/stock_ret.pqt", columns=[]).index.get_level_values('DataDate').unique()

    days = ft_days.intersection(g_days).intersection(lb_days)

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
    """并行生成每个交易日的截面数据，按日期顺序保存为单个 .pt 文件。

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


class DailyHistDataset(Dataset):
    """加载 build_and_save_split 保存的 .pt 文件，按交易日提供截面。"""

    def __init__(self, split: str, root: str = None):
        root = root or SAVE_DIR
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
    root = root or SAVE_DIR
    return DataLoader(
        DailyHistDataset(split, root=root),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=lambda batch: batch[0])


if __name__ == "__main__":
    args = parse_args()
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