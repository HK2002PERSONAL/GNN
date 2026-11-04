"""HIST数据加载工具

按交易日提供横截面样本，数据目录为 data/hist_{version}/{split}.pt。

支持的 version 形如：
    {freq}_lag{spans}_v0          单概念（行业）
    {freq}_lag{spans}_v0v{n}      双概念（行业 + 第 n 版边连通分量），供 HIST_a6 等
    {freq}_lag{spans}_v0v12v{n}   三概念（行业 + v12 风险边 + 第 n 版边），供 HIST_a10

样本字段（按 version 不同，industry_v* 键不同）：
    x             [N, F] 特征
    industry_v0   [N]    行业整数编码（概念 1_1）
    industry_v12  [N]    v12 风险边连通分量标签（概念 1_2，仅三概念数据）
    industry_v{n} [N]    第 n 版边连通分量标签（概念 2）
    market_value  [N]    流通市值
    y             [N]    标签（test 集为股票代码）
    date          [3]    年/月/日
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader

class DailyHistDataset(Dataset):
    """按交易日提供横截面样本，每个样本对应一个交易日。"""

    def __init__(self, split: str, version: str):
        path = os.path.join('data', f"hist_{version}", f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {path}，请先构建数据")
        payload = torch.load(path, weights_only=False)
        self.days = payload["days"]
        self.data_list = payload["data_list"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloader(
    split: str,
    version: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """daily batch 训练用 batch_size=1；collate 直接返回单个样本。"""
    return DataLoader(
        DailyHistDataset(split, version=version),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=lambda batch: batch[0])