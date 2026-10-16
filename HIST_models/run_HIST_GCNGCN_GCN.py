"""HIST_models：HIST_a10（GCN[Industry] + GCN[Risk] + GCN[corr]）训练脚本（daily batch）

模型 HIST_a10 用三个 GCN_InfoShare_block，需要三种预定义概念（边）：
    概念 1_1 (v0)  : 一级行业         —— IndustryCodeNew
    概念 1_2 (v12) : 风险因子相关边     —— data/edges/edges_v12/{day}.npz 的连通分量
    概念 2   (v{n}) : 可选相关性边连通分量 —— data/edges/edges_v{n}/{day}.npz 的连通分量

数据由 DataBuilder/hist_data_builder/build_GCN_GCN_GCN_data.py 产出：
    data/hist_{freq}_lag{lags}_v0v12v{n}/{train,valid,test}.pt
    样本含 industry_v0 / industry_v12 / industry_v{n} 三个概念标签，已在同一股票集合上对齐。

    样本字段：
        x             [N, F] 特征
        industry_v0   [N]    行业整数编码（概念 1_1）
        industry_v12  [N]    v12 风险边连通分量标签（概念 1_2）
        industry_v{n} [N]    第 n 版边连通分量标签（概念 2）
        market_value  [N]    流通市值
        y             [N]    标签（test 集为股票代码）
        date          [3]    年/月/日

运行：
    python HIST_models/run_HIST_GCNGCN_GCN.py --edge-version 2
    python HIST_models/run_HIST_GCNGCN_GCN.py --edge-version 2 --hidden_size 64 --clip 0.04

输出：model_configs/best_hist_a10_{freq}_lag{lags}_v0v12v{n}_{clip}.pt、
      model_outputs/hist_test_a10_{freq}_lag{lags}_v0v12v{n}_{clip}.pqt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                              # 项目根目录（utils 等）
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 本目录 models / data_utils

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from models import HIST_a10
from data_utils import get_dataloader
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch, seed_torch_all
from utils.ModelUtils import z_score

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)

# 第三边支持的版本（与 build_GCNGCN_GCN_data.py 保持一致，不含 12）
SUPPORTED_EDGE_VERSIONS = [2, 3, 7, 9, 10]


# ---------------- 概念 -> 概念矩阵 ----------------

def industry_to_concept(industry: torch.Tensor) -> torch.Tensor:
    """把每日概念标签 [N]（整数）变形为股票-概念 0/1 矩阵 [N, C]。

    C = 当日出现的概念数。股票 A 属于概念 c 时 [A, c] = 1，否则 0。
    """
    n = industry.shape[0]
    c = int(industry.max().item()) + 1
    concept = torch.zeros(n, c, dtype=torch.float, device=industry.device)
    concept.scatter_(1, industry.reshape(-1, 1), 1.0)
    return concept


# ---------------- 损失 ----------------

class ICLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, clamp: float = 5):
        super().__init__()
        self.eps = eps
        self.clamp = clamp

    def cs_zscore(self, y):
        """截面标准化标签（对训练更稳，不改变截面排序）"""
        mask = torch.isfinite(y)
        m, s = y[mask].mean(), y[mask].std().clamp_min(1e-8)
        return ((y - m) / s).clamp(-self.clamp, self.clamp)

    def forward(self, x, y):
        x = self.cs_zscore(x).reshape(-1).float()
        y = self.cs_zscore(y).reshape(-1).float()
        xm = x - x.mean()
        ym = y - y.mean()
        denom = (xm.norm(p=2) * ym.norm(p=2)).clamp_min(self.eps)
        return - xm.dot(ym) / denom


# ---------------- 前向 ----------------

def model_forward(_model, _sample, _device, _edge_version: int):
    """把单个日截面样本喂给 HIST_a10，返回预测值 [N]。

    概念 1_1 = v0 行业，概念 1_2 = v12 风险边连通分量，概念 2 = v{n} 连通分量，
    三者已在同一股票集合上对齐。
    """
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v12 = _sample["industry_v12"].to(_device)
    industry_v3 = _sample[f"industry_v{_edge_version}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1_1 = industry_to_concept(industry_v0)
    concept_1_2 = industry_to_concept(industry_v12)
    concept_2 = industry_to_concept(industry_v3)
    return _model(x, concept_1_1, concept_1_2, concept_2, market_value)


# ---------------- 训练 ----------------

def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _edge_version):
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue
        _pred = model_forward(_model, _sample, _device, _edge_version)
        loss = _loss(_pred, y)
        if loss is None or not torch.isfinite(loss):
            continue
        _optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            _optimizer.zero_grad()
            continue
        _optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(_model, _loader, _loss, _device, _edge_version):
    _model.eval()
    ics = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        _pred = model_forward(_model, _sample, _device, _edge_version)
        ics.append(_loss(_pred, y).item())
    ics = np.asarray(ics, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a10（GCN+GCN+GCN，v0 行业 + v12 风险边 + v{n} 相关性边）训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version", "-v", type=int, default=2, choices=SUPPORTED_EDGE_VERSIONS,
                        help=f"第三边版本号，可选 {SUPPORTED_EDGE_VERSIONS}，默认 2")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a10"
    edge_version = args.edge_version

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    # 数据 version 与 build_GCN_GCN_GCN_data.py 的 SAVE_DIR 保持一致：
    # data/hist_{freq}_lag{lags}_v0v12v{n}
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v12v{edge_version}"

    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size}")

    seed_all_non_torch(root_seed)

    smart_log("Loading Data...")
    train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=True); smart_log("train done")
    valid_loader = get_dataloader("valid", version=data_version, batch_size=1, shuffle=False); smart_log("valid done")

    # 特征维度从训练集首个样本动态获取
    d_feat = train_loader.dataset[0]["x"].shape[1]
    smart_log(f"d_feat = {d_feat}")

    criterion = ICLoss()

    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    for torch_seed in torch_seed_range:
        tpmd_pth = Path(ckpt_path) / f"{torch_seed}.pt"
        if Path(tpmd_pth).is_file():
            smart_log(f"{torch_seed} already done")
            continue

        smart_log(f"training {torch_seed}")
        seed_torch_all(torch_seed)

        model = HIST_a10(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, edge_version)
            ic, icir = evaluate(model, valid_loader, criterion, device, edge_version)
            smart_log(f"epoch {epoch:3d} | train loss {train_loss:.4f} | valid IC {ic:.4f} | ICIR {icir:.4f}")

            if ic < best_ic:
                smart_log(f"saving model {best_ic:.6f} -> {ic:.6f}")
                best_ic, wait = ic, 0
                torch.save(model.state_dict(), tpmd_pth)
            else:
                wait += 1
                if wait >= args.patience:
                    smart_log("early stopping")
                    break

    modelS = {iter_seed: HIST_a10(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip).to(device) for iter_seed in torch_seed_range}
    for model_num in torch_seed_range:
        modelS[model_num].load_state_dict(torch.load(
            Path(ckpt_path) / f"{model_num}.pt",
            map_location=device))
        modelS[model_num].eval()

    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=data_version, batch_size=1, shuffle=False):
            for i, model_num in enumerate(torch_seed_range):
                if i == 0:
                    pred = z_score(model_forward(modelS[model_num], sample, device, edge_version).reshape(-1)).cpu().numpy()
                else:
                    pred += z_score(model_forward(modelS[model_num], sample, device, edge_version).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")