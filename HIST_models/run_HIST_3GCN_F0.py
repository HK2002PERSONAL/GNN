"""HIST_models：HIST_FactorGCL_V0（3GCN + 正交损失）训练脚本（daily batch）

模型 HIST_FactorGCL_V0 继承自 HIST_a11（三个 GCN_InfoShare_block 串联），
输出分解为 gnn_info_fore（三个 GCN 信息之和）与 i_info_fore（个股特异信息）。

损失函数：
    L = IC_loss + λ * Orth_loss(gnn_info_fore, i_info_fore)
    其中 Orth_loss 为 gnn_info_fore 与 i_info_fore 的余弦相似度平方均值，
    鼓励两个信息分量正交（解耦）。

需要三种预定义概念（边）：
    概念 1 (v0)  : 一级行业      —— IndustryCodeNew（固定）
    概念 2 (v{n1}): 第二版边连通分量（开放选择）
    概念 3 (v{n2}): 第三版边连通分量（开放选择）

数据由 DataBuilder/hist_data_builder/build_GCN_GCN_GCN_data.py 产出：
    data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{n1}v{n2}_{label_name}/{train,valid,test}.pt

    样本字段：
        x             [N, F] 特征
        industry_v0   [N]    行业整数编码（概念 1，固定）
        industry_v{n1}[N]    第二边连通分量标签（概念 2）
        industry_v{n2}[N]    第三边连通分量标签（概念 3）
        market_value  [N]    流通市值
        y             [N]    标签（test 集为股票代码）
        date          [3]    年/月/日

运行：
    python HIST_models/run_HIST_3GCN_F0.py --edge-version-2 2 --edge-version-3 3
    python HIST_models/run_HIST_3GCN_F0.py -v2 2 -v3 3 --hidden_size 64 --clip 0.04
    python HIST_models/run_HIST_3GCN_F0.py -v2 2 -v3 3 --orth_lambda 0.1

输出：model_configs/best_hist_3gcn_f0_{data_version}_{clip_mark}{ew_mark}/
      model_outputs/hist_test_3gcn_f0_{data_version}_{clip_mark}{ew_mark}.pqt
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                              # 项目根目录（utils 等）
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 本目录 models / data_utils

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from models import HIST_FactorGCL_V0
from data_utils import get_dataloader
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch, seed_torch_all
from utils.ModelUtils import z_score

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)


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


class OrthLoss(nn.Module):
    """正交损失：鼓励 gnn_info_fore 与 i_info_fore 正交。

    对每个样本计算两个隐向量的余弦相似度，取平方均值作为损失。
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, gnn_info: torch.Tensor, i_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_info: [N, H] GCN 信息分量
            i_info:   [N, H] 个股特异信息分量
        Returns:
            余弦相似度平方的均值（标量）
        """
        # 1. 在股票截面维度 (dim=0) 上中心化
        gnn_centered = gnn_info - gnn_info.mean(dim=0, keepdim=True)
        i_centered = i_info - i_info.mean(dim=0, keepdim=True)

        # 2. 在股票截面维度 (dim=0) 做 L2 Normalization
        gnn_norm = nn.functional.normalize(gnn_centered, dim=0, eps=self.eps)
        i_norm = nn.functional.normalize(i_centered, dim=0, eps=self.eps)

        # 3. 计算每个特征维度在截面上的相关性，求平方均值
        cov_matrix = (gnn_norm * i_norm).sum(dim=0)  # [H]
        return (cov_matrix ** 2).mean()


# ---------------- 前向 ----------------

def model_forward(_model, _sample, _device, _edge_v2: int, _edge_v3: int):
    """把单个日截面样本喂给 HIST_FactorGCL_V0。

    返回:
        pred:          [N]   预测值
        gnn_info_fore: [N,H] GCN 信息分量
        i_info_fore:   [N,H] 个股特异信息分量

    概念 1 = v0 行业（固定），概念 2 = v{edge_v2}，概念 3 = v{edge_v3}。
    """
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample[f"industry_v{_edge_v2}"].to(_device)
    industry_v3 = _sample[f"industry_v{_edge_v3}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    concept_3 = industry_to_concept(industry_v3)
    return _model(x, concept_1, concept_2, concept_3, market_value)


# ---------------- 训练 ----------------

def train_one_epoch(
    _model, _loader, _ic_loss, _orth_loss, _orth_lambda,
    _optimizer, _device, _edge_v2: int, _edge_v3: int
):
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue
        y_hat, gnn_info_fore, i_info_fore = model_forward(
            _model, _sample, _device, _edge_v2, _edge_v3)
        loss_ic = _ic_loss(y_hat, y)
        loss_orth = _orth_loss(gnn_info_fore, i_info_fore)
        loss = loss_ic + _orth_lambda * loss_orth
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
def evaluate(_model, _loader, _ic_loss, _orth_loss, _device, _edge_v2: int, _edge_v3: int):
    """验证集评估：返回 IC均值、ICIR、正交损失均值（模型选择仍只看 IC）。"""
    _model.eval()
    ics, orth_losses = [], []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        y_hat, gnn_info_fore, i_info_fore = model_forward(
            _model, _sample, _device, _edge_v2, _edge_v3)
        ics.append(_ic_loss(y_hat, y).item())
        orth_losses.append(_orth_loss(gnn_info_fore, i_info_fore).item())
    ics = np.asarray(ics, dtype=float)
    orth_mean = float(np.nanmean(orth_losses)) if orth_losses else float("nan")
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12), orth_mean  # IC均值, ICIR, OrthLoss均值


def parse_args():
    parser = argparse.ArgumentParser(
        description="HIST_FactorGCL_V0（3GCN + 正交损失，v0 行业 + 第二边 + 第三边）训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120],
                        help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14,
                        choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第二边版本（开放选择）")
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2,
                        choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第三边版本（开放选择）")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--orth_lambda", type=float, default=0.1,
                        help="正交损失权重 λ（默认 0.1）")
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1],
                        help="第一层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1],
                        help="第二层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew3", type=int, default=1, choices=[0, 1],
                        help="第三层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--label", type=str, default="data/label/stock_ret.pqt",
                        help="标签文件路径（.pqt 格式），默认 data/label/stock_ret.pqt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "3gcn_f0"

    # 校验：第二边和第三边版本不能相同
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    orth_lambda = args.orth_lambda

    # 标签名称（不含扩展名），用于数据版本与输出路径命名
    if not args.label.endswith(".pqt"):
        raise ValueError(f"标签文件必须为 .pqt 格式: {args.label}")
    label_name = os.path.splitext(os.path.basename(args.label))[0]

    # 数据版本：{freq}_lag{spans}_v0v{edge_v2}v{edge_v3}_{label_name}
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v{args.edge_version_2}v{args.edge_version_3}_{label_name}"

    ew1 = bool(args.ew1); ew_mark = 'ew1[off]' if not ew1 else ''
    ew2 = bool(args.ew2); ew_mark = ew_mark + 'ew2[off]' if not ew2 else ew_mark
    ew3 = bool(args.ew3); ew_mark = ew_mark + 'ew3[off]' if not ew3 else ew_mark
    orth_mark = f"ol{orth_lambda:g}"
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}_{orth_mark}{'_' + ew_mark if len(ew_mark) else ''}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}_{orth_mark}{'_' + ew_mark if len(ew_mark) else ''}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_FactorGCL_V0 ({model_id}) | data = {data_version} | hidden_size = {args.hidden_size} | ew1 = {ew1} | ew2 = {ew2} | ew3 = {ew3} | orth_lambda = {orth_lambda}")
    smart_log(f"label = {args.label}")

    seed_all_non_torch(root_seed)

    smart_log("Loading Data...")
    train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=True); smart_log("train done")
    valid_loader = get_dataloader("valid", version=data_version, batch_size=1, shuffle=False); smart_log("valid done")

    # 特征维度从训练集首个样本动态获取
    d_feat = train_loader.dataset[0]["x"].shape[1]
    smart_log(f"d_feat = {d_feat}")

    ic_criterion = ICLoss()
    orth_criterion = OrthLoss()

    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    edge_v2 = args.edge_version_2
    edge_v3 = args.edge_version_3

    for torch_seed in torch_seed_range:
        tpmd_pth = Path(ckpt_path) / f"{torch_seed}.pt"
        if Path(tpmd_pth).is_file():
            smart_log(f"{torch_seed} already done")
            continue

        smart_log(f"training {torch_seed}")
        seed_torch_all(torch_seed)

        model = HIST_FactorGCL_V0(
            d_feat=d_feat, hidden_size=args.hidden_size, clip=clip,
            ew1=ew1, ew2=ew2, ew3=ew3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, ic_criterion, orth_criterion, orth_lambda,
                optimizer, device, edge_v2, edge_v3)
            ic, icir, orth_loss_valid = evaluate(
                model, valid_loader, ic_criterion, orth_criterion, device, edge_v2, edge_v3)
            smart_log(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} | valid IC {ic:.4f} | "
                f"ICIR {icir:.4f} | valid OrthLoss {orth_loss_valid:.4f}")

            if ic < best_ic:
                smart_log(f"saving model {best_ic:.6f} -> {ic:.6f}")
                best_ic, wait = ic, 0
                torch.save(model.state_dict(), tpmd_pth)
            else:
                wait += 1
                if wait >= args.patience:
                    smart_log("early stopping")
                    break

    modelS = {
        iter_seed: HIST_FactorGCL_V0(
            d_feat=d_feat, hidden_size=args.hidden_size, clip=clip,
            ew1=ew1, ew2=ew2, ew3=ew3).to(device)
        for iter_seed in torch_seed_range
    }
    for model_num in torch_seed_range:
        modelS[model_num].load_state_dict(torch.load(
            Path(ckpt_path) / f"{model_num}.pt",
            map_location=device))
        modelS[model_num].eval()

    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=data_version, batch_size=1, shuffle=False):
            for i, model_num in enumerate(torch_seed_range):
                pred, _, _ = model_forward(modelS[model_num], sample, device, edge_v2, edge_v3)
                if i == 0:
                    pred_arr = z_score(pred.reshape(-1)).cpu().numpy()
                else:
                    pred_arr += z_score(pred.reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred_arr}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")