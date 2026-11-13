"""HIST_models：HIST_a11_tn（GCN + GCN + GCN + TabNet head）训练脚本（daily batch）

模型 HIST_a11_tn 在 HIST_a11 基础上，将输出头替换为 TabNet，
训练时需要加入稀疏正则项：loss = reg_criterion(pred, y) + model.sparsity * M_loss
验证时使用 ICLoss。

模型需要三种预定义概念（边）：
    概念 1 (v0)  : 一级行业      —— IndustryCodeNew（固定）
    概念 2 (v{n1}): 第二版边连通分量（开放选择）
    概念 3 (v{n2}): 第三版边连通分量（开放选择）

数据由 DataBuilder/hist_data_builder/build_GCN_GCN_GCN_data.py 产出：
    data/hist_{freq}_lag{lag1}_{lag2}_..._v0v{n1}v{n2}_{label_name}/{train,valid,test}.pt
    其中 label_name 为标签文件名（不含扩展名），如 stock_ret、spec_ret（默认 stock_ret）

    样本字段：
        x             [N, F] 特征
        industry_v0   [N]    行业整数编码（概念 1，固定）
        industry_v{n1}[N]    第二边连通分量标签（概念 2）
        industry_v{n2}[N]    第三边连通分量标签（概念 3）
        market_value  [N]    流通市值
        y             [N]    标签（test 集为股票代码）
        date          [3]    年/月/日

运行：
    python HIST_models/run_HIST_3GCN_tn.py --edge-version-2 2 --edge-version-3 3
    python HIST_models/run_HIST_3GCN_tn.py -v2 2 -v3 3 --hidden_size 64 --clip 0.04
    python HIST_models/run_HIST_3GCN_tn.py -v2 2 -v3 3 --sparsity 1e-3

输出：model_configs/best_hist_a11_tn_{data_version}_{clip_mark}{ew_mark}/
      model_outputs/hist_test_a11_tn_{data_version}_{clip_mark}{ew_mark}.pqt
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

from models import HIST_a11_tn
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


class CCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, clamp: float = 5.0):
        super().__init__()
        self.eps = eps
        self.clamp = clamp

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: 预测值，[N] 或 [N, 1]
        y: 真实标签，[N] 或 [N, 1]
        """
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()

        # 仅保留预测值和标签都有效的样本
        mask = torch.isfinite(x) & torch.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.numel() < 2:
            return x.sum() * 0.0

        # 仅截断极值，不做均值移除或截面标准化
        x = x.clamp(-self.clamp, self.clamp)
        y = y.clamp(-self.clamp, self.clamp)

        x_mean = x.mean()
        y_mean = y.mean()

        x_centered = x - x_mean
        y_centered = y - y_mean

        x_var = x_centered.pow(2).mean()
        y_var = y_centered.pow(2).mean()
        covariance = (x_centered * y_centered).mean()

        denominator = (
            x_var
            + y_var
            + (x_mean - y_mean).pow(2)
        ).clamp_min(self.eps)

        ccc = 2.0 * covariance / denominator

        return 1.0 - ccc


# ---------------- 前向 ----------------

def model_forward(_model, _sample, _device, _edge_v2: int, _edge_v3: int):
    """把每日截面样本喂给 HIST_a11_tn。

    概念 1 = v0 行业（固定），概念 2 = v{edge_v2}，概念 3 = v{edge_v3}。
    
    训练模式下模型返回 (pred, M_loss)；推理模式下只返回 pred。
    return_M_loss=True 时（训练用）返回 (pred, M_loss)。
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

def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _edge_v2: int, _edge_v3: int):
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue
        # 训练模式下模型返回 (pred, M_loss)
        _pred, M_loss = model_forward(_model, _sample, _device, _edge_v2, _edge_v3)
        # 损失 = 回归损失 + 稀疏正则项
        loss = _loss(_pred, y) + _model.sparsity * M_loss
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
def evaluate(_model, _loader, _loss, _device, _edge_v2: int, _edge_v3: int):
    _model.eval()
    ics = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        # 推理模式下模型只返回 pred
        _pred = model_forward(_model, _sample, _device, _edge_v2, _edge_v3)
        ics.append(_loss(_pred, y).item())
    ics = np.asarray(ics, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a11_tn（GCN+GCN+GCN+TabNet head，v0 行业 + 第二边 + 第三边）训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第二边版本（开放选择）")
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第三边版本（开放选择）")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1],
                        help="第一层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1],
                        help="第二层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew3", type=int, default=1, choices=[0, 1],
                        help="第三层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--label", type=str, default="data/label/stock_ret.pqt",
                        help="标签文件路径（.pqt 格式），默认 data/label/stock_ret.pqt")
    parser.add_argument("--loss", type=str, default="ICLoss", choices=["ICLoss", "CCCLoss"],
                        help="训练损失函数选择（验证集仍使用 ICLoss），默认 ICLoss")
    # TabNet 参数
    parser.add_argument("--n_d", type=int, default=8, help="TabNet decision 层宽度")
    parser.add_argument("--n_a", type=int, default=8, help="TabNet attention 层宽度")
    parser.add_argument("--n_steps", type=int, default=3, help="TabNet 步数")
    parser.add_argument("--n_independent", type=int, default=2, help="TabNet 独立层数")
    parser.add_argument("--n_shared", type=int, default=2, help="TabNet 共享层数")
    parser.add_argument("--gamma", type=float, default=1.3, help="TabNet gamma 系数")
    parser.add_argument("--momentum", type=float, default=0.02, help="TabNet BN momentum")
    parser.add_argument("--mask_type", type=str, default="sparsemax", choices=["sparsemax", "entmax"],
                        help="TabNet mask 类型")
    parser.add_argument("--virtual_batch_size", type=int, default=64, help="TabNet virtual batch size")
    parser.add_argument("--sparsity", type=float, default=1e-3, help="TabNet 稀疏正则系数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a11_tn"

    # 校验：第二边和第三边版本不能相同
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    # 标签名称（不含扩展名），用于数据版本与输出路径命名
    if not args.label.endswith(".pqt"):
        raise ValueError(f"标签文件必须为 .pqt 格式: {args.label}")
    label_name = os.path.splitext(os.path.basename(args.label))[0]

    # 数据版本：{freq}_lag{spans}_v0v{edge_v2}v{edge_v3}_{label_name}
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v{args.edge_version_2}v{args.edge_version_3}_{label_name}"

    ew1 = bool(args.ew1); ew_mark = 'ew1[off]' if not ew1 else ''
    ew2 = bool(args.ew2); ew_mark = ew_mark + 'ew2[off]' if not ew2 else ew_mark
    ew3 = bool(args.ew3); ew_mark = ew_mark + 'ew3[off]' if not ew3 else ew_mark
    
    # 损失函数标记（使用非默认损失时添加到输出路径）
    loss_mark = f"_{args.loss}"
    
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size} | ew1 = {ew1} | ew2 = {ew2} | ew3 = {ew3}")
    smart_log(f"label = {args.label} | train_loss = {args.loss} | sparsity = {args.sparsity}")
    smart_log(f"TabNet: n_d = {args.n_d} | n_a = {args.n_a} | n_steps = {args.n_steps} | mask_type = {args.mask_type}")

    seed_all_non_torch(root_seed)

    smart_log("Loading Data...")
    train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=True); smart_log("train done")
    valid_loader = get_dataloader("valid", version=data_version, batch_size=1, shuffle=False); smart_log("valid done")

    # 特征维度从训练集首个样本动态获取
    d_feat = train_loader.dataset[0]["x"].shape[1]
    smart_log(f"d_feat = {d_feat}")

    # 训练损失函数（可选），验证集统一使用 ICLoss
    if args.loss == "ICLoss":
        criterion = ICLoss()
    elif args.loss == "CCCLoss":
        criterion = CCCLoss()
    else:
        raise ValueError(f"不支持的损失函数: {args.loss}")
    
    # 验证集统一使用 ICLoss
    valid_criterion = ICLoss()

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

        model = HIST_a11_tn(
            d_feat=d_feat, hidden_size=args.hidden_size, clip=clip,
            ew1=ew1, ew2=ew2, ew3=ew3,
            n_d=args.n_d, n_a=args.n_a, n_steps=args.n_steps,
            n_independent=args.n_independent, n_shared=args.n_shared,
            gamma=args.gamma, momentum=args.momentum, mask_type=args.mask_type,
            virtual_batch_size=args.virtual_batch_size,
            sparsity=args.sparsity,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, edge_v2, edge_v3)
            ic, icir = evaluate(model, valid_loader, valid_criterion, device, edge_v2, edge_v3)
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

    modelS = {iter_seed: HIST_a11_tn(
            d_feat=d_feat, hidden_size=args.hidden_size, clip=clip,
            ew1=ew1, ew2=ew2, ew3=ew3,
            n_d=args.n_d, n_a=args.n_a, n_steps=args.n_steps,
            n_independent=args.n_independent, n_shared=args.n_shared,
            gamma=args.gamma, momentum=args.momentum, mask_type=args.mask_type,
            virtual_batch_size=args.virtual_batch_size,
            sparsity=args.sparsity,
        ).to(device) for iter_seed in torch_seed_range}
    for model_num in torch_seed_range:
        modelS[model_num].load_state_dict(torch.load(
            Path(ckpt_path) / f"{model_num}.pt",
            map_location=device))
        modelS[model_num].eval()

    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=data_version, batch_size=1, shuffle=False):
            for i, model_num in enumerate(torch_seed_range):
                # 推理模式下模型只返回 pred
                if i == 0:
                    pred = z_score(model_forward(modelS[model_num], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
                else:
                    pred += z_score(model_forward(modelS[model_num], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")