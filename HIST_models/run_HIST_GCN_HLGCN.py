"""HIST_V_ablation：HIST_a6（GCN + GCN）训练脚本（daily batch）

模型 HIST_a6 用两个 GCN_InfoShare_block 串联，需要两种预定义概念（边）：
    概念 1 (v0) : 一级行业      —— IndustryCodeNew
    概念 2 (v2) : 相关性边连通分量 —— data/edges_v2/{day}.npz 的连通分量

数据由 data_utils.py 的 v0v2 版本提供：
    data/hist_daily_dual/{train,valid,test}.pt
    （由 HIST_V_ablation/step1_build_dual_data.py 产出，样本含
      industry_v0 / industry_v2 两个概念标签，已在同一股票集合上对齐）

    样本字段：
        x            [N, F] 特征
        industry_v0  [N]    行业整数编码（概念 1）
        industry_v2  [N]    连通分量标签（概念 2）
        market_value [N]    流通市值
        y            [N]    标签（test 集为股票代码）
        date         [3]    年/月/日

运行：
    python HIST_V_ablation/run_HIST_GCN_GCN.py
    python HIST_V_ablation/run_HIST_GCN_GCN.py --hidden_size 64 --clip 5

输出：model_configs/best_hist_a6_v0v2_<clip>.pt、model_outputs/hist_test_a6_v0v2_<clip>.pqt
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

from models import HIST_a6_v2, HIST_a6_v2_2
from data_utils import get_dataloader
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch, seed_torch_all
from utils.ModelUtils import z_score

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)
model_dict = {1: HIST_a6_v2, 2: HIST_a6_v2_2}

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

def model_forward(_model, _sample, _device):
    """把单个日截面样本喂给 HIST_a6，返回预测值 [N]。

    概念 1 = v0 行业，概念 2 = v2 连通分量，两者已在同一股票集合上对齐。
    """
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample["industry_v2"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    return _model(x, concept_1, concept_2, market_value)


# ---------------- 训练 ----------------

def train_one_epoch(_model, _loader, _loss, _optimizer, _device):
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue
        _pred = model_forward(_model, _sample, _device)
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
def evaluate(_model, _loader, _loss, _device):
    _model.eval()
    ics = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        _pred = model_forward(_model, _sample, _device)
        ics.append(_loss(_pred, y).item())
    ics = np.asarray(ics, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a6_v2（GCN+HLGCN，v0 行业 + v2 相关性边）训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--data", type=str, default="v0v3", choices=["v0v3"],
                        help="数据集版本：v0v3（双概念对齐数据）")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--hlgcn_version", "-hlv", type=int, default=1)
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1],
                        help="第一层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1],
                        help="第二层是否使用 edge_weight，1=使用（默认），0=不使用")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a6_v2" if args.hlgcn_version == 1 else f"a6_v2_{args.hlgcn_version}"

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_{args.data}"

    ew1 = bool(args.ew1); ew_mark = 'ew1[off]' if not ew1 else ''
    ew2 = bool(args.ew2); ew_mark = ew_mark + 'ew2[off]' if not ew2 else ew_mark
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{ew_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}{ew_mark}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size} | ew1 = {ew1} | ew2 = {ew2}")

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

        model = model_dict[args.hlgcn_version](d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            ic, icir = evaluate(model, valid_loader, criterion, device)
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

    modelS = {iter_seed: model_dict[args.hlgcn_version](d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2).to(device) for iter_seed in torch_seed_range}
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
                    pred = z_score(model_forward(modelS[model_num], sample, device).reshape(-1)).cpu().numpy()
                else:
                    pred += z_score(model_forward(modelS[model_num], sample, device).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")