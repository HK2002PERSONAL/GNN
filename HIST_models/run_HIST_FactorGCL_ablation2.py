"""HIST_models：HIST_FactorGCL（GCN + GCN + 截面对比学习）训练脚本（daily batch）

模型 HIST_FactorGCL 用两个 GCN_InfoShare_block 串联 + 截面对比学习（TRCL）：
    概念 1 (v0)  : 一级行业 —— IndustryCodeNew（固定）
    概念 2 (v{n1}): 第二版边连通分量（开放选择）

数据由 DataBuilder/hist_data_builder/build_GCN_GCN_GCN_data.py 产出：
    data/hist_{freq}_lag{lags}_v0v{n1}v{n2}/{train,valid,test}.pt

    样本字段：
        x              [N, F] 特征
        industry_v0    [N]    行业整数编码（概念 1）
        industry_v{n1} [N]    第二边连通分量标签（概念 2）
        industry_v{n2} [N]    第三边连通分量标签（本模型不使用）
        market_value   [N]    流通市值
        y              [N]    标签（test 集为股票代码）
        date           [3]    年/月/日

训练损失：TRCL_Loss = ICLoss(pred, y) + gamma * TCInfoNCE(i_info)
验证损失：ICLoss（仅用 pred 与 y 计算 IC）

运行：
    python HIST_models/run_HIST_FactorGCL_V3.py --edge-version-2 2 --edge-version-3 3
    python HIST_models/run_HIST_FactorGCL_V3.py -v2 2 -v3 3 --hidden_size 64 --clip 0.04 --gamma 0.1

输出：model_configs/best_hist_FactorGCL_{data_version}_{clip_mark}{ew_mark}/
      model_outputs/hist_test_FactorGCL_{data_version}_{clip_mark}{ew_mark}.pqt
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

from models import HIST_Orth
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


class TCInfoNCE(nn.Module):
    """Cross-section InfoNCE for TRCL.

    forward(i_info) -> l_cross

    i_info: [B, F] individual alpha features
    B must be the stocks of a single cross-section (one trading day).
    """

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    @staticmethod
    def _cosine(a, b):
        a = nn.functional.normalize(a, dim=-1, p=2)
        b = nn.functional.normalize(b, dim=-1, p=2)
        return a @ b.t()  # [B, B]

    def forward(self, i_info):
        pre = i_info  # [B, F]

        # ---- cross-section InfoNCE: pos = (pre_i, pre_i), neg = (pre_i, pre_j) ----
        sim_pp = self._cosine(pre, pre) / self.tau          # [B, B]
        m2 = sim_pp.max(dim=1, keepdim=True).values.detach()
        l_cross = -(torch.diag(sim_pp) - m2.squeeze(1)
                    - torch.logsumexp(sim_pp - m2, dim=1)).mean()

        return l_cross


class TRCL_Loss(nn.Module):
    """TRCL = ICLoss(pred, y) + gamma * TCInfoNCE(i_info)"""

    def __init__(self, gamma: float, tau: float = 0.2, eps: float = 1e-8, clamp: float = 5):
        super().__init__()
        self.alpha_loss = ICLoss(eps, clamp)
        self.info_loss = TCInfoNCE(tau)
        self.gamma = gamma

    def forward(self, y, y_hat, i_info):
        return self.alpha_loss(y_hat, y) + self.gamma * self.info_loss(i_info)


# ---------------- 前向 ----------------

def model_forward_train(_model, _sample, _device, _edge_v2: int):
    """训练模式：直接调用 real_forward，返回 (pred, i_info)。

    概念 1 = v0 行业（固定），概念 2 = v{edge_v2}。
    """
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample[f"industry_v{_edge_v2}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    # real_forward -> (batch_result, i_info)
    return _model.real_forward(x, concept_1, concept_2, market_value)


def model_forward_eval(_model, _sample, _device, _edge_v2: int):
    """推理模式：output=True，返回预测值 [N, 1]。测试集无 x_fut，使用 output=True。"""
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample[f"industry_v{_edge_v2}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    # output=True -> batch_result only
    return _model(concept_1, concept_2, market_value, x, output=True)


# ---------------- 训练 ----------------

def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _edge_v2: int):
    """训练：使用 TRCL_Loss = ICLoss + gamma * TCInfoNCE(i_info)。"""
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue

        pred, i_info = model_forward_train(
            _model, _sample, _device, _edge_v2)
        pred = pred.reshape(-1)

        loss = _loss(y, pred, i_info)
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
def evaluate(_model, _loader, _ic_loss, _device, _edge_v2: int):
    """验证：使用 ICLoss（output=True，无需 x_fut）。"""
    _model.eval()
    ics = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        pred = model_forward_eval(_model, _sample, _device, _edge_v2).reshape(-1)
        ics.append(_ic_loss(pred, y).item())
    ics = np.asarray(ics, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_FactorGCL（GCN+GCN+截面对比学习）训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第二边版本（概念 2）")
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第三边版本（数据构建用，模型不使用）")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="TRCL_Loss 中对比学习损失权重")
    parser.add_argument("--tau", type=float, default=0.2,
                        help="TCInfoNCE 温度参数")
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1],
                        help="第一层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1],
                        help="第二层是否使用 edge_weight，1=使用（默认），0=不使用")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = f"FactorGCL_a2g[{args.gamma}]"

    # 校验：第二边和第三边版本不能相同
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    # 数据版本：{freq}_lag{spans}_v0v{edge_v2}v{edge_v3}（与 run_HIST_GCN_GCN_GCN.py 相同）
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v{args.edge_version_2}v{args.edge_version_3}"

    ew1 = bool(args.ew1)
    ew2 = bool(args.ew2)
    ew_mark = ''
    if not ew1:
        ew_mark += 'ew1[off]'
    if not ew2:
        ew_mark += ('_' if ew_mark else '') + 'ew2[off]'

    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if ew_mark else ''}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if ew_mark else ''}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size}")
    smart_log(f"gamma = {args.gamma} | tau = {args.tau} | ew1 = {ew1} | ew2 = {ew2}")

    seed_all_non_torch(root_seed)

    smart_log("Loading Data...")
    train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=True); smart_log("train done")
    valid_loader = get_dataloader("valid", version=data_version, batch_size=1, shuffle=False); smart_log("valid done")

    # 特征维度从训练集首个样本动态获取
    d_feat = train_loader.dataset[0]["x"].shape[1]
    smart_log(f"d_feat = {d_feat}")

    # 训练用 TRCL_Loss，验证用 ICLoss
    trcl_criterion = TRCL_Loss(gamma=args.gamma, tau=args.tau)
    ic_criterion = ICLoss()

    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    edge_v2 = args.edge_version_2

    for torch_seed in torch_seed_range:
        tpmd_pth = Path(ckpt_path) / f"{torch_seed}.pt"
        if Path(tpmd_pth).is_file():
            smart_log(f"{torch_seed} already done")
            continue

        smart_log(f"training {torch_seed}")
        seed_torch_all(torch_seed)

        model = HIST_Orth(
            d_feat=d_feat, hidden_size=args.hidden_size,
            clip=clip, ew1=ew1, ew2=ew2
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, trcl_criterion, optimizer, device, edge_v2)
            ic, icir = evaluate(model, valid_loader, ic_criterion, device, edge_v2)
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

    modelS = {
        iter_seed: HIST_Orth(
            d_feat=d_feat, hidden_size=args.hidden_size,
            clip=clip, ew1=ew1, ew2=ew2
        ).to(device)
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
                pred_i = z_score(
                    model_forward_eval(modelS[model_num], sample, device, edge_v2).reshape(-1)
                ).cpu().numpy()
                if i == 0:
                    prediction = pred_i
                else:
                    prediction += pred_i
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": prediction}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")