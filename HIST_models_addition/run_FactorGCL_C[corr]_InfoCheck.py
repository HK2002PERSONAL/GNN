"""HIST_models_addition：FactorGCL_C[corr] InfoCheck 版本

本版本基于 run_FactorGCL_C[corr].py，新增：
    - 继承 HIST_FactorGCL 创建 HIST_FactorGCL_InfoVer 类
    - 在验证集上记录 p1_info_fore, p2_info_fore, p3_info_fore, i_info_fore 的均值方差
    - 每个 epoch 输出这些统计量的均值

运行：
    python HIST_models_addition/run_FactorGCL_C[corr]_InfoCheck.py --edge-version-2 2 --edge-version-3 3
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

# 未来特征偏移（与 build_FactorGCL_data.py 保持一致，仅用于数据目录名后缀）
FUT_SHIFT = 5


# ---------------- InfoVer 模型类 ----------------

class HIST_FactorGCL_InfoVer(HIST_Orth):
    """继承 HIST_FactorGCL，增加记录各层 info_fore 的均值方差功能。
    
    在验证时记录：
        - p1_info_fore: 第一层 GCN_InfoShare_block 输出
        - p2_info_fore: 第二层 GCN_InfoShare_block 输出
        - p3_info_fore: 第三层 GCN_InfoShare_block 输出
        - i_info_fore: 最终个体特征
    """
    # pylint:disable=W0621
    def __init__(self, d_feat=6, hidden_size=64, clip=5,
                 ew1: bool = True, ew2: bool = True, ew3: bool = True):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, clip=clip,
                         ew1=ew1, ew2=ew2, ew3=ew3)
        # 存储各层 info_fore 的统计信息
        self.info_stats = {
            'p1_info_fore': {'mean': [], 'std': []},
            'p2_info_fore': {'mean': [], 'std': []},
            'p3_info_fore': {'mean': [], 'std': []},
            'i_info_fore': {'mean': [], 'std': []},
        }
    
    def real_forward_with_stats(
        self,
        x: torch.Tensor, # [N, F]
        concept_matrix_1: torch.Tensor, # [N, C] 
        concept_matrix_2: torch.Tensor, # [N, C] 
        concept_matrix_3: torch.Tensor, # [N, C] 
        market_value: torch.Tensor, # [N] 
        record_stats: bool = False
    ) -> torch.Tensor:
        """带统计信息的前向传播。
        
        Args:
            record_stats: 是否记录统计信息（验证时为 True）
        
        Returns:
            (pred, i_info_fore, info_dict) 其中 info_dict 包含各层 info_fore
        """
        x_hidden = self.encoder(self.fclip(x))

        # Layer 1
        p1_info_fore, p1_info_back = self.predefined_info_block_1(x_hidden, concept_matrix_1, market_value)
        x_hidden_dp1 = x_hidden - p1_info_back

        # Layer 2
        p2_info_fore, p2_info_back = self.predefined_info_block_2(x_hidden_dp1, concept_matrix_2, market_value)
        x_hidden_dp1p2 = x_hidden_dp1 - p2_info_back

        # Layer 3
        p3_info_fore, p3_info_back = self.predefined_info_block_3(x_hidden_dp1p2, concept_matrix_3, market_value)
        x_hidden_dp1p2p3 = x_hidden_dp1p2 - p3_info_back

        # final layer
        i_info_fore = nn.functional.leaky_relu(self.fc_i_fore(x_hidden_dp1p2p3))

        # output layer
        all_info_fore = p1_info_fore + p2_info_fore + p3_info_fore + i_info_fore

        # 记录统计信息
        if record_stats:
            with torch.no_grad():
                self.info_stats['p1_info_fore']['mean'].append(p1_info_fore.mean().item())
                self.info_stats['p1_info_fore']['std'].append(p1_info_fore.std().item())
                self.info_stats['p2_info_fore']['mean'].append(p2_info_fore.mean().item())
                self.info_stats['p2_info_fore']['std'].append(p2_info_fore.std().item())
                self.info_stats['p3_info_fore']['mean'].append(p3_info_fore.mean().item())
                self.info_stats['p3_info_fore']['std'].append(p3_info_fore.std().item())
                self.info_stats['i_info_fore']['mean'].append(i_info_fore.mean().item())
                self.info_stats['i_info_fore']['std'].append(i_info_fore.std().item())

        return self.fc_out(all_info_fore), i_info_fore, {
            'p1_info_fore': p1_info_fore,
            'p2_info_fore': p2_info_fore,
            'p3_info_fore': p3_info_fore,
            'i_info_fore': i_info_fore
        }
    
    def clear_stats(self):
        """清空统计信息。"""
        for key in self.info_stats:
            self.info_stats[key]['mean'] = []
            self.info_stats[key]['std'] = []
    
    def get_stats_summary(self):
        """获取当前统计信息的均值（用于每个 epoch 结束后输出）。"""
        summary = {}
        for key in self.info_stats:
            means = self.info_stats[key]['mean']
            stds = self.info_stats[key]['std']
            summary[key] = {
                'mean_of_mean': float(np.mean(means)) if means else float('nan'),
                'mean_of_std': float(np.mean(stds)) if stds else float('nan'),
            }
        return summary


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
    """Cross-section InfoNCE for TRCL (C[corr]: 仅保留 Cross，删除 Temporal).

    forward(i_info) -> l_cross

    i_info: [B, F] individual alpha features
    B must be the stocks of a single cross-section (one trading day).

    C[corr] 版本：使用原始余弦相似度（区别于 C[orth] 的平方相似度），
    且删除对角线正样本项（对角线为常数 1/tau，理论上无梯度），
    仅保留 l_cross = logsumexp(sim_pp) 作为正则。
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

        # ---- cross-section InfoNCE（C[corr]）：删除对角线正样本项，只保留 logsumexp ----
        sim_pp = self._cosine(pre, pre) / self.tau          # [B, B]
        u = sim_pp.max(dim=1, keepdim=True).values.detach()
        l_cross = (u.squeeze(1) + torch.logsumexp(sim_pp - u, dim=1)).mean()

        return l_cross


class TRCL_Loss(nn.Module):
    """TRCL = ICLoss(pred, y) + gamma * CrossInfoNCE(i_info)
    
    返回: (total_loss, ic_loss, cross_loss)
    """

    def __init__(self, gamma: float, tau: float = 0.2, eps: float = 1e-8, clamp: float = 5):
        super().__init__()
        self.alpha_loss = ICLoss(eps, clamp)
        self.info_loss = TCInfoNCE(tau)
        self.gamma = gamma

    def forward(self, y, y_hat, i_info):
        ic_loss = self.alpha_loss(y_hat, y)
        cross_loss = self.info_loss(i_info)
        total_loss = ic_loss + self.gamma * cross_loss
        return total_loss, ic_loss, cross_loss


# ---------------- 前向 ----------------

def model_forward_train(_model, _sample, _device, _edge_v2: int, _edge_v3: int, record_stats: bool = False):
    """训练模式：直接调用 real_forward，返回 (pred, i_info)。

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
    # real_forward -> (batch_result, i_info, info_dict)
    pred, i_info, _ = _model.real_forward_with_stats(x, concept_1, concept_2, concept_3, market_value, record_stats)
    return pred, i_info


def model_forward_eval(_model, _sample, _device, _edge_v2: int, _edge_v3: int):
    """推理模式：output=True，返回预测值 [N, 1]。测试集无 x_fut，使用 output=True。"""
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample[f"industry_v{_edge_v2}"].to(_device)
    industry_v3 = _sample[f"industry_v{_edge_v3}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    concept_3 = industry_to_concept(industry_v3)
    # output=True -> batch_result only
    return _model(concept_1, concept_2, concept_3, market_value, x, output=True)


# ---------------- 训练 ----------------

def compute_grad_norm(_model, _loss_component, _gamma: float = 1.0):
    """对单个损失分量进行反向传播并计算梯度范数（不更新参数）。
    
    注意：调用后需要 _model.zero_grad() 清除梯度。
    """
    _model.zero_grad()
    scaled_loss = _gamma * _loss_component
    scaled_loss.backward(retain_graph=True)
    total_norm = 0.0
    for p in _model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    _model.zero_grad()
    return total_norm


def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _edge_v2: int, _edge_v3: int):
    """训练：使用 TRCL_Loss = ICLoss + gamma * CrossInfoNCE(i_info)。
    
    返回: (total_loss_mean, ic_grad_norm_mean, cross_grad_norm_mean)
    
    对每个样本：
        1. 分别计算两个损失分量的梯度范数（用于日志）
        2. 用总损失进行实际梯度更新
    """
    _model.train()
    total_losses = []
    ic_grad_norms = []
    cross_grad_norms = []
    
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue

        pred, i_info = model_forward_train(
            _model, _sample, _device, _edge_v2, _edge_v3, record_stats=False)
        pred = pred.reshape(-1)

        total_loss, ic_loss, cross_loss = _loss(y, pred, i_info)
        if total_loss is None or not torch.isfinite(total_loss):
            continue
        
        # ---- 计算各损失分量的梯度范数（仅用于日志，不影响训练） ----
        # IC loss 的梯度范数
        ic_gn = compute_grad_norm(_model, ic_loss)
        # Cross InfoNCE 的梯度范数（乘以 gamma 以反映实际贡献）
        cross_gn = compute_grad_norm(_model, cross_loss, _loss.gamma)
        
        # ---- 实际训练步骤：用总损失更新参数 ----
        _optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            _optimizer.zero_grad()
            continue
        _optimizer.step()
        
        total_losses.append(total_loss.item())
        ic_grad_norms.append(ic_gn)
        cross_grad_norms.append(cross_gn)
    
    if not total_losses:
        return float("nan"), float("nan"), float("nan")
    
    return (
        float(np.mean(total_losses)),
        float(np.mean(ic_grad_norms)),
        float(np.mean(cross_grad_norms))
    )


@torch.no_grad()
def evaluate(_model, _loader, _ic_loss, _cross_loss_fn, _device, _edge_v2: int, _edge_v3: int):
    """验证：计算 IC loss 和 Cross InfoNCE loss，同时记录 info_fore 统计。
    
    使用 real_forward_with_stats 获取 i_info 以计算 cross_loss。
    返回: (IC均值, ICIR, cross_loss均值)
    """
    _model.eval()
    _model.clear_stats()  # 清空之前的统计
    ics = []
    cross_losses = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        # 使用 real_forward_with_stats 获取 pred 和 i_info，同时记录统计
        pred, i_info = model_forward_train(_model, _sample, _device, _edge_v2, _edge_v3, record_stats=True)
        pred = pred.reshape(-1)
        ics.append(_ic_loss(pred, y).item())
        cross_losses.append(_cross_loss_fn(i_info).item())
    ics = np.asarray(ics, dtype=float)
    cross_losses = np.asarray(cross_losses, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12), np.nanmean(cross_losses)


def format_info_stats(summary):
    """格式化 info 统计信息用于日志输出。"""
    parts = []
    for key in ['p1_info_fore', 'p2_info_fore', 'p3_info_fore', 'i_info_fore']:
        stats = summary[key]
        parts.append(f"{key}(mean:{stats['mean_of_mean']:.4f},std:{stats['mean_of_std']:.4f})")
    return " | ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="FactorGCL_C[corr] InfoCheck 版本 - 记录 info_fore 统计")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第二边版本（概念 2）")
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第三边版本（概念 3）")
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
    parser.add_argument("--ew3", type=int, default=1, choices=[0, 1],
                        help="第三层是否使用 edge_weight，1=使用（默认），0=不使用")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = f"FactorGCL_C[corr]g[{args.gamma}]_InfoLog"

    # 校验：第二边和第三边版本不能相同
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    # 数据版本：{freq}_lag{spans}_v0v{edge_v2}v{edge_v3}_fut{FUT_SHIFT}（与 build_FactorGCL_data.py 输出目录一致）
    data_version = (
        f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}"
        f"_v0v{args.edge_version_2}v{args.edge_version_3}_fut{FUT_SHIFT}"
    )

    ew1 = bool(args.ew1)
    ew2 = bool(args.ew2)
    ew3 = bool(args.ew3)
    ew_mark = ''
    if not ew1:
        ew_mark += 'ew1[off]'
    if not ew2:
        ew_mark += ('_' if ew_mark else '') + 'ew2[off]'
    if not ew3:
        ew_mark += ('_' if ew_mark else '') + 'ew3[off]'

    # 添加 InfoCheck 标识
    exp_name = f"{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if ew_mark else ''}_InfoCheck"
    ckpt_path = f"model_configs_GCL/{exp_name}"
    out_path = f"model_outputs_GCL/{exp_name}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id}_InfoVer | data = {data_version} | hidden_size = {args.hidden_size}")
    smart_log(f"gamma = {args.gamma} | tau = {args.tau} | ew1 = {ew1} | ew2 = {ew2} | ew3 = {ew3}")
    smart_log("InfoCheck 模式（C[corr]）：每 epoch 记录验证集 p1/p2/p3/i_info_fore 的均值方差")

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
    edge_v3 = args.edge_version_3

    for torch_seed in torch_seed_range:
        tpmd_pth = Path(ckpt_path) / f"{torch_seed}.pt"
        if Path(tpmd_pth).is_file():
            smart_log(f"{torch_seed} already done")
            continue

        smart_log(f"training {torch_seed}")
        seed_torch_all(torch_seed)

        model = HIST_FactorGCL_InfoVer(
            d_feat=d_feat, hidden_size=args.hidden_size,
            clip=clip, ew1=ew1, ew2=ew2, ew3=ew3
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss, ic_gn_mean, cross_gn_mean = train_one_epoch(
                model, train_loader, trcl_criterion, optimizer, device, edge_v2, edge_v3)
            ic, icir, valid_cross_loss = evaluate(
                model, valid_loader, ic_criterion, trcl_criterion.info_loss, device, edge_v2, edge_v3)
            
            # 获取 info 统计信息
            stats_summary = model.get_stats_summary()
            info_stats_str = format_info_stats(stats_summary)
            
            # 记录：训练集总损失、验证集 IC/ICIR/cross_loss、info_fore 统计
            smart_log(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                f"valid IC {ic:.4f} | ICIR {icir:.4f} | valid cross_loss {valid_cross_loss:.4f} | "
                f"IC grad norm {ic_gn_mean:.4f} | Cross grad norm {cross_gn_mean:.4f}"
            )
            smart_log(f"    info stats: {info_stats_str}")

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
        iter_seed: HIST_FactorGCL_InfoVer(
            d_feat=d_feat, hidden_size=args.hidden_size,
            clip=clip, ew1=ew1, ew2=ew2, ew3=ew3
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
                    model_forward_eval(modelS[model_num], sample, device, edge_v2, edge_v3).reshape(-1)
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