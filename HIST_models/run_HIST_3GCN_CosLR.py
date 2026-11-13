"""HIST_models：HIST_a11（GCN + GCN + GCN）训练脚本（多种学习率调度策略）

基于 run_HIST_3GCN.py，添加多种学习率调度器支持：
    - CosineAnnealingWarmRestarts: 余弦退火带热重启（支持快照集成）
    - CosineAnnealingLR: 标准余弦退火（不支持快照集成）
    - ReduceLROnPlateau: 验证集指标停滞时降低学习率（不支持快照集成）

模型 HIST_a11 用三个 GCN_InfoShare_block 串联，需要三种预定义概念（边）：
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
    # CosineAnnealingWarmRestarts（默认）
    python HIST_models/run_HIST_3GCN_CosLR.py --edge-version-2 14 --edge-version-3 2
    python HIST_models/run_HIST_3GCN_CosLR.py -v2 14 -v3 2 --T_0 10 --T_mult 2 --eta_min 1e-5
    python HIST_models/run_HIST_3GCN_CosLR.py -v2 14 -v3 2 --snapshot  # 启用快照集成
    
    # CosineAnnealingLR（不支持快照集成）
    python HIST_models/run_HIST_3GCN_CosLR.py -v2 14 -v3 2 --scheduler cosine
    
    # ReduceLROnPlateau（不支持快照集成）
    python HIST_models/run_HIST_3GCN_CosLR.py -v2 14 -v3 2 --scheduler plateau --plateau_patience 5 --plateau_factor 0.5

快照集成 (Snapshot Ensembling)：
    在每个学习率周期结束时保存模型快照，推理时集成所有快照的预测结果。
    快照文件命名：{torch_seed}_snapshot_{id}.pt
    注意：仅 CosineAnnealingWarmRestarts 调度器支持快照集成。

输出：model_configs/best_hist_a11_{data_version}_{clip_mark}{ew_mark}{loss_mark}_{scheduler_mark}[_snap]/
      model_outputs/hist_test_a11_{data_version}_{clip_mark}{ew_mark}{loss_mark}_{scheduler_mark}[_snap].pqt
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

from models import HIST_a11
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
    """把单个日截面样本喂给 HIST_a11，返回预测值 [N]。

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

def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _edge_v2: int, _edge_v3: int):
    _model.train()
    losses = []
    for _sample in _loader:                  # batch_size=1，每天一个截面
        y = _sample["y"].to(_device)
        sample_num = torch.isfinite(y).sum()
        if sample_num < 10:
            continue
        _pred = model_forward(_model, _sample, _device, _edge_v2, _edge_v3)
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
def evaluate(_model, _loader, _loss, _device, _edge_v2: int, _edge_v3: int):
    _model.eval()
    ics = []
    for _sample in _loader:
        y = _sample["y"].to(_device)
        if torch.isfinite(y).sum() < 5:
            continue
        _pred = model_forward(_model, _sample, _device, _edge_v2, _edge_v3)
        ics.append(_loss(_pred, y).item())
    ics = np.asarray(ics, dtype=float)
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a11（GCN+GCN+GCN，v0 行业 + 第二边 + 第三边）训练（多种学习率调度策略）")
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
    # 学习率调度器选择
    parser.add_argument("--scheduler", type=str, default="warm_restarts",
                        choices=["warm_restarts", "cosine", "plateau"],
                        help="学习率调度器：warm_restarts=CosineAnnealingWarmRestarts（默认），"
                             "cosine=CosineAnnealingLR，plateau=ReduceLROnPlateau")
    # CosineAnnealingWarmRestarts / CosineAnnealingLR 共用参数
    parser.add_argument("--T_0", type=int, default=10,
                        help="第一次重启的周期长度（默认 10 epochs），仅 warm_restarts 使用")
    parser.add_argument("--T_mult", type=int, default=2,
                        help="每次重启后周期倍增系数（默认 2，即周期变为 10→20→40...），仅 warm_restarts 使用")
    parser.add_argument("--eta_min", type=float, default=1e-5,
                        help="最小学习率（默认 1e-5），warm_restarts 和 cosine 使用")
    # ReduceLROnPlateau 参数
    parser.add_argument("--plateau_mode", type=str, default="min", choices=["min", "max"],
                        help="ReduceLROnPlateau 监控模式（默认 min，即监控验证损失/IC 越小越好）")
    parser.add_argument("--plateau_patience", type=int, default=5,
                        help="ReduceLROnPlateau 学习率降低前的等待轮数（默认 5）")
    parser.add_argument("--plateau_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau 学习率衰减因子（默认 0.5）")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6,
                        help="ReduceLROnPlateau 最小学习率（默认 1e-6）")
    # 快照集成参数
    parser.add_argument("--snapshot", action="store_true",
                        help="启用快照集成（每个周期结束保存快照），默认不启用，仅支持 warm_restarts 调度器")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a11"

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
    
    # 调度器标记 + 快照集成标记
    scheduler_type = args.scheduler
    # 快照集成仅支持 warm_restarts 调度器
    use_snapshot = args.snapshot and scheduler_type == "warm_restarts"
    if args.snapshot and scheduler_type != "warm_restarts":
        print(f"[警告] 快照集成仅支持 warm_restarts 调度器，当前使用 {scheduler_type}，快照集成已禁用")
    
    # 根据调度器类型生成标记
    scheduler_mark_map = {
        "warm_restarts": "_CosLR",
        "cosine": "_CosineLR",
        "plateau": "_PlateauLR",
    }
    scheduler_mark = scheduler_mark_map[scheduler_type] + ("_snap" if use_snapshot else "")
    
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}{scheduler_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}{scheduler_mark}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size} | ew1 = {ew1} | ew2 = {ew2} | ew3 = {ew3}")
    smart_log(f"label = {args.label} | train_loss = {args.loss}")
    # 根据调度器类型输出不同日志
    if scheduler_type == "warm_restarts":
        smart_log(f"scheduler = CosineAnnealingWarmRestarts | T_0 = {args.T_0} | T_mult = {args.T_mult} | eta_min = {args.eta_min}")
    elif scheduler_type == "cosine":
        smart_log(f"scheduler = CosineAnnealingLR | T_max = {args.epochs} | eta_min = {args.eta_min}")
    else:  # plateau
        smart_log(f"scheduler = ReduceLROnPlateau | mode = {args.plateau_mode} | patience = {args.plateau_patience} | factor = {args.plateau_factor} | min_lr = {args.plateau_min_lr}")
    smart_log(f"snapshot_ensemble = {use_snapshot}")

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

        model = HIST_a11(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2, ew3=ew3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 根据调度器类型创建学习率调度器
        if scheduler_type == "warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.T_0,
                T_mult=args.T_mult,
                eta_min=args.eta_min,
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.eta_min,
            )
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=args.plateau_mode,
                patience=args.plateau_patience,
                factor=args.plateau_factor,
                min_lr=args.plateau_min_lr,
            )

        best_ic, wait = np.inf, 0
        snapshot_count = 0  # 快照计数器
        
        # 计算当前周期结束点（用于快照保存，仅 warm_restarts 使用）
        # 初始周期长度为 T_0，之后每次乘以 T_mult
        cycle_length = args.T_0 if scheduler_type == "warm_restarts" else 0
        epoch_in_cycle = 0
        
        for epoch in range(1, args.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, edge_v2, edge_v3)
            ic, icir = evaluate(model, valid_loader, valid_criterion, device, edge_v2, edge_v3)
            smart_log(f"epoch {epoch:3d} | lr {current_lr:.6f} | train loss {train_loss:.4f} | valid IC {ic:.4f} | ICIR {icir:.4f}")

            # 每个 epoch 结束后更新学习率
            # plateau 调度器需要传入验证指标（IC 损失，越小越好）
            if scheduler_type == "plateau":
                scheduler.step(ic)
            else:
                scheduler.step()
            
            # 快照集成：在周期结束时保存快照（仅 warm_restarts 支持）
            if scheduler_type == "warm_restarts" and use_snapshot:
                epoch_in_cycle += 1
                if epoch_in_cycle >= cycle_length:
                    snapshot_count += 1
                    snapshot_pth = Path(ckpt_path) / f"{torch_seed}_snapshot_{snapshot_count}.pt"
                    torch.save(model.state_dict(), snapshot_pth)
                    smart_log(f"  [snapshot {snapshot_count}] saved at epoch {epoch} | IC {ic:.4f}")
                    # 更新下一个周期长度
                    cycle_length = int(cycle_length * args.T_mult)
                    epoch_in_cycle = 0

            if ic < best_ic:
                smart_log(f"saving model {best_ic:.6f} -> {ic:.6f}")
                best_ic, wait = ic, 0
                torch.save(model.state_dict(), tpmd_pth)
            else:
                wait += 1
                if wait >= args.patience:
                    smart_log("early stopping")
                    break

    # 加载模型用于推理
    if use_snapshot:
        # 快照集成模式：加载所有种子的所有快照
        smart_log("Loading snapshot models for ensemble...")
        all_models = []  # [(model, seed, snapshot_id), ...]
        for seed in torch_seed_range:
            # 查找该种子的所有快照
            snapshot_files = sorted(Path(ckpt_path).glob(f"{seed}_snapshot_*.pt"))
            if not snapshot_files:
                smart_log(f"  [warning] no snapshots found for seed {seed}, using best model")
                best_pth = Path(ckpt_path) / f"{seed}.pt"
                if best_pth.exists():
                    temp_model = HIST_a11(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2, ew3=ew3).to(device)
                    temp_model.load_state_dict(torch.load(best_pth, map_location=device))
                    temp_model.eval()
                    all_models.append((temp_model, seed, "best"))
            else:
                for snap_pth in snapshot_files:
                    temp_model = HIST_a11(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2, ew3=ew3).to(device)
                    temp_model.load_state_dict(torch.load(snap_pth, map_location=device))
                    temp_model.eval()
                    snap_id = snap_pth.stem.split("_snapshot_")[1]
                    all_models.append((temp_model, seed, snap_id))
                smart_log(f"  seed {seed}: loaded {len(snapshot_files)} snapshots")
        smart_log(f"Total ensemble models: {len(all_models)}")
    else:
        # 标准模式：加载每个种子的最佳模型
        modelS = {iter_seed: HIST_a11(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2, ew3=ew3).to(device) for iter_seed in torch_seed_range}
        for model_num in torch_seed_range:
            modelS[model_num].load_state_dict(torch.load(
                Path(ckpt_path) / f"{model_num}.pt",
                map_location=device))
            modelS[model_num].eval()

    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=data_version, batch_size=1, shuffle=False):
            if use_snapshot:
                # 快照集成：平均所有快照的预测
                pred = None
                for temp_model, seed, snap_id in all_models:
                    p = z_score(model_forward(temp_model, sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
                    if pred is None:
                        pred = p
                    else:
                        pred += p
            else:
                # 标准集成：平均所有种子的预测
                for i, model_num in enumerate(torch_seed_range):
                    if i == 0:
                        pred = z_score(model_forward(modelS[model_num], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
                    else:
                        pred += z_score(model_forward(modelS[model_num], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")
