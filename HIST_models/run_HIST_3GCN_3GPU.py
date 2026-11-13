"""HIST_models：HIST_a11（GCN + GCN + GCN）3GPU并行训练脚本

基于 run_HIST_3GCN.py，使用 torch.multiprocessing 在3张GPU上并行训练3个种子模型。

运行：
    python HIST_models/run_HIST_3GCN_3GPU.py --edge-version-2 2 --edge-version-3 3
    python HIST_models/run_HIST_3GCN_3GPU.py -v2 2 -v3 3 --hidden_size 64 --clip 0.04
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from models import HIST_a11
from data_utils import get_dataloader
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch, seed_torch_all
from utils.ModelUtils import z_score

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)  # [10086, 10090, 10094]

SUPPORTED_HORIZONS = ["5d", "20d"]
DEFAULT_HORIZON = "5d"


# ---------------- 概念 -> 概念矩阵 ----------------

def industry_to_concept(industry: torch.Tensor) -> torch.Tensor:
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
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        mask = torch.isfinite(x) & torch.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.numel() < 2:
            return x.sum() * 0.0
        x = x.clamp(-self.clamp, self.clamp)
        y = y.clamp(-self.clamp, self.clamp)
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        x_var = x_centered.pow(2).mean()
        y_var = y_centered.pow(2).mean()
        covariance = (x_centered * y_centered).mean()
        denominator = (x_var + y_var + (x_mean - y_mean).pow(2)).clamp_min(self.eps)
        ccc = 2.0 * covariance / denominator
        return 1.0 - ccc


# ---------------- 前向 ----------------

def model_forward(_model, _sample, _device, _edge_v2: int, _edge_v3: int):
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
    for _sample in _loader:
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
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)


# ---------------- 单GPU训练进程 ----------------

def train_worker(gpu_rank: int, _seed: int, shared_args_dict: dict):
    """单个GPU上的训练进程。
    
    Args:
        rank: GPU编号 (0, 1, 2)
        seed: 随机种子
        args_dict: 参数字典
    """
    _device = f"cuda:{gpu_rank}"
    torch.cuda.set_device(gpu_rank)
    
    # 设置进程日志标识
    log_prefix = f"[GPU{gpu_rank}|Seed{_seed}] "
    
    # 检查是否已完成
    _ckpt_path = shared_args_dict["ckpt_path"]
    tpmd_pth = Path(_ckpt_path) / f"{_seed}.pt"
    if tpmd_pth.is_file():
        print(f"{log_prefix}已完成，跳过")
        return
    
    print(f"{log_prefix}开始训练，设备: {_device}")
    
    # 设置种子
    seed_all_non_torch(_seed)
    seed_torch_all(_seed)
    
    # 加载数据（每个进程独立加载）
    _data_version = shared_args_dict["data_version"]
    _train_loader = get_dataloader("train", version=_data_version, batch_size=1, shuffle=True, num_workers=2)
    _valid_loader = get_dataloader("valid", version=_data_version, batch_size=1, shuffle=False, num_workers=2)
    
    _d_feat = _train_loader.dataset[0]["x"].shape[1]
    
    # 损失函数
    if shared_args_dict["loss"] == "ICLoss":
        criterion = ICLoss()
    elif shared_args_dict["loss"] == "CCCLoss":
        criterion = CCCLoss()
    else:
        raise ValueError(f"不支持的损失函数: {shared_args_dict['loss']}")
    valid_criterion = ICLoss()
    
    # 模型
    model = HIST_a11(
        d_feat=_d_feat,
        hidden_size=shared_args_dict["hidden_size"],
        clip=shared_args_dict["clip"],
        ew1=shared_args_dict["ew1"],
        ew2=shared_args_dict["ew2"],
        ew3=shared_args_dict["ew3"]
    ).to(_device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=shared_args_dict["lr"], weight_decay=shared_args_dict["weight_decay"])
    
    _edge_v2 = shared_args_dict["edge_version_2"]
    _edge_v3 = shared_args_dict["edge_version_3"]
    
    best_ic, wait = np.inf, 0
    for epoch in range(1, shared_args_dict["epochs"] + 1):
        train_loss = train_one_epoch(model, _train_loader, criterion, optimizer, _device, _edge_v2, _edge_v3)
        ic, icir = evaluate(model, _valid_loader, valid_criterion, _device, _edge_v2, _edge_v3)
        print(f"{log_prefix}epoch {epoch:3d} | train loss {train_loss:.4f} | valid IC {ic:.4f} | ICIR {icir:.4f}")
        
        if ic < best_ic:
            print(f"{log_prefix}saving model {best_ic:.6f} -> {ic:.6f}")
            best_ic, wait = ic, 0
            torch.save(model.state_dict(), tpmd_pth)
        else:
            wait += 1
            if wait >= shared_args_dict["patience"]:
                print(f"{log_prefix}early stopping")
                break
    
    print(f"{log_prefix}训练完成，最佳IC: {best_ic:.6f}")


# ---------------- 参数解析 ----------------

def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a11 3GPU并行训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120])
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14, choices=[2, 3, 7, 9, 10, 12, 14])
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2, choices=[2, 3, 7, 9, 10, 12, 14])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ew3", type=int, default=1, choices=[0, 1])
    parser.add_argument("--label", type=str, default="data/label/stock_ret.pqt")
    parser.add_argument("--horizon", "-hz", type=str, default=DEFAULT_HORIZON, choices=SUPPORTED_HORIZONS)
    parser.add_argument("--loss", type=str, default="ICLoss", choices=["ICLoss", "CCCLoss"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a11"
    
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")
    
    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"
    
    if not args.label.endswith(".pqt"):
        raise ValueError(f"标签文件必须为 .pqt 格式: {args.label}")
    label_name = os.path.splitext(os.path.basename(args.label))[0]
    horizon = args.horizon
    
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v{args.edge_version_2}v{args.edge_version_3}_{label_name}_{horizon}"
    
    ew1 = bool(args.ew1); ew_mark = 'ew1[off]' if not ew1 else ''
    ew2 = bool(args.ew2); ew_mark = ew_mark + 'ew2[off]' if not ew2 else ew_mark
    ew3 = bool(args.ew3); ew_mark = ew_mark + 'ew3[off]' if not ew3 else ew_mark
    loss_mark = f"_{args.loss}"
    
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}.pqt"
    
    make_log_file()
    
    n_gpus = torch.cuda.device_count()
    n_seeds = len(torch_seed_range)
    smart_log(f"可用GPU数量: {n_gpus}, 待训练种子数量: {n_seeds}")
    
    if n_gpus < n_seeds:
        smart_log(f"警告: GPU数量({n_gpus})少于种子数量({n_seeds})，部分种子将排队训练")
    
    smart_log(f"model = HIST_{model_id} | data = {data_version} | hidden_size = {args.hidden_size}")
    smart_log(f"label = {args.label} | horizon = {horizon} | train_loss = {args.loss}")
    
    seed_all_non_torch(root_seed)
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    
    # 构建参数字典（传递给子进程）
    args_dict = {
        "data_version": data_version,
        "ckpt_path": ckpt_path,
        "hidden_size": args.hidden_size,
        "clip": clip,
        "ew1": ew1,
        "ew2": ew2,
        "ew3": ew3,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "edge_version_2": args.edge_version_2,
        "edge_version_3": args.edge_version_3,
        "loss": args.loss,
    }
    
    # 并行训练
    seeds = list(torch_seed_range)
    smart_log(f"启动并行训练，种子: {seeds}")
    
    mp.set_start_method("spawn", force=True)
    processes = []
    
    for rank, seed in enumerate(seeds):
        gpu_id = rank % max(n_gpus, 1)  # 如果GPU不足，循环使用
        p = mp.Process(target=train_worker, args=(gpu_id, seed, args_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    smart_log("所有种子训练完成")
    
    # 推理阶段（使用第一个GPU）
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 获取特征维度
    train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=False)
    d_feat = train_loader.dataset[0]["x"].shape[1]
    del train_loader
    
    edge_v2 = args.edge_version_2
    edge_v3 = args.edge_version_3
    
    smart_log("加载模型进行推理...")
    modelS = {}
    for seed in torch_seed_range:
        modelS[seed] = HIST_a11(d_feat=d_feat, hidden_size=args.hidden_size, clip=clip, ew1=ew1, ew2=ew2, ew3=ew3).to(device)
        modelS[seed].load_state_dict(torch.load(Path(ckpt_path) / f"{seed}.pt", map_location=device))
        modelS[seed].eval()
    
    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=data_version, batch_size=1, shuffle=False):
            for i, seed in enumerate(torch_seed_range):
                if i == 0:
                    pred = z_score(model_forward(modelS[seed], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
                else:
                    pred += z_score(model_forward(modelS[seed], sample, device, edge_v2, edge_v3).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")