"""HIST_V_ablation：HIST 消融实验训练脚本（daily batch）

数据由 data_utils.py 提供（--data 可选，复用已构建的截面数据）：
    v0 -> data/hist_daily_v0  （HIST_V0/step1_build_data.py 产出）
    v1 -> data/hist_daily_v1  （HIST_V1/step1_build_data.py 产出）

模型来自本目录 models.py：
    a1: 仅保留 Predefined Concept
    a2: 仅保留 Hidden Concept（forward 仅需 x）
    a3: Hidden Concept 替换为 GAT（横截面信息共享）
    a4: Predefined Concept 替换为 GCN
    a5: GCN + GAT

运行：
    python HIST_V_ablation/run_ablation.py --model a1
    python HIST_V_ablation/run_ablation.py --model a3 --data v1 --hidden_size 64

输出：model_configs/best_hist_<model>_<data>.pt、model_output/hist_test_<model>_<data>.pqt
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

from models import HIST_a0, HIST_a1, HIST_a2, HIST_a3, HIST_a4, HIST_a5
from data_utils import get_dataloader, REGISTERED_DATA_SET
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch, seed_torch_all
from utils.ModelUtils import z_score

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)

# ---------------- 模型注册表 ----------------

MODEL_REGISTRY = {
    "a0": HIST_a0,
    "a1": HIST_a1,
    "a2": HIST_a2,
    "a3": HIST_a3,
    "a4": HIST_a4,
    "a5": HIST_a5,
}

# a2 仅保留 Hidden Concept，forward 只需要 x；其余模型需要概念矩阵与市值
NEEDS_CONCEPT = {"a0": True, "a1": True, "a2": False, "a3": True, "a4": True, "a5": True}


# ---------------- 行业 -> 概念矩阵 ----------------

def industry_to_concept(industry: torch.Tensor) -> torch.Tensor:
    """把每日行业标签 [N]（整数）变形为股票-概念 0/1 矩阵 [N, C]。

    C = 当日出现的行业数。股票 A 属于行业 c 时 [A, c] = 1，否则 0。
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

def model_forward(_model, _sample, _device, _model_id):
    """把单个日截面样本喂给消融模型，返回预测值 [N]。"""
    x = _sample["x"].to(_device)
    if not NEEDS_CONCEPT[_model_id]:          # a2 仅需 x
        return _model(x)
    industry = _sample["industry"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept = industry_to_concept(industry)
    return _model(x, concept, market_value)


# ---------------- 训练 ----------------

# def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _model_id):
#     _model.train()
#     losses = []
#     for _sample in _loader:                  # batch_size=1，每天一个截面
#         y = _sample["y"].to(_device)
#         sample_num = torch.isfinite(y).sum()
#         if sample_num < 10:
#             continue
#         _pred = model_forward(_model, _sample, _device, _model_id)
#         loss = _loss(_pred, y)
#         if loss is None or not torch.isfinite(loss):
#             continue
#         _optimizer.zero_grad()
#         loss.backward()
#         grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
#         if not torch.isfinite(grad_norm):
#             _optimizer.zero_grad()
#             continue
#         _optimizer.step()
#         losses.append(loss.item())
#     return float(np.mean(losses)) if losses else float("nan")

def train_one_epoch(_model, _loader, _loss, _optimizer, _device, _model_id):
    _model.train()
    losses = []
    for _sample in _loader:
        y_cpu = _sample["y"]
        if torch.isfinite(y_cpu).sum() < 10:   # 在 CPU 侧判断，不触发 GPU 同步
            continue
        y = y_cpu.to(_device)
        _pred = model_forward(_model, _sample, _device, _model_id)
        loss = _loss(_pred, y)
        _optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):      # NaN loss → NaN 梯度，这里一并拦住
            _optimizer.zero_grad()
            continue
        _optimizer.step()
        losses.append(loss.detach())           # 不同步，留在 GPU 上
    return float(torch.stack(losses).mean()) if losses else float("nan")


# @torch.no_grad()
# def evaluate(_model, _loader, _loss, _device, _model_id):
#     _model.eval()
#     ics = []
#     for _sample in _loader:
#         y = _sample["y"].to(_device)
#         if torch.isfinite(y).sum() < 5:
#             continue
#         _pred = model_forward(_model, _sample, _device, _model_id)
#         ics.append(_loss(_pred, y).item())
#     ics = np.asarray(ics, dtype=float)
#     return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)  # IC均值, ICIR


@torch.no_grad()
def evaluate(_model, _loader, _loss, _device, _model_id):
    _model.eval()
    ics = []
    for _sample in _loader:
        if torch.isfinite(_sample["y"]).sum() < 5:   # CPU 侧判断
            continue
        y = _sample["y"].to(_device)
        _pred = model_forward(_model, _sample, _device, _model_id)
        ics.append(_loss(_pred, y).detach())
    if not ics:
        return float("nan"), float("nan")
    ics = torch.stack(ics).cpu().numpy()
    return np.nanmean(ics), np.nanmean(ics) / (np.nanstd(ics) + 1e-12)


def parse_args():
    parser = argparse.ArgumentParser(description="HIST 消融实验（daily batch 训练）")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="消融模型编号：a0~a5")
    parser.add_argument("--data", type=str, default="v0",
                        choices=list(REGISTERED_DATA_SET.keys()),
                        help="数据集版本：v0 / v1")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip", "-c", type=float, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = args.model

    clip = args.clip

    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    ckpt_path = f"model_configs/best_hist_{model_id}_{args.data}_{clip_mark}"
    out_path = f"model_outputs/hist_test_{model_id}_{args.data}_{clip_mark}.pqt"

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} | data = {args.data} | hidden_size = {args.hidden_size}")

    seed_all_non_torch(root_seed)

    smart_log("Loading Data...")
    train_loader = get_dataloader("train", version=args.data, batch_size=1, shuffle=True); smart_log("train done")
    valid_loader = get_dataloader("valid", version=args.data, batch_size=1, shuffle=False); smart_log("valid done")

    # 特征维度从训练集首个样本动态获取
    d_feat = train_loader.dataset[0]["x"].shape[1]
    smart_log(f"d_feat = {d_feat}")

    criterion = ICLoss()

    Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    for torch_seed in torch_seed_range:
        temp_pth = Path(ckpt_path) / "tmp.pt"
        tpmd_pth = Path(ckpt_path) / f"{torch_seed}.pt"
        if Path(tpmd_pth).is_file():
            smart_log(f"{torch_seed} already done")
            continue

        smart_log(f"training {torch_seed}")
        seed_torch_all(torch_seed)

        model = MODEL_REGISTRY[model_id](d_feat=d_feat, hidden_size=args.hidden_size, clip=clip).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_ic, wait = np.inf, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, model_id)
            ic, icir = evaluate(model, valid_loader, criterion, device, model_id)
            smart_log(f"epoch {epoch:3d} | train loss {train_loss:.4f} | valid IC {ic:.4f} | ICIR {icir:.4f}")

            if ic < best_ic:
                smart_log(f"saving model {best_ic:.6f} -> {ic:.6f}")
                best_ic, wait = ic, 0
                torch.save(model.state_dict(), temp_pth)
            else:
                wait += 1
                if wait >= args.patience:
                    smart_log("early stopping")
                    break

        torch.save(model.state_dict(), tpmd_pth)

    modelS = {iter_seed: MODEL_REGISTRY[model_id](d_feat=d_feat, hidden_size=args.hidden_size, clip=clip).to(device) for iter_seed in torch_seed_range}
    for model_num in torch_seed_range:
        modelS[model_num].load_state_dict(torch.load( # pylint: disable=used-before-assignment
            Path(ckpt_path) / f"{model_num}.pt",
            map_location=device))
        modelS[model_num].eval()

    rows = []
    with torch.no_grad():
        for sample in get_dataloader("test", version=args.data, batch_size=1, shuffle=False):
            for i, model_num in enumerate(torch_seed_range):
                if i == 0:
                    pred = z_score(model_forward(modelS[model_num], sample, device, model_id).reshape(-1)).cpu().numpy()
                else:
                    pred += z_score(model_forward(modelS[model_num], sample, device, model_id).reshape(-1)).cpu().numpy()
            date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
            rows.append(pd.DataFrame({"date": date, "code": sample["y"].numpy().astype(int), "pred": pred}))
    result = pd.concat(rows)
    result.sort_index().to_parquet(out_path)
    smart_log(f"test 预测已保存 -> {out_path}，共 {len(result)} 行")