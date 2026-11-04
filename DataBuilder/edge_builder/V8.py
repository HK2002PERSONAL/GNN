"""GCN_V8 step0：滚动 240 日窗口构建股票间价格相关系数边（最低 5%）

与 V7 的唯一区别：
- 选边方向反转：从 **相关系数最高的前 5%** 改为 **相关系数最低的后 5%**
- 其余逻辑与 V7 完全一致

规则：
- 对每个目标交易日 t，取 **截止 t-1** 的最近 240 个交易日 DailyReturn（避免前视）
- 窗口内有效交易日 >= MIN_OBS(200) 的股票保留，其余排除；保留股票的剩余 NaN 填 0
- 用 (1 + DailyReturn).cumprod() 重建价格路径，计算价格路径的 Pearson 相关系数矩阵
- 取下半三角（不含对角线），按相关系数数值取后 5%（最低，固定比例）作为边
- 逐日保存 data/edges/edges_v8/{date_int}.npz：edge_index(2, E) int32 + codes

运行：conda activate testbench_gpu && python DataBuilder/edge_builder/V8.py
"""
import os
from time import time

import numpy as np
import pandas as pd
import torch

WINDOW   = 240      # 滚动窗口长度（交易日）
MIN_OBS  = 200      # 窗口内最少有效观测，不足则排除该股票
EDGE_PCT = 0.05     # 保留相关系数最低 5% 的配对作为边
START    = "2010-01-01"  # 起始日期
SAVE_DIR = "data/edges/edges_v8"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def day_int(_d: pd.Timestamp) -> int:
    return _d.year * 10000 + _d.month * 100 + _d.day


def build_edges_for_day(
    ret_array: np.ndarray,        # (T, N_all) float32，含 NaN
    position: int,                # 目标日 t 在交易日序列中的位置
) -> tuple[np.ndarray, np.ndarray] | None:
    """返回 (edge_index(2,E) int32, codes)，位置空间为当日保留股票

    价格相关系数：先 (1 + r).cumprod() 重建价格路径，再对价格路径求 Pearson 相关
    """
    win = ret_array[position - WINDOW:position]                      # (240, N_all)，截止 t-1
    obs = np.isfinite(win).sum(axis=0)                  # 每只股票有效天数
    columns = np.where(obs >= MIN_OBS)[0]
    if len(columns) < 2:
        return None

    R = win[:, columns]                                    # (240, n)
    R = np.where(np.isfinite(R), R, 0.0)                # 剩余 NaN 填 0（停牌无收益）

    # 价格序列 = (1 + r).cumprod()，从收益率累乘重建价格路径
    P = np.cumprod(1.0 + R, axis=0)                     # (240, n)

    Pt = torch.from_numpy(P).to(DEVICE).float()         # (240, n)
    Pc = Pt - Pt.mean(dim=0, keepdim=True)              # 按股票中心化
    norm = Pc.norm(dim=0)                               # (n,)
    corr = (Pc.T @ Pc) / torch.outer(norm, norm).clamp_min(1e-12)   # (n, n)

    n = corr.shape[0]
    rows, cols_tr = torch.tril_indices(n, n, offset=-1, device=DEVICE)
    vals = corr[rows, cols_tr]
    vals = torch.nan_to_num(vals, nan=1.0)              # NaN 相关（零方差）不参与选边（置 1 使其不会被当作最低）
    k = max(1, int(rows.numel() * EDGE_PCT))
    top_idx = torch.topk(vals, k, largest=False).indices  # 后 5% 最小相关系数

    _edge_index = torch.stack([rows[top_idx], cols_tr[top_idx]], dim=0)  # (2, k)，i > j
    return _edge_index.cpu().numpy().astype(np.int32), columns


if __name__ == "__main__":
    t0 = time()
    print(f"device = {DEVICE}")

    # 1. 读取日收益率，unstack 成 (T, N)
    print("读取 DailyReturn ...")
    ret = pd.read_parquet(
        path="data/raw_data/stock_data.pqt", columns=["DailyReturn"]
    )["DailyReturn"].unstack()                          # (T, N)
    trade_days = ret.index.sort_values()
    ret = ret.loc[trade_days]
    codes_all = ret.columns.to_numpy()
    if codes_all.dtype == object:
        try:
            codes_all = codes_all.astype(np.int64)
        except (TypeError, ValueError):
            codes_all = codes_all.astype(str)

    pos_map = {d: i for i, d in enumerate(trade_days)}  # 目标日 -> 位置

    # 2. 目标日 = stock_data 中 >= START 的全部交易日
    target_days = trade_days[trade_days >= pd.Timestamp(START)]
    print(f"目标交易日: {len(target_days)} 天 ({target_days[0].date()} ~ {target_days[-1].date()})")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 3. 预转 numpy，加速逐日切片
    ret_np = ret.to_numpy(dtype=np.float32)

    n_edges_total, skipped = 0, 0
    for i, d in enumerate(target_days):
        pos = pos_map[d]
        if pos < WINDOW:
            print(f"[skip] {d.date()} 预热不足 {pos} < {WINDOW}")
            skipped += 1
            continue

        out = build_edges_for_day(ret_np, pos)
        if out is None:
            skipped += 1
            continue
        edge_index, cols = out
        np.savez_compressed(
            os.path.join(SAVE_DIR, f"{day_int(d)}.npz"),
            edge_index=edge_index,
            codes=codes_all[cols],
        )
        n_edges_total += edge_index.shape[1]

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(target_days)}] {d.date()} | "
                  f"保留股票 {len(cols)} | 当日边 {edge_index.shape[1]:,} | "
                  f"累计边 {n_edges_total:,} | {time()-t0:.0f}s")

    meta = {
        "window": WINDOW, "min_obs": MIN_OBS, "edge_pct": EDGE_PCT,
        "graph_type": "price_corr_bottom5pct",
        "price_method": "cumprod(1+DailyReturn)",
        "days": [day_int(d) for d in target_days],
    }
    pd.Series(meta).to_json(os.path.join(SAVE_DIR, "meta.json"))

    print(f"\n完成: {len(target_days) - skipped} 天, 总边数 {n_edges_total:,} (单向), "
          f"跳过 {skipped} 天, 耗时 {time()-t0:.0f}s")
    print(f"输出目录: {SAVE_DIR}")