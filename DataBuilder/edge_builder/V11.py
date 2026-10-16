"""GCN_V11 step0：构建行业图 ∩ 收益相关性top5%图 的交集边

规则：
- 行业图：同行业（IndustryCodeNew）内所有股票两两连边
- 收益相关性图：滚动 240 日窗口（截止 t-1），Pearson 相关系数前 5%
- 交集：仅保留同时出现在两个图中的边（即同行业 且 相关性top5%）
- 逐日保存 data/edges/edges_v11/{date_int}.npz：edge_index(2, E) int32 + codes

运行：conda activate testbench_gpu && python DataBuilder/edge_builder/V11.py
"""
import os
from time import time

import numpy as np
import pandas as pd
import torch

WINDOW   = 240      # 滚动窗口长度（交易日）
MIN_OBS  = 200      # 窗口内最少有效观测，不足则排除该股票
EDGE_PCT = 0.05     # 保留相关系数前 5% 的配对作为边
START    = "2010-01-01"  # 起始日期
SAVE_DIR = "data/edges/edges_v11"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def to_code_str(c):
    """将股票代码统一为 6 位字符串"""
    if isinstance(c, (int, np.integer)):
        return str(c).zfill(6)
    return str(c)


def day_int(_d: pd.Timestamp) -> int:
    return _d.year * 10000 + _d.month * 100 + _d.day


def build_corr_topk_pairs(
    ret_array: np.ndarray,        # (T, N_all) float32，含 NaN
    position: int,                # 目标日 t 在交易日序列中的位置
):
    """计算收益相关性 top5% 的配对。

    返回 (sel_rows, sel_cols, keep_cols)：
    - sel_rows, sel_cols: 选中的下三角行/列索引（位置空间）
    - keep_cols: 当日保留股票在原始矩阵中的列索引
    """
    win = ret_array[position - WINDOW:position]                      # (240, N_all)，截止 t-1
    obs = np.isfinite(win).sum(axis=0)                  # 每只股票有效天数
    columns = np.where(obs >= MIN_OBS)[0]
    if len(columns) < 2:
        return None

    X = win[:, columns]                                    # (240, n)
    X = np.where(np.isfinite(X), X, 0.0)                # 剩余 NaN 填 0（停牌无收益）

    Xt = torch.from_numpy(X).to(DEVICE)                 # (240, n)
    Xc = Xt - Xt.mean(dim=0, keepdim=True)              # 按股票中心化
    norm = Xc.norm(dim=0)                               # (n,)
    corr = (Xc.T @ Xc) / torch.outer(norm, norm).clamp_min(1e-12)   # (n, n)

    n = corr.shape[0]
    rows, cols_tr = torch.tril_indices(n, n, offset=-1, device=DEVICE)
    vals = corr[rows, cols_tr]
    vals = torch.nan_to_num(vals, nan=-1.0)             # NaN 相关（零方差）不参与选边
    k = max(1, int(rows.numel() * EDGE_PCT))
    top_idx = torch.topk(vals, k).indices               # 前 5% 最大相关系数

    sel_rows = rows[top_idx].cpu().numpy()
    sel_cols = cols_tr[top_idx].cpu().numpy()

    return sel_rows, sel_cols, columns


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
    codes_all_str = np.array([to_code_str(c) for c in codes_all])

    pos_map = {d: i for i, d in enumerate(trade_days)}  # 目标日 -> 位置

    # 2. 读取行业数据
    print("读取 IndustryCodeNew ...")
    ind_ser = pd.read_parquet(
        path="data/raw_data/stock_data.pqt", columns=["IndustryCodeNew"]
    )["IndustryCodeNew"]

    # 3. 目标日 = stock_data 中 >= START 的全部交易日
    target_days = trade_days[trade_days >= pd.Timestamp(START)]
    print(f"目标交易日: {len(target_days)} 天 ({target_days[0].date()} ~ {target_days[-1].date()})")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 4. 预转 numpy，加速逐日切片
    ret_np = ret.to_numpy(dtype=np.float32)

    n_edges_total, skipped = 0, 0
    for i, d in enumerate(target_days):
        pos = pos_map[d]
        if pos < WINDOW:
            print(f"[skip] {d.date()} 预热不足 {pos} < {WINDOW}")
            skipped += 1
            continue

        # --- 收益相关性 top5% 边 ---
        corr_out = build_corr_topk_pairs(ret_np, pos)
        if corr_out is None:
            skipped += 1
            continue
        sel_rows, sel_cols, keep_cols = corr_out
        codes_keep_str = codes_all_str[keep_cols]       # 当日保留股票代码字符串

        # --- 当日行业映射 ---
        try:
            ind_day = ind_ser.xs(d, level='DataDate')
        except KeyError:
            skipped += 1
            continue

        # code_str -> industry
        ind_map = {}
        for code_val, ind_val in ind_day.items():
            ind_map[to_code_str(code_val)] = ind_val

        # --- 交集：逐条相关性边检查是否同行业 ---
        edge_src, edge_dst = [], []
        for r, c in zip(sel_rows, sel_cols):
            code_r = codes_keep_str[r]
            code_c = codes_keep_str[c]
            ind_r = ind_map.get(code_r)
            ind_c = ind_map.get(code_c)
            if ind_r is not None and ind_c is not None and ind_r == ind_c:
                edge_src.append(int(r))
                edge_dst.append(int(c))

        if len(edge_src) == 0:
            # 交集为空，保存空边文件（后续 step1 会过滤）
            np.savez_compressed(
                os.path.join(SAVE_DIR, f"{day_int(d)}.npz"),
                edge_index=np.empty((2, 0), dtype=np.int32),
                codes=codes_all[keep_cols],
            )
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(target_days)}] {d.date()} | "
                      f"交集边 0 | 累计边 {n_edges_total:,} | {time()-t0:.0f}s")
            continue

        edge_index = np.array([edge_src, edge_dst], dtype=np.int32)

        np.savez_compressed(
            os.path.join(SAVE_DIR, f"{day_int(d)}.npz"),
            edge_index=edge_index,
            codes=codes_all[keep_cols],
        )
        n_edges_total += edge_index.shape[1]

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(target_days)}] {d.date()} | "
                  f"相关性边 {len(sel_rows):,} | "
                  f"交集边 {edge_index.shape[1]:,} | 累计边 {n_edges_total:,} | {time()-t0:.0f}s")

    meta = {
        "window": WINDOW, "min_obs": MIN_OBS, "edge_pct": EDGE_PCT,
        "desc": "intersection of industry graph and return-correlation top5% graph",
        "days": [day_int(d) for d in target_days],
    }
    pd.Series(meta).to_json(os.path.join(SAVE_DIR, "meta.json"))

    print(f"\n完成: {len(target_days) - skipped} 天, 总边数 {n_edges_total:,} (单向), "
          f"跳过 {skipped} 天, 耗时 {time()-t0:.0f}s")
    print(f"输出目录: {SAVE_DIR}")