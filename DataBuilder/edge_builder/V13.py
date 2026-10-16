"""GCN_V13 step0：基于每日风格因子截面计算股票间余弦相似度边

规则：
- 对每个目标交易日 t，取 **t-1 日** 的风格因子截面数据（避免前视）
- 每只股票的 11 维风格因子向量视为样本，计算股票两两余弦相似度
- 有效因子数 < MIN_OBS(8) 的股票排除；剩余 NaN 填 0
- 取下半三角（不含对角线），按余弦相似度数值取前 5%（固定比例）作为边
- 逐日保存 data/edges/edges_v13/{date_int}.npz：edge_index(2, E) int32 + codes

风格因子列表（来自 stock_risk.pqt）：
beta, bp, ey, growth, leverage, liquidity, lncap, momentum, nls, rv, short_rev

运行：python DataBuilder/edge_builder/V13.py [--workers 32]
纯 CPU + joblib.Parallel 并行
"""
import os
from time import time
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

MIN_OBS  = 8        # 11 个因子中至少 8 个有效，不足则排除该股票
EDGE_PCT = 0.05     # 保留余弦相似度前 5% 的配对作为边
START    = "2010-01-01"  # 起始日期
SAVE_DIR = "data/edges/edges_v13"

FACTOR_COLS = [
    "beta", "bp", "ey", "growth", "leverage", "liquidity",
    "lncap", "momentum", "nls", "rv", "short_rev",
]


def day_int(_d: pd.Timestamp) -> int:
    return _d.year * 10000 + _d.month * 100 + _d.day


def build_edges_for_day(factor_array):
    """纯 numpy 实现：计算因子余弦相似度并取前 5% 边
    factor_np: (N_all, 11) float32，当日（t-1）的风格因子截面
    返回 (edge_index(2,E) int32, rows_keep)，位置空间为当日保留股票；无效时返回 None
    """
    obs = np.isfinite(factor_array).sum(axis=1)              # 每只股票有效因子数
    rows_keep = np.where(obs >= MIN_OBS)[0]
    if len(rows_keep) < 2:
        return None

    X = factor_array[rows_keep]                              # (n, 11)
    X = np.where(np.isfinite(X), X, 0.0)                  # 剩余 NaN 填 0

    # 计算余弦相似度：直接对原始向量归一化（不做中心化）
    norm = np.linalg.norm(X, axis=1)                        # (n,)
    norm = np.maximum(norm, 1e-12)                          # 避免除零
    Xn = X / norm[:, None]                                  # 归一化
    cos_sim = Xn @ Xn.T                                     # (n, n) 余弦相似度

    n = cos_sim.shape[0]
    rows, cols = np.tril_indices(n, k=-1)                   # 下半三角（不含对角线）
    vals = cos_sim[rows, cols]
    vals = np.nan_to_num(vals, nan=-1.0)                    # NaN 相似度（零向量）不参与选边

    k = max(1, int(len(vals) * EDGE_PCT))
    # argpartition 是 O(n)，比完整排序快
    top_idx = np.argpartition(vals, -k)[-k:]
    # 按余弦相似度降序排列（可选，保持输出一致性）
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

    edge_index = np.stack([rows[top_idx], cols[top_idx]], axis=0).astype(np.int32)
    return edge_index, rows_keep


def process_one_day(_d, _, factor_array, codes):
    """供 joblib delayed 调用：计算一天的边并保存"""
    out = build_edges_for_day(factor_array)
    if out is None:
        return 0
    edge_index, keep_rows = out
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"{day_int(_d)}.npz"),
        edge_index=edge_index,
        codes=codes[keep_rows],
    )
    return edge_index.shape[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32, help="并行进程数")
    args = parser.parse_args()

    t0 = time()

    # 1. 读取风格因子数据，unstack 成 (T, N, F)
    print("读取风格因子数据 ...")
    risk = pd.read_parquet(path="data/raw_data/stock_risk.pqt", columns=FACTOR_COLS)

    print("构建按日期索引的因子截面 ...")
    factor_mat_dict = {}
    for f in FACTOR_COLS:
        factor_mat_dict[f] = risk[f].unstack()            # (T, N)
    factor_tensor = np.stack(
        [factor_mat_dict[f].to_numpy(dtype=np.float32) for f in FACTOR_COLS], axis=-1
    )                                                     # (T, N, F)
    risk_days = factor_mat_dict[FACTOR_COLS[0]].index.sort_values()
    codes_all = factor_mat_dict[FACTOR_COLS[0]].columns.to_numpy()
    pos_map = {d: i for i, d in enumerate(risk_days)}     # 因子日 -> 位置
    print(f"因子数据: {len(risk_days)} 天, {len(codes_all)} 只股票, tensor shape: {factor_tensor.shape}")
    del risk, factor_mat_dict                             # 释放内存

    # 2. 目标日 = stock_risk 中 >= START 的全部日期
    target_days = risk_days[risk_days >= pd.Timestamp(START)]
    print(f"目标交易日: {len(target_days)} 天 ({target_days[0].date()} ~ {target_days[-1].date()})")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 3. 准备任务列表：每个目标日 t 使用 t-1 日的因子截面
    tasks = []
    skipped_prep = 0
    for d in target_days:
        prev_day_idx = risk_days.searchsorted(d) - 1
        if prev_day_idx < 0:
            skipped_prep += 1
            continue
        prev_day = risk_days[prev_day_idx]
        if prev_day >= d:
            prev_day_idx -= 1
            if prev_day_idx < 0:
                skipped_prep += 1
                continue
            prev_day = risk_days[prev_day_idx]
        factor_np = factor_tensor[pos_map[prev_day]]
        tasks.append((d, prev_day, factor_np))

    if skipped_prep > 0:
        print(f"[准备阶段跳过 {skipped_prep} 天（无 t-1 日因子数据）]")

    # 4. joblib 并行计算
    print(f"使用 {args.workers} 个进程并行计算 {len(tasks)} 天 ...")
    edge_counts = Parallel(n_jobs=args.workers, verbose=10)(
        delayed(process_one_day)(d, prev_day, factor_np, codes_all)
        for d, prev_day, factor_np in tasks
    )
    n_edges_total = sum(edge_counts)
    skipped = sum(1 for c in edge_counts if c == 0)

    meta = {
        "min_obs": MIN_OBS, "edge_pct": EDGE_PCT,
        "factor_cols": FACTOR_COLS,
        "desc": "daily cross-sectional factor cosine similarity top 5% edges",
        "days": [day_int(d) for d in target_days],
    }
    pd.Series(meta).to_json(os.path.join(SAVE_DIR, "meta.json"))

    print(f"\n完成: {len(tasks) - skipped} 天, 总边数 {n_edges_total:,} (单向), "
          f"跳过 {skipped + skipped_prep} 天, 耗时 {time()-t0:.0f}s")
    print(f"输出目录: {SAVE_DIR}")