import os
import argparse
import numpy as np
import polars as pl
import pandas as pd
import iisignature
from typing import Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from joblib import Parallel, delayed

from gen_path_sig_alpha import get_signature_columns

#####################################################################
########################### Configuration ###########################
#####################################################################

# 固定参数
ordo = 3
features = ["open_price", "high_price", "low_price", "close_price", "turnover_volume", "turnover_value"]
sig_features, _ = get_signature_columns(features, ordo)
N_CORES = 32
# 流水线实际使用的列（显式 select，避免多文件合并时因额外列导致类型冲突）
REQUIRED_COLS = ["security_code", "trading_day", "start_time"] + features


@dataclass(frozen=True)
class PipelineConfig:
    """
    流水线配置（可被 pickle 序列化，用于多进程传递）
    
    使用 dataclass 封装所有配置，避免依赖全局变量。
    在 joblib loky 多进程环境下，全局变量不会自动传递到子进程，
    通过显式传递配置对象确保子进程能正确获取所有参数。
    """
    source_data_dir: str
    target_data_dir: str
    freq_count: int
    source_freq_count: int
    source_cutoff_time: int
    resample_bin_minutes: Optional[int]
    avail_dates: Tuple[int, ...]  # 使用 tuple 以便 pickle
    look_backs: Tuple[int, ...]   # 回溯天数列表


def setup_config() -> PipelineConfig:
    """
    解析命令行参数并构建流水线配置
    
    Returns:
        config: PipelineConfig 配置对象（包含所有配置，含 look_backs）
    """
    parser = argparse.ArgumentParser(description="Path Signature Feature Builder")
    parser.add_argument("--freq", type=str, required=True,
                        choices=["one_minute", "five_minute", "half_hour"],
                        help="数据频率: one_minute / five_minute / half_hour")
    parser.add_argument("--spans", type=int, nargs="+", required=True,
                        help="回溯天数列表，如: 5 10 20")
    args = parser.parse_args()

    freq = args.freq
    if freq in ['half_hour']:
        # 降频频率以 five_minute 数据为源
        source_data_dir = os.path.join("/data", "cephfs", "minute", "five_minute")
    elif freq in ['one_minute', 'five_minute']:
        source_data_dir = os.path.join("/data", "cephfs", "minute", freq)
    else:
        raise NotImplementedError(
            f"Freq {freq} not supported yet. Only 'one_minute', 'five_minute', 'half_hour' are supported."
        )

    # 目标频率下每只股票每个交易日的 bar 数量
    # 使用左闭区间（floor）分箱：9:30~9:55 → 09:30 窗口，13:00~13:25 → 13:00 窗口，午休不产生空 bar
    freq_count = {"one_minute": 237, "five_minute": 48, "half_hour": 8}[freq]
    # 源数据频率下每只股票每个交易日的 bar 数量（降频前用于完整性过滤）
    # half_hour 以 five_minute 为源，截断逻辑统一由 BAR_LAYOUT 决定
    source_layout = {
        "one_minute": {"cutoff": 1456, "max_bars": 237}, # one_minute 数据原始有 237/240 根两种：1457/1458/1459 三根（14:57~14:59，盘尾集合竞价时段）仅部分股票存在，统一砍掉后所有股票对齐为 237 根
        "five_minute": {"cutoff": 1455, "max_bars": 48},
    }["five_minute" if freq == "half_hour" else freq]

    source_freq_count = source_layout["max_bars"]
    source_cutoff_time = source_layout["cutoff"]
    # 降频窗口（分钟）：None 表示无需降频
    resample_bin_minutes = {"half_hour": 30}.get(freq)

    # 预制日期列表（转为 tuple 以便 pickle）
    avail_dates = tuple(sorted(
        int(f[:8]) for f in os.listdir(source_data_dir)
        if len(f) >= 8 and f.startswith("20") and f[:8].isdigit()
    ))

    target_data_dir = f"data/feature/CAT_{freq}_LAG{'_'.join([str(_span) for _span in args.spans])}"
    os.makedirs(target_data_dir, exist_ok=True)

    config = PipelineConfig(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        freq_count=freq_count,
        source_freq_count=source_freq_count,
        source_cutoff_time=source_cutoff_time,
        resample_bin_minutes=resample_bin_minutes,
        avail_dates=avail_dates,
        look_backs=tuple(args.spans),
    )

    return config

#####################################################################
########################### Configuration ###########################
#####################################################################


def resample_stock_data_polars(df: pl.DataFrame, bin_minutes: int = 30) -> pl.DataFrame:
    """
    将 5 分钟数据降频至 half_hour（Polars 实现）

    使用左闭区间（closed='left' / floor）分箱，每天产生恰好 8 根 bar：
        09:30, 10:00, 10:30, 11:00（上午 4 根，覆盖 09:30~11:30）
        13:00, 13:30, 14:00, 14:30（下午 4 根，覆盖 13:00~15:00）

    分箱规则：
    - 09:30~09:55 的 6 根 5 分钟 bar → 09:30 窗口
    - 10:00~10:25 的 6 根 5 分钟 bar → 10:00 窗口
    - ...以此类推，午休（11:30~13:00）无数据不产生空 bar

    实现原理：
    - closed='left' 语义：时间戳为 t 的 bar 归属到起始于 floor(t / bin) * bin 的窗口
      （例如 09:30 bar 归入 09:30 窗口，13:00 bar 归入 13:00 窗口）
    - 直接用整数分钟运算计算窗口标签并分组聚合，避免 datetime resample 的开销，
      也不依赖 group_by_dynamic 的窗口对齐行为，结果确定且高效

    OHLCV 聚合规则：
        open=first, high=max, low=min, close=last, turnover_volume=sum, turnover_value=sum
    """
    # 当日内的分钟数（start_time 为 HHMM 格式整数）
    minutes = (pl.col("start_time") // 100) * 60 + (pl.col("start_time") % 100)
    # closed='left'：bar 归属到起始于 floor(t / bin) * bin 的窗口
    bin_label_minutes = (minutes // bin_minutes) * bin_minutes

    resampled = (
        df.with_columns(bin_label_minutes.alias("_bin_minutes"))
        # 关键：在每个窗口内按 start_time 升序排序，确保 first()/last() 取到时间上最早/最新的 bar
        .sort(["security_code", "trading_day", "_bin_minutes", "start_time"])
        .group_by(["security_code", "trading_day", "_bin_minutes"], maintain_order=True)
        .agg([
            pl.col("open_price").first(),
            pl.col("high_price").max(),
            pl.col("low_price").min(),
            pl.col("close_price").last(),
            pl.col("turnover_volume").sum(),
            pl.col("turnover_value").sum(),
        ])
        .sort(["security_code", "trading_day", "_bin_minutes"])
        # 将窗口标签还原为 HHMM 格式的 start_time，保持与原始数据相同的结构
        .with_columns(
            ((pl.col("_bin_minutes") // 60) * 100 + (pl.col("_bin_minutes") % 60))
            .cast(pl.Int32)
            .alias("start_time")
        )
        .drop("_bin_minutes")
    )
    return resampled


def build_daily_single_lag_feature(
    date: int,
    look_back: int,
    config: PipelineConfig,
    z_score: bool = True
) -> Optional[pd.DataFrame]:
    """
    构建单个回溯天数的路径签名特征
    
    Args:
        date: 目标日期 (YYYYMMDD 格式)
        look_back: 回溯天数
        config: 流水线配置对象
        z_score: 是否进行 z-score 标准化
        
    Returns:
        包含路径签名特征的 DataFrame，如果数据不足或不匹配则返回 None
    """
    # 使用 tuple 的 avail_dates 进行过滤
    look_back_dates = [d for d in config.avail_dates if d <= date]
    if len(look_back_dates) < look_back:
        return None
    look_back_dates = look_back_dates[-look_back:]

    # 构建文件路径列表
    file_paths = [os.path.join(config.source_data_dir, f"{_date}.parquet") for _date in look_back_dates]
    # 逐文件扫描并只保留必需列后纵向拼接，避免一次性扫描多文件时
    # 因文件间额外列/类型差异导致的 SchemaMismatch，以及空分区的索引越界
    lf = pl.concat(
        [pl.scan_parquet(fp).select(REQUIRED_COLS) for fp in file_paths],
        how="vertical",
    )

    # 将 date 转换为 Date 类型用于过滤（parquet 中 trading_day 是 Date 类型）
    date_filter = pl.lit(pd.to_datetime(date, format="%Y%m%d").date()).cast(pl.Date)

    # 惰性执行：排序、类型转换、前向填充、日期过滤、股票过滤
    df = (
        lf
        .sort(["security_code", "trading_day", "start_time"])
        # 将 Decimal 类型转换为 Float64（polars 的 Decimal 类型运算较慢）
        .with_columns([
            pl.col(col).cast(pl.Float64) for col in features
        ])
        # 按股票代码分组进行前向填充
        .with_columns([
            pl.col(col).fill_null(strategy="forward").over("security_code")
            for col in features
        ])
        # 按股票代码分组进行反向填充（处理开头的 null）
        .with_columns([
            pl.col(col).fill_null(strategy="backward").over("security_code")
            for col in features
        ])
        # 删除填充后仍为 null 的行（理论上不应再有 null，但作为安全网保留）
        .drop_nulls(subset=features)
        # 过滤日期 <= 当前处理日期（与原 pandas 代码一致）
        .filter(pl.col("trading_day") <= date_filter)
        # 截断尾部不完整的 bar（如 one_minute 的 1457~1459 盘尾集合竞价），
        # 使所有股票在相同时间范围内对齐，确保后续完整性过滤保留全部股票
        .filter(pl.col("start_time") <= config.source_cutoff_time)
        # 计算每只股票的记录数（按源频率统计）
        .with_columns(pl.len().over("security_code").alias("qualify"))
        # 过滤满足完整数据要求的股票
        .filter(pl.col("qualify") == config.source_freq_count * look_back)
        .collect()  # 默认引擎更稳定
        # .collect(engine="streaming")  # streaming 引擎更适合大数据量
    )

    # 降频频率（如 half_hour）：先对 5 分钟数据降频
    # 注意：turnover_value / turnover_volume 在降频时按原始值求和，之后再取对数
    if config.resample_bin_minutes is not None:
        df = resample_stock_data_polars(df, config.resample_bin_minutes)

    # 对成交额取对数
    df = df.with_columns(pl.col("turnover_value").log1p())

    # 空数据检查
    if len(df) == 0:
        tqdm.write(f"No valid stocks at {date} with look_back={look_back}, skipping")
        return None

    # 获取股票列表（保持顺序）
    stock_list = df["security_code"].unique(maintain_order=True).to_numpy()
    n_stocks = len(stock_list)
    total_rows_per_stock = config.freq_count * look_back
    n_features = len(features)

    feature_array = df.select(features).to_numpy() # 数据已按 security_code, trading_day, start_time 排序，可以直接 reshape

    # 防御性检查：确保行数符合预期
    expected_rows = n_stocks * total_rows_per_stock
    if feature_array.shape[0] != expected_rows:
        tqdm.write(f"Row mismatch at {date}: expected {expected_rows}, got {feature_array.shape[0]}, skipping")
        return None

    batch_data = feature_array.reshape(n_stocks, total_rows_per_stock, n_features)

    if z_score:
        mean = np.nanmean(batch_data, axis=1, keepdims=True)
        std = np.nanstd(batch_data, axis=1, ddof=1, keepdims=True)
        batch_data = (batch_data - mean) / (std + 1e-8)
        batch_data = np.clip(batch_data, -5, 5)

    # 计算 path signature
    sig_data = pd.DataFrame(
        iisignature.sig(batch_data, ordo),
        index=pd.MultiIndex.from_product([[pd.to_datetime(date, format="%Y%m%d")], stock_list], names=['DataDate', 'SecuCode']),
        columns=[f"{sig_feature}_LAG{look_back}" for sig_feature in sig_features]
    )

    return sig_data


def build_daily_feature(
    date: int,
    config: PipelineConfig,
    z_score: bool = True
) -> None:
    """
    构建指定日期的所有路径签名特征并保存为 parquet
    """
    results = [
        build_daily_single_lag_feature(date, look_back, config, z_score)
        for look_back in config.look_backs
    ]
    
    # 检查是否有 None 结果
    if any(r is None for r in results):
        failed_lags = [lb for lb, r in zip(config.look_backs, results) if r is None]
        raise ValueError(f"Incomplete data at {date}, failed lags: {failed_lags}")

    daily_sig_data = pd.concat(results, axis=1, join="inner")
    daily_sig_data.to_parquet(f"{config.target_data_dir}/{date}.pqt")


def update_daily_feature(
    config: PipelineConfig,
    z_score: bool = True,
    n_jobs: int = N_CORES
) -> None:
    """
    对比 target_data_dir 中已存在的文件与 avail_dates，用 joblib 多进程补齐缺失日期
    
    Args:
        config: 流水线配置对象（可被 pickle 序列化传递到子进程，含 look_backs）
        z_score: 是否进行 z-score 标准化
        n_jobs: 并行进程数
    """
    os.makedirs(config.target_data_dir, exist_ok=True)

    # 已构建的日期集合（从文件名解析，如 20260101.pqt -> 20260101）
    existing_dates = {
        int(f[:-4]) for f in os.listdir(config.target_data_dir)
        if f.endswith(".pqt") and f[:-4].isdigit()
    }

    # 需要构建的日期：存在于源数据中但尚未构建
    pending_dates = [d for d in config.avail_dates if d not in existing_dates]

    print(f"共 {len(config.avail_dates)} 个可用日期，已构建 {len(existing_dates)} 个，待构建 {len(pending_dates)} 个")
    if not pending_dates:
        return

    def _build_one_date(date: int) -> str:
        """
        joblib loky 子进程入口
        
        注意：config 通过闭包捕获，由 joblib 的 cloudpickle 序列化传递到子进程
        """
        try:
            build_daily_feature(date, config, z_score)
            return f"[OK] {date}"
        except Exception as e:  # pylint:disable=W0703
            return f"[FAIL] {date}: {e}"

    print("Dispatching")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_build_one_date)(_day) for _day in pending_dates)

    n_ok = sum(r.startswith("[OK]") for r in results)
    print(f"完成 {n_ok}/{len(results)}，失败 {len(results) - n_ok} 个")
    for r in results:
        if not r.startswith("[OK]"):
            print(r)


if __name__ == "__main__":
    global_config = setup_config()

    update_daily_feature(
        config=global_config,
        z_score=True,
        n_jobs=N_CORES
    )
