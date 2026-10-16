import os
from typing import List

import pandas as pd
import numpy as np
import iisignature
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit
# from factory.back_testV2 import batch_test

def get_signature_columns(
    seq_names: List[str], 
    depth: int, 
    prefix: str = ''
):
    """
    生成路径签名的列名。
    """
    columns = []
    chosen = []
    
    # 从 1 阶遍历到最高阶 depth
    for order in range(1, depth + 1):
        # itertools.product 生成可重复的排列组合
        # repeat=order 表示从 seq_names 中取 order 个元素组合
        for combo in itertools.product(seq_names, repeat=order):
            # 将组合拼接成字符串，例如 ('X', 'Y') 变成 'XY'
            term = "@".join(combo)+prefix
            columns.append(term)
            if len(set(combo)) > 1:
                chosen.append(term)
            
    return columns, chosen

#####################################################################
########################### Configuration ###########################
#####################################################################
ordo = 3
freq = 'five_minute'
if freq in ['half_hour', 'one_hour']:
    data_root_dir = os.path.join("/data", "cephfs", "minute", "five_minute")
elif freq in ['one_minute', 'five_minute']:
    data_root_dir = os.path.join("/data", "cephfs", "minute", freq)
else:
    raise NotImplementedError(f"Freq {freq} not supported yet")

features = ["open_price", "high_price", "low_price", "close_price", "turnover_volume", "turnover_value"]
iisig_features, _ = get_signature_columns(features, ordo)
#####################################################################
########################### Configuration ###########################
#####################################################################

#####################################################################
######################### Prepare Chen Data #########################
#####################################################################



def resample_stock_data(data, target_freq):
    """将5分钟数据降频至半小时或一小时

    Parameters:
    data (pd.DataFrame): 包含 MultiIndex ["security_code", "start_time"]
    的Dataframe
    target_freq (str): 'half_hour' 或 'one_hour'
    """
    # 1. 映射频率字串到 pandas 标准频率
    freq_map = {"half_hour": "30min", "one_hour": "1H"}
    p_freq = freq_map[target_freq]

    # 2. 重置索引，方便按股票和时间处理
    df_reset = data.reset_index()

    # 3. 将时间列显式转换为 datetime 类型（resample 必需）
    df_reset['start_time'] = pd.to_datetime(df_reset['trading_day']) + pd.to_timedelta(df_reset['start_time'] // 100, unit='h') + pd.to_timedelta(df_reset['start_time'] % 100, unit='m')

    # 4. 设置时间为单索引，以便进行 resample
    df_reset = df_reset.set_index("start_time")

    # 5. 定义不同字段的聚合规则（开高低收成交量的标准标准）
    agg_rules = {
        "high_price": "max",
        "low_price": "min",
        "close_price": "last",
        "turnover_value": "sum",
    }

    # 6. 按股票代码分组，并在组内按目标频率降频
    # label='right' 或 'left' 决定K线时间戳显示周期开始还是结束， closed='right' 符合国内标准
    resampled = (
        df_reset.groupby("security_code")
        .resample(p_freq, closed="right", label="right")
        .agg(agg_rules)
    )

    # 7. 恢复原始的 MultiIndex 结构和顺序
    resampled = resampled.reorder_levels(["security_code", "start_time"])

    return resampled.reset_index(drop=False)

def cal_iisig(df: pd.DataFrame) -> pd.Series:
    return pd.Series(iisignature.sig(df.values, ordo).reshape(-1), index=iisig_features)

def process_single_file(file):

    file_path = os.path.join(data_root_dir, file)
    
    data = pd.read_parquet(file_path, columns=["security_code", "trading_day", "start_time"] + features)
    if freq in ['half_hour', 'one_hour']:
        data = resample_stock_data(data, freq)
    else:
        data = data.drop(columns=['trading_day'])
    data = data.set_index(["security_code", "start_time"]).astype(float)
    data['turnover_value'] = data['turnover_value'].apply(np.log)
    data.dropna(inplace=True)

    # ===============================
    # 构造专门用于 agg 的特征矩阵
    # ===============================
    agg_data = data.copy()
    # 批量生成平方项列
    sq_features = [f"{f}_sq" for f in features]
    agg_data[sq_features] = data[features] ** 2

    grouped = data.groupby(level='security_code')
    grouped_agg = agg_data.groupby(level='security_code')

    # 1. 签名计算 (仍然用原始 data 算)
    data_iisig = grouped.apply(cal_iisig)

    # 2. 构造 agg 字典
    # 原始特征需要 'sum' 和 'last'，平方特征只需要 'sum'
    agg_funcs = {f: ['sum', 'first', 'last'] for f in features}
    agg_funcs.update({f: ['sum'] for f in sq_features})

    # 一次性算出所有 META 数据 (全 C 语言底层，极速)
    meta_df = grouped_agg.agg(agg_funcs)

    # 展平列名并美化 (例如: sum_high_price, last_high_price, sum2_high_price)
    new_cols = []
    for col, func in meta_df.columns:
        if '_sq' in col:
            new_cols.append(f"sum2_{col.replace('_sq', '')}")
        else:
            new_cols.append(f"{func}_{col}")
    meta_df.columns = new_cols

    # 补充 N
    meta_df['N'] = grouped.size()

    file_key = int(file.split('.')[0])
    return file_key, data_iisig, meta_df

# 2. 主执行入口
def build_daily_factor():
    files = sorted(os.listdir(data_root_dir))
    
    # 确保保存目录存在
    os.makedirs(F'factor_feature/HF_{freq}', exist_ok=True)
    
    # n_jobs 设置：如果是纯计算，推荐设置为你 CPU 的物理核心数，例如 n_jobs=-1 拉满
    parallel_output = Parallel(n_jobs=8, batch_size='auto')(
        delayed(process_single_file)(file) 
        for file in tqdm(files, desc="Processing files")
    )
    
    # 分别创建两个字典来收集 签名数据 和 META 数据
    results_sig = {}
    results_meta = {}
    
    for output in parallel_output:
        if output is None:  # 防御性编程：万一某个文件全空，函数返回了 None
            continue
            
        key, sig_df, meta_df = output
        results_sig[key] = sig_df
        results_meta[key] = meta_df

    if results_sig and results_meta:
        # 1. 拼接并保存原始签名数据 (Raw Signature)
        print("\n=== Merging Signature Data ===")
        results_df = pd.concat(results_sig).dropna()
        results_df.index.names = ['DataDate', 'SecuCode']
        print(results_df.info())
        # reset_index 展平包含 file_key 和 security_code 的 MultiIndex
        results_df.reset_index().to_feather(F'factor_feature/HF_{freq}/HF_Raw.ftr') 
        
        # 2. 拼接并保存 META 数据 (用于流式合并的统计量)
        print("\n=== Merging META Data ===")
        # META 数据通常不会有 NaN，所以不需要 dropna
        meta_df_all = pd.concat(results_meta)
        meta_df_all.index.names = ['DataDate', 'SecuCode']
        print(meta_df_all.info())
        meta_df_all.reset_index().to_feather(F'factor_feature/HF_{freq}/HF_meta.ftr')
    else:
        print("No data processed.")

#####################################################################
######################### Prepare Chen Data #########################
#####################################################################

#####################################################################
########################### Chen Multiply ###########################
#####################################################################

# ==========================================
# 1. 核心数学算子 (Numba @njit 编译为 C 语言)
# ==========================================
# 优化点：避免使用 np.tensordot(axes=0) 因为 Numba 不支持，替换为等价的 outer+reshape
# 优化点：提前创建 res 空数组赋值，避免 concatenate 内存分配碎片

# ==========================================
# 1. 核心数学算子 (修复 Numba 连续内存报错)
# ==========================================
@njit
def calc_chen_product_numba(s_a, s_b, d):
    res = np.empty(d + d**2 + d**3, dtype=np.float64)
    
    a1 = s_a[:d]
    b1 = s_b[:d]
    
    # 【修复点】：增加 np.ascontiguousarray 强制内存连续
    a2 = np.ascontiguousarray(s_a[d : d+d**2]).reshape((d, d))
    b2 = np.ascontiguousarray(s_b[d : d+d**2]).reshape((d, d))
    a3 = np.ascontiguousarray(s_a[d+d**2 : d+d**2+d**3]).reshape((d, d, d))
    b3 = np.ascontiguousarray(s_b[d+d**2 : d+d**2+d**3]).reshape((d, d, d))

    res[:d] = a1 + b1
    res[d : d+d**2] = (a2 + b2 + np.outer(a1, b1)).flatten()
    
    # np.outer 返回的必定是连续数组，所以这里的 reshape 是安全的
    t1 = np.outer(a1, b2.flatten()).reshape((d, d, d))
    t2 = np.outer(a2.flatten(), b1).reshape((d, d, d))
    res[d+d**2 :] = (a3 + b3 + t1 + t2).flatten()
    return res

@njit
def calc_gap_signature_numba(delta, d):
    res = np.empty(d + d**2 + d**3, dtype=np.float64)
    res[:d] = delta
    
    g2 = np.outer(delta, delta) / 2.0
    res[d : d+d**2] = g2.flatten()
    
    g3 = np.outer(delta, g2.flatten()).reshape((d, d, d)) / 3.0
    res[d+d**2 :] = g3.flatten()
    return res

@njit
def calc_sig_inverse_numba(s, d):
    res = np.empty(d + d**2 + d**3, dtype=np.float64)
    
    s1 = s[:d]
    
    # 【修复点】：增加 np.ascontiguousarray
    s2 = np.ascontiguousarray(s[d : d+d**2]).reshape((d, d))
    s3 = np.ascontiguousarray(s[d+d**2 : d+d**2+d**3]).reshape((d, d, d))

    inv1 = -s1
    res[:d] = inv1
    
    inv2 = -s2 - np.outer(s1, inv1)
    res[d : d+d**2] = inv2.flatten()
    
    t1 = np.outer(s1, inv2.flatten()).reshape((d, d, d))
    t2 = np.outer(s2.flatten(), inv1).reshape((d, d, d))
    res[d+d**2 :] = (-s3 - t1 - t2).flatten()
    return res

# ==========================================
# 2. 时序内层循环 (Numba @njit 彻底消灭 Python 循环开销)
# ==========================================
@njit
def compute_stock_trajectory(sig_vals, first_vals, last_vals, spans, roll_N_mat, roll_sum_mat, roll_sum2_mat, d):
    T = sig_vals.shape[0]
    sig_dim = d + d**2 + d**3
    num_spans = len(spans)
    
    # 预分配前缀数组 P
    P = np.zeros((T, sig_dim), dtype=np.float64)
    P[0] = sig_vals[0]
    
    # 阶段一：计算前缀积 P (C语言级速度)
    for t in range(1, T):
        gap = first_vals[t] - last_vals[t-1]
        gap_sig = calc_gap_signature_numba(gap, d)
        temp = calc_chen_product_numba(P[t-1], gap_sig, d)
        P[t] = calc_chen_product_numba(temp, sig_vals[t], d)
        
    # 阶段二：计算各个滚动窗口
    out_zscores = np.full((num_spans, T, sig_dim), np.nan, dtype=np.float64)
    
    for w_idx in range(num_spans):
        w = spans[w_idx]
        if T < w: continue
            
        roll_N = roll_N_mat[:, w_idx]
        roll_sum = roll_sum_mat[:, w_idx, :]
        roll_sum2 = roll_sum2_mat[:, w_idx, :]
        
        for t in range(w - 1, T):
            if t - w + 1 == 0:
                window_raw_sig = P[t]
            else:
                L = t - w
                gap_L_next = first_vals[L+1] - last_vals[L]
                gap_sig_L = calc_gap_signature_numba(gap_L_next, d)
                base_to_remove = calc_chen_product_numba(P[L], gap_sig_L, d)
                base_inv = calc_sig_inverse_numba(base_to_remove, d)
                window_raw_sig = calc_chen_product_numba(base_inv, P[t], d)
            
            c_N = roll_N[t]
            mu = roll_sum[t] / c_N
            var = (roll_sum2[t] / c_N) - mu**2
            
            # 手动提取 sigma，规避负数浮点误差
            sigma = np.zeros(d, dtype=np.float64)
            for i in range(d):
                sigma[i] = np.sqrt(max(var[i], 1e-8))
            
            # 构造权重并缩放
            w2 = np.outer(sigma, sigma).flatten()
            w3 = np.outer(sigma, w2).flatten()
            
            weights = np.empty(sig_dim, dtype=np.float64)
            weights[:d] = sigma
            weights[d:d+d**2] = w2
            weights[d+d**2:] = w3
            
            out_zscores[w_idx, t, :] = window_raw_sig / weights
            
    return out_zscores

# ==========================================
# 3. 多进程包裹函数
# ==========================================
def process_single_stock(secu_code, g_sig, g_meta, spans, _features, d):
    """供 Joblib 调用的单支股票处理函数"""
    sig_vals = g_sig.values.astype(np.float64)
    first_vals = g_meta[[f'first_{f}' for f in _features]].values.astype(np.float64)
    last_vals = g_meta[[f'last_{f}' for f in _features]].values.astype(np.float64)
    
    T = len(sig_vals)
    if T == 0: 
        return secu_code, g_sig.index, None
    
    # 提前在 pandas 层面算出 Numba 需要的 rolling 矩阵
    roll_N_mat = np.zeros((T, len(spans)), dtype=np.float64)
    roll_sum_mat = np.zeros((T, len(spans), d), dtype=np.float64)
    roll_sum2_mat = np.zeros((T, len(spans), d), dtype=np.float64)
    
    for idx, w in enumerate(spans):
        roll_N_mat[:, idx] = g_meta['N'].rolling(w).sum().values
        roll_sum_mat[:, idx, :] = g_meta[[f'sum_{f}' for f in _features]].rolling(w).sum().values
        roll_sum2_mat[:, idx, :] = g_meta[[f'sum2_{f}' for f in _features]].rolling(w).sum().values
        
    # 送入 C++ 级黑盒计算
    out_zscores = compute_stock_trajectory(
        sig_vals, first_vals, last_vals, np.array(spans),
        roll_N_mat, roll_sum_mat, roll_sum2_mat, d
    )
    return secu_code, g_sig.index, out_zscores

# ==========================================
# 4. 主干管道 (维持原接口不变)
# ==========================================
def Chen_Roll_join_daily(sig_df: pd.DataFrame, meta_df: pd.DataFrame, spans: List[int], _freq: str, _features: List[str]):
    d = len(_features)
    results = {f'Chen_{0}': sig_df}  
    
    # 1. 准备并行任务
    tasks = []
    # 这里优化：不再使用 meta_df.xs，因为直接 loc 对齐 index 快得多
    for secu_code, g_sig in tqdm(sig_df.groupby(level='SecuCode'), desc="Preparing tasks"):
        # 确保 meta 对应的片段被准确提取
        g_meta = meta_df.loc[g_sig.index]
        tasks.append((secu_code, g_sig, g_meta, spans, _features, d))
        
    # 2. 多核并发计算 (n_jobs=-1 表示使用所有可用 CPU 核心)
    # 第一次运行 Numba 会触发编译，会有 1-2 秒的停顿，后续如飞一般。
    out_results = Parallel(n_jobs=16)(
        delayed(process_single_stock)(*task) 
        for task in tqdm(tasks, desc="Parallel Numba Computing")
    )
    
    # 3. 结果合并重组
    indices_list = []
    rolled_data = {w: [] for w in spans}
    
    for secu_code, idx, out_zscores in out_results:
        if out_zscores is None: continue
        indices_list.append(idx)
        for w_idx, w in enumerate(spans):
            rolled_data[w].append(out_zscores[w_idx])
            
    concat_index = indices_list[0].append(indices_list[1:]) if len(indices_list) > 1 else indices_list[0]
    
    for w in spans:
        mat = np.vstack(rolled_data[w])
        df_w = pd.DataFrame(mat, index=concat_index, columns=sig_df.columns)
        results[f'Chen_{w}'] = df_w.loc[sig_df.index] 

    results_df = pd.concat(results, axis=1, join='inner')
    results_df = results_df.swaplevel(axis=1).sort_index(axis=1)

    save_dir = f'factor_feature/HF_{_freq}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/HF_Zscore_ChenRoll_{"_".join(map(str, spans))}.ftr'
    
    results_df.columns = [f"{col[0]}_{col[1]}" for col in results_df.columns]
    results_df.reset_index().to_feather(save_path)
    
    print(f"✅ Rolling computation completed and saved to {save_path}")
    return results_df

#####################################################################
########################### Chen Multiply ###########################
#####################################################################

if __name__ == "__main__":

    print("Processing Daily sig")
    build_daily_factor()

    print("Loading Daily Sig")
    feature_df = pd.read_feather(f"factor_feature/HF_{freq}/HF_Raw.ftr")
    feature_df = feature_df.set_index(['SecuCode', 'DataDate']).sort_index()

    info_df = pd.read_feather(f"factor_feature/HF_{freq}/HF_meta.ftr")
    info_df = info_df.set_index(['SecuCode', 'DataDate']).sort_index()

    common_index = feature_df.index.intersection(info_df.index)

    feature_df = feature_df.loc[common_index]
    info_df = info_df.loc[common_index]

    print("Chen Rolling")
    print(Chen_Roll_join_daily(feature_df, info_df, [40], freq, features).info())

    