import re
import itertools
from typing import Optional, List

import pandas as pd 
import numpy as np
import iisignature
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view


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


def gen_path_sig_alpha(data: pd.DataFrame, lag: int, ordo: Optional[int] = None, z_score: bool = True):
    """
    生成路径签名因子 (保留交错项)
    """
    Factor = []
    
    # 提取特征列，提前判断并约束
    marks = data.columns.to_list()
    num_features = data.shape[1]
    assert 0 < num_features < 4, "num of indicator shouldn't be larger than 3 for now"
    single_signal = (num_features == 1)
    ordo = num_features + 1 if ordo is None else ordo 

    # 提前获取所需特征列并在循环外拿到索引，避免在 DataFrame 上根据字符串切片的开销
    columns, chosen = get_signature_columns(marks + ['T'] if single_signal else marks, ordo, f"_lag{lag}")
    chosen_idx = [columns.index(c) for c in chosen]
    
    # 提前为单信号生成时间特征序列，避免在循环中重复 np.tile 生成
    if single_signal:
        t_arr = np.linspace(0, 1, lag)

    for _, SecuData in tqdm(data.groupby('SecuCode'), position=1, leave=False):
        N = len(SecuData)
        if N < lag: 
            continue
            
        # 1. 提取 NumPy 数组并 ffill。相当于原代码中对每一个列分别 shift 之后再 ffill 的效果
        vals = SecuData[marks].ffill().values
        
        # 2. 生成滑动窗口，直接截取无 NaN 的有效区间，避免无意义的运算
        # window 形状: (N - lag + 1, lag, num_features)
        windows = sliding_window_view(vals, window_shape=lag, axis=0)
        windows = np.swapaxes(windows, 1, 2)
        
        # 3. 沿 lag 维度计算 Z-score (等价于原版的 ts_zscore)
        # 这里使用 nanmean 和 nanstd 以防序列最开头有无法 ffill 的 np.nan 干扰计算
        # 注意: pandas 的 std 默认 ddof=1，需在 numpy 中显式指定以对齐原始逻辑
        if z_score:
            mean = np.nanmean(windows, axis=1, keepdims=True)
            std = np.nanstd(windows, axis=1, ddof=1, keepdims=True)
            windows = (windows - mean) / (std + 1e-8)
        
        # 4. 如果是单信号，拼接时间向量
        if single_signal:
            # 利用广播(broadcast_to)零拷贝生成 T_feat, 形状: (N - lag + 1, lag, 1)
            T_feat = np.broadcast_to(t_arr, (windows.shape[0], lag))[..., np.newaxis]
            windows = np.concatenate([windows, T_feat], axis=-1)
            
        # 5. 调用 iisignature，只计算能凑齐 lag 长度的有效序列，节约算力
        sig_result = iisignature.sig(windows, ordo)
        
        # 6. 按所需的特征索引切片
        sig_chosen = sig_result[:, chosen_idx]
        
        # 有效计算区间对应原序列的 [lag - 1 : ]
        valid_index = SecuData.index[lag - 1:]
        
        # 保存这组的 DataFrame (最后如果内部依然存在极个别 NaN，则在此处统一 dropna)
        Factor.append(pd.DataFrame(sig_chosen, index=valid_index, columns=chosen).dropna())
        
    if not Factor:
        return pd.DataFrame(columns=chosen)
        
    return pd.concat(Factor)


def gen_path_sig_feature(data: pd.DataFrame, lag: int, ordo: Optional[int] = None, z_score: bool = True):
    """
    生成路径签名特征 (保留所有签名项)
    """
    Factor = []
    
    # 提取特征列，提前判断并约束
    marks = data.columns.to_list()
    num_features = data.shape[1]
    # assert 0 < num_features < 5, "num of indicator shouldn't be larger than 3 for now"
    single_signal = (num_features == 1)
    ordo = num_features + 1 if ordo is None else ordo 

    # 获取所有的 signature columns（忽略第二个返回值 chosen）
    columns, _ = get_signature_columns(marks + ['T'] if single_signal else marks, ordo, f"_lag{lag}")
    
    # 提前为单信号生成时间特征序列，避免在循环中重复 np.tile 生成
    if single_signal:
        t_arr = np.linspace(0, 1, lag)

    for _, SecuData in tqdm(data.groupby('SecuCode'), position=1, leave=False):
        N = len(SecuData)
        if N < lag: 
            continue
            
        # 1. 提取 NumPy 数组并 ffill。
        vals = SecuData[marks].ffill().values
        
        # 2. 生成滑动窗口
        # window 形状: (N - lag + 1, lag, num_features)
        windows = sliding_window_view(vals, window_shape=lag, axis=0)
        windows = np.swapaxes(windows, 1, 2)
        
        # 3. 沿 lag 维度计算 Z-score
        if z_score:
            mean = np.nanmean(windows, axis=1, keepdims=True)
            std = np.nanstd(windows, axis=1, ddof=1, keepdims=True)
            windows = (windows - mean) / (std + 1e-8)
        
        # 4. 如果是单信号，拼接时间向量
        if single_signal:
            # 利用广播(broadcast_to)零拷贝生成 T_feat, 形状: (N - lag + 1, lag, 1)
            T_feat = np.broadcast_to(t_arr, (windows.shape[0], lag))[..., np.newaxis]
            windows = np.concatenate([windows, T_feat], axis=-1)
            
        # 5. 调用 iisignature 计算全部签名项
        sig_result = iisignature.sig(windows, ordo)
        
        # 有效计算区间对应原序列的 [lag - 1 : ]
        valid_index = SecuData.index[lag - 1:]
        
        # 6. 直接使用全部 sig_result 构建 DataFrame
        Factor.append(pd.DataFrame(sig_result, index=valid_index, columns=columns).dropna())
        
    if not Factor:
        return pd.DataFrame(columns=columns)
        
    return pd.concat(Factor)



def get_log_signature_columns(
    seq_names: List[str], 
    depth: int, 
    prefix: str = ''
):
    """
    生成对数签名的列名。
    利用 iisignature 自带的 basis 输出，并将数字索引替换为实际的变量名。
    例如：'1' -> 'X', '[1,2]' -> '[X,Y]'
    """
    channels = len(seq_names)
    
    # 生成对数签名的结构准备对象
    s = iisignature.prepare(channels, depth)
    
    # basis() 返回的是类似于 ('1', '2', '[1,2]', '[1,[1,2]]', ...) 的元组
    basis_tuples = iisignature.basis(s)
    
    columns = []
    for basis_str in basis_tuples:
        term = basis_str
        # 倒序遍历替换，防止当通道数>=10时，'1' 把 '10' 里的 1 替换掉（利用正则 \b 匹配独立单词边界更安全）
        for i in range(channels, 0, -1):
            term = re.sub(rf'\b{i}\b', seq_names[i-1], term)
        columns.append(term + prefix)
            
    return columns

def gen_path_logsig_feature(
    data: pd.DataFrame, 
    lag: int, 
    ordo: Optional[int] = None, 
    z_score: bool = True
):
    """
    生成路径对数签名特征 (Log Signature)
    """
    Factor = []
    
    # 提取特征列
    marks = data.columns.to_list()
    num_features = len(marks)
    single_signal = (num_features == 1)
    
    # 如果是单信号，自动增加时间维度 T
    seq_names = marks + ['T'] if single_signal else marks
    channels = len(seq_names)
    
    # 确定阶数
    ordo = num_features + 1 if ordo is None else ordo 

    # 1. 提前生成 Log Signature 的 preparation object
    # 这是一个非常关键的优化：不需要在循环里反复生成 basis
    logsig_prep = iisignature.prepare(channels, ordo)
    
    # 2. 获取对数签名的列名
    columns = get_log_signature_columns(seq_names, ordo, f"_lag{lag}")
    
    # 提前为单信号生成时间特征序列，避免在循环中重复 np.tile 生成
    if single_signal:
        t_arr = np.linspace(0, 1, lag)

    for _, SecuData in tqdm(data.groupby('SecuCode'), position=1, leave=False):
        N = len(SecuData)
        if N < lag: 
            continue
            
        # 提取 NumPy 数组并 ffill
        vals = SecuData[marks].ffill().values
        
        # 生成滑动窗口 (N - lag + 1, lag, num_features)
        windows = sliding_window_view(vals, window_shape=lag, axis=0)
        windows = np.swapaxes(windows, 1, 2)
        
        # 计算 Z-score
        if z_score:
            mean = np.nanmean(windows, axis=1, keepdims=True)
            std = np.nanstd(windows, axis=1, ddof=1, keepdims=True)
            windows = (windows - mean) / (std + 1e-8)
        
        # 如果是单信号，拼接时间向量
        if single_signal:
            T_feat = np.broadcast_to(t_arr, (windows.shape[0], lag))[..., np.newaxis]
            windows = np.concatenate([windows, T_feat], axis=-1)
            
        # ================= 核心修改点 =================
        # 调用 iisignature.logsig 计算对数签名
        # 注意这里第二个参数传入的是前面生成的 logsig_prep 对象
        sig_result = iisignature.logsig(windows, logsig_prep)
        # ==============================================
        
        # 有效计算区间对应原序列的 [lag - 1 : ]
        valid_index = SecuData.index[lag - 1:]
        
        # 构建 DataFrame
        Factor.append(pd.DataFrame(sig_result, index=valid_index, columns=columns).dropna())
        
    if not Factor:
        return pd.DataFrame(columns=columns)
        
    return pd.concat(Factor)