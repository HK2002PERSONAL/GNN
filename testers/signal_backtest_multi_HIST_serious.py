import sys
import os
import pandas as pd

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackTester.back_testV2 import batch_test

if __name__ == "__main__":

    rival = "-LGBM"
    Signal_Mark_0 = "0-HIST"
    Signal_Mark_1 = "1-Predefined"
    Signal_Mark_2 = "2-Hidden"
    Signal_Mark_3 = "3-Predefined_GAT"
    Signal_Mark_4 = "4-GCN_Hidden"
    Signal_Mark_5 = "5-GCN_GAT"
    Signal_Mark_6 = "6-GCN_GCN"
    Signal_Mark_6_1 = "6-GCN_GCN(neg_corr)"
    Signal_Mark_6_2 = "6_2-GCN_HLGCN"
    Signal_Mark_7 = "7-GCN_RAT"
    Signal_Mark_8 = "8-GCN_RGAT"
    Signal_Mark_9 = "9-GCN_GCN_DRAT"
    

    SMOOTH_DECAY = 15

    clip = 0.04

    if clip < 1:
        clip_mark = f"{clip*100:.1f}%"
    else:
        clip_mark = f"{clip:.1f}"

    TEST_MARK = f"HIST_ablation_c{clip_mark}"

    signal_data_rival = pd.read_parquet("VPureETE2_r5/Tech_LGBM_Regression_Rank_pred.parquet")
    signal_data_rival['SecuCode'] = signal_data_rival['SecuCode'].astype(int)
    signal_data_rival = signal_data_rival.set_index(['SecuCode', 'DataDate']).sort_index()
    signal_data_rival.index.names = ['code', 'date']

    signal_data_0 = pd.read_parquet(f"model_outputs/hist_test_a0_v0_c{clip_mark}.pqt")
    signal_data_0['date'] = pd.to_datetime(signal_data_0['date'], format="%Y%m%d")
    signal_data_0 = signal_data_0.set_index(['code', 'date']).sort_index()

    signal_data_1 = pd.read_parquet(f"model_outputs/hist_test_a1_v0_c{clip_mark}.pqt")
    signal_data_1['date'] = pd.to_datetime(signal_data_1['date'], format="%Y%m%d")
    signal_data_1 = signal_data_1.set_index(['code', 'date']).sort_index()

    signal_data_2 = pd.read_parquet(f"model_outputs/hist_test_a2_v0_c{clip_mark}.pqt")
    signal_data_2['date'] = pd.to_datetime(signal_data_2['date'], format="%Y%m%d")
    signal_data_2 = signal_data_2.set_index(['code', 'date']).sort_index()

    signal_data_3 = pd.read_parquet(f"model_outputs/hist_test_a3_v0_c{clip_mark}.pqt")
    signal_data_3['date'] = pd.to_datetime(signal_data_3['date'], format="%Y%m%d")
    signal_data_3 = signal_data_3.set_index(['code', 'date']).sort_index()

    signal_data_4 = pd.read_parquet(f"model_outputs/hist_test_a4_v0_c{clip_mark}.pqt")
    signal_data_4['date'] = pd.to_datetime(signal_data_4['date'], format="%Y%m%d")
    signal_data_4 = signal_data_4.set_index(['code', 'date']).sort_index()

    signal_data_5 = pd.read_parquet(f"model_outputs/hist_test_a5_v0_c{clip_mark}.pqt")
    signal_data_5['date'] = pd.to_datetime(signal_data_5['date'], format="%Y%m%d")
    signal_data_5 = signal_data_5.set_index(['code', 'date']).sort_index()

    signal_data_6 = pd.read_parquet(f"model_outputs/hist_test_a6_v0v2_c{clip_mark}.pqt")
    signal_data_6['date'] = pd.to_datetime(signal_data_6['date'], format="%Y%m%d")
    signal_data_6 = signal_data_6.set_index(['code', 'date']).sort_index()

    signal_data_6_1 = pd.read_parquet(f"model_outputs/hist_test_a6_v0v3_c{clip_mark}.pqt")
    signal_data_6_1['date'] = pd.to_datetime(signal_data_6_1['date'], format="%Y%m%d")
    signal_data_6_1 = signal_data_6_1.set_index(['code', 'date']).sort_index()

    signal_data_6_2 = pd.read_parquet(f"model_outputs/hist_test_a6_v2_v0v3_c{clip_mark}.pqt")
    signal_data_6_2['date'] = pd.to_datetime(signal_data_6_2['date'], format="%Y%m%d")
    signal_data_6_2 = signal_data_6_2.set_index(['code', 'date']).sort_index()

    signal_data_7 = pd.read_parquet(f"model_outputs/hist_test_a7_v0rat_c{clip_mark}.pqt")
    signal_data_7['date'] = pd.to_datetime(signal_data_7['date'], format="%Y%m%d")
    signal_data_7 = signal_data_7.set_index(['code', 'date']).sort_index()

    signal_data_8 = pd.read_parquet(f"model_outputs/hist_test_a8_v0rat_c{clip_mark}.pqt")
    signal_data_8['date'] = pd.to_datetime(signal_data_8['date'], format="%Y%m%d")
    signal_data_8 = signal_data_8.set_index(['code', 'date']).sort_index()

    signal_data_9 = pd.read_parquet(f"model_outputs/hist_test_a9_v0v2rat_c{clip_mark}.pqt")
    signal_data_9['date'] = pd.to_datetime(signal_data_9['date'], format="%Y%m%d")
    signal_data_9 = signal_data_9.set_index(['code', 'date']).sort_index()

    signal_data = pd.concat([
        signal_data_rival['pred'],
        signal_data_0['pred'], 
        signal_data_1['pred'], 
        signal_data_2['pred'], 
        signal_data_3['pred'], 
        signal_data_4['pred'], 
        signal_data_5['pred'], 
        signal_data_6['pred'], 
        signal_data_6_1['pred'], 
        signal_data_6_2['pred'], 
        signal_data_7['pred'], 
        signal_data_8['pred'], 
        signal_data_9['pred'], 
    ], axis=1, join="inner")
    signal_data.columns = [
        rival,
        Signal_Mark_0, 
        Signal_Mark_1, 
        Signal_Mark_2, 
        Signal_Mark_3, 
        Signal_Mark_4, 
        Signal_Mark_5, 
        Signal_Mark_6, 
        Signal_Mark_6_1, 
        Signal_Mark_6_2, 
        Signal_Mark_7, 
        Signal_Mark_8, 
        Signal_Mark_9, 
    ]
    signal_data.index.names = ['SecuCode', 'DataDate']

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test(signal_data, f"{TEST_MARK}_lncap", neutral_factor=['lncap'])

    batch_test(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])