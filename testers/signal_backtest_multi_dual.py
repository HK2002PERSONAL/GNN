import sys
import os
import pandas as pd

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackTester.back_testV2 import batch_test

if __name__ == "__main__":

    Signal_Mark_1 = "[C[IJ[corr]]]"
    Signal_Mark_2 = "[C[IJ[orth]]]"
    
    TEST_MARK = "FactorGCL_Ablation_gamma[1.0]"

    SMOOTH_DECAY = 15

    signal_data_1 = pd.read_parquet("model_outputs/hist_test_FactorGCL_a2g[1.0]_1D_lag20_60_120_v0v14v2_c4.0%.pqt")
    signal_data_1['date'] = pd.to_datetime(signal_data_1['date'], format="%Y%m%d")
    signal_data_1 = signal_data_1.set_index(['code', 'date']).sort_index()

    # signal_data_rival = pd.read_parquet("VHF[five_minute]PureETEChen_r5/Tech_LGBM_Regression_Rank_pred.parquet")
    # signal_data_rival['SecuCode'] = signal_data_rival['SecuCode'].astype(int)
    # signal_data_rival = signal_data_rival.set_index(['SecuCode', 'DataDate']).sort_index()
    # signal_data_rival.index.names = ['code', 'date']

    signal_data_2 = pd.read_parquet("model_outputs/hist_test_FactorGCL_a3g[1.0]_1D_lag20_60_120_v0v14v2_c4.0%.pqt")
    signal_data_2['date'] = pd.to_datetime(signal_data_2['date'], format="%Y%m%d")
    signal_data_2 = signal_data_2.set_index(['code', 'date']).sort_index()

    signal_data = pd.concat([
        signal_data_1['pred'], 
        # signal_data_rival['pred'], 
        signal_data_2['pred'], 
    ], axis=1, join="inner")
    signal_data.columns = [
        Signal_Mark_1, 
        Signal_Mark_2, 
    ]
    signal_data.index.names = ['SecuCode', 'DataDate']

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test(signal_data, f"{TEST_MARK}_lncap", neutral_factor=['lncap'])

    batch_test(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])