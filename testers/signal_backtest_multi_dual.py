import sys
import os
import pandas as pd

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackTester.back_testV2 import batch_test_multi_field

if __name__ == "__main__":

    Signal_Mark_1 = "ICLoss"
    Signal_Mark_2 = "CSRNICLoss"
    
    TEST_MARK = "LossTest"

    SMOOTH_DECAY = 35

    signal_data_1 = pd.read_parquet("model_outputs/hist_test_a11_1D_lag20_60_120_v0v14v2_stock_ret_5d_c4.0%_ICLoss.pqt")
    signal_data_1['date'] = pd.to_datetime(signal_data_1['date'], format="%Y%m%d")
    signal_data_1 = signal_data_1.set_index(['code', 'date']).sort_index()

    signal_data_2 = pd.read_parquet("model_outputs/hist_test_a11_1D_lag20_60_120_v0v14v2_stock_ret_5d_c4.0%_CSRNICLoss.pqt")
    signal_data_2['date'] = pd.to_datetime(signal_data_2['date'], format="%Y%m%d")
    signal_data_2 = signal_data_2.set_index(['code', 'date']).sort_index()

    signal_data = pd.concat([
        signal_data_1['pred'], 
        signal_data_2['pred'], 
    ], axis=1, join="inner")
    signal_data.columns = [
        Signal_Mark_1, 
        Signal_Mark_2, 
    ]
    signal_data.index.names = ['SecuCode', 'DataDate']

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test_multi_field(signal_data, f"{TEST_MARK}_cap", neutral_factor=['lncap', 'nls'])

    batch_test_multi_field(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])