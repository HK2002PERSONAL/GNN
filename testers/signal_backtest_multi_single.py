import sys
import os
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BackTester.back_testV2 import batch_test_multi_field

if __name__ == "__main__":

    TEST_MARK = Signal_Mark = "3GCN_DELIMIT"

    SMOOTH_DECAY = 15

    signal_data = pd.read_parquet("model_outputs/hist_test_a11_1D_lag20_60_120_v0v14v2_stock_ret_delimit_c4.0%.pqt")
    signal_data['date'] = pd.to_datetime(signal_data['date'], format="%Y%m%d")
    signal_data = signal_data.set_index(['code', 'date']).sort_index()

    signal_data = signal_data[['pred']]
    signal_data.columns = [Signal_Mark]
    signal_data.index.names = ['SecuCode', 'DataDate']

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test_multi_field(signal_data, f"{TEST_MARK}_cap", neutral_factor=['lncap', 'nls'])
    batch_test_multi_field(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])