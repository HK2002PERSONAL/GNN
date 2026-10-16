import pandas as pd

from BackTester.back_testV2 import batch_test

if __name__ == "__main__":

    TEST_MARK = Signal_Mark = "V0V2"

    SMOOTH_DECAY = 15

    signal_data = pd.read_parquet("model_outputs/hist_test_a6_1D_lag20_60_120_v0v2_c4.0%.pqt")
    signal_data['date'] = pd.to_datetime(signal_data['date'], format="%Y%m%d")
    signal_data = signal_data.set_index(['code', 'date']).sort_index()

    signal_data = signal_data[['pred']]
    signal_data.columns = [Signal_Mark]
    signal_data.index.names = ['SecuCode', 'DataDate']

    signal_data = signal_data.groupby("SecuCode").ewm(span=SMOOTH_DECAY).mean().droplevel(0)

    signal_data = signal_data[signal_data.index.get_level_values('DataDate') > pd.to_datetime("2021-12-31")]

    batch_test(signal_data, f"{TEST_MARK}_lncap", neutral_factor=['lncap'])
    batch_test(signal_data, f"{TEST_MARK}_cap", neutral_factor=['lncap', 'nls'])
    batch_test(signal_data, f"{TEST_MARK}_allneut", 
        neutral_factor=['beta', 'bp', 'ey', 'growth', 'leverage', 'liquidity', 'lncap', 'momentum', 'nls', 'rv', 'contrarian', 'short_rev'])