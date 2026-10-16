import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from gen_path_sig_alpha import gen_path_sig_feature
# from factory.gen_path_sig_alpha import gen_path_logsig_feature


if __name__ == "__main__":

    TEST_NAME = "TechPureETE2"
    print(f"Processing {TEST_NAME}")

    # cleaning
    data = pd.read_feather(os.path.join('data', 'stock', "stock_data.ftr")).set_index(['SecuCode', 'DataDate']).sort_index()
    data = data[data['IfSuspended']==0]

    # preprocessing
    data['AdjClosePrice'] = data.groupby('SecuCode')['DailyReturn'].transform(lambda x: (1+x).cumprod())
    data['AdjHighPrice'] = data['HighPrice'] / data['ClosePrice'] * data['AdjClosePrice']
    data['AdjLowPrice'] = data['LowPrice'] / data['ClosePrice'] * data['AdjClosePrice']
    data['AdjOpenPrice'] = data['OpenPrice'] / data['ClosePrice'] * data['AdjClosePrice']
    data['TurnoverValue'] = data['TurnoverValue'].apply(np.log1p)
    data.loc[(data['HighPrice']==data['LowPrice'])&(data['DailyReturn']!=0), 'TurnoverValue'] = data.groupby('SecuCode')['TurnoverValue'].transform(lambda x: x.rolling(120).max())
    data['TurnoverVolume'] = data['TurnoverVolume'].apply(np.log1p)
    data.loc[(data['HighPrice']==data['LowPrice'])&(data['DailyReturn']!=0), 'TurnoverVolume'] = data.groupby('SecuCode')['TurnoverVolume'].transform(lambda x: x.rolling(120).max())

    # data = data.groupby('SecuCode').ffill()
    data = data.dropna()

    features = ['AdjOpenPrice', 'AdjClosePrice', 'AdjHighPrice', 'AdjLowPrice', 'TurnoverValue', 'TurnoverVolume']
    # features = ['AdjClosePrice', 'AdjHighPrice', 'AdjLowPrice', 'TurnoverValue']

    feature_data = pd.concat([gen_path_sig_feature(data[features], lag=_lag, ordo=3, z_score=True) for _lag in tqdm([20, 60, 120])], axis=1, join='inner')

    # feature_data.swaplevel().sort_index().to_parquet(f"{TEST_NAME}.pqt")
    feature_data.to_feather(f"factor_feature/{TEST_NAME}.ftr")


