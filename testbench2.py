import pandas as pd
# import numpy as np

print("读取数据...")
raw = pd.read_parquet("data/raw_data/stock_edge.pqt")

# 股票名称映射（取最新日期）
latest_date = raw['DataDate'].max()

# 与热力图脚本相同口径：最近240日收益，去NaN
data = raw.set_index(["SecuCode", "DataDate"]).sort_index()
ret = data['DailyReturn'].unstack().iloc[:, -240:].dropna().T

ret_corr = ret.corr().stack()

barra_data = pd.read_parquet("data/raw_data/stock_risk.pqt")
latest_barra = barra_data[barra_data.index.get_level_values('DataDate') == latest_date]
barra_corr = latest_barra.droplevel(0).T.corr().stack()


print(ret_corr.head())
print(barra_corr.head())

print(ret_corr.corr(barra_corr, method="spearman"))