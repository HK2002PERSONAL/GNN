import pandas as pd 


# df = pd.read_feather("data/feature/HF_Zscore_Polars_5_20_60.ftr")
model_mark = "hist_test_a6_1D_lag20_60_120_v0v2_c4.0%"
df = pd.read_parquet(f"model_outputs/{model_mark}.pqt")
df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
df.sort_values(['date', 'code'], inplace=True)

df['pct_rank'] = df.groupby('date')['pred'].rank(pct=True)
df['weight'] = (df['pct_rank'] > 0.95).astype(float)

df['weight'] = df['weight'] / df.groupby('date')['weight'].transform("sum")
df = df[['date', 'code', 'weight']].reset_index(drop=True)
df['code'] = df['code'].astype(str).str.zfill(6)

# print(df.info())
# print(df.head())

df.to_parquet(f"weight[{model_mark}].parquet")