import pandas as pd

df = pd.read_parquet(
    path="data/label/stock_ret.pqt"
)



df[['Return1d', 'Return5d', 'Return20d']] = df[['Return1d', 'Return5d', 'Return20d']].groupby("DataDate").rank(pct=True)

print(df.info())
print(df.head())