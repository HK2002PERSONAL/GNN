import pandas as pd

df = pd.read_parquet(
        path="data/raw_data/stock_data.pqt", columns=["DailyReturn"]
    )

print(df.info())
print(df.head())