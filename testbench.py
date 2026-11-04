import pandas as pd

df = pd.read_parquet(
    path="data/label/spec_ret.pqt"
)

print(df.info())
print(df.head())