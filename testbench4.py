import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

folder = "data/feature/CAT_five_minute_LAG20_60_120"
query = f"""
    SELECT * FROM read_parquet('{folder}/*.pqt')
    ORDER BY DataDate, SecuCode
"""

con = duckdb.connect()
con.execute("PRAGMA enable_progress_bar")
# con.execute("SET memory_limit = '8GB'")          # 可选，超出会溢出到磁盘

# 1. 用空 DataFrame 生成带 MultiIndex 的 pandas 元数据
empty = con.execute(query + " LIMIT 0").df().set_index(["DataDate", "SecuCode"])
pandas_meta = pa.Table.from_pandas(empty).schema.metadata   # 含 b'pandas' 键

# 2. 拿到 DuckDB 结果的真实 schema，贴上 pandas 元数据
duck_schema = con.execute(query + " LIMIT 0").arrow().schema
final_schema = duck_schema.with_metadata({**(duck_schema.metadata or {}), **pandas_meta})

# 3. 流式写出
with pq.ParquetWriter("CAT_five_minute_LAG20_60_120.pqt", final_schema, compression="zstd") as writer:
    reader = con.execute(query).to_arrow_reader(batch_size=500_000)
    for batch in reader:
        writer.write_batch(batch)

con.close()