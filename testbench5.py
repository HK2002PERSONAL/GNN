"""按 testbench5.py 中的三个问题顺序，诊断 HIST 信号 TOP 组表现不佳的原因。

约定：
- 信号 signal_neutralized_standardized 越大越 TOP；每日按信号十分位(10%一档)分组，TOP=第10组。
- stock_ret.ftr 来自 ForecastReturn 表，Return5d 为"未来5日"前向收益；
  因此"前5日收益"(过去5日)用 Return5d 按股票向前平移 5 个交易日近似。
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# ---------- 数据准备（与 testbench5.py 一致） ----------
df = pd.read_parquet("HIST_ablation_allneut/factor_result/HIST.parquet/signal_neut.parquet")
df['data_date'] = pd.to_datetime(df['data_date'])
df = df.set_index(["data_date", "inner_code"]).sort_index()
df.index.names = ["DataDate", "InnerCode"]
SIG = "signal_neutralized_standardized"

lb = pd.read_feather("data/label/stock_ret.ftr")
lb["DataDate"] = pd.to_datetime(lb["DataDate"])
lb = lb.set_index(["DataDate", "InnerCode"]).sort_index()

data = df.join(lb, how='inner')
print("合并后数据形状:", data.shape)

# 过去5日收益：Return5d 按股票向前平移5个交易日
data["PastRet5d"] = data.groupby(level="InnerCode")["Return5d"].shift(5)

RET = "Return5d"  # 评估用前向收益


def decile_group(g: pd.Series) -> pd.Series:
    """每日按信号十分位分组，1=最低 ... 10=最高(TOP)。"""
    return pd.qcut(g, 10, labels=False, duplicates="drop") + 1


data["grp"] = data.groupby(level="DataDate")[SIG].transform(decile_group)

SEP = "=" * 90

# ---------- 问题 1 ----------
print("\n" + SEP)
print("问题1: 各组中『过去5日收益位于全市场前10%』的股票占比（动量/追高检查）")
print(SEP)
# 每日全市场过去5日收益前10%阈值
thr = data.groupby(level="DataDate")["PastRet5d"].transform(lambda s: s.quantile(0.9))
data["is_past_top10"] = data["PastRet5d"] >= thr

occ = data.groupby("grp")["is_past_top10"].mean()
cnt = data.groupby("grp")["is_past_top10"].count()
res1 = pd.DataFrame({"占比_过去5日前10%": occ, "样本数": cnt})
print(res1.T.to_markdown(floatfmt=".2f"))
print(f"\nTOP组(10) 占比 = {occ.loc[10]:.4%}，第9组占比 = {occ.loc[9]:.4%}，"
      f"差值 = {occ.loc[10]-occ.loc[9]:.4%}")

# ---------- 问题 2 ----------
print("\n" + SEP)
print("问题2: 组收益的 均值 / 中位数 / 1%缩尾均值 的单调性（前向 Return5d）")
print(SEP)


def winsor_mean(s: pd.Series, q: float = 0.01) -> float:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi).mean()


g_mean = data.groupby("grp")[RET].mean()
g_med = data.groupby("grp")[RET].median()
g_wins = data.groupby("grp")[RET].apply(winsor_mean)
res2 = pd.DataFrame({"均值": g_mean, "中位数": g_med, "1%缩尾均值": g_wins})
print(res2.to_string(float_format=lambda x: f"{x:,.5f}"))


def monotonic_violations(series: pd.Series) -> int:
    """统计相邻组之间不满足递增的次数。"""
    vals = series.sort_index().values
    return int((np.diff(vals) < 0).sum())


for col in ["均值", "中位数", "1%缩尾均值"]:
    print(f"{col}: 单调递增违反次数 = {monotonic_violations(res2[col])} / {len(res2)-1}，"
          f"第10组-第9组 = {res2[col].loc[10]-res2[col].loc[9]:.5f}")

# ---------- 问题 3 ----------
print("\n" + SEP)
print("问题3: TOP组 vs 第9组 预测值方差；TOP组内按预测值细分5档")
print(SEP)
sig_std = data.groupby("grp")[SIG].std()
sig_mean = data.groupby("grp")[SIG].mean()
sig_span = data.groupby("grp")[SIG].agg(lambda s: s.max() - s.min())
res3 = pd.DataFrame({"信号均值": sig_mean, "信号标准差": sig_std, "信号极差": sig_span})
print(res3.to_markdown(floatfmt=".2f"))
print(f"\nTOP组信号标准差 = {sig_std.loc[10]:.5f}，第9组 = {sig_std.loc[9]:.5f}，"
      f"比值 = {sig_std.loc[10]/sig_std.loc[9]:.3f}")

top = data[data["grp"] == 10].copy()
top["sub"] = top.groupby(level="DataDate")[SIG].transform(
    lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1)
sub_stats = top.groupby("sub").agg(
    样本数=(RET, "size"),
    信号均值=(SIG, "mean"),
    前向收益均值=(RET, "mean"),
    前向收益中位数=(RET, "median"),
    过去5日收益均值=("PastRet5d", "mean"),
)
print("\nTOP组内细分5档（1=TOP中最低信号, 5=最极端/最高信号）:")
print(sub_stats.to_markdown(floatfmt=".2f"))
print(f"\n最极端档(5) 前向收益均值 = {sub_stats['前向收益均值'].loc[5]:.5f}，"
      f"档4 = {sub_stats['前向收益均值'].loc[4]:.5f}，"
      f"档1 = {sub_stats['前向收益均值'].loc[1]:.5f}")
if sub_stats["前向收益均值"].loc[5] < sub_stats["前向收益均值"].loc[4]:
    print("=> 最极端档收益反而低于次高档：存在饱和/外推迹象")
else:
    print("=> 最极端档收益未显著变差")