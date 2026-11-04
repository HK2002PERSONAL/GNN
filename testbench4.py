import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 读取数据，计算收益率/价格相关性矩阵 ==========
print("读取数据...")
data = pd.read_parquet("data/raw_data/stock_edge.pqt")
data = data.reset_index(drop=False).set_index(["SecuCode", "DataDate"]).sort_index()
ret = data['DailyReturn'].unstack().iloc[:, -240:].dropna().T
prc = (ret + 1).cumprod()
ret_corr_matrix = ret.corr()
prc_corr_matrix = prc.corr()

# ========== 绘制上半三角相关系数分布直方图 ==========
def get_upper_triangle_values(corr_matrix):
    """提取相关系数矩阵上半三角（不含对角线）的所有值"""
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    return corr_matrix.values[mask]

def plot_corr_histogram(ax, values, title, color):
    """绘制相关系数分布直方图，标注5%和95%分位数分割线"""
    q05 = np.percentile(values, 5)
    q95 = np.percentile(values, 95)
    
    ax.hist(values, bins=100, color=color, edgecolor='white', alpha=0.7)
    ax.axvline(q05, color='red', linestyle='--', linewidth=2, 
               label=f'5%分位数: {q05:.4f}')
    ax.axvline(q95, color='green', linestyle='--', linewidth=2, 
               label=f'95%分位数: {q95:.4f}')
    ax.set_xlabel('相关系数')
    ax.set_ylabel('频数')
    ax.set_title(title)
    ax.legend()
    return q05, q95

# 提取上半三角相关系数
ret_values = get_upper_triangle_values(ret_corr_matrix)
prc_values = get_upper_triangle_values(prc_corr_matrix)

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 绘制直方图
q05_ret, q95_ret = plot_corr_histogram(ax1, ret_values, '收益率相关系数分布（上半三角）', 'steelblue')
q05_prc, q95_prc = plot_corr_histogram(ax2, prc_values, '价格相关系数分布（上半三角）', 'coral')

plt.tight_layout()
plt.savefig('correlation_distribution.png', dpi=150, bbox_inches='tight')
print("图表已保存至: correlation_distribution.png")

# 显示统计信息
print("\n========== 统计信息 ==========")
print(f"股票数量: {ret_corr_matrix.shape[0]}")
print(f"收益率相关系数: 5%分位数={q05_ret:.4f}, 95%分位数={q95_ret:.4f}, 均值={np.mean(ret_values):.4f}, 中位数={np.median(ret_values):.4f}")
print(f"价格相关系数:   5%分位数={q05_prc:.4f}, 95%分位数={q95_prc:.4f}, 均值={np.mean(prc_values):.4f}, 中位数={np.median(prc_values):.4f}")

plt.show()
