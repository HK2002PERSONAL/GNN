import pandas as pd 
import numpy as np
import torch
import signatory

from testbench2 import cal_segmented_levy_fast

def cal_abs_levy(asset_path: np.ndarray) -> np.ndarray:
    asset_path = torch.from_numpy(asset_path)
    B = asset_path.shape[0]
    norm = (asset_path - asset_path.mean(dim=-1, keepdim=True)) / \
           (asset_path.std(dim=-1, keepdim=True) + 1e-8)
    norm = torch.clip(norm, -5, 5).transpose(0, 1)          # (H, B)
    sig = signatory.signature(norm.unsqueeze(0), depth=2).squeeze(0)
    cross_sig = sig[B:].reshape(B, B)
    return torch.abs(cross_sig - cross_sig.T).numpy()



data = pd.read_parquet("data/raw_data/stock_edge.pqt", columns=["IndustryNameNew", "SecuAbbr", "DataDate", "DailyReturn"])
data = data.reset_index(drop=False).set_index(["IndustryNameNew", "SecuAbbr", "DataDate"])
data = data.sort_index()['DailyReturn'].unstack().iloc[:, -240:].dropna().T
corr_matrix = data.corr()
price_path = (data+1).cumprod()
# levy_matrix = pd.DataFrame(cal_abs_levy(price_path.T.values), corr_matrix.index, corr_matrix.columns)
seg_levy_matrix = pd.DataFrame(
    cal_segmented_levy_fast(torch.from_numpy(price_path.T.values), device="cuda", reference_line="TLS").detach().cpu().numpy(), 
    corr_matrix.index, corr_matrix.columns)


def get_top_pairs(matrix: pd.DataFrame, top_n: int = 10, largest: bool = True,
                  extra_matrix: pd.DataFrame = None, extra_name: str = None) -> pd.DataFrame:
    """提取上三角(排除对角线)中最大/最小的 top_n 对资产
    
    extra_matrix: 可选，用于附带查询另一个矩阵中相同资产对的值（如相关系数）
    """
    vals = matrix.values
    n = vals.shape[0]
    # 取上三角索引 (i < j)，排除对角线
    iu = np.triu_indices(n, k=1)
    pairs = pd.DataFrame({
        '资产1': matrix.index[iu[0]],
        '资产2': matrix.columns[iu[1]],
        '值': vals[iu],
        '_i': iu[0],
        '_j': iu[1],
    })
    if extra_matrix is not None and extra_name is not None:
        pairs[extra_name] = extra_matrix.values[pairs['_i'], pairs['_j']]
    pairs[['行业1', '证券名称1']] = pd.DataFrame(pairs['资产1'].tolist(), index=pairs.index)
    pairs[['行业2', '证券名称2']] = pd.DataFrame(pairs['资产2'].tolist(), index=pairs.index)
    pairs = pairs.sort_values('值', ascending=not largest).head(top_n).reset_index(drop=True)
    cols = ['证券名称1', '行业1', '证券名称2', '行业2', '值']
    if extra_name is not None:
        cols.append(extra_name)
    return pairs[cols]


print("=" * 100)
print("相关性矩阵最大的前10对资产 (附带分段Levy):")
print("=" * 100)
print(get_top_pairs(corr_matrix, top_n=10, largest=True,
                    extra_matrix=seg_levy_matrix, extra_name='分段Levy').to_string())

# print("\n" + "=" * 100)
# print("Levy矩阵最小的前10对资产:")
# print("=" * 100)
# print(get_top_pairs(levy_matrix, top_n=10, largest=False).to_string())

print("\n" + "=" * 100)
print("分段Levy矩阵最小的前10对资产 (附带相关系数):")
print("=" * 100)
print(get_top_pairs(seg_levy_matrix, top_n=10, largest=True,
                    extra_matrix=corr_matrix, extra_name='相关系数').to_string())
