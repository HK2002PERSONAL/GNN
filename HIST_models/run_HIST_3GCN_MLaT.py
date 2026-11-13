"""HIST_models：HIST_a11（GCN + GCN + GCN）特征提取 + LightGBM 训练（MLaT V1/V2/V3）

假设 run_HIST_3GCN.py 已经训练完成。本脚本：
1. 加载已训练的 HIST_a11 模型权重
2. 使用 HIST_MLaT_V1/V2/V3（继承自 HIST_a11，forward 返回最后一层前的特征）
   - V1: all_info_fore = p1 + p2 + p3 + i，维度 [N, hidden_size]
   - V2: cat([gnn_info_fore, i_info_fore])，维度 [N, hidden_size * 2]
   - V3: cat([p1, p2, p3, i])，维度 [N, hidden_size * 4]
3. 提取 train/valid/test 的特征并保存为 parquet
4. 训练 LightGBM（参考 pipeline_ML_base_LightGBM.py，不滚动训练，训练集、验证集、测试集和原范围保持一致）

数据划分（与 build_GCN_GCN_GCN_data.py 一致）：
    train: 2010-01-01 ~ 2019-12-31
    valid: 2020-01-31 ~ 2021-12-31
    test:  2022-01-31 起

运行：
    python HIST_models/run_HIST_3GCN_MLaT.py -v2 14 -v3 2 --mlat-version V1
    python HIST_models/run_HIST_3GCN_MLaT.py -v2 14 -v3 2 --mlat-version V2
    python HIST_models/run_HIST_3GCN_MLaT.py -v2 14 -v3 2 --mlat-version V3
    python HIST_models/run_HIST_3GCN_MLaT.py -v2 14 -v3 2 --mlat-version V1 --skip-extract  # 跳过特征提取

输出：model_features/hist_a11_MLaT{V1|V2|V3}_{data_version}_{clip_mark}{ew_mark}{loss_mark}/
      ├── train.pqt          训练集特征（date, y=标签, f_0~f_{hidden_size-1}）
      ├── valid.pqt          验证集特征
      ├── test.pqt           测试集特征（y=股票代码）
      ├── LGBM_pred.parquet  测试集预测（DataDate, SecuCode, pred）
      └── LGBM_IC_stats.csv  测试集 IC 统计
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                              # 项目根目录（utils 等）
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 本目录 models / data_utils

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from models import HIST_MLaT_V1, HIST_MLaT_V2, HIST_MLaT_V3

# MLaT 模型类映射
MLAT_MODELS = {
    "V1": HIST_MLaT_V1,
    "V2": HIST_MLaT_V2,
    "V3": HIST_MLaT_V3,
}

# MLaT 各版本输出维度倍数（相对于 hidden_size）
MLAT_DIM_MULTIPLIER = {
    "V1": 1,  # [N, hidden_size]
    "V2": 2,  # [N, hidden_size * 2]
    "V3": 4,  # [N, hidden_size * 4]
}
from data_utils import get_dataloader
from utils.Logger import smart_log, make_log_file
from utils.SeedManager import seed_all_non_torch

root_seed = 10086
torch_seed_range = range(root_seed, root_seed + 10, 4)


# ---------------- 概念 -> 概念矩阵 ----------------

def industry_to_concept(industry: torch.Tensor) -> torch.Tensor:
    """把每日概念标签 [N]（整数）变形为股票-概念 0/1 矩阵 [N, C]。

    C = 当日出现的概念数。股票 A 属于概念 c 时 [A, c] = 1，否则 0。
    """
    n = industry.shape[0]
    c = int(industry.max().item()) + 1
    concept = torch.zeros(n, c, dtype=torch.float, device=industry.device)
    concept.scatter_(1, industry.reshape(-1, 1), 1.0)
    return concept


# ---------------- 特征提取前向 ----------------

def model_forward_features(_model, _sample, _device, _edge_v2: int, _edge_v3: int):
    """把单个日截面样本喂给 HIST_MLaT_V1，返回最后一层前的特征 [N, hidden_size]。

    概念 1 = v0 行业（固定），概念 2 = v{edge_v2}，概念 3 = v{edge_v3}。
    """
    x = _sample["x"].to(_device)
    industry_v0 = _sample["industry_v0"].to(_device)
    industry_v2 = _sample[f"industry_v{_edge_v2}"].to(_device)
    industry_v3 = _sample[f"industry_v{_edge_v3}"].to(_device)
    market_value = _sample["market_value"].to(_device)
    concept_1 = industry_to_concept(industry_v0)
    concept_2 = industry_to_concept(industry_v2)
    concept_3 = industry_to_concept(industry_v3)
    return _model(x, concept_1, concept_2, concept_3, market_value)


# ---------------- LightGBM 工具函数（参考 pipeline_ML_base_LightGBM.py） ----------------

def cross_sectional_rank(s):
    """
    将收益率转换为截面百分位排名 (0 到 1 之间)。
    相同的收益率取平均排名，避免排序噪音。
    """
    return s.rank(pct=True).values


def custom_ic_eval(preds, train_data):
    """
    自定义 LightGBM 评估函数：计算预测值与真实 Label 的 Pearson IC。
    由于我们在外部已经把 Label 转换为了每日截面 Rank，
    所以这里的全局 Pearson IC 在数学上非常接近每日 RankIC 的均值。
    """
    labels = train_data.get_label()
    # 避免标准差为 0 导致的除零错误
    if np.std(preds) <= 1e-8 or np.std(labels) <= 1e-8:
        return 'IC', 0.0, True

    # 计算相关系数
    ic = np.corrcoef(preds, labels)[0, 1]
    return 'IC', float(ic), True  # True 表示指标越大越好


def parse_args():
    parser = argparse.ArgumentParser(description="HIST_a11 特征提取（MLaT_V1）+ LightGBM 训练")
    parser.add_argument("--freq", type=str, default="1D", choices=["1D", "five_minute"])
    parser.add_argument("--spans", type=int, nargs="+", default=[20, 60, 120], help="回溯天数列表，如: 20 60 120")
    parser.add_argument("--edge-version-2", "-v2", type=int, default=14, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第二边版本（开放选择）")
    parser.add_argument("--edge-version-3", "-v3", type=int, default=2, choices=[2, 3, 7, 9, 10, 12, 14],
                        help="第三边版本（开放选择）")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--clip", "-c", type=float, default=0.04)
    parser.add_argument("--ew1", type=int, default=1, choices=[0, 1],
                        help="第一层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew2", type=int, default=1, choices=[0, 1],
                        help="第二层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--ew3", type=int, default=1, choices=[0, 1],
                        help="第三层是否使用 edge_weight，1=使用（默认），0=不使用")
    parser.add_argument("--label", type=str, default="data/label/stock_ret.pqt",
                        help="标签文件路径（.pqt 格式），默认 data/label/stock_ret.pqt")
    parser.add_argument("--loss", type=str, default="ICLoss", choices=["ICLoss", "CCCLoss"],
                        help="原训练损失函数（用于定位模型路径），默认 ICLoss")
    parser.add_argument("--mlat-version", "-m", type=str, default="V1", choices=["V1", "V2", "V3"],
                        help="MLaT 版本：V1=[p1+p2+p3+i], V2=cat([gnn, i]), V3=cat([p1,p2,p3,i])，默认 V1")
    parser.add_argument("--label-span", type=int, default=5,
                        help="LightGBM 评估用标签前瞻天数（需与 GNN 数据标签一致），默认 5")
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过特征提取，直接使用已有特征训练 LightGBM")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = "a11"

    # 校验：第二边和第三边版本不能相同
    if args.edge_version_2 == args.edge_version_3:
        raise ValueError(f"第二边版本 (v{args.edge_version_2}) 和第三边版本 (v{args.edge_version_3}) 不能相同")

    clip = args.clip
    if clip < 1:
        clip_mark = f"c{clip*100:.1f}%"
    else:
        clip_mark = f"c{clip:.1f}"

    # 标签名称（不含扩展名），用于数据版本与输出路径命名
    if not args.label.endswith(".pqt"):
        raise ValueError(f"标签文件必须为 .pqt 格式: {args.label}")
    label_name = os.path.splitext(os.path.basename(args.label))[0]

    # 数据版本：{freq}_lag{spans}_v0v{edge_v2}v{edge_v3}_{label_name}
    data_version = f"{args.freq}_lag{'_'.join([str(span) for span in args.spans])}_v0v{args.edge_version_2}v{args.edge_version_3}_{label_name}"

    ew1 = bool(args.ew1); ew_mark = 'ew1[off]' if not ew1 else ''
    ew2 = bool(args.ew2); ew_mark = ew_mark + 'ew2[off]' if not ew2 else ew_mark
    ew3 = bool(args.ew3); ew_mark = ew_mark + 'ew3[off]' if not ew3 else ew_mark

    # 损失函数标记（使用非默认损失时添加到输出路径）
    loss_mark = f"_{args.loss}"

    # MLaT 版本
    mlat_version = args.mlat_version
    mlat_model_class = MLAT_MODELS[mlat_version]
    mlat_dim_multiplier = MLAT_DIM_MULTIPLIER[mlat_version]
    feat_dim = args.hidden_size * mlat_dim_multiplier  # 特征维度

    # 原训练模型路径（run_HIST_3GCN.py 的输出）
    ckpt_path = f"model_configs/best_hist_{model_id}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}"
    # 特征输出目录
    feat_dir = Path(f"model_features/hist_{model_id}_MLaT{mlat_version}_{data_version}_{clip_mark}{'_' + ew_mark if len(ew_mark) else ''}{loss_mark}")

    make_log_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smart_log(f"device = {device}")
    smart_log(f"model = HIST_{model_id} -> HIST_MLaT_{mlat_version} | data = {data_version} | hidden_size = {args.hidden_size} | feat_dim = {feat_dim}")
    smart_log(f"ckpt_path = {ckpt_path}")
    smart_log(f"feat_dir = {feat_dir}")

    seed_all_non_torch(root_seed)

    edge_v2 = args.edge_version_2
    edge_v3 = args.edge_version_3

    # ============================================
    # 1. 特征提取（可选跳过）
    # ============================================
    if not args.skip_extract:
        smart_log("Loading Data...")
        train_loader = get_dataloader("train", version=data_version, batch_size=1, shuffle=False)
        smart_log("train done")

        # 特征维度从训练集首个样本动态获取
        d_feat = train_loader.dataset[0]["x"].shape[1]
        smart_log(f"d_feat = {d_feat}")

        # 检查模型文件是否存在
        for seed in torch_seed_range:
            pth = Path(ckpt_path) / f"{seed}.pt"
            if not pth.is_file():
                raise FileNotFoundError(f"未找到已训练的模型: {pth}，请先运行 run_HIST_3GCN.py")

        # 加载模型（HIST_MLaT_V1/V2/V3 继承自 HIST_a11，可直接加载其权重）
        modelS = {}
        for seed in torch_seed_range:
            model = mlat_model_class(
                d_feat=d_feat, hidden_size=args.hidden_size, clip=clip,
                ew1=ew1, ew2=ew2, ew3=ew3
            ).to(device)
            model.load_state_dict(torch.load(
                Path(ckpt_path) / f"{seed}.pt",
                map_location=device))
            model.eval()
            modelS[seed] = model
        smart_log(f"已加载 {len(modelS)} 个模型: {list(torch_seed_range)}")

        # 提取特征
        feat_cols = [f"f_{i}" for i in range(feat_dim)]
        n_seeds = len(torch_seed_range)

        for split in ["train", "valid", "test"]:
            smart_log(f"提取 {split} 特征...")
            rows = []
            loader = get_dataloader(split, version=data_version, batch_size=1, shuffle=False)

            with torch.no_grad():
                for sample in loader:
                    # 多 seed 特征平均
                    feats = None
                    for seed in torch_seed_range:
                        f = model_forward_features(modelS[seed], sample, device, edge_v2, edge_v3)
                        feats = f if feats is None else feats + f
                    feats = (feats / n_seeds).cpu().numpy()

                    date = int(sample["date"][0] * 10000 + sample["date"][1] * 100 + sample["date"][2])
                    df = pd.DataFrame(feats, columns=feat_cols)
                    df["date"] = date
                    df["y"] = sample["y"].numpy()  # train/valid: 标签; test: 股票代码
                    rows.append(df)

            result = pd.concat(rows, ignore_index=True)
            feat_dir.mkdir(parents=True, exist_ok=True)
            result.to_parquet(feat_dir / f"{split}.pqt")
            smart_log(f"{split} 特征已保存 -> {feat_dir / (split + '.pqt')}，共 {len(result)} 行")

    # ============================================
    # 2. LightGBM 训练（不滚动，训练/验证/测试范围与 GNN 数据一致）
    # ============================================
    smart_log("加载特征，开始 LightGBM 训练...")

    train_df = pd.read_parquet(feat_dir / "train.pqt")
    valid_df = pd.read_parquet(feat_dir / "valid.pqt")
    test_df = pd.read_parquet(feat_dir / "test.pqt")

    feature_cols = [c for c in train_df.columns if c.startswith("f_")]
    smart_log(f"特征数量: {len(feature_cols)}")

    # 过滤标签非有限的样本（train/valid 的 y 是标签）
    train_mask = np.isfinite(train_df["y"].values)
    valid_mask = np.isfinite(valid_df["y"].values)

    train_df = train_df.loc[train_mask].reset_index(drop=True)
    valid_df = valid_df.loc[valid_mask].reset_index(drop=True)

    smart_log(f"train 样本数: {len(train_df)} | valid 样本数: {len(valid_df)} | test 样本数: {len(test_df)}")

    # Label 处理：截面 Rank 编码 (0~1)
    train_l = train_df.groupby("date")["y"].transform(cross_sectional_rank).values
    valid_l = valid_df.groupby("date")["y"].transform(cross_sectional_rank).values

    train_X = train_df[feature_cols].values
    valid_X = valid_df[feature_cols].values
    test_X = test_df[feature_cols].values

    # LGBM Dataset
    train_data_lgb = lgb.Dataset(
        train_X,
        label=train_l,
        feature_name=feature_cols,
        free_raw_data=False
    )

    valid_data_lgb = lgb.Dataset(
        valid_X,
        label=valid_l,
        feature_name=feature_cols,
        reference=train_data_lgb,
        free_raw_data=False
    )

    # 训练参数 (Regression)
    params = {
        'objective': 'regression',
        'metric': 'None',                 # 禁用默认 metric，完全依赖自定义的 IC
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': root_seed,
        'n_jobs': 32
    }

    # 训练并根据 valid 的 IC 进行早停
    model = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=1500,
        valid_sets=[train_data_lgb, valid_data_lgb],
        valid_names=['train', 'valid'],
        feval=custom_ic_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    smart_log(f"LightGBM 训练完成，best_iteration = {model.best_iteration}")

    # ============================================
    # 3. 测试集预测与评估
    # ============================================
    test_pred = model.predict(test_X)

    # 构建预测 DataFrame（test 的 y 是股票代码）
    test_codes = test_df["y"].astype(np.int64).astype(str).str.zfill(6).values
    test_dates_int = test_df["date"].values

    pred_df = pd.DataFrame({
        "date": test_dates_int,
        "code": test_codes,
        "pred": test_pred,
    })

    # 保存预测结果（转换为与 pipeline_ML_base_LightGBM.py 兼容的格式）
    pred_df["DataDate"] = pd.to_datetime(pred_df["date"].astype(str), format="%Y%m%d")
    pred_df["SecuCode"] = pred_df["code"]
    pred_out = pred_df.set_index(["DataDate", "SecuCode"])[["pred"]].sort_index()

    feat_dir.mkdir(parents=True, exist_ok=True)
    pred_out.reset_index().to_parquet(feat_dir / "LGBM_pred.parquet", index=False)
    smart_log(f"测试集预测已保存 -> {feat_dir / 'LGBM_pred.parquet'}，共 {len(pred_out)} 行")

    # 评估：加载标签对齐计算每日 IC
    label = pd.read_parquet(args.label)
    if 'DataDate' in label.columns and 'SecuCode' in label.columns:
        label = label.set_index(['DataDate', 'SecuCode'])
    label = label.sort_index()

    label_col = f"Return{args.label_span}d"
    if label_col not in label.columns:
        smart_log(f"警告: 标签文件中未找到 {label_col} 列，跳过 IC 评估")
    else:
        # 对齐预测与标签
        avail_idx = pred_out.index.intersection(label.index)
        pred_eval = pred_out.loc[avail_idx].copy()
        label_eval = label.loc[avail_idx, label_col].dropna()
        pred_eval = pred_eval.loc[pred_eval.index.intersection(label_eval.index)]
        pred_eval["label"] = label_eval.loc[pred_eval.index].values

        if len(pred_eval) > 0:
            # 每日截面 Pearson IC
            daily_ic = pred_eval.groupby(level='DataDate').apply(
                lambda df: df['pred'].corr(df['label'], method='pearson')
            ).dropna()

            # 每日截面 Spearman RankIC
            daily_rankic = pred_eval.groupby(level='DataDate').apply(
                lambda df: df['pred'].corr(df['label'], method='spearman')
            ).dropna()

            # 汇总统计
            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std(ddof=1)
            rankic_mean = daily_rankic.mean()
            rankic_std = daily_rankic.std(ddof=1)

            ic_ir = ic_mean / ic_std if ic_std > 1e-8 else np.nan
            rankic_ir = rankic_mean / rankic_std if rankic_std > 1e-8 else np.nan

            ic_records = [{
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'IC_IR': ic_ir,
                'RankIC_mean': rankic_mean,
                'RankIC_std': rankic_std,
                'RankIC_IR': rankic_ir,
                'n_days': len(daily_ic),
                'best_iteration': model.best_iteration
            }]

            ic_df = pd.DataFrame(ic_records)
            ic_df.to_csv(feat_dir / "LGBM_IC_stats.csv", index=False)

            smart_log(f"测试集评估: IC={ic_mean:.4f} (IR={ic_ir:.3f}) | "
                      f"RankIC={rankic_mean:.4f} (RankIR={rankic_ir:.3f}) | "
                      f"n_days={len(daily_ic)} | best_iter={model.best_iteration}")
            smart_log(f"IC 统计已保存 -> {feat_dir / 'LGBM_IC_stats.csv'}")
        else:
            smart_log("警告: 预测与标签无交集，跳过 IC 评估")

    smart_log("全部任务完成")