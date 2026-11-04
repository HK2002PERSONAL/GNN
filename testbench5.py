"""
训练日志解析与可视化工具
- 解析 training_logs.log 中3个随机种子的训练记录
- 绘制3张并列子图：左轴 Train Loss，右轴 Valid IC
- 底部显示损失函数拆分出来的范数均值表格
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_training_log(log_path):
    """
    解析训练日志文件，按随机种子分组提取数据
    
    Returns:
        dict: {seed: {'epochs': [], 'train_loss': [], 'valid_ic': [], 
                      'ic_grad_norm': [], 'temporal_grad_norm': []}}
    """
    seed_data = {}
    current_seed = None
    
    # 正则表达式匹配
    seed_pattern = re.compile(r'^training (\d+)')
    epoch_pattern = re.compile(
        r'^epoch\s+(\d+)\s*\|\s*train loss\s+(-?[\d.]+)\s*\|\s*valid IC\s+(-?[\d.]+)\s*\|\s*'
        r'ICIR\s+(-?[\d.]+)\s*\|\s*IC grad norm\s+([\d.]+)\s*\|\s*'
        r'Temporal grad norm\s+([\d.]+)'
    )
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 匹配随机种子行
            seed_match = seed_pattern.match(line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                seed_data[current_seed] = {
                    'epochs': [],
                    'train_loss': [],
                    'valid_ic': [],
                    'ic_grad_norm': [],
                    'temporal_grad_norm': []
                }
                continue
            
            # 匹配epoch数据行（支持负数损失值）
            epoch_match = epoch_pattern.match(line)
            if epoch_match and current_seed is not None:
                data = seed_data[current_seed]
                data['epochs'].append(int(epoch_match.group(1)))
                data['train_loss'].append(float(epoch_match.group(2)))
                data['valid_ic'].append(float(epoch_match.group(3)))
                # group(4) 是 ICIR，跳过
                data['ic_grad_norm'].append(float(epoch_match.group(5)))
                data['temporal_grad_norm'].append(float(epoch_match.group(6)))
    
    return seed_data


def compute_norm_means(seed_data):
    """
    计算每个随机种子的范数均值
    
    Returns:
        list: 表格行数据 [[seed, ic_mean, temporal_mean, total_mean], ...]
    """
    table_rows = []
    for seed, data in seed_data.items():
        ic_mean = np.mean(data['ic_grad_norm'])
        temporal_mean = np.mean(data['temporal_grad_norm'])
        total_mean = ic_mean + temporal_mean
        table_rows.append([seed, ic_mean, temporal_mean, total_mean])
    return table_rows


def plot_training_logs(seed_data, save_path='training_logs_analysis.png'):
    """
    绘制3个随机种子的并列训练日志图
    - 每张图左轴: Train Loss
    - 每张图右轴: Valid IC
    - 底部: 范数均值表格
    """
    seeds = list(seed_data.keys())
    n_seeds = len(seeds)
    
    # 创建图形: 上方是子图，下方是表格
    fig = plt.figure(figsize=(6 * n_seeds, 6))
    
    # 使用 GridSpec: 上面放子图，下面放表格
    gs = fig.add_gridspec(2, n_seeds, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    
    # 绘制每个随机种子的子图
    colors_loss = ['#1f77b4', '#2ca02c', '#9467bd']
    colors_ic = ['#d62728', '#ff7f0e', '#8c564b']
    
    for idx, seed in enumerate(seeds):
        ax1 = fig.add_subplot(gs[0, idx])
        data = seed_data[seed]
        
        # 左轴: Train Loss
        color_loss = colors_loss[idx % len(colors_loss)]
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color_loss)
        ax1.plot(data['epochs'], data['train_loss'], color=color_loss, 
                 linewidth=2, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.set_title(f'Seed {seed}', fontsize=12, fontweight='bold')
        
        # 右轴: Valid IC
        ax2 = ax1.twinx()
        color_ic = colors_ic[idx % len(colors_ic)]
        ax2.set_ylabel('Valid IC', color=color_ic)
        ax2.plot(data['epochs'], data['valid_ic'], color=color_ic, 
                 linewidth=2, linestyle='--', label='Valid IC')
        ax2.tick_params(axis='y', labelcolor=color_ic)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
    
    # 底部: 范数均值表格
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    ax_table.set_title('Gradient Norm Means by Seed', fontsize=11, pad=10)
    
    # 计算范数均值
    table_rows = compute_norm_means(seed_data)
    
    # 构建表格
    col_labels = ['Seed', 'IC Grad Norm', 'Temporal Grad Norm', 'Total']
    cell_text = []
    for row in table_rows:
        cell_text.append([
            str(row[0]),
            f'{row[1]:.4f}',
            f'{row[2]:.4f}',
            f'{row[3]:.4f}'
        ])
    
    # 添加均值行
    mean_row = [
        'Mean',
        f'{np.mean([r[1] for r in table_rows]):.4f}',
        f'{np.mean([r[2] for r in table_rows]):.4f}',
        f'{np.mean([r[3] for r in table_rows]):.4f}'
    ]
    cell_text.append(mean_row)
    
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 设置表头样式
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # 设置最后一行（均值行）样式
    for j in range(len(col_labels)):
        table[(len(cell_text), j)].set_facecolor('#D9E2F3')
        table[(len(cell_text), j)].set_text_props(fontweight='bold')
    
    plt.suptitle('Training Logs Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图表已保存至: {save_path}")


def main():
    log_path = 'training_logs.log'
    
    # 解析日志
    seed_data = parse_training_log(log_path)
    
    print(f"解析到 {len(seed_data)} 个随机种子: {list(seed_data.keys())}")
    for seed, data in seed_data.items():
        print(f"  Seed {seed}: {len(data['epochs'])} epochs")
    
    # 绘制图表
    plot_training_logs(seed_data)


if __name__ == '__main__':
    main()