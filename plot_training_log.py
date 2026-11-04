import re
import matplotlib.pyplot as plt
import numpy as np

# 解析日志文件
def parse_log(filepath):
    epochs = []
    train_losses = []
    valid_ics = []
    valid_icirs = []
    valid_cross_losses = []
    ic_grad_norms = []
    cross_grad_norms = []
    
    pattern = r'epoch\s+(\d+)\s+\|\s+train loss\s+([\d.]+)\s+\|\s+valid IC\s+(-?[\d.]+)\s+\|\s+ICIR\s+(-?[\d.]+)\s+\|\s+valid cross_loss\s+([\d.]+)\s+\|\s+IC grad norm\s+([\d.]+)\s+\|\s+Cross grad norm\s+([\d.]+)'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                valid_ics.append(float(match.group(3)))
                valid_icirs.append(float(match.group(4)))
                valid_cross_losses.append(float(match.group(5)))
                ic_grad_norms.append(float(match.group(6)))
                cross_grad_norms.append(float(match.group(7)))
    
    return epochs, train_losses, valid_ics, valid_icirs, valid_cross_losses, ic_grad_norms, cross_grad_norms

epochs, train_losses, valid_ics, valid_icirs, valid_cross_losses, ic_grad_norms, cross_grad_norms = parse_log('txtlog.log')

# 创建图表：一行3列
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 第一张：train_loss 和 valid IC（共享X轴）
ax1 = axes[0]
ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Train Loss & Valid IC')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(epochs, valid_ics, 'r-', linewidth=2, label='Valid IC')
ax1_twin.set_ylabel('Valid IC', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')

# 第二张：valid_icir 和 valid cross（共享X轴）
ax2 = axes[1]
ax2.plot(epochs, valid_icirs, 'g-', linewidth=2, label='Valid ICIR')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Valid ICIR', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_title('Valid ICIR & Valid Cross Loss')
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
ax2_twin.plot(epochs, valid_cross_losses, 'm-', linewidth=2, label='Valid Cross Loss')
ax2_twin.set_ylabel('Valid Cross Loss', color='m')
ax2_twin.tick_params(axis='y', labelcolor='m')

# 合并图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left')

# 第三张：IC grad norm 和 Cross grad norm 堆叠图
ax3 = axes[2]
ax3.stackplot(epochs, ic_grad_norms, cross_grad_norms, 
              labels=['IC Grad Norm', 'Cross Grad Norm'],
              colors=['#ff9999', '#66b3ff'], alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Grad Norm')
ax3.set_title('IC & Cross Grad Norm (Stacked)')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_log_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('training_log_analysis.pdf', bbox_inches='tight')
print("图表已保存为 training_log_analysis.png 和 training_log_analysis.pdf")