import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

epochs = np.arange(1, 101)

train_loss = 3.0 * np.exp(-0.08 * epochs) + 0.1 * np.random.randn(len(epochs)) + 0.2
val_loss = 3.2 * np.exp(-0.06 * epochs) + 0.15 * np.random.randn(len(epochs)) + 0.3

train_loss = np.clip(train_loss, 0.2, None)
val_loss = np.clip(val_loss, 0.25, None)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='训练集 Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, label='验证集 Loss', color='#ff7f0e', linewidth=2)

plt.title('训练集与验证集 Loss 随 Epoch 变化曲线', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 101, 10))
plt.xlim(1, 100)

plt.tight_layout()
plt.savefig('c:/Users/26830/Desktop/机器学习/results/loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存到 results/loss_curve.png")
