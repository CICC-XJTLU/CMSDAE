# 作者:王勇
# 开发时间:2024/11/26 10:01
import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 Excel 文件
file_path = 'urban.xlsx'  # 文件路径
df = pd.read_excel(file_path)

# 提取数据
epochs = df['Epoch']
learning_rates = df.columns[1:]  # 获取所有 learning rate 列名

# 绘制图形
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    plt.plot(epochs, df[lr], label=lr, marker='o')  # 绘制每个 learning rate 的曲线

# 图表设置
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error across Different Learning Rates and Epochs')
plt.legend(title='Learning Rate')
plt.grid(True)
plt.tight_layout()

plt.savefig('urban.svg', format='svg')

# 显示图表
plt.show()