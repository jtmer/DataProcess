import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('dp_search_results.csv')

# 解析config列中的JSON字符串
df['config_dict'] = df['config'].apply(ast.literal_eval)
config_df = pd.json_normalize(df['config_dict'])

# 合并回原DataFrame
df = pd.concat([df, config_df], axis=1)

# 选择我们关心的超参数列
hyperparameter_columns = [
    'trimmer_seq_len',
    'aligner_mode',
    'inputer_detect_method',
    'warper_method',
    'normalizer_method',
    'clip_factor'
]

# 提取这些列并合并到主表中
df_analysis = df[['trial', 'mse', 'mae', 'smape', 'time_sec'] + hyperparameter_columns].copy()


# 绘制箱线图：不同trimmer_seq_len对MSE的影响
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_analysis, x='trimmer_seq_len', y='mse')
plt.title('MSE by Trimmer Sequence Length')
plt.xticks(rotation=45)
plt.show()

# 绘制箱线图：不同normalizer_method对MSE的影响
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_analysis, x='normalizer_method', y='mse')
plt.title('MSE by Normalizer Method')
plt.show()

# 绘制散点图：MSE vs. Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_analysis, x='time_sec', y='mse', hue='warper_method', style='normalizer_method', s=100)
plt.title('MSE vs. Time by Warper and Normalizer')
plt.show()

# 绘制热力图：超参数与MSE的相关性（数值型）
numeric_cols = ['trimmer_seq_len', 'mse', 'mae', 'smape', 'time_sec']
corr = df_analysis[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# 绘制多面板图：不同inputer_detect_method下的MSE分布
g = sns.FacetGrid(df_analysis, col='inputer_detect_method', height=4, aspect=1)
g.map(sns.histplot, 'mse', kde=True)
g.fig.suptitle('MSE Distribution by Inputer Detect Method', y=1.05)
plt.show()