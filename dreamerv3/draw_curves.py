import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 文件路径 (保持不变)
# 'CartPole/logs/datas/CartPole, 1h, 1w6, avg 500/Train-Episode-Rewards_rank_0_env-0.csv'
file_path_0 = './logs/datas/CartPole, 1h, 1w6, avg 500/Train-Episode-Rewards_rank_0_env-0.csv'
file_path_1 = './logs/datas/CartPole, 1h, 1w6, avg 500/Train-Episode-Rewards_rank_0_env-1.csv'
file_path_2 = './logs/datas/CartPole, 1h, 1w6, avg 500/Train-Episode-Rewards_rank_0_env-2.csv'
file_path_3 = './logs/datas/CartPole, 1h, 1w6, avg 500/Train-Episode-Rewards_rank_0_env-3.csv'

file_paths = [file_path_0, file_path_1, file_path_2, file_path_3]

# 读取并合并 CSV 数据 (保持不变)
all_data = []
for path in file_paths:
    try:
        df = pd.read_csv(path)
        all_data.append(df)
    except FileNotFoundError:
        print(f"文件未找到: {path}")

if not all_data:
    print("没有找到任何 CSV 文件，程序结束。")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)
combined_df = combined_df[:230]

# 1. 计算滑动平均 (Rolling Average)
window_size = 100000 # 设置滑动窗口大小，可以根据需要调整 (没用)
combined_df['Smoothed_Value'] = combined_df.groupby('Step')['Value'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
#  解释:
#  - `combined_df.groupby('Step')['Value']`:  按 'Step' 列分组，并选择 'Value' 列。虽然这里按 Step 分组好像没必要，因为 step 本来就是递增的... 但如果你的 step 不是严格递增的，分组是必要的
#  - `.transform(lambda x: ...)`:  对每个分组应用 lambda 函数。
#  - `x.rolling(window=window_size, min_periods=1).mean()`:  对每个分组（也就是 Value 列的一系列值），计算窗口大小为 `window_size` 的滑动平均。 `min_periods=1` 表示即使窗口内的数据点少于 `window_size`，也进行计算 (对于数据点少于 window size 的起始部分)。

# 2. 绘制平滑后的阴影折线图
# plt.figure(figsize=(10, 6))

sns.lineplot(data=combined_df, x='Step', y='Smoothed_Value', estimator='mean', errorbar='sd') # y轴改为 'Smoothed_Value'

# plt.title('Smoothed DRL Training Rewards over Steps (Rolling Average)')
plt.xlabel('Step')
plt.ylabel('Rewards')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
