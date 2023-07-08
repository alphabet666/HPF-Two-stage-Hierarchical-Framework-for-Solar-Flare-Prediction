import pandas as pd
from scipy.stats import wilcoxon

# 读取CSV文件
data = pd.read_csv(r'C:\Users\Administrator\Desktop\base_hpf.csv',header=None)
print(data)

#获取需要进行Wilcoxon检验的列索引
col_indices1 = list(range(1, 8))   # 第2到8列的索引（从0开始）
col_indices2 = list(range(8, 15))  # 第9到15列的索引（从0开始）

# 执行Wilcoxon检验
results = []
for _, row in data.iterrows():
    values1 = row[col_indices1]
    values2 = row[col_indices2]
    statistic, p_value = wilcoxon(values1, values2)
    results.append((statistic, p_value))

# 将结果添加到原始数据的新列
data['Wilcoxon Statistic'] = [result[0] for result in results]
data['Wilcoxon P-value'] = [result[1] for result in results]
print(data)

# 保存结果为CSV文件
data.to_csv('wilcoxon_results.csv', index=False)
