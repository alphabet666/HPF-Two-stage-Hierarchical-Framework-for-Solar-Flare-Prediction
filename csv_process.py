import pandas as pd

# 读取CSV文件
data = pd.read_csv(r'D:\HPF\HPF\SMOTE.csv')

# 根据第一列进行分组
grouped_data = data.groupby(data.columns[0])

# 计算相同名字的第三列的均值和标准差
result = grouped_data[data.columns[2]].agg(['mean', 'std'])

# 打印结果
print(result)
