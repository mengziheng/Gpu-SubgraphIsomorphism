# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取Excel数据
# df = pd.read_excel('speedup.xlsx')

# # 提取横坐标和纵坐标数据
# x = df.columns[1:]  # 第一行作为横坐标
# print(x)
# y = df.mean()[0:]  # 每一列的平均值作为纵坐标
# print(y)

# # 绘制折线图
# plt.plot(x, y)
# plt.xlabel('chunksize')
# plt.ylabel('speedup')
# # plt.title('')
# plt.show()
# plt.savefig('line_plot.png')


import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel数据
df = pd.read_excel('speedup.xlsx')

# 提取横坐标和纵坐标数据
x = df.columns[1:]  # 第一行作为横坐标
print(x)
rows = [1,10,21,31,40]
y_values = df.iloc[rows, 1:]  # 每隔五行取值作为纵坐标
files = df.iloc[rows, 0].values  # 第一列为文件名

# 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图形大小

# 为每个文件绘制折线图
for i in range(len(files)):
    plt.plot(x, y_values.iloc[i], label=files[i])

plt.xlabel('Chunksize')
plt.ylabel('Speedup')
plt.legend()  # 显示图例

plt.savefig('line_plot.png')
plt.show()