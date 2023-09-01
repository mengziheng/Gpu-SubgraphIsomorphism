import re
from openpyxl import Workbook

# 创建一个Workbook对象
workbook = Workbook()

# 获取默认的工作表
worksheet = workbook.active

import csv

# 定义正则表达式模式
pattern = r'\/.*?\/(.*?)\.mmio:.*?=(.*?) ms'

# 打开文本文件并逐行读取
with open('gunrock_result_tri_big.txt', 'r') as file:
    lines = file.readlines()

# 遍历每一行，提取信息并写入Excel表格
for i in range(len(lines)):
    line = lines[i]
    if ":0" in line:
        last_slash_index = line.rfind("/")
        after_last_slash = line[last_slash_index + 1:]

        # 提取".mmio"之前的内容
        dot_mmio_index = after_last_slash.find(".mmio")
        graph_name = after_last_slash[:dot_mmio_index]

        materialize_time_line = lines[i + 1]
        materialize_time = materialize_time_line.split("elapsed_time=")[1].strip().split("ms")[0]
        # 在Excel表格中写入提取的数据
        worksheet.append([graph_name, materialize_time])

# 保存Excel文件
workbook.save('output.xlsx')
