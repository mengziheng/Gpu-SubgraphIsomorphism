#测试"我们的代码"在"小图"上所有的"三角形"的时间与个数

import os
import subprocess
import re
import csv

folder_path = "/data/zh_dataset/graph_challenge_snapdata"
output_file = "../results.csv"  # 输出的CSV文件名

# 获取文件夹中的所有文件
file_names = os.listdir(folder_path)
# file_names = ["cit-HepPh_adj.mmio"]

# 创建空的结果列表
results = []

i = 0
# 遍历文件夹中的每个文件
for file_name in file_names:
    # 预处理文件名
    file_name = os.path.splitext(file_name)[0]
    adj_index = file_name.find("_adj")
    if adj_index != -1:
        file_name = file_name[:adj_index]

    print(file_name + " order : " + str(i))
    
    # 构造命令
    command = f"mpirun -n 1 ./subgraphmatch.bin {file_name} triangle 1 0.25 8 216 1024 10"

    # 使用正则表达式提取图的名称、时间和个数
    regex_pattern = r"graph : ([\w\-]+) time is : (\d+\.\d+) ms,count is : (\d+)"
    # 执行C++文件并捕获输出
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    # 逐行读取输出并提取信息
    for line in iter(process.stdout.readline, b''):
        line = line.decode().strip()
        match = re.search(regex_pattern, line)
        if match:
            graph_name = match.group(1)
            time = match.group(2)
            count = match.group(3)
            results.append([graph_name, time, count])
            print([graph_name, time, count])
    i = i + 1

# 将结果写入CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入表头
    writer.writerow(['File', 'Time', 'Triangle Count'])
    
    # 写入每行结果
    writer.writerows(results)

print("CSV文件已生成。")