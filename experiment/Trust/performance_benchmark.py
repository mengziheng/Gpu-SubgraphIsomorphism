import os
import subprocess
import re
import pandas as pd
import sys

folder_path = "/data/zh_dataset/TRUST_processed_graph_challenge_dataset/Synthetic"

# folder_path = sys.argv[1]
output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/Trust"
output_file = os.path.join(output_path, folder_path.split('/')[-1]+".xlsx")

# 获取文件夹中的所有文件
file_names = os.listdir(folder_path)
# file_names = ["flickrEdges_adj.mmio"]

# 创建空的结果列表
results = []

i = 0
# 遍历文件夹中的每个文件
for file_name in file_names:

    print(file_name + " order : " + str(i))
    dir_name = os.path.join(folder_path, file_name)

    # 构造命令
    command = f"mpirun -n 1 ./trianglecounting.bin {dir_name}/ 1 1024 1024 10"
    print(command)
    # 使用正则表达式提取图的名称、时间和个数
    regex_pattern = r"graph : ([\w\-\/\.]+) time is : (\d+\.\d+) ms,count is : (\d+)"
    # 执行C++文件并捕获输出
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    # 逐行读取输出并提取信息
    for line in iter(process.stdout.readline, b''):
        line = line.decode().strip()
        match = re.search(regex_pattern, line)
        if match:
            graph_name = match.group(1)
            graph_name = graph_name.replace(folder_path,"")
            graph_name = graph_name.replace("/","")
            time = match.group(2)
            count = match.group(3)
            results.append([graph_name, time, count])
            print([graph_name, time, count])
    i = i + 1
    # if(i == 2):
    #         break

# 创建DataFrame对象
df = pd.DataFrame(results, columns=["graph_name", "time", "count"])

# 对"Filename"列进行升序排序
df = df.sort_values(by="graph_name", ascending=True)

df.to_excel(output_file, index=False)

print("excel文件已生成。")