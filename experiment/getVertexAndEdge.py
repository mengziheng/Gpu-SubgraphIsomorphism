import os
import subprocess
import re
import pandas as pd
import sys

folder_path = "/data/zh_dataset/graph_challenge_dataset/Synthetic"

# folder_path = sys.argv[1]
output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/"
output_file = output_path + "vertexAndEdgeCountForSynthetic.xlsx"

results = []

file_names = os.listdir(folder_path)

i = 0

# 遍历文件夹中的每个文件
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        first_line = file.readline()
        values = first_line.split()
        vertex = values[0]
        edge = values[2]
        results.append([file_name, vertex, edge])
        print([file_name, vertex, edge])
    i = i + 1
    # if(i == 2):
    #         break

# 创建DataFrame对象
df = pd.DataFrame(results, columns=["graph_name", "vertex", "edge"])

# 对"Filename"列进行升序排序
df = df.sort_values(by="graph_name", ascending=True)

df.to_excel(output_file, index=False)

print("excel文件已生成。")