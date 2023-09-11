# get vertex and edge information from folder that contain all unprocessed graph

import os
import subprocess
import re
import pandas as pd
import sys

# folder_path = sys.argv[1]
folder_path = "/data/zh_dataset/graph_challenge_dataset/Synthetic"
output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/"
output_file = output_path + folder_path.split('/')[-1] + ".xlsx"

results = []

file_names = os.listdir(folder_path)

i = 0

for file_name in file_names:
    if file_name.endswith('.mmio'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            first_line = file.readline()
            values = first_line.split()
            vertex = values[0]
            edge = values[2]
            results.append([file_name, vertex, edge])
            print([file_name, vertex, edge])
        i = i + 1

df = pd.DataFrame(results, columns=["graph_name", "vertex", "edge"])

df = df.sort_values(by="graph_name", ascending=True)

df.to_excel(output_file, index=False)

print("excel has already been generated")