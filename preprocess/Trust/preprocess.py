import os
import subprocess
import re
import csv
import sys

# input_path ： 包含全部需要预处理的图的文件夹路径
# output_file ： 包含全部预处理完的图的文件夹路径

# input_path = "/data/zh_dataset/graph_challenge_dataset/snap"
# output_path = "/data/zh_dataset/TRUST_processed_graph_challenge_dataset/snap"

input_path = sys.argv[1]
output_path = sys.argv[2] 
file_names = os.listdir(input_path)

command = f'g++ sort.cpp -o sort'
os.system(command)

i = 0
# 预处理
for file_name in file_names:
        file_path = os.path.join(input_path, file_name)
        file_name = file_name.rsplit('.', 1)[0]
        file_name = file_name.rstrip('_adj')
        output_file_path = os.path.join(output_path, file_name)
        print("processing : " + file_name)
        command = f'sudo chmod 777 {file_path}'
        os.system(command)
        command = f'./fromDirectToUndirect {file_path}'
        if not os.path.exists(output_file_path):
                os.system(command)
                command = f'mkdir {output_file_path}'
        os.system(command)
        command = f'./sort 1.mmio {output_file_path}'
        os.system(command)
        command = f'rm 1.mmio'
        os.system(command)

                


