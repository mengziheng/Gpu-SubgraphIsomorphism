# 用于对初始的图进行预处理，初始图的数据格式是.mmio或者.tsv格式
# 输入格式为if_dir,input_path,output_path

import os
import sys

input_path = sys.argv[1]
output_path = sys.argv[2] 
# input_path = "/data/zh_dataset/graph_challenge_dataset/snap"
# output_path = "/data/zh_dataset/processed_graph_challenge_dataset/snap"
if_dir = 1

command = "g++ sort.cpp -o sort"
os.system(command)
command = "g++ fromDirectToUndirect.cpp -o fromDirectToUndirect"
os.system(command)

i = 0


file_names = os.listdir(input_path)
for file_name in file_names:
        print(file_name)
        graph_path = os.path.join(input_path, file_name)
        graph_name = os.path.splitext(file_name)[0]
        last_dot_index = graph_name.rfind(".")
        if last_dot_index != -1:
            graph_name = graph_name[:last_dot_index]
        else:
            graph_name = graph_name
        adj_index = graph_name.find("_adj")
        if adj_index != -1:
            graph_name = graph_name[:adj_index]
        
        file_path = os.path.join(output_path, graph_name)
        if os.path.exists(file_path):
            print(file_path + "文件夹存在")
            continue
        os.makedirs(file_path, exist_ok=True)
        command = f"./fromDirectToUndirect {graph_path} {graph_path}"
        print(graph_path)
        os.system(command)
        command = f"./sort {graph_path} {output_path}"
        os.system(command)
        i = i + 1


