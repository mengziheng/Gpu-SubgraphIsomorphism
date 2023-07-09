import os
import subprocess
import re
import pandas as pd
import sys

# input_path = "/data/zh_dataset/Hindex_processed_graph_challenge_dataset/snap"
input_path = "/data/zh_dataset/Hindex_processed_graph_challenge_dataset/Synthetic"

# input_path = sys.argv[1]
output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/Hindex"
another_output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/TC.result"
output_file = os.path.join(output_path, input_path.split('/')[-1]+".xlsx")
print(output_file)
file_names = os.listdir(input_path)
results = []
i = 3
num_iterations = 10

# 执行
for root, dirs, files in os.walk(input_path):
    for dir_name in dirs:
        dir_name = os.path.join(input_path, dir_name)
        print(dir_name)
        command = f'mpirun -n 1 ./trianglecounting.bin {dir_name}/ 1 1024 1024 32 0 0 '
        regex_pattern = r"graph : ([\w\-\/\.]+) time is : (\d+\.\d+) ms,count is : (\d+)"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        # 逐行读取输出并提取信息
        for line in iter(process.stdout.readline, b''):
            line = line.decode().strip()
            match = re.search(regex_pattern, line)
            if match:
                graph_name = match.group(1)
                graph_name = graph_name.replace(input_path,"")
                graph_name = graph_name.replace("/","")
                time = match.group(2)
                count = match.group(3)
                results.append([graph_name, time, count])
                print([graph_name, time, count])
        i = i + 1
        # if(i == 2):
        #     break
        

# 创建DataFrame对象
df = pd.DataFrame(results, columns=["graph_name", "time", "count"])

# 对"Filename"列进行升序排序
df = df.sort_values(by="graph_name", ascending=True)

df.to_excel(output_file, index=False)

print("excel文件已生成。")