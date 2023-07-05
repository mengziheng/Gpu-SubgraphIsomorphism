import os
import subprocess
import re
import csv
import sys


input_path = sys.argv[1]
output_path = "/home/zhmeng/GPU/Gpu-SubgraphIsomorphism/result/Trust"
output_file = os.path.join(output_path, input_path.split('/')[-1])
file_names = os.listdir(input_path)
results = []

# 执行
for root, dirs, files in os.walk(input_path):
    for dir_name in dirs:
        dir_name = os.path.join(input_path, dir_name)
        print(dir_name)
        command = f'mpirun -n 1 ./trianglecounting.bin {dir_name}/ 1 1024 108 32 0 0 '
        regex_pattern = r"graph : ([\w\-\/]+) time is : (\d+\.\d+) ms,count is : (\d+)"
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
        

# 将结果写入CSV文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['File', 'Time', 'Triangle Count'])
    # 写入每行结果
    writer.writerows(results)

print("CSV文件已生成。")