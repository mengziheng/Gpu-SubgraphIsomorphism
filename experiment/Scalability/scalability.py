import os
import subprocess
import re
import matplotlib.pyplot as plt

# 定义文件名列表和文件前缀
file_list_path = "graph_dataset.txt"
file_prefix = "/data/zh_dataset/processed_graph_challenge_dataset/snap/"

# 定义参数范围
param_range = range(7)

# 存储测试结果
results = []

# 从文件中逐行读取数据集文件名
with open(file_list_path, 'r') as file:
    file_names = file.read().splitlines()

# 遍历每个数据集
for file_name in file_names:
    # 构建文件路径
    file_path = os.path.join(file_prefix, file_name)

    # 存储当前数据集的测试结果
    dataset_results = []

    # 遍历每个参数
    for param in param_range:
        # 执行测试并获取结果时间
        command = f"mpirun -n {param} ./subgraphmatch.bin {file_path} triangle {param} 0.05 4 216 1024 10"
        regex_pattern = r"graph : ([\w\-\/]+) time is : (\d+\.\d+) ms,count is : (\d+)"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        # 逐行读取输出并提取信息
        for line in iter(process.stdout.readline, b''):
            line = line.decode().strip()
            match = re.search(regex_pattern, line)
            if match:
                result_time = match.group(2)
                count = match.group(3)
                results.append([file_name, result_time, count])
                print([file_name, result_time, count])
        # 存储结果时间
        dataset_results.append(result_time)

    # 存储当前数据集的结果
    results.append(dataset_results)

# 计算速度提升
speedup = []
baseline = results[0]  # 参数为0时的结果作为基准

for dataset_results in results:
    dataset_speedup = [dataset_results[0] / t for t in dataset_results]
    speedup.append(dataset_speedup)

# 绘制速度提升图
for i, dataset_speedup in enumerate(speedup):
    plt.plot(param_range, dataset_speedup, marker='o', label=file_names[i])

plt.xlabel('Parameter')
plt.ylabel('Speedup')
plt.title('Speedup vs Parameter')
plt.grid(True)
plt.legend()
plt.show()
