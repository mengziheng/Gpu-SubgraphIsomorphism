import re
import pandas as pd

# 文件路径
file_path = 'rps_result_big.txt'

# 读取文件内容
with open(file_path, 'r') as file:
    text = file.read()

    # 使用正则表达式匹配所有的结果
    pattern = r"\/data\/graph_challenge_bigdata\/([\w\-\/\.]+)\.txt:0\nprocess_level_time=(\d+\.\d+)ms\nmaterialize_time=\d+\.\d+ms\ntotal_match_count=(\d+)"
    matches = re.findall(pattern, text)

    # 存储所有匹配结果的列表
    results = []

    # 遍历每个匹配结果
    for match in matches:
        filename = match[0]
        process_level_time = float(match[1])
        total_match_count = int(match[2])

        # 添加到结果列表
        result = {'filename': filename, 'process_level_time': process_level_time, 'total_match_count': total_match_count}
        results.append(result)

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 将DataFrame写入Excel文件
    df.to_excel('results.xlsx', index=False)
    print("结果已保存到 results.xlsx 文件")
