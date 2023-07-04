import os

# 需手动写入图的路径
folder_path = "/data/zh_dataset/graph_challenge_otherdata/"


command = "g++ sort.cpp -o sort"
os.system(command)
command = "g++ fromDirectToUndirect.cpp -o fromDirectToUndirect"
os.system(command)

i = 0

file_names = os.listdir(folder_path)
for file_name in file_names:
    graph_path = os.path.join(folder_path, file_name)
    graph_name = os.path.splitext(file_name)[0]
    last_dot_index = graph_name.rfind(".")
    if last_dot_index != -1:
        graph_name = graph_name[:last_dot_index]
    else:
        graph_name = graph_name
    adj_index = graph_name.find("_adj")
    if adj_index != -1:
        graph_name = graph_name[:adj_index]
    # 需手动写入预处理后的数据的保存路径
    clique_folder_path = os.path.join("/data/zh_dataset/graph_preprocessed/clique_graph_preprocessed/", graph_name)
    dataforgeneral_folder_path = os.path.join("/data/zh_dataset/graph_preprocessed/generic_graph_preprocessed/", graph_name)
    os.makedirs(clique_folder_path, exist_ok=True)
    print("im" + clique_folder_path)
    os.makedirs(dataforgeneral_folder_path, exist_ok=True)
    command = f"./fromDirectToUndirect {file_name} {file_name}"
    os.system(command)
    command = f"./sort {file_name}"
    os.system(command)
    i = i + 1