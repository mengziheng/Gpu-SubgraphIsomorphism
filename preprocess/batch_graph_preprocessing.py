import os

folder_path = "/data/zh_dataset/graph_challenge_data"


file_names = os.listdir(folder_path)

for file_name in file_names:
    graph_path = os.path.join(folder_path, file_name)
    graph_name = os.path.splitext(file_name)[0]
    clique_folder_path = os.path.join("/data/zh_dataset/dataforclique", graph_name)
    dataforgeneral_folder_path = os.path.join("/data/zh_dataset/dataforgeneral", graph_name)
    os.makedirs(clique_folder_path, exist_ok=True)
    os.makedirs(dataforgeneral_folder_path, exist_ok=True)
    command = f"./fromDirectToUndirect {graph_name} {graph_name}"
    os.system(command)
    command = f"./sort {graph_name}"
    os.system(command)