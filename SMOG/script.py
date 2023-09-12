import sys
import os
import argparse


def generate_header_file(i, intersection_size, restriction_size,
                         intersection_orders, intersection_offset, restriction,
                         reuse, *args):
    header_file = "constants.h"
    with open(header_file, "w") as f:
        f.write("#ifndef CONSTANTS_HEADER\n#define CONSTANTS_HEADER\n")
        f.write(f"#define H {i}\n")
        for arg in args:
            f.write(f"#define {arg}\n")
        f.write(
            "extern int intersection_size;\nextern int restriction_size;\nextern int *intersection_orders;\nextern int *intersection_offset;\nextern int *restriction;\nextern int *reuse;\n#endif"
        )

    header_file = "constants.cpp"
    with open(header_file, "w") as f:
        f.write("#include\"constants.h\"\n\n")
        f.write(f"int intersection_size = {intersection_size};\n")
        f.write(f"int restriction_size = {restriction_size};\n")
        f.write(
            f"int *intersection_orders = new int[{intersection_size}]{intersection_orders};\n"
        )
        f.write(
            f"int *intersection_offset = new int[{i}]{intersection_offset};\n")
        f.write(
            f"int *restriction = new int[{restriction_size}]{restriction};\n")
        f.write(f"int *reuse = new int[{i}]{reuse};\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument('--input_graph_folder', type=str, default="")
    parser.add_argument('--input_pattern', type=str, default='default_value2')
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--load_factor', type=float, default=0.1)
    parser.add_argument(
        '--bucket_size',
        type=int,
        default=8,
    )
    parser.add_argument('--block_number_for_kernel', type=int, default=216)
    parser.add_argument('--block_size_for_kernel', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=10)

    # 解析命令行参数
    args = parser.parse_args()

    # 检查graph和pattern是否被提供
    if 'input_graph_folder' not in args:
        parser.error("--input_graph_folder is required.")
    if 'input_pattern' not in args:
        parser.error("--input_pattern is required.")

    input_graph_folder = args.input_graph_folder
    input_pattern = args.input_pattern
    N = args.N  # 记录并行处理的GPU数量
    load_factor = args.load_factor
    bucket_size = args.bucket_size
    block_number_for_kernel = args.block_number_for_kernel
    block_size_for_kernel = args.block_size_for_kernel
    chunk_size = args.chunk_size

    intersection_size = 1
    restriction_size = 1
    intersection_orders = ""
    intersection_offset = ""
    restriction = ""
    reuse = ""

    # 根据输入的pattern，自动改变参数
    match input_pattern:
        case "Q0":
            i = 3
            defs = []
            intersection_size = 4
            intersection_orders = "{0, 0, 1}"
            intersection_offset = "{0, 1, 3}"
            reuse = "{-1, -1, -1}"
        case "Q1":
            i = 4
            defs = ["withRestriction"]
            intersection_size = 5
            restriction_size = 5
            intersection_orders = "{0, 0, 1, 2}"
            intersection_offset = "{0, 1, 2, 4}"
            restriction = "{-1, 0, 1, 0}"
        case "Q2":
            i = 4
            defs = ["withRestriction"]
            intersection_size = 5
            restriction_size = 4
            intersection_orders = "{0, 0, 1, 1, 2}"
            intersection_offset = "{0, 1, 3, 5}"
            restriction = "{-1, -1, 1, 0}"
        case "Q3":
            i = 4
            defs = []
            intersection_size = 6
            intersection_orders = "{0, 0, 1, -1, 2}"
            intersection_offset = "{0, 1, 3, 5}"
            reuse = "{-1, -1, -1, 2}"
        case "Q4":
            i = 5
            defs = ["withRestriction", "withDuplicate"]
            intersection_size = 8
            restriction_size = 5
            intersection_orders = "{0, 0, 1, -1, 2, -1}"
            intersection_offset = "{0, 1, 3, 5, 6}"
            restriction = "{-1, 0, -1, 2, -1}"
            reuse = "{-1, -1, -1, 2, 2}"
        case "Q5":
            i = 5
            defs = []
            intersection_size = 7
            intersection_orders = "{0, 0, 1, -1, 2, -1, 3}"
            intersection_offset = "{0, 1, 3, 5, 7}"
            reuse = "{-1, -1, -1, 2, 3}"
        case "Q6":
            i = 5
            defs = ["withRestriction", "withDuplicate"]
            intersection_size = 8
            restriction_size = 5
            intersection_orders = "{0, 0, 1, -1, 2, 3}"
            intersection_offset = "{0, 1, 3, 4, 6}"
            restriction = "{-1, 0, -1, 2, -1}"
            reuse = "{-1, -1, -1, 2, -1}"
        # case "Q7":
        #     i = 4
        #     args = []
        # case "Q8":
        #     i = 6
        #     defs = ["withRestriction", "withDuplicate"]
        #     intersection_size = 9
        #     restriction_size = 6
        #     intersection_orders = "{0, 0, 1, 2, 0, 3, 1, 3, 4}"
        #     intersection_offset = "{0, 1, 3, 4, 6, 9}"
        #     restriction = "{-1, 0, -1, 2, -1, 4}"
        # case "Q9":
        #     i = 5
        #     defs = ["withRestriction", "withDuplicate"]
        #     intersection_size = 7
        #     restriction_size = 6
        #     intersection_orders = "{0, 0, 1, 2, 0, 1, 4}"
        #     intersection_offset = "{0, 1, 2, 4, 5, 7}"
        #     restriction = "{-1, 0, -1, -1, 2, -1}"
        # case "Q10":
        #     i = 5
        #     defs = ["withRestriction", "withDuplicate"]
        #     intersection_size = 8
        #     restriction_size = 5
        #     intersection_orders = "{0, 0, 0, 1, 2, 0, 1, 4}"
        #     intersection_offset = "{0, 1, 2, 3, 5, 6, 9}"
        #     restriction = "{-1, -1, 1, -1, 3}"
        #     reuse = "{-1, -1, -1, -1, 3}"
        # case "Q11":
        #     i = 5
        #     defs = ["withRestriction", "withDuplicate"]
        #     intersection_size = 8
        #     restriction_size = 5
        #     intersection_orders = "{0, 0, 1, 2, -1, 3}"
        #     intersection_offset = "{0, 1, 2, 4, 6}"
        #     restriction = "{-1, -1, 1, -1, 3}"
        #     reuse = "{-1, -1, -1, -1, 3}"
        # case "Q12":
        #     i = 5
        #     defs = ["withRestriction", "withDuplicate"]
        #     intersection_size = 8
        #     restriction_size = 5
        #     intersection_orders = "{0, 0, 1, 2, -1, 3}"
        #     intersection_offset = "{0, 1, 2, 4, 6}"
        #     restriction = "{-1, -1, 1, -1, 3}"
        #     reuse = "{-1, -1, -1, -1, 3}"

    generate_header_file(i, intersection_size, restriction_size,
                         intersection_orders, intersection_offset, restriction,
                         reuse, *defs)

    command = f"make"
    os.system(command)
    command = f"mpirun -n {N} ./subgraphmatch.bin {input_graph_folder} {input_pattern} {N} {load_factor} {bucket_size} {block_number_for_kernel} {block_size_for_kernel} {chunk_size}"
    # command = f"mpirun -n {N} compute-sanitizer ./subgraphmatch.bin {input_graph_folder} {input_pattern} {N} {load_factor} {bucket_size} {block_number_for_kernel} {block_size_for_kernel} {chunk_size}"
    os.system(command)
