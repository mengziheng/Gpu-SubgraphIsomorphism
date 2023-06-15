#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <cooperative_groups.h>
using namespace cooperative_groups;

using namespace std;

__device__ int getNewVertex(int wid, int cur_vertex, int stride, int vertex_count)
{
    if (cur_vertex == -1)
        return wid;
    if (cur_vertex + stride < vertex_count)
        return cur_vertex + stride;
    else
        return -1;
}

// h : height of subtree; h = pattern vertex number
__global__ void DFSKernel(int vertex_count, int edge_count, int max_degree, int h, int parameter, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *ir, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int lid = tid % 32;                              // landid
    int level = 1;                                   // level of subtree,start from 0,not root for tree,but root for subtree
    int warp_sum = 0;                                // 记录一下每个warp记录得到的数量
    int cur_vertex = -1;
    int stride = blockDim.x * gridDim.x / 32;
    int *ir_number = new int[h]; // 记录一下每一层保存的数据大小

    // 初始化

    int *buffer = new int[h]; // 记录每次的中间结果

    // each warp process a subtree (an probe item)

    while (true)
    {
        // 当前层为空
        if (ir_number[level] == 0)
        {
            if (level == 1)
            {
                cur_vertex = getNewVertex(wid, cur_vertex, stride, vertex_count);
                if (cur_vertex == -1)
                    break;
                // 选择了一个新的节点作为probe item，作为初始节点,需要初始化ir_number
                for (int i = 0; i < h; i++)
                    ir_number[i] = 0;
                buffer[level - 1] = cur_vertex;
            }

            // 计算需要做交集的元素即其邻居节点数目
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_end = intersection_offset[level];
            int intersection_order_length = intersection_order_end - intersection_order_start;
            int *intersection_order = new int[intersection_order_length]; // wzb: refine it
            for (int i = 0; i < intersection_order_length; i++)
                intersection_order[i] = intersection_orders[intersection_order_start + i];
            int *neighbour_numbers = new int[intersection_order_length];
            for (int i = 0; i < intersection_order_length; i++)
            {
                neighbour_numbers[i] = csr_row_value[buffer[intersection_order[i]]];
            }

            // 只有一个元素，不需要输出，只需要写入中间结果。
            if (intersection_order_length == 1)
            {
                if (level == h - 1)
                {
                    warp_sum = warp_sum + neighbour_numbers[0];
                    level--;
                    continue;
                }
                else
                {
                    for (int i = lid; i < neighbour_numbers[0]; i += 32)
                    {
                        ir[wid * max_degree + (level - 1) * 1024 * 216 / 32 * max_degree + i] = csr_column_index[csr_row_offset[buffer[intersection_order[0]]] + i];
                    }
                    if (neighbour_numbers[0] == 0)
                    {
                        if (level > 1)
                            level--;
                        else
                            ir_number[level] = 0;
                        continue;
                    }
                }
                ir_number[level] = neighbour_numbers[0];
                buffer[level] = ir[wid * max_degree + (level - 1) * 1024 * 216 / 32 * max_degree + ir_number[level] - 1];
                continue;
            }
            // 不止一个元素，需要做交集
            else
            {
                // 用一个线程去完成冒泡排序，记录排序后的顶点顺序，即最终的intersection顺序，这需要修改，这不是并行的了
                int min;
                int min_index;
                for (int i = 0; i < intersection_order_length; i++)
                {
                    min = neighbour_numbers[i];
                    min_index = i;
                    for (int j = i + 1; j < intersection_order_length; j++)
                    {
                        if (neighbour_numbers[j] < min)
                        {
                            min = neighbour_numbers[j];
                            min_index = j;
                        }
                    }
                    int tmp = neighbour_numbers[i];
                    neighbour_numbers[i] = neighbour_numbers[min_index];
                    neighbour_numbers[min_index] = tmp;
                    tmp = intersection_order[i];
                    intersection_order[i] = intersection_order[min_index];
                    intersection_order[min_index] = tmp;
                }

                // 开始按intersection_order做join操作
                // 首先取第一个intersection_order的顶点的邻居，每个顶点平均分配这些邻居节点,用local_meemory保存
                int cur_vertex = buffer[intersection_order[0]];
                int neighbor_num = neighbour_numbers[0];
                if (neighbor_num == 0)
                {
                    level--;
                    continue;
                }
                int thread_cache_size = neighbor_num / 32; // wzb: can not maintain in register, reuse buffer
                int remainder = neighbor_num % 32;
                // 判断是否需要向上取整
                if (remainder > 0)
                {
                    thread_cache_size += 1;
                }
                int *thread_cache = new int[thread_cache_size];
                // 初始化cache
                for (int i = 0; i < thread_cache_size; i++)
                    thread_cache[i] = -1;
                int index_for_thread_cache = 0;
                // 初始化cache
                for (int i = lid; i < neighbor_num; i += 32)
                {
                    thread_cache[index_for_thread_cache] = csr_column_index[csr_row_offset[cur_vertex] + i];
                    index_for_thread_cache++;
                }
                // 对所有的邻居集合都要进行一次intersection，因此需要for循环
                // 如果有三个顶点，那最终只需要intersection两次。而最后一次需要写回，因此是3-2。这解释了下面为什么-2
                int cur_order_index;
                for (cur_order_index = 1; cur_order_index < intersection_order_length - 2; cur_order_index++)
                {
                    // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                    for (int i = 0; i < thread_cache_size; i++)
                    {
                        int k = 0;
                        if (thread_cache[i] == -1)
                            continue;
                        int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                        int hash_tables_start = hash_tables_offset[buffer[intersection_order[cur_order_index]]];
                        int *cmp = &hash_tables[hash_tables_start + value + edge_count]; // wzb: remove edge count
                        printf("%d\n", *cmp);
                        while (*cmp != -1)
                        {
                            if (*cmp == thread_cache[i])
                            {
                                break;
                            }
                            if (*cmp == -1)
                                thread_cache[i] = -1;
                        }
                    }
                }
                // 对最后一个顶点进行交集操作之后，就可以直接写回了。
                // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                // 这边可以再优化一下，每次存储的值都写到数组的前几位，这样可以减少循环。
                if (cur_order_index != intersection_order_length)
                {
                    int ptr = 0;
                    for (int i = 0; i < thread_cache_size; i++)
                    {
                        if (thread_cache[i] == -1)
                            continue;
                        int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                        int hash_tables_start = hash_tables_offset[buffer[intersection_order[cur_order_index]]];
                        int *cmp = &hash_tables[hash_tables_start + value + edge_count];
                        while (*cmp != -1)
                        {
                            if (*cmp == thread_cache[i])
                            {
                                if (level == h - 1)
                                {
                                    // 最后一层，直接输出好了，不过目前实现的还是计数
                                    coalesced_group active = coalesced_threads();
                                    warp_sum = warp_sum + active.size();
                                }
                                else
                                {
                                    coalesced_group active = coalesced_threads();
                                    ir[active.thread_rank() + ptr + wid * max_degree + level * 1024 * 216 / 32 * max_degree] = thread_cache[index_for_thread_cache];
                                    ptr = ptr + active.size();
                                }
                            }
                            cmp = cmp + edge_count;
                            if (*cmp == -1)
                                thread_cache[i] = -1;
                        }
                    }
                    if (level == h - 1)
                        level--;
                    else
                    {
                        if (ptr == 0)
                        {
                            if (level > 1)
                                level = level - 1;
                            else
                                ir_number[level] = ptr;
                        }
                        continue;
                    }
                }
                free(thread_cache);
            }
            free(intersection_order);
            free(neighbour_numbers);
        }
        // 当前层不为空，取下一个元素
        else
        {
            ir_number[level] = ir_number[level] - 1;
        }
        // 更新临时结果
        buffer[level] = ir[(level - 1) * 1024 * 216 / 32 * max_degree + wid * max_degree + ir_number[level] - 1];
        level++;
        ir_number[level] = 0;
    }
    if (lid == 1)
    {
        atomicAdd(sum, warp_sum);
    }
    free(ir_number);
    free(buffer);
}

__global__ void DFSKernel(int vertex_count, int edge_count, int max_degree, int h, int parameter, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *ir, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int lid = tid % 32;                              // landid
    int level = 1;                                   // level of subtree,start from 0,not root for tree,but root for subtree
    int warp_sum = 0;                                // 记录一下每个warp记录得到的数量
    int stride = blockDim.x * gridDim.x / 32;
    int *ir_number = new int[h]; // 记录一下每一层保存的数据大小

    // 初始化
    for (int i = 0; i < h; i++)
        ir_number[i] = 0;
    int *buffer = new int[h]; // 记录每次的中间结果

    // each warp process a subtree (an probe item)
    for (int first_vertex = wid; first_vertex < vertex_count; first_vertex += stride) // wzb: change to dynamic load
    {
        buffer[0] = first_vertex;
        ir_number[0] = 1;
        while (true)
        {
            // 不是最后一层，将中间结果写回global memory。不过也可以先写回share memory，再写回global memory，分层次写
            if (ir_number[level] == 0)
            {
                // 这里是每一个线程去执行一个用hash tabel进行交集的运算
                // 用buffer存储当前的一个中间结果
                // 首先对buffer[]按照neighbour排序，按照neighbour从小到大顺序进行交集。
                // 是否可以用warp去优化一下这个排序
                // 用一个提前给定的结构给出每次做交集的点的索引。
                // printf("level : %d\n", level);
                int intersection_order_start = intersection_offset[level - 1];
                int intersection_order_end = intersection_offset[level];
                int intersection_order_length = intersection_order_end - intersection_order_start;
                int *intersection_order = new int[intersection_order_length]; // wzb: refine it

                for (int i = 0; i < intersection_order_length; i++)
                    intersection_order[i] = intersection_orders[intersection_order_start + i];

                int *neighbour_numbers = new int[intersection_order_length];
                // 将需要做交集的点的邻居数量都读入到寄存器中
                for (int i = 0; i < intersection_order_length; i++)
                {
                    neighbour_numbers[i] = csr_row_value[buffer[intersection_order[i]]];
                }

                // 不需要交集，直接写回
                if (intersection_order_length == 1)
                {
                    for (int i = lid; i < neighbour_numbers[0]; i += 32)
                    {
                        ir[wid * max_degree + (level - 1) * 1024 * 216 / 32 * max_degree + i] = csr_column_index[csr_row_offset[buffer[intersection_order[0]]] + i];
                    }
                    ir_number[level] = neighbour_numbers[0];
                    buffer[level] = ir[wid * max_degree + (level - 1) * 1024 * 216 / 32 * max_degree + ir_number[level] - 1];
                    continue;
                }
                // 需要做交集
                else
                {
                    // 用一个线程去完成冒泡排序，记录排序后的顶点顺序，即最终的intersection顺序
                    // 这需要修改，这不是并行的了
                    int min;
                    int min_index;
                    for (int i = 0; i < intersection_order_length; i++)
                    {
                        min = neighbour_numbers[i];
                        min_index = i;
                        for (int j = i + 1; j < intersection_order_length; j++)
                        {
                            if (neighbour_numbers[j] < min)
                            {
                                min = neighbour_numbers[j];
                                min_index = j;
                            }
                        }
                        int tmp = neighbour_numbers[i];
                        neighbour_numbers[i] = neighbour_numbers[min_index];
                        neighbour_numbers[min_index] = tmp;
                        tmp = intersection_order[i];
                        intersection_order[i] = intersection_order[min_index];
                        intersection_order[min_index] = tmp;
                    }

                    // 开始按intersection_order做join操作
                    // 首先取第一个intersection_order的顶点的邻居，每个顶点平均分配这些邻居节点,用local_meemory保存
                    int cur_vertex = buffer[intersection_order[0]];
                    int neighbor_num = neighbour_numbers[0];
                    if (neighbor_num == 0)
                    {
                        level--;
                        continue;
                    }
                    int thread_cache_size = neighbor_num / 32; // wzb: can not maintain in register, reuse buffer
                    int remainder = neighbor_num % 32;
                    // 判断是否需要向上取整
                    if (remainder > 0)
                    {
                        thread_cache_size += 1;
                    }
                    int *thread_cache = new int[thread_cache_size];
                    // 初始化cache
                    for (int i = 0; i < thread_cache_size; i++)
                        thread_cache[i] = -1;
                    int index_for_thread_cache = 0;
                    // 初始化cache
                    for (int i = lid; i < neighbor_num; i += 32)
                    {
                        thread_cache[index_for_thread_cache] = csr_column_index[csr_row_offset[cur_vertex] + i];
                        index_for_thread_cache++;
                    }
                    // 对所有的邻居集合都要进行一次intersection，因此需要for循环
                    // 如果有三个顶点，那最终只需要intersection两次。而最后一次需要写回，因此是3-2。这解释了下面为什么-2
                    int cur_order_index;
                    for (cur_order_index = 1; cur_order_index < intersection_order_length - 2; cur_order_index++)
                    {
                        // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                        for (int i = 0; i < thread_cache_size; i++)
                        {
                            int k = 0;
                            if (thread_cache[i] == -1)
                                continue;
                            int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                            int hash_tables_start = hash_tables_offset[buffer[intersection_order[cur_order_index]]];
                            int *cmp = &hash_tables[hash_tables_start + value + edge_count]; // wzb: remove edge count
                            printf("%d\n", *cmp);
                            while (*cmp != -1)
                            {
                                if (tid == 0)
                                    printf("%d\n", *cmp);
                                if (*cmp == thread_cache[i])
                                {
                                    break;
                                }

                                // cmp = cmp + edge_count;
                                if (*cmp == -1)
                                    thread_cache[i] = -1;
                            }
                        }
                    }
                    // 对最后一个顶点进行交集操作之后，就可以直接写回了。
                    // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                    // 这边可以再优化一下，每次存储的值都写到数组的前几位，这样可以减少循环。
                    if (cur_order_index != intersection_order_length)
                    {
                        int ptr = 0;
                        for (int i = 0; i < thread_cache_size; i++)
                        {
                            if (thread_cache[i] == -1)
                                continue;
                            int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                            int hash_tables_start = hash_tables_offset[buffer[intersection_order[cur_order_index]]];
                            int *cmp = &hash_tables[hash_tables_start + value + edge_count];
                            while (*cmp != -1)
                            {
                                if (*cmp == thread_cache[i])
                                {
                                    if (level != h - 1)
                                    {
                                        coalesced_group active = coalesced_threads();
                                        ir[active.thread_rank() + ptr + wid * max_degree + level * 1024 * 216 / 32 * max_degree] = thread_cache[index_for_thread_cache];
                                        ptr = ptr + active.size();
                                    }
                                    else
                                    {
                                        // 最后一层，直接输出好了，不过目前实现的还是计数
                                        coalesced_group active = coalesced_threads();
                                        warp_sum = warp_sum + active.size();
                                    }
                                }
                                cmp = cmp + edge_count;
                                if (*cmp == -1)
                                    thread_cache[i] = -1;
                            }
                        }
                        ir_number[level] = ptr;
                        if (ptr == 0)
                            if (level > 0)
                                level = level - 1;
                            else
                                break;
                        continue;
                    }
                    free(intersection_order);
                    free(neighbour_numbers);
                    free(thread_cache);
                }
            }
            // 当前层不为空，取下一个元素
            else
            {
                if (lid == 0)
                    ir_number[level] = ir_number[level] - 1;
            }
            // 更新临时结果
            buffer[level] = ir[(level - 1) * 1024 * 216 / 32 * max_degree + wid * max_degree + ir_number[level] - 1];
            level++;
        }
        if (lid == 1)
        {
            atomicAdd(sum, warp_sum);
            break;
        }
    }
    free(ir_number);
    free(buffer);
    // }
}

int main()
{
    unsigned int N = 1024;
    int *h_array;
    int *d_array;
    int *h_max;
    int *d_max;
    int *d_mutex;

    // allocate memory
    h_array = (int *)malloc(N * sizeof(int));
    h_max = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMalloc((void **)&d_max, sizeof(int));
    cudaMalloc((void **)&d_mutex, sizeof(int));
    cudaMemset(d_max, 0, sizeof(int));
    cudaMemset(d_mutex, 0, sizeof(int));

    // fill host array with data
    for (unsigned int i = 0; i < N; i++)
    {
        h_array[i] = N * int(rand()) / RAND_MAX;
    }

    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // call kernel

    dim3 gridSize = 256;
    dim3 blockSize = 256;
    find_maximum_kernel<<<gridSize, blockSize>>>(d_array, d_max, d_mutex, N);

    // copy from device to host
    cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    // report results
    std::cout << "Maximum number found on gpu was: " << *h_max << std::endl;

    // run cpu version
    clock_t cpu_start = clock();

    *h_max = -1.0;
    for (unsigned int i = 0; i < N; i++)
    {
        if (h_array[i] > *h_max)
        {
            *h_max = h_array[i];
        }
    }

    // free memory
    free(h_array);
    free(h_max);
    cudaFree(d_array);
    cudaFree(d_max);
    cudaFree(d_mutex);
}
