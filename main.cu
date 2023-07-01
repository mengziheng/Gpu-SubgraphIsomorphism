#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cub/cub.cuh>
#include <math.h>
#include <cooperative_groups.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define FULL_MASK 0xffffffff
#define MIN_BUCKET_NUM 8
#define MAX_SIZE_FOR_ARRAY 10
#define map_value(hash_value, bucket_num) (((hash_value) % MIN_BUCKET_NUM) * (bucket_num / MIN_BUCKET_NUM) + (hash_value) / MIN_BUCKET_NUM)
#define H 3

using namespace cooperative_groups;
using namespace std;

int vertex_count = 0, edge_count;
float load_factor = 0.25;
int bucket_size = 4;
int block_size = 216;
int block_number = 1024;
int chunk_size = 1;

inline off_t fsize(const char *filename)
{
    struct stat st;
    if (stat(filename, &st) == 0)
    {
        return st.st_size;
    }
    return -1;
}

static void HandError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("\n%s in %s at line %d\n",
               cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HRR(err) (HandError(err, __FILE__, __LINE__))

tuple<int *, int *, int *, int *> loadGraphWithName(string Infilename, string pattern)
{
    int *csr_column_index;
    int *csr_row_offset;
    int *csr_row_value;
    int *vertex_list;

    string folder;
    if (pattern.compare("triangle") == 0 || pattern.find("clique") != std::string::npos)
        folder = "/data/zh_dataset/dataforclique/" + Infilename;
    else
        folder = "/data/zh_dataset/dataforgeneral/" + Infilename;
    string s_begin = folder + "/begin.bin";  // 记录索引，而且已经维护了最后一位
    string s_adj = folder + "/adjacent.bin"; // 记录邻居节点
    string s_degree = folder + "/edge";      // 记录邻居节点数目
    string s_vertex = folder + "/vertex";

    char *begin_file = const_cast<char *>(s_begin.c_str());
    char *adj_file = const_cast<char *>(s_adj.c_str());
    char *degree_file = const_cast<char *>(s_degree.c_str());
    char *vertex_file = const_cast<char *>(s_vertex.c_str());

    vertex_count = fsize(begin_file) / sizeof(int) - 1;
    edge_count = fsize(adj_file) / sizeof(int);

    FILE *pFile1 = fopen(begin_file, "rb");
    if (!pFile1)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_row_offset = (int *)malloc(fsize(begin_file));
    size_t x = fread(csr_row_offset, sizeof(int), vertex_count + 1, pFile1);
    fclose(pFile1);

    FILE *pFile2 = fopen(adj_file, "rb");
    if (!pFile2)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_column_index = (int *)malloc(fsize(adj_file));
    x = fread(csr_column_index, sizeof(int), edge_count, pFile2);
    fclose(pFile2);

    // int j = 0;
    // int z = 0;
    // for (int i = 0; i < edge_count; i++)
    // {
    //     printf("%d ", csr_column_index[i]);
    //     j++;
    //     if (j == csr_row_offset[z + 1])
    //     {
    //         printf("is neighbour of %d\n", z);
    //         j == 0;
    //         z++;
    //     }
    // }
    // printf("\n");

    FILE *pFile3 = fopen(degree_file, "rb");
    if (!pFile3)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_row_value = (int *)malloc(fsize(degree_file));
    x = fread(csr_row_value, sizeof(int), vertex_count, pFile3);
    fclose(pFile3);

    FILE *pFile4 = fopen(vertex_file, "rb");
    if (!pFile4)
    {
        cout << "error" << endl;
        // return 0;
    }
    vertex_list = (int *)malloc(fsize(vertex_file));
    x = fread(vertex_list, sizeof(int), edge_count, pFile4);
    fclose(pFile4);
    int *d_csr_column_index, *d_csr_row_value, *d_vertex_list, *d_csr_row_offset;

    HRR(cudaMalloc((void **)&d_csr_row_offset, sizeof(int) * (vertex_count + 1)));
    HRR(cudaMalloc((void **)&d_csr_column_index, sizeof(int) * (edge_count)));
    HRR(cudaMalloc((void **)&d_vertex_list, sizeof(int) * (edge_count)));
    HRR(cudaMalloc((void **)&d_csr_row_value, sizeof(int) * (vertex_count + 1)));

    HRR(cudaMemcpy(d_csr_row_value, csr_row_value, sizeof(int) * (vertex_count + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_csr_row_offset, csr_row_offset, sizeof(int) * (vertex_count + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_csr_column_index, csr_column_index, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_vertex_list, vertex_list, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    cout << "graph vertex number is : " << vertex_count << endl;
    cout << "graph edge number is : " << edge_count << endl;
    cout << "graph bucket size is : " << bucket_size << endl;
    return make_tuple(d_csr_column_index, d_csr_row_value, d_vertex_list, d_csr_row_offset);
}

void printGpuInfo()
{
    // 查看下可用share memory的最大值
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 假设设备号为0
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    cout << "share memory size per block : " << sharedMemPerBlock << endl;
    cout << "registers number per block : " << deviceProp.regsPerBlock << endl;
}

__global__ void buildHashTableLensKernel(int *hash_tables_lens, int *csr_row_value, int vertex_count, float load_factor, int bucket_size)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int len;
    int log;
    for (int i = tid; i < vertex_count; i += stride)
    {
        len = (int)(csr_row_value[i] / load_factor / bucket_size);
        if (csr_row_value[i] / load_factor / bucket_size != (float)len)
            len++;
        log = log2f(len);
        len = powf(2, log) == len ? len : powf(2, log + 1);
        if (len < 8 && len != 0)
            len = 8;
        hash_tables_lens[i] = len;
    }
}

__global__ void buildHashTableKernel(long long *hash_tables_offset, int *hash_tables, int *csr_column_index, int *vertex_list, int edge_count, long long bucket_num, int bucket_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int key;
    int vertex;
    long long hash_table_start;
    long long hash_table_end;
    int hash_table_length;
    int value;
    int mapped_value;
    for (int i = tid; i < edge_count; i += stride)
    {
        key = csr_column_index[i];
        vertex = vertex_list[i];
        hash_table_start = hash_tables_offset[vertex];
        hash_table_end = hash_tables_offset[vertex + 1];
        hash_table_length = hash_table_end - hash_table_start;
        // hash function = % k
        value = key % hash_table_length;
        mapped_value = map_value(value, hash_table_length);
        int index = 0;
        while (atomicCAS(&hash_tables[hash_table_start + mapped_value + bucket_num * index], -1, key) != -1)
        {
            index++;
            if (index == bucket_size)
            {
                index = 0;
                mapped_value++;
                if (mapped_value == hash_table_length)
                    mapped_value = 0;
            }
        }
    }
}

tuple<long long *, int *, int, long long> buildHashTable(int *d_csr_column_index, int *d_vertex_list, int *d_csr_row_value)
{
    // compute max degree
    int *d_max_degree;
    int max_degree;
    cudaMalloc(&d_max_degree, sizeof(int));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_max_degree, vertex_count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_max_degree, vertex_count);
    HRR(cudaMemcpy(&max_degree, d_max_degree, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "max degree is: " << max_degree << std::endl;

    // get bucket_num
    int *d_hash_tables_lens;
    HRR(cudaMalloc(&d_hash_tables_lens, vertex_count * sizeof(int)));
    buildHashTableLensKernel<<<216, 1024>>>(d_hash_tables_lens, d_csr_row_value, vertex_count, load_factor, bucket_size);
    // exclusiveSum
    long long *d_hash_tables_offset;
    HRR(cudaMalloc(&d_hash_tables_offset, (vertex_count + 1) * sizeof(long long)));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);

    // get sum
    int last_count;
    long long last_sum;
    long long bucket_num;
    HRR(cudaMemcpy(&last_count, d_hash_tables_lens + vertex_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
    HRR(cudaMemcpy(&last_sum, d_hash_tables_offset + vertex_count - 1, sizeof(long long), cudaMemcpyDeviceToHost));
    bucket_num = last_sum + last_count;
    HRR(cudaMemcpy(d_hash_tables_offset + vertex_count, &bucket_num, sizeof(long long), cudaMemcpyHostToDevice));
    // build hash table in device
    int *d_hash_tables;
    HRR(cudaMalloc(&d_hash_tables, bucket_size * bucket_num * sizeof(int)));
    HRR(cudaMemset(d_hash_tables, -1, bucket_size * bucket_num * sizeof(int)));
    cout << "bucket_num is : " << bucket_num << endl;
    cout << "hash table size is : " << bucket_size * bucket_num * sizeof(int) << endl;

    buildHashTableKernel<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables, d_csr_column_index, d_vertex_list, edge_count, bucket_num, bucket_size);
    return make_tuple(d_hash_tables_offset, d_hash_tables, max_degree, bucket_num);
}

__inline__ __device__ void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

__inline__ __device__ bool search_in_hashtable(int x, int bucket_size, long long bucket_num, int hash_table_len, int *hash_table)
{
    int mapped_value = map_value(x % hash_table_len, hash_table_len);
    int *cmp = hash_table + mapped_value;
    int index = 0;
    while (*cmp != -1)
    {
        if (*cmp == x)
        {
            return true;
        }
        cmp = cmp + bucket_num;
        index++;
        if (index == bucket_size)
        {
            mapped_value++;
            index = 0;
            if (mapped_value == hash_table_len)
                mapped_value = 0;
            cmp = &hash_table[mapped_value];
        }
    }
    return false;
}

// h : height of subtree; h = pattern vertex number
__global__ void DFSKernelForGeneral(int chunk_size, int vertex_count, int bucket_size, long long bucket_num, int max_degree, int *intersection_orders, int *intersection_offset, int *restriction, int *csr_row_offset, int *csr_row_value, int *csr_column_index, long long *hash_tables_offset, int *hash_tables, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int mapping[H];                                         // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    unsigned long long my_count = 0;
    int start_vertex = wid * chunk_size;
    int vertex_end = start_vertex + chunk_size;
    int level;
    // each warp process a subtree (an probe item)
    while (start_vertex < vertex_count)
    {
        mapping[0] = start_vertex;
        level = 0;
        for (;;)
        {
            level++;
            int candidate_number;
            candidate_number = 0;
            // find possible connection and maintain in S
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_length = intersection_offset[level] - intersection_order_start;
            // get degree
            int min_degree;
            int min_degree_vertex = mapping[intersection_orders[intersection_order_start]];
            min_degree = csr_row_value[min_degree_vertex];
            for (int i = 1; i < intersection_order_length; i++)
            {
                if (csr_row_value[mapping[intersection_orders[intersection_order_start + i]]] < min_degree)
                {
                    min_degree_vertex = mapping[intersection_orders[intersection_order_start + i]];
                    min_degree = csr_row_value[min_degree_vertex];
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_candidates = my_candidates_for_all_mapping + level * max_degree;
            int cur_vertex;
            int *cur_neighbor_list = csr_column_index + csr_row_offset[min_degree_vertex];

            if (intersection_order_length == 1)
            {
                for (int i = lid; i < min_degree; i += 32)
                {
                    int flag = restriction[level] != -1 && cur_neighbor_list[i] > mapping[restriction[level]];
                    // 改成小于，使用break
                    if (flag)
                    {
                        coalesced_group active = coalesced_threads();
                        my_candidates[active.thread_rank() + candidate_number] = cur_neighbor_list[i];
                    }
                    candidate_number += __reduce_add_sync(__activemask(), flag);
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
            }
            else
            {
                candidate_number = min_degree;
                // if (wid == 0 && lid == 0)
                //     printf("level : %d candidate_number : %d\n", candidate_number);
                for (int j = 0; j < intersection_order_length; j++)
                {
                    cur_vertex = mapping[intersection_orders[intersection_order_start + j]];
                    if (cur_vertex == min_degree_vertex)
                        continue;
                    int *cur_hashtable = hash_tables + hash_tables_offset[cur_vertex];
                    int len = int(hash_tables_offset[cur_vertex + 1] - hash_tables_offset[cur_vertex]); // len记录当前hash_table的长度

                    int candidate_number_previous = candidate_number;
                    candidate_number = 0;
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item;
                        if (j == 0 || j == 1 && mapping[intersection_orders[intersection_order_start]] == min_degree_vertex)
                            item = cur_neighbor_list[i];
                        else
                            item = my_candidates[i];
                        int cmp;
                        if (restriction[level] == -1)
                            cmp = -1;
                        else
                            cmp = mapping[restriction[level]];
                        int is_exist;
                        if (item <= cmp)
                            is_exist = false;
                        else
                            is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                        if (level != H - 1 && is_exist)
                        {
                            coalesced_group active = coalesced_threads();
                            // 最后一层不写回
                            my_candidates[active.thread_rank() + candidate_number] = item;
                        }
                        candidate_number += __reduce_add_sync(__activemask(), is_exist);
                    }
                    // 这一行是否可以删掉？？？
                    candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                }
            }
            next_candidate_array[level] = candidate_number;
            // if (wid == 0 && lid == 0)
            // printf("level : %d mapping : %d %d %d %d\n", level, mapping[0], mapping[1], mapping[2], mapping[3]);
            if (level == H - 1)
            {
                if (lid == 0)
                {
                    my_count += candidate_number;
                }
                level--;
            }
            for (;; level--)
            {
                if (level == 0)
                    break;
                next_candidate_array[level]--;
                if (next_candidate_array[level] > -1)
                {
                    mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == 0)
                break;
        }
        start_vertex++;
        if (start_vertex == vertex_end)
        {
            if (lid == 0)
            {
                start_vertex = atomicAdd(&G_INDEX[0], chunk_size);
            }
            start_vertex = __shfl_sync(0xffffffff, start_vertex, 0);
            vertex_end = start_vertex + chunk_size;
        }
    }
    if (lid == 0)
    {
        atomicAdd(sum, my_count);
    }
}

__global__ void DFSKernelForClique(int chunk_size, int vertex_count, int bucket_size, long long bucket_num, int max_degree, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, long long *hash_tables_offset, int *hash_tables, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int mapping[H];                                         // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    unsigned long long my_count = 0;
    // each warp process a subtree (an probe item)
    int start_vertex = wid * chunk_size;
    int vertex_end = start_vertex + chunk_size;
    int level;
    // each warp process a subtree (an probe item)
    while (start_vertex < vertex_count)
    {
        mapping[0] = start_vertex;
        level = 0;
        for (;;)
        {
            level++;
            int candidate_number;
            candidate_number = 0;
            // find possible connection and maintain in S
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_length = intersection_offset[level] - intersection_order_start;
            // get degree
            int min_neighbour_numbers;
            int min_neighbour_vertex = mapping[intersection_orders[intersection_order_start]];
            min_neighbour_numbers = csr_row_value[min_neighbour_vertex];
            for (int i = 1; i < intersection_order_length; i++)
            {
                if (csr_row_value[mapping[intersection_orders[intersection_order_start + i]]] < min_neighbour_numbers)
                {
                    min_neighbour_vertex = mapping[intersection_orders[intersection_order_start + i]];
                    min_neighbour_numbers = csr_row_value[min_neighbour_vertex];
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_candidates = my_candidates_for_all_mapping + level * max_degree;
            int cur_vertex;
            int *cur_neighbor_list = csr_column_index + csr_row_offset[min_neighbour_vertex];

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (intersection_order_length == 1 && lid == 0)
            {
                my_candidates[0] = -1;
                my_candidates[1] = (int)((long)cur_neighbor_list >> 32);
                my_candidates[2] = (int)((long)cur_neighbor_list & 0xFFFFFFFF);
            }
            // intersect
            candidate_number = min_neighbour_numbers;
            for (int j = 0; j < intersection_order_length; j++)
            {
                cur_vertex = mapping[intersection_orders[intersection_order_start + j]];
                if (cur_vertex == min_neighbour_vertex)
                    continue;
                int *cur_hashtable = hash_tables + hash_tables_offset[cur_vertex];
                int len = int(hash_tables_offset[cur_vertex + 1] - hash_tables_offset[cur_vertex]); // len记录当前hash_table的长度

                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                for (int i = lid; i < candidate_number_previous; i += 32)
                {
                    int item;
                    if (j == 0 || j == 1 && mapping[intersection_orders[intersection_order_start]] == min_neighbour_vertex)
                        item = cur_neighbor_list[i];
                    else
                        item = my_candidates[i];
                    int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                    if (level != H - 1 && is_exist)
                    {
                        coalesced_group active = coalesced_threads();
                        // 最后一层不写回
                        my_candidates[active.thread_rank() + candidate_number] = item;
                    }
                    candidate_number += __reduce_add_sync(__activemask(), is_exist);
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
            }
            next_candidate_array[level] = candidate_number;
            if (level == H - 1)
            {
                if (lid == 0)
                {
                    my_count += candidate_number;
                }
                level--;
            }
            for (;; level--)
            {
                if (level == 0)
                    break;
                next_candidate_array[level]--;
                if (next_candidate_array[level] > -1)
                {
                    // 如果是-1，说明没有写入，直接从neighbour list读出的
                    if (my_candidates_for_all_mapping[level * max_degree] == -1)
                    {
                        int *result;
                        result = (int *)(((long)my_candidates_for_all_mapping[level * max_degree + 1] << 32) | ((long)my_candidates_for_all_mapping[level * max_degree + 2] & 0xFFFFFFFF));
                        mapping[level] = result[next_candidate_array[level]];
                    }
                    else
                        mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == 0)
                break;
        }
        start_vertex++;
        if (start_vertex == vertex_end)
        {
            if (lid == 0)
            {
                start_vertex = atomicAdd(&G_INDEX[0], chunk_size);
            }
            start_vertex = __shfl_sync(0xffffffff, start_vertex, 0);
            vertex_end = start_vertex + chunk_size;
        }
    }
    if (lid == 0)
    {
        atomicAdd(sum, my_count);
    }
}

void SubgraphMatching(string pattern, int max_degree, long long bucket_num, int *d_csr_row_offset, int *d_csr_row_value, int *d_csr_column_index, long long *d_hash_tables_offset, int *d_hash_tables)
{
    int *d_ir; // intermediate result;
    // refine the malloc
    HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));
    cout << "ir memory size is : " << 216 * 32 * max_degree * H / 1024 / 1024 << "MB" << endl;
    int intersection_orders[MAX_SIZE_FOR_ARRAY];
    int intersection_offset[MAX_SIZE_FOR_ARRAY];
    int restriction[MAX_SIZE_FOR_ARRAY];
    int intersection_size;
    int restriction_size;
    // 先提前假定一下三角形的顺序
    if (pattern.compare("triangle") == 0)
    {
        int tmp_intersection_orders[4] = {0, 0, 1};
        int tmp_intersection_offset[4] = {0, 1, 3};
        intersection_size = 4;
        memcpy(intersection_orders, tmp_intersection_orders, intersection_size * sizeof(int));
        memcpy(intersection_offset, tmp_intersection_offset, intersection_size * sizeof(int));
    }
    else if (pattern.compare("4-cycle") == 0)
    {

        // int tmp_intersection_orders[5] = {0, 1, 0, 2};
        // int tmp_intersection_offset[5] = {0, 1, 2, 4};
        // int tmp_restriction[5] = {-1, 0, 0, 1};
        int tmp_intersection_orders[5] = {0, 0, 1, 2};
        int tmp_intersection_offset[5] = {0, 1, 2, 4};
        int tmp_restriction[5] = {-1, 0, 1, 0};
        intersection_size = 5;
        restriction_size = 5;
        memcpy(intersection_orders, tmp_intersection_orders, intersection_size * sizeof(int));
        memcpy(intersection_offset, tmp_intersection_offset, intersection_size * sizeof(int));
        memcpy(restriction, tmp_restriction, restriction_size * sizeof(int));
    }
    int *d_intersection_orders;
    HRR(cudaMalloc(&d_intersection_orders, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    HRR(cudaMalloc(&d_intersection_offset, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_restriction;
    HRR(cudaMalloc(&d_restriction, restriction_size * sizeof(int)));
    HRR(cudaMemcpy(d_restriction, restriction, restriction_size * sizeof(int), cudaMemcpyHostToDevice));
    int *G_INDEX;
    int temp = block_size * block_number / 32 * chunk_size;
    HRR(cudaMalloc(&G_INDEX, sizeof(int)));
    HRR(cudaMemcpy(G_INDEX, &temp, sizeof(int), cudaMemcpyHostToDevice));

    unsigned long long *d_sum;
    HRR(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    // double start_time = wtime();

    double cmp_time;
    double time_start;
    if (pattern.compare("triangle") == 0 || pattern.find("clique") != std::string::npos)
    {
        time_start = clock();
        DFSKernelForClique<<<block_size, block_number>>>(chunk_size, vertex_count, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum, G_INDEX);
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
    }
    else
    {
        time_start = clock();
        DFSKernelForGeneral<<<block_size, block_number>>>(chunk_size, vertex_count, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum, G_INDEX);
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
    }

    std::cout << "time: " << cmp_time * 1000 << " ms" << std::endl;
    long long sum;
    cudaMemcpy(&sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    cout << pattern << " count is " << sum << endl;
}

int main(int argc, char *argv[])
{

    string Infilename = "4-cycle";
    string pattern = "4-cycle";
    if (argc > 1)
    {
        Infilename = argv[1];
        pattern = argv[2];
        load_factor = atof(argv[3]);
        bucket_size = atoi(argv[4]);
        block_size = atoi(argv[5]);
        block_number = atoi(argv[6]);
        chunk_size = atoi(argv[7]);
    }

    int *d_csr_column_index, *d_csr_row_value, *d_vertex_list, *d_csr_row_offset;
    tie(d_csr_column_index, d_csr_row_value, d_vertex_list, d_csr_row_offset) = loadGraphWithName(Infilename, pattern);

    printGpuInfo();

    int *d_hash_tables;
    long long *d_hash_tables_offset;
    int max_degree;
    long long bucket_num;
    tie(d_hash_tables_offset, d_hash_tables, max_degree, bucket_num) = buildHashTable(d_csr_column_index, d_vertex_list, d_csr_row_value);

    SubgraphMatching(pattern, max_degree, bucket_num, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables);
}