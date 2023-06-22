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

// 添加了location
// 添加了写回到candidate时候，先写到share memory，再写到global memory
// 添加了多层次的存储

#define FULL_MASK 0xffffffff
#define MIN_BUCKET_NUM 8
#define map_value(hash_value, bucket_num) ((hash_value % MIN_BUCKET_NUM) * (bucket_num / MIN_BUCKET_NUM) + hash_value / MIN_BUCKET_NUM)

using namespace cooperative_groups;

using namespace std;

int vertex_count = 0, vCount = 0, uMax = 0, edge_count;
float load_factor = 0.25;
int load_factor_inverse = 1 / load_factor;
int bucket_size = 4;
int bucket_num;
int parameter = load_factor_inverse / bucket_size;
int max_degree;                // 记录最大度数
int pattern_vertex_number = 3; // pattern的节点数量

inline off_t fsize(const char *filename)
{
    struct stat st;
    if (stat(filename, &st) == 0)
    {
        return st.st_size;
    }
    return -1;
}

double wtime()
{
    double time[2];
    struct timeval time1;
    gettimeofday(&time1, NULL);

    time[0] = time1.tv_sec;
    time[1] = time1.tv_usec;

    return time[0] + time[1] * 1.0e-6;
}

double getDeltaTime(double &startTime)
{
    double deltaTime = wtime() - startTime;
    startTime = wtime();
    return deltaTime;
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

__global__ void buildHashTablelens(int *hash_tables_lens, int *csr_row_value, int vertex_count, int parameter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int len;
    int log;
    for (int i = tid; i < vertex_count + 1; i += stride)
    {
        len = csr_row_value[i] * parameter;
        log = log2f(len);
        len = powf(2, log) == len ? len : powf(2, log + 1);
        if (len < 8 && len != 0)
            len = 8;
        hash_tables_lens[i] = len;
    }
}

__global__ void buildHashTable(int *hash_tables_offset, int *hash_tables, int *csr_column_index, int *vertex_list, int edge_count, int bucket_size, int load_factor_inverse, int bucket_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int key;
    int vertex;
    int hash_table_start;
    int hash_table_end;
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
        while (atomicCAS(&hash_tables[hash_table_start + mapped_value + index * bucket_num], -1, key) != -1)
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

__inline__ __device__ void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

__inline__ __device__ bool search_in_hashtable(int x, int bucket_num, int bucket_size, int hash_table_len, int *hash_table)
{
    if (x == -1)
        return false;
    int value = x % hash_table_len;
    int mapped_value = map_value(value, hash_table_len);
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

// candidates_of_all_warp
// my_candidates_for_all_mapping
// my_candidates

// 还没有解决如何动态的传入数组
// h : height of subtree; h = pattern vertex number
__global__ void DFSKernel(int vertex_count, int bucket_num, int max_degree, int h, int bucket_size, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *candidates_of_all_warp, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int lid = tid % 32;                              // landid
    int inblock_wid = threadIdx.x / 32;
    int stride = blockDim.x * gridDim.x / 32;
    int candidate_number_array[3]; // 记录一下每一层保存的数据大小
    int next_candidate_array[3];   // 记录一下每一层保存的数据大小
    int mapping[3];                // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * h * max_degree;
    int my_count = 0;
    // 分层次存储：local memory每个线程一个int，share memory每个线程128个int,global memory
    int l_cache;
    extern __shared__ int s_cache[];
    // each warp process a subtree (an probe item)
    for (int start_vertex = wid; start_vertex < vertex_count; start_vertex += stride)
    {
        mapping[0] = start_vertex;
        int level = 0;
        for (;;)
        {
            level++;
            int &candidate_number = candidate_number_array[level];
            next_candidate_array[level] = -1;
            candidate_number = 0;
            __syncwarp();
            // find possible connection and maintain in S
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_length = intersection_offset[level] - intersection_order_start;
            int mapped_intersection_order[2]; // wzb: refine it
            for (int i = 0; i < intersection_order_length; i++)
            {
                mapped_intersection_order[i] = mapping[intersection_orders[intersection_order_start + i]];
            }

            // get degree
            int neighbour_numbers[2];
            neighbour_numbers[0] = csr_row_value[mapped_intersection_order[0]];
            for (int i = 1; i < intersection_order_length; i++)
            {
                neighbour_numbers[i] = csr_row_value[mapped_intersection_order[i]];
                if (neighbour_numbers[i] < neighbour_numbers[0])
                {
                    swap(mapped_intersection_order[i], mapped_intersection_order[0]);
                    swap(neighbour_numbers[i], neighbour_numbers[0]);
                }
            }

            // Start intersection, load neighbor of m1 into my_candidates
            int *my_candidates = my_candidates_for_all_mapping + level * max_degree;
            int cur_vertex = mapped_intersection_order[0];
            int *cur_neighbor_list = csr_column_index + csr_row_offset[cur_vertex];

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (intersection_order_length == 1)
            {
                my_candidates[0] = -1;
                my_candidates[1] = (int)((long)cur_neighbor_list >> 32);
                my_candidates[2] = (int)((long)cur_neighbor_list & 0xFFFFFFFF);
            }
            // 要做交集，则分层次存储
            else
            {
                if (lid < neighbour_numbers[0])
                {
                    l_cache = cur_neighbor_list[lid];
                }
                // s_cache是否要列式存储
                for (int i = lid + 32; i < 150 || i < neighbour_numbers[0]; i += 32)
                    s_cache[inblock_wid * 128 + i - 32] = cur_neighbor_list[i];
            }
            // intersect
            candidate_number = neighbour_numbers[0];
            for (int j = 1; j < intersection_order_length; j++)
            {
                cur_vertex = mapped_intersection_order[j];
                int *cur_hashtable = hash_tables + hash_tables_offset[cur_vertex];
                int len = hash_tables_offset[cur_vertex + 1] - hash_tables_offset[cur_vertex]; // len记录当前hash_table的长度
                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                for (int i = lid; i < candidate_number_previous; i += 32)
                {
                    int item;
                    if (i < 32)
                        item = l_cache;
                    if (i >= 32 && i < 150)
                        item = s_cache[inblock_wid * 128 + i - 32];
                    int is_exist = search_in_hashtable(item, bucket_num, bucket_size, len, cur_hashtable);
                    int count = __reduce_add_sync(__activemask(), is_exist);
                    if (!is_exist)
                    {
                        if (i < 32)
                            l_cache = -1;
                        if (i >= 32 && i < 150)
                            s_cache[inblock_wid * 128 + i - 32] = -1;
                    }
                    // need to write back
                    if (is_exist && intersection_order_length == intersection_order_length - 1)
                    {
                        coalesced_group active = coalesced_threads();
                        // 最后一层不写回
                        if (level != h - 1)
                            my_candidates[active.thread_rank() + candidate_number] = item;
                    }
                    candidate_number += count;
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
            }
            __syncwarp();

            if (level == h - 1)
            {
                if (lid == 0)
                {
                    my_count += candidate_number;
                }
                __syncwarp();
                level--;
            }
            for (;; level--)
            {
                if (level == 0)
                    break;
                next_candidate_array[level]++;
                if (next_candidate_array[level] < candidate_number_array[level])
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
    }
    if (lid == 0)
    {
        atomicAdd(sum, my_count);
    }
}

int main(int argc, char *argv[])
{
    int *csr_column_index;
    int *csr_row_offset;
    int *csr_row_value;
    int *vertex_list;

    string s_begin = "./data/begin.bin";  // 记录索引，而且已经维护了最后一位
    string s_adj = "./data/adjacent.bin"; // 记录邻居节点
    string s_degree = "./data/edge";      // 记录邻居节点数目
    string s_vertex = "./data/vertex";

    char *begin_file = const_cast<char *>(s_begin.c_str());
    char *adj_file = const_cast<char *>(s_adj.c_str());
    char *degree_file = const_cast<char *>(s_degree.c_str());
    char *vertex_file = const_cast<char *>(s_vertex.c_str());

    int vertex_count = fsize(begin_file) / sizeof(int) - 1;
    int edge_count = fsize(adj_file) / sizeof(int);

    FILE *pFile1 = fopen(begin_file, "rb");
    csr_row_offset = (int *)malloc(fsize(begin_file));
    fread(csr_row_offset, sizeof(int), vertex_count + 1, pFile1);
    fclose(pFile1);

    FILE *pFile2 = fopen(adj_file, "rb");
    csr_column_index = (int *)malloc(fsize(adj_file));
    fread(csr_column_index, sizeof(int), edge_count, pFile2);
    fclose(pFile2);

    FILE *pFile3 = fopen(degree_file, "rb");
    csr_row_value = (int *)malloc(fsize(degree_file));
    fread(csr_row_value, sizeof(int), vertex_count, pFile3);
    fclose(pFile3);

    FILE *pFile4 = fopen(vertex_file, "rb");
    vertex_list = (int *)malloc(fsize(vertex_file));
    fread(vertex_list, sizeof(int), edge_count, pFile4);
    fclose(pFile4);

    int *d_csr_row_offset;
    int *d_csr_column_index, *d_csr_row_value, *d_vertex_list;
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
    cout << "graph load_factor_inverse is : " << load_factor_inverse << endl;
    cout << "graph bucket_size is : " << bucket_size << endl;
    cout << "graph parameter is : " << parameter << endl;

    // 查看下可用share memory的最大值
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 假设设备号为0
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    cout << "share memory size : " << sharedMemPerBlock << endl;

    // compute max degree
    int *d_max_degree;
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
    buildHashTablelens<<<216, 1024>>>(d_hash_tables_lens, d_csr_row_value, vertex_count, parameter);
    // exclusiveSum
    int *d_hash_tables_offset;
    HRR(cudaMalloc(&d_hash_tables_offset, (vertex_count + 1) * sizeof(int)));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    // get sum
    int last_count;
    int last_sum;
    HRR(cudaMemcpy(&last_count, d_hash_tables_lens + vertex_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
    HRR(cudaMemcpy(&last_sum, d_hash_tables_offset + vertex_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int bucket_num = last_count + last_sum;
    HRR(cudaMemset(d_hash_tables_offset + vertex_count, bucket_num, sizeof(int)));

    // build hash table in device
    int *d_hash_tables;
    HRR(cudaMalloc(&d_hash_tables, (long long)bucket_size * bucket_num * sizeof(int)));
    HRR(cudaMemset(d_hash_tables, -1, (long long)bucket_size * bucket_num * sizeof(int)));

    buildHashTable<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables, d_csr_column_index, d_vertex_list, edge_count, bucket_size, load_factor_inverse, bucket_num);

    // DFS
    int *d_ir; // intermediate result;
    // refine the malloc
    HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * pattern_vertex_number * sizeof(int)));
    cout << "ir memory size is : " << 216 * 32 * max_degree * pattern_vertex_number << endl;
    // 先提前假定一下三角形的顺序
    int intersection_orders[4] = {0, 0, 1};
    int intersection_offset[4] = {0, 1, 3};
    int *d_intersection_orders;
    int intersection_size = 4;
    HRR(cudaMalloc(&d_intersection_orders, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    HRR(cudaMalloc(&d_intersection_offset, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int h = 3;
    int *d_sum;
    HRR(cudaMalloc(&d_sum, 4));
    HRR(cudaMemset(d_sum, 0, 4));
    // double start_time = wtime();

    double t1 = wtime();
    double cmp_time;
    double time_start;
    double max_time = 0;
    double min_time = 1000;
    double ave_time = 0;
    for (int i = 0; i < 1000; i++)
    {
        HRR(cudaMemset(d_sum, 0, 4));
        time_start = clock();
        // share memory分配每个warp有128个int，每个block有4096个int
        DFSKernel<<<216, 1024, 4096 * sizeof(int)>>>(vertex_count, bucket_num, max_degree, h, bucket_size, d_intersection_orders, d_intersection_offset, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum);
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
        if (cmp_time > max_time)
            max_time = cmp_time;
        if (cmp_time < min_time)
            min_time = cmp_time;
        ave_time += cmp_time;
        HRR(cudaFree(d_ir));
        HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * pattern_vertex_number * sizeof(int)));
    }
    std::cout << "max time: " << max_time * 1000 << " ms" << std::endl;
    std::cout << "min time: " << min_time * 1000 << " ms" << std::endl;
    std::cout << "average time: " << ave_time << " ms" << std::endl;

    int sum;
    cudaMemcpy(&sum, d_sum, 4, cudaMemcpyDeviceToHost);
    printf("triangle count is %d\n", sum);
}