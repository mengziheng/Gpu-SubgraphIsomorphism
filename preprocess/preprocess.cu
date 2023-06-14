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

#define __max(a, b) (((a) > (b)) ? (a) : (b))

using namespace cooperative_groups;

using namespace std;

struct edge
{
    int u, v;
};
vector<edge> edgelist;
int uCount = 0, vCount = 0, uMax = 0, vMax = 0, edgeCount;
float load_factor = 0.25;
int load_factor_inverse = 1 / load_factor;
int bucket_size = 4;
int parameter = load_factor_inverse / bucket_size;
int local_register_num = 2;
int max_degree;            // 记录最大度数
int pattern_vertex_number; // pattern的节点数量

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

void loadgraph(string filename)
{
    ifstream inFile(filename.c_str(), ios::in);
    if (!inFile)
    {
        cout << "error" << endl;
    }
    string line;
    stringstream ss;
    while (getline(inFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        else
        {
            ss << line;
            ss >> uCount >> vCount >> edgeCount;
            break;
        }
    }
    while (getline(inFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        ss.str("");
        ss.clear();
        ss << line;

        edge e;
        ss >> e.u >> e.v;
        e.u--;
        e.v--;
        edgelist.push_back(e);
    }
}

// 这一步骤应该放外面处理，不算总时间内
__global__ void translateIntoCSRKernel(int *edgelist, int edgeCount, int vertexCount, int *csr_column_index, int *csr_row_value)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    // extern __shared__ int s_csr_row_value[];
    for (int i = tid; i < edgeCount; i += stride)
    {
        atomicAdd(csr_row_value + edgelist[i * 2 + 1], 1);
        csr_column_index[i] = edgelist[i * 2];
    }
    // for (int i = threadIdx.x; i < vertexCount; i += blockDim.x)
    // {
    //     atomicAdd(&csr_row_value[i], s_csr_row_value[i]);
    // }
}

__global__ void buildHashTableOffset(int *hash_tables_offset, int *csr_row_offset, int vertex_count, int parameter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < vertex_count; i += stride)
        hash_tables_offset[i] = csr_row_offset[i] * parameter;
    // if (tid == 0)
    //     printf("%d\n", parameter);
}

__global__ void buildHashTable(int *hash_tables_offset, int *hash_tables, int *hash_table_parameters, int *csr_row_offset, int *edgelist, int vertex_count, int edge_count, int bucket_size, int load_factor_inverse)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int key;
    int vertex;
    int hash_table_start;
    int hash_table_end;
    int hash_table_length;
    int value;
    for (int i = tid; i < edge_count; i += stride)
    {
        // edgelist要不要用share_memory赌气来
        key = edgelist[i * 2];
        vertex = edgelist[i * 2 + 1];
        hash_table_start = hash_tables_offset[vertex];
        hash_table_end = hash_tables_offset[vertex + 1] - 1;
        hash_table_length = hash_table_end - hash_table_start + 1;
        // hash function就选择为%k好了
        value = key % hash_table_length;
        // 按列存储
        // len这一行可以存到shared memory中加快速度
        int bucket_len = hash_tables[hash_table_start + value];
        // 找到当前hash_tables中不满的bucket
        // if (i == 10000)
        //     printf("start %d end %d len %d key %d vertex %d\n", hash_table_start, hash_table_end, hash_table_length, key, vertex);
        while (bucket_len == bucket_size - 1)
        {
            // 这里要注意，因为是向后增加元素，万一这是最后一个元素怎么办？会造成前面的元素多，后面的元素少。
            value++;
            if (value == hash_table_length)
                value = 0;
            bucket_len = hash_tables[hash_table_start + value];
        }
        hash_tables[hash_table_start + value + (bucket_len + 1) * edge_count] = key;
        atomicAdd(hash_tables + hash_table_start + value, 1);
    }
}

// h : height of subtree; h = pattern vertex number
__global__ void DFSKernel(int vertex_count, int edge_count, int max_degree, int h, int parameter, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *ir, int *sum)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int lid = tid % 32;                              // landid
    int level = 1;                                   // level of subtree,start from 0,not root for tree,but root for subtree
    int warp_sum = 0;                                // 记录一下每个warp记录得到的数量
    int stride = blockDim.x * gridDim.x % 32;
    int *ir_number = new int[h]; // 记录一下每一层保存的数据大小

    // 初始化
    for (int i = 0; i < h; i++)
        ir_number[i] = 0;
    int *buffer = new int[h]; // 记录每次的中间结果

    if (tid == 0)
        printf("%d %d %d\n", intersection_offset[0], intersection_offset[1], intersection_offset[2]);
    if (tid == 0)
        printf("%d\n", vertex_count);
    if (tid == 0)
        printf("value : %d %d %d\n", csr_row_value[0], csr_row_value[1], csr_row_value[2]);

    // each warp process a subtree (an probe item)
    for (int first_vertex = wid; first_vertex < vertex_count; first_vertex += stride) // wzb: change to dynamic load
    {
        buffer[0] = first_vertex;
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
                int intersection_order_start = intersection_offset[level - 1];
                int intersection_order_end = intersection_offset[level];
                int intersection_order_length = intersection_order_end - intersection_order_start;
                int *intersection_order = new int[intersection_order_length]; // wzb: refine it
                for (int i = 0; i < intersection_order_length; i++)
                    intersection_order[i] = intersection_orders[intersection_order_start + i];
                int *neighbour_numbers = new int[intersection_order_length];
                printf("level : %d intersection_order_length :%d\n", level, intersection_order_length);
                // 将需要做交集的点的邻居数量都读入到寄存器中
                for (int i = 0; i < intersection_order_length; i++)
                {
                    neighbour_numbers[i] = csr_row_value[buffer[intersection_order[i]]];
                    printf("neighbour_number for %d  is %d\n", buffer[intersection_order[i]], neighbour_numbers[i]);
                }

                // if (tid == 1)
                // {
                //     printf("level %d intersection_order_length %d\n", level, intersection_order_length);
                //     printf("level : %d buffer[0] %d buffer[intersection_order[i]] %d  %d\n", level, buffer[0], buffer[intersection_order[0]], neighbour_numbers[0]);
                //     printf("%d %d %d", csr_row_offset[0], csr_row_offset[1], csr_row_offset[2]);
                // }

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
                int thread_cache_size = neighbor_num / 32; // wzb: can not maintain in register, reuse buffer
                int remainder = neighbor_num % 32;
                // 判断是否需要向上取整
                if (remainder > 0)
                {
                    thread_cache_size += 1;
                }
                int *thread_cache = new int[thread_cache_size];
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
                        if (thread_cache[i] == -1)
                            continue;
                        int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                        int hash_tables_start = hash_tables_offset[buffer[intersection_order[i]]];
                        int *cmp = &hash_tables[hash_tables_start + value + edge_count]; // wzb: remove edge count
                        while (*cmp != -1)
                        {
                            if (*cmp == thread_cache[i])
                                break;
                            cmp = cmp + edge_count;
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
                        int hash_tables_start = hash_tables_offset[buffer[intersection_order[i]]];
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
            // 是最后一层，将结果写入global
            // 当前层不为空，取下一个元素
            else
            {
                if (lid == 0)
                    ir_number[level] = ir_number[level] - 1;
            }
            // 更新临时结果
            buffer[level] = ir[level * 1024 * 216 / 32 * max_degree + wid * max_degree + ir_number[level]];
            level++;
        }
        if (lid == 1)
        {
            atomicAdd(sum, warp_sum);
        }
    }
    free(ir_number);
    free(buffer);
}

int main(int argc, char *argv[])
{
    // load graph file
    // string infilename = "../dataset/graph/as20000102_adj.mmio";
    // string infilename = "../dataset/graph/cit-Patents_adj.mmio";
    string infilename = "../dataset/graph/test.mmio";
    loadgraph(infilename);
    cout << "graph vertex number is : " << uCount << endl;
    cout << "graph edge number is : " << edgeCount << endl;
    cout << "graph load_factor_inverse is : " << load_factor_inverse << endl;
    cout << "graph bucket_size is : " << bucket_size << endl;
    cout << "graph parameter is : " << parameter << endl;

    int *d_edgelist;
    HRR(cudaMalloc(&d_edgelist, edgeCount * 2 * sizeof(int)));
    HRR(cudaMemcpy(d_edgelist, &edgelist[0], edgeCount * 2 * sizeof(int), cudaMemcpyHostToDevice));

    // get CSR and other structure in device
    int *d_csr_column_index;
    int *d_csr_row_value;
    int *d_csr_row_offset;
    HRR(cudaMalloc(&d_csr_column_index, edgeCount * sizeof(int)));
    HRR(cudaMalloc(&d_csr_row_value, uCount * sizeof(int)));
    HRR(cudaMalloc(&d_csr_row_offset, uCount * sizeof(int)));
    printf("-------------translate into CSR and saved in Device---------------------\n");
    // 查看下可用share memory的最大值
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 假设设备号为0
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    cout << "share memory size : " << sharedMemPerBlock << endl;
    translateIntoCSRKernel<<<216, 1024>>>(d_edgelist, edgeCount, uCount, d_csr_column_index, d_csr_row_value);
    // translateIntoCSRKernel<<<216, 1024, uCount * sizeof(int)>>>(d_edgelist, edgeCount, uCount, d_csr_column_index, d_csr_row_value);

    // mzh:可以重写为一个kernel，因为本质上都是reduction
    int *d_max_degree;
    // compute max degree
    cudaMalloc((void **)&d_max_degree, sizeof(int));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_max_degree, uCount);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_max_degree, uCount);
    int *h_max = (int *)malloc(sizeof(int));
    HRR(cudaMemcpy(h_max, d_max_degree, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Maximum number found on gpu was: " << *h_max << std::endl;

    // get row_offset for CSR
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);

    // build hash table in device
    int *d_hash_tables;
    HRR(cudaMalloc(&d_hash_tables, bucket_size * edgeCount * sizeof(int)));
    int *d_hash_tables_offset;
    HRR(cudaMalloc(&d_hash_tables_offset, uCount * sizeof(int)));
    int *d_hash_table_parameters;
    HRR(cudaMalloc(&d_hash_table_parameters, uCount * sizeof(int)));
    // 写hash_table_offset
    buildHashTableOffset<<<216, 1024>>>(d_hash_tables_offset, d_csr_row_offset, uCount, parameter);
    buildHashTable<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables, d_hash_table_parameters, d_csr_row_offset, d_edgelist, uCount, edgeCount, bucket_size, load_factor_inverse);

    // DFS
    int *d_ir; // intermediate result;
    HRR(cudaMalloc(&d_ir, 216 * 1024 / 32 * max_degree * pattern_vertex_number));
    HRR(cudaMemset(&d_ir, -1, 216 * 1024 / 32 * max_degree * pattern_vertex_number)); // 初始值默认为-1
    // int *final_result; // 暂时先用三角形考虑。
    // HRR(cudaMalloc(&d_ir_ptr, 216 * 1024 / 32 * pattern_vertex_number));
    // 先提前假定一下三角形的顺序
    int intersection_orders[3] = {0, 0, 1};
    int intersection_offset[3] = {0, 1, 3};
    int *d_intersection_orders;
    cudaMalloc(&d_intersection_orders, 3 * 4);
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, 12, cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    cudaMalloc(&d_intersection_offset, 3 * 4);
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, 12, cudaMemcpyHostToDevice));
    int h = 2;
    int *d_sum;
    cudaMalloc(&d_sum, 4);
    cudaMemset(d_sum, 0, 4);
    int csr_row_value[uCount];
    cudaMemcpy(csr_row_value, d_csr_row_value, uCount * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++)
    {
        printf("%d ", csr_row_value[i]);
    }
    printf("\n");
    DFSKernel<<<216, 512>>>(uCount, edgeCount, max_degree, h, parameter, d_intersection_orders, d_intersection_offset, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum);
    int sum;
    cudaMemcpy(&sum, d_sum, 4, cudaMemcpyDeviceToHost);
    printf("triangle count is %d\n", sum);
    cudaDeviceSynchronize();
    // verify
    // int csr_column_index[edgeCount];
    // cudaMemcpy(csr_column_index, d_csr_column_index, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", csr_column_index[i]);
    // }
    // printf("\n");

    // int csr_row_value[uCount];
    // cudaMemcpy(csr_row_value, d_csr_row_value, uCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", csr_row_value[i]);
    // }
    // printf("\n");

    // int csr_row_offset[uCount];
    // int csr_row_offset[377476];
    // cudaMemcpy(csr_row_offset, d_csr_row_offset, 377476 * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 377476; i++)
    // {
    //     printf("%d ",csr_row_offset[i]);
    // }
    // printf("\n");
    cudaDeviceSynchronize();
}