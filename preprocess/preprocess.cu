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
int bucket_num;
int parameter = load_factor_inverse / bucket_size;
int local_register_num = 2;
int max_degree;                // 记录最大度数
int pattern_vertex_number = 3; // pattern的节点数量

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
}

__global__ void buildHashTableOffset(int *hash_tables_offset, int *csr_row_offset, int *csr_row_value, int vertex_count, int parameter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    csr_row_offset[vertex_count] = csr_row_offset[vertex_count - 1] + csr_row_value[vertex_count - 1];
    for (int i = tid; i < vertex_count + 1; i += stride)
        hash_tables_offset[i] = csr_row_offset[i] * parameter;
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
    int bucket_number = edge_count * load_factor_inverse / bucket_size;
    int index;
    for (int i = tid; i < edge_count; i += stride)
    {
        // edgelist要不要用share_memory?
        key = edgelist[i * 2];
        vertex = edgelist[i * 2 + 1];
        hash_table_start = hash_tables_offset[vertex];
        hash_table_end = hash_tables_offset[vertex + 1];
        hash_table_length = hash_table_end - hash_table_start;
        // hash function就选择为%k好了
        value = key % hash_table_length;
        // 按列存储
        index = 0;
        while (atomicCAS(&hash_tables[hash_table_start + value + index * edge_count], -1, key) != -1)
        {
            index++;
            if (index == bucket_size)
            {
                index = 0;
                value++;
                if (value == hash_table_length)
                    value = 0;
            }
        }
    }
}

// __device__ int getNewVertex(int wid, int cur_vertex, int stride, int vertex_count)
// {
//     if (cur_vertex == -1)
//         if (wid < vertex_count)
//             return wid;
//         else
//             return -1;
//     if (cur_vertex + stride < vertex_count)
//         return cur_vertex + stride;
//     else
//         return -1;
// }

__device__ int getNewVertex(int wid, int cur_vertex, int stride, int vertex_count)
{
    if (cur_vertex == -1)
        if (wid < vertex_count)
            return wid;
        else
            return -1;
    if (cur_vertex + stride < vertex_count)
        return cur_vertex + stride;
    else
        return -1;
}

// h : height of subtree; h = pattern vertex number
__global__ void DFSKernel(int vertex_count, int edge_count, int max_degree, int h, int bucket_size, int parameter, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *ir, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int in_block_wid = threadIdx.x / 32;
    int lid = tid % 32;
    int stride = blockDim.x * gridDim.x / 32;
    int level = 1;
    int cur_vertex = -1;
    int ir_number[3]; // 记录一下每一层保存的数据大小
    int buffer[3];    // 记录每次的中间结果
    ir_number[level] = 0;
    __shared__ int warpsum[32];
    __shared__ int ptr[32];
    warpsum[in_block_wid] = 0;
    int FLAG = 0;

    while (true)
    {
        // 当前层为空
        if (ir_number[level] == 0)
        {
            if (level == 1)
            {
                cur_vertex = getNewVertex(wid, cur_vertex, stride, vertex_count);
                if (cur_vertex == 2)
                {
                    break;
                }
                // if (lid == 0)
                //     printf("CHANGE new vertex ! cur_vertex is %d level is %d \n", cur_vertex, level);
                __syncwarp();
                // 选择了一个新的节点作为probe item，作为初始节点,需要初始化ir_number
                for (int i = 0; i < h; i++)
                    ir_number[i] = 0;
                buffer[level - 1] = cur_vertex;
            }

            // 预处理，计算需要做交集的元素即其邻居节点数目
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_end = intersection_offset[level];
            int intersection_order_length = intersection_order_end - intersection_order_start;
            int intersection_order[2]; // wzb: refine it
            for (int i = 0; i < intersection_order_length; i++)
            {
                intersection_order[i] = intersection_orders[intersection_order_start + i];
            }
            int neighbour_numbers[2];
            for (int i = 0; i < intersection_order_length; i++)
            {
                neighbour_numbers[i] = csr_row_value[buffer[intersection_order[i]]];
            }
            // if (intersection_order_length == 2 && lid == 0)
            // {
            //     printf("neighbour_number[0] : %d neighbour_number[1] : %d\n", neighbour_numbers[0], neighbour_numbers[1]);
            //     printf("buffer[0] : %d buffer[1] : %d\n", buffer[0], buffer[1]);
            // }
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
            int cur_vertex = buffer[intersection_order[0]];
            int neighbor_num = neighbour_numbers[0];

            int thread_cache_size = neighbor_num / 32 + 1;
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
            __syncwarp();
            int cur_order_index;
            ptr[in_block_wid] = 0;
            for (int i = 0; i < thread_cache_size; i++)
            {
                // cur_order_index从1开始是因为0已经被存入cache。
                for (cur_order_index = 1; cur_order_index < intersection_order_length; cur_order_index++)
                {
                    int k = 0;
                    if (thread_cache[i] == -1)
                        continue;
                    int value = thread_cache[i] % (neighbour_numbers[cur_order_index] * parameter);
                    int hash_table_start = hash_tables_offset[buffer[intersection_order[cur_order_index]]];
                    int hash_table_end = hash_tables_offset[buffer[intersection_order[cur_order_index]] + 1];
                    int hash_table_length = hash_table_end - hash_table_start;
                    // if (buffer[1] == 791 && thread_cache[i] == 1112)
                    // printf("value : %d parameter is %d vertex : %d start : %d index : %d\n", value, neighbour_numbers[cur_order_index] * parameter, buffer[intersection_order[cur_order_index]], hash_table_start, hash_table_start + value + edge_count);
                    int *cmp = &hash_tables[hash_table_start + value]; // wzb: remove edge count
                    int index = 0;
                    // printf("tid %d cmp : %d && cache is %d\n", tid, *cmp, thread_cache[i]);
                    while (*cmp != -1)
                    {

                        if (*cmp == thread_cache[i])
                        {
                            break;
                        }
                        cmp = cmp + edge_count;
                        index++;
                        if (index == bucket_size)
                        {
                            value++;
                            index = 0;
                            if (value == hash_table_length)
                                value = 0;
                            cmp = &hash_tables[hash_table_start + value];
                        }
                        // printf("tid %d cmp : %d && cache is %d\n", tid, *cmp, thread_cache[i]);
                    }
                    if (*cmp == -1)
                        thread_cache[i] = -1;
                }
                __syncwarp();
                if (thread_cache[i] != -1)
                {
                    coalesced_group active = coalesced_threads();
                    // 如果是最后一层，则写入中间结果
                    if (level == h - 1)
                    {
                        // printf("level is %d cache is %d active.size is %d\n", level, thread_cache[i], active.size());
                        if (active.thread_rank() == 0)
                            warpsum[in_block_wid] = warpsum[in_block_wid] + active.size();
                    }
                    // 如果不是，则写回
                    else
                    {
                        ir[active.thread_rank() + ptr[in_block_wid] + wid * max_degree + 216 * 1024 / 32 * max_degree * level] = thread_cache[i];
                        // ir[active.thread_rank() + ptr[in_block_wid] + wid * max_degree + level * 1024 * 216 / 32 * max_degree] = thread_cache[i];
                        ptr[in_block_wid] = ptr[in_block_wid] + active.size();
                    }
                }
                __syncwarp();
            }
            if (level == h - 1)
            {
                // if (lid == 0)
                // printf("buffer[0] is %d buffer[1] is %d number is %d\n", buffer[0], buffer[1], warpsum[in_block_wid]);
                level--;
                delete (thread_cache);
                continue;
            }
            else
            {
                ir_number[level] = ptr[in_block_wid] - 1;
                // if (lid == 0)
                // printf("not last vertex , neighbor is %d\n", ptr[in_block_wid]);
                if (ptr[in_block_wid] == 0)
                {
                    if (level > 1)
                        level--;
                    else
                        ir_number[level] = 0;
                    delete (thread_cache);
                    continue;
                }
            }
            delete (thread_cache);
        }
        // 当前层不为空，取下一个元素
        else
        {
            ir_number[level] = ir_number[level] - 1;
        }
        __syncwarp();
        // 更新临时结果
        int tmp2 = ir_number[level];
        int tmp1 = ir[(level)*1024 * 216 / 32 * max_degree + wid * max_degree + tmp2];
        buffer[level] = tmp1;
        // if (lid == 0)
        // {
        //     if (level == 1)
        //         printf("Current buffer is {%d,%d}\n", buffer[0], buffer[1]);
        //     if (level == 2)
        //         printf("Current buffer is {%d,%d,%d}\n", buffer[0], buffer[1], buffer[2]);
        // }
        level++;
        __syncwarp();
        ir_number[level] = 0;
        // if (lid == 0)
        // printf("-------------------\n");
        FLAG++;
    }
    delete (ir_number);
    delete (buffer);
    if (lid == 0)
    {
        atomicAdd(sum, warpsum[in_block_wid]);
    }
}

int main(int argc, char *argv[])
{
    // load graph file
    string infilename = "../dataset/graph/as20000102_adj.mmio";
    // string infilename = "/data/zh_dataset/cit-Patents_adj.mmio";
    // string infilename = "../dataset/graph/test2.mmio";
    // string infilename = "../dataset/graph/test3.mmio";
    // string infilename = "../dataset/graph/clique_6.mmio";
    loadgraph(infilename);
    bucket_num = edgeCount * load_factor_inverse / bucket_size;
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
    HRR(cudaMalloc(&d_csr_row_offset, (uCount + 1) * sizeof(int)));
    printf("-------------translate into CSR and saved in Device---------------------\n");
    // 查看下可用share memory的最大值
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 假设设备号为0
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    cout << "share memory size : " << sharedMemPerBlock << endl;
    translateIntoCSRKernel<<<216, 1024>>>(d_edgelist, edgeCount, uCount, d_csr_column_index, d_csr_row_value);
    // translateIntoCSRKernel<<<216, 1024, uCount * sizeof(int)>>>(d_edgelist, edgeCount, uCount, d_csr_column_index, d_csr_row_value);
    int *csr_row_value = new int[uCount];
    HRR(cudaMemcpy(csr_row_value, d_csr_row_value, uCount * sizeof(int), cudaMemcpyDeviceToHost));
    printf("csr_value is : ");
    for (int i = 0; i < 20; i++)
    {
        printf("%d ", csr_row_value[i]);
    }
    delete[] csr_row_value;
    printf("\n");
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
    HRR(cudaMemcpy(&max_degree, d_max_degree, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "max degree is: " << max_degree << std::endl;

    // get row_offset for CSR
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);

    // build hash table in device
    int *d_hash_tables;
    HRR(cudaMalloc(&d_hash_tables, load_factor_inverse * edgeCount * sizeof(int)));
    HRR(cudaMemset(d_hash_tables, -1, load_factor_inverse * edgeCount * sizeof(int)));
    int *d_hash_tables_offset;
    HRR(cudaMalloc(&d_hash_tables_offset, (uCount + 1) * sizeof(int)));
    int *d_hash_table_parameters;
    HRR(cudaMalloc(&d_hash_table_parameters, uCount * sizeof(int)));
    printf("uCount : %d\n", uCount);
    // 写hash_table_offset
    buildHashTableOffset<<<216, 1024>>>(d_hash_tables_offset, d_csr_row_offset, d_csr_row_value, uCount, parameter);
    buildHashTable<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables, d_hash_table_parameters, d_csr_row_offset, d_edgelist, uCount, edgeCount, bucket_size, load_factor_inverse);

    // int hash_tables[load_factor_inverse * edgeCount];
    // cudaMemcpy(hash_tables, d_hash_tables, load_factor_inverse * edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", hash_tables[i]);
    // }
    // printf("\n");

    // DFS
    int *d_ir; // intermediate result;
    // mzh : 这里有问题
    HRR(cudaMalloc(&d_ir, 216 * 10240 / 32 * max_degree * pattern_vertex_number));
    HRR(cudaMemset(d_ir, -1, 216 * 10240 / 32 * max_degree * pattern_vertex_number)); // 初始值默认为-1
    cout << "ir memory size is : " << 216 * 1024 / 32 * max_degree * pattern_vertex_number << endl;
    // int *final_result; // 暂时先用三角形考虑。
    // HRR(cudaMalloc(&d_ir_ptr, 216 * 1024 / 32 * pattern_vertex_number));
    // 先提前假定一下三角形的顺序
    int intersection_orders[4] = {0, 0, 1};
    int intersection_offset[4] = {0, 1, 3};
    int *d_intersection_orders;
    int intersection_size = 4;
    cudaMalloc(&d_intersection_orders, intersection_size * sizeof(int));
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, 16, cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    cudaMalloc(&d_intersection_offset, intersection_size * sizeof(int));
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, 16, cudaMemcpyHostToDevice));
    int h = 3;
    int *d_sum;
    cudaMalloc(&d_sum, 4);
    cudaMemset(d_sum, 0, 4);

    // set up timing variables
    float gpu_elapsed_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    // copy from host to device
    cudaEventRecord(gpu_start, 0);
    DFSKernel<<<216, 256>>>(uCount, edgeCount, max_degree, h, bucket_size, parameter, d_intersection_orders, d_intersection_offset, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    // report results
    std::cout << "The gpu took: " << gpu_elapsed_time << " milli-seconds" << std::endl;

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