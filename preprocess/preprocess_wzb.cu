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
    // for (int i = threadIdx.x; i < vertexCount; i += blockDim.x)
    // {
    //     atomicAdd(&csr_row_value[i], s_csr_row_value[i]);
    // }
}

__global__ void buildHashTableOffset(int *hash_tables_offset, int *csr_row_offset, int *csr_row_value, int vertex_count, int parameter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    csr_row_offset[vertex_count] = csr_row_offset[vertex_count - 1] + csr_row_value[vertex_count - 1];
    for (int i = tid; i < vertex_count + 1; i += stride)
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
    int bucket_number = edge_count * load_factor_inverse / bucket_size;
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
        if (key == 1112 && vertex == 0)
            printf("value : %d parameter is %d vertex : 0 bucket_len = %d start : %d index is :%d\n", value, hash_table_length, hash_tables[hash_table_start + value], hash_table_start, hash_table_start + value + (hash_tables[hash_table_start + value] + 1) * edge_count);
        // 按列存储

        int index = 0;
        // 找到当前hash_tables中不满的bucket
        // if (i == 10000)
        //     printf("start %d end %d len %d key %d vertex %d\n", hash_table_start, hash_table_end, hash_table_length, key, vertex);
        // while (hash_tables[hash_table_start + value + index * edge_count] != -1 )
        if (hash_table_start + value + index * edge_count > 4 * edge_count)
            printf("FUCK!!! index : %d value : %d\n", index, value);
        while (atomicCAS(&hash_tables[hash_table_start + value + index * edge_count], -1, key) != -1)
        {
            if (hash_table_start + value + index * edge_count > 4 * edge_count)
                printf("FUCK!!! index : %d value : %d\n", index, value);
            // 这里要注意，因为是向后增加元素，万一这是最后一个元素怎么办？会造成前面的元素多，后面的元素少。
            index++;
            if (index == bucket_size)
            {
                index = 0;
                value++;
                if (value == hash_table_length)
                    value = 0;
            }
        }
        if (key == 1112 && vertex == 0)
            printf("True index is : %d, value is %d\n", hash_table_start + value + index * edge_count, value);
        if (hash_table_start + value + index * edge_count == 12773)
            printf("why : %d %d\n", key, vertex);
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

__inline__ __device__ void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

__device__ bool search_in_hashtable(int x, int edge_count, int bucket_size, int k, int hash_table_len, int *hash_table)
{
    int value = x % k;
    int *cmp = hash_table;
    int index = 0;
    while (*cmp != -1)
    {
        if (*cmp == x)
        {
            return true;
        }
        cmp = cmp + edge_count;
        index++;
        if (index == bucket_size)
        {
            value++;
            index = 0;
            if (value == hash_table_len)
                value = 0;
            cmp = &hash_table[value];
        }
        // printf("tid %d cmp : %d && cache is %d\n", tid, *cmp, thread_cache[i]);
    }
    return false;
}
// h : height of subtree; h = pattern vertex number
__global__ void DFSKernel(int vertex_count, int edge_count, int max_degree, int h, int bucket_size, int parameter, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, int *hash_tables_offset, int *hash_tables, int *candidates_of_all_warp, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadid
    int wid = tid / 32;                              // warpid
    int in_block_wid = threadIdx.x / 32;
    int lid = tid % 32; // landid
    int stride = blockDim.x * gridDim.x / 32;
    int *candidate_number_array = new int[h]; // 记录一下每一层保存的数据大小
    int *next_candidate_array = new int[h];   // 记录一下每一层保存的数据大小
    int *mapping = new int[h];                // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + wid * h * 32 * 216 * max_degree;
    int my_count = 0;
    __shared__ int warpsum[32];
    warpsum[in_block_wid] = 0;
    // each warp process a subtree (an probe item)

    for (int start_vertex = wid; start_vertex < vertex_count; start_vertex += stride)
    {
        mapping[0] = start_vertex;
        int level = 0;
        for (;;)
        {
            level++;
            if (lid == 0)
                printf("start_vertex %d level %d\n", start_vertex, level);
            int &candidate_number = candidate_number_array[level];
            candidate_number = 0;

            // find possible connection and maintain in S
            int intersection_order_start = intersection_offset[level - 1];
            int intersection_order_end = intersection_offset[level];
            int intersection_order_length = intersection_order_end - intersection_order_start;
            int intersection_order[2]; // wzb: refine it
            for (int i = 0; i < intersection_order_length; i++)
            {
                intersection_order[i] = intersection_orders[intersection_order_start + i];
            }
            // get degree
            int neighbour_numbers[2];
            neighbour_numbers[0] = csr_row_value[mapping[intersection_order[0]]];
            for (int i = 1; i < intersection_order_length; i++)
            {
                int degree = csr_row_value[mapping[intersection_order[i]]];
                neighbour_numbers[i] = degree;
                if (degree < neighbour_numbers[0])
                {
                    swap(intersection_order[i], intersection_order[0]);
                    swap(neighbour_numbers[i], neighbour_numbers[0]);
                }
            }

            // Start intersection, load neighbor of m1 into my_candidates
            int *my_candidates = my_candidates_for_all_mapping + level * max_degree;
            int cur_vertex = intersection_order[0];
            int *cur_neighbor_list = csr_column_index + csr_row_offset[mapping[cur_vertex]];
            for (int i = lid; i < neighbour_numbers[0]; i += 32)
            {
                my_candidates[i] = cur_neighbor_list[i];
            }
            // intersect
            int candidate_number_previous = neighbour_numbers[0];
            candidate_number = candidate_number_previous;
            if (lid == 0)
                printf("candidate_number %d \n", candidate_number);
            for (int j = 1; j < intersection_order_length; j++)
            {
                cur_vertex = intersection_order[j];
                int *cur_hashtable = hash_tables + hash_tables_offset[mapping[cur_vertex]];
                int len = hash_tables_offset[mapping[cur_vertex] + 1] - hash_tables_offset[mapping[cur_vertex]]; // len记录当前hash_table的长度
                for (int i = lid; i < candidate_number_previous; i += 32)
                {
                    search_in_hashtable(my_candidates[i], edge_count, bucket_size, len * parameter, len, cur_hashtable);
                }
                candidate_number_previous = candidate_number;
            }

            if (level == h - 1)
            {
                if (lid == 0)
                    my_count += candidate_number;
                level--;
            }
            for (;; level--)
            {
                if (level == 0)
                    break;
                next_candidate_array[level]++;
                if (next_candidate_array[level] < candidate_number_array[level])
                {
                    mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == 0)
                break;
        }
    }
    delete (mapping);
    delete (candidate_number_array);
    delete (next_candidate_array);
    // if (wid == 0)
    // {
    //     // atomicAdd(sum, warp_sum);
    //     printf("final sum is %d tid : %d\n", warp_sum, tid);
    // }
    if (lid == 0)
    {
        atomicAdd(sum, warpsum[in_block_wid]);
    }
}

int main(int argc, char *argv[])
{
    // load graph file
    // string infilename = "../dataset/graph/as20000102_adj.mmio";
    // string infilename = "../dataset/graph/cit-Patents_adj.mmio";
    string infilename = "../dataset/graph/test.mmio";
    // string infilename = "../dataset/graph/test3.mmio";
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

    int hash_tables[load_factor_inverse * edgeCount];
    cudaMemcpy(hash_tables, d_hash_tables, load_factor_inverse * edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++)
    {
        printf("%d ", hash_tables[i]);
    }
    printf("\n");

    // DFS
    int *d_ir; // intermediate result;
    HRR(cudaMalloc(&d_ir, 216 * 1024 / 32 * max_degree * pattern_vertex_number));
    HRR(cudaMemset(d_ir, -1, 216 * 1024 / 32 * max_degree * pattern_vertex_number)); // 初始值默认为-1
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
    int csr_row_value[uCount];
    HRR(cudaMemcpy(csr_row_value, d_csr_row_value, uCount * sizeof(int), cudaMemcpyDeviceToHost));
    // printf("csr_value is : ");
    // for (int i = 0; i < uCount; i++)
    // {
    //     printf("%d ", csr_row_value[i]);
    // }
    // printf("\n");
    DFSKernel<<<1, 32>>>(uCount, edgeCount, max_degree, h, bucket_size, parameter, d_intersection_orders, d_intersection_offset, d_csr_row_offset, d_csr_row_value, d_csr_column_index, d_hash_tables_offset, d_hash_tables, d_ir, d_sum);
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
}