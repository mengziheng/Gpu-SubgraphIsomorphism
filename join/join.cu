#include <iostream>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/set_operations.h>

float load_factor = 0.25;
int load_factor_inverse = 1 / load_factor;
int bucket_size = 4;
int local_register_num = 2;

__device__ int getNeighborNum(int index, int *offset)
{
    return offset[index + 1] - offset[index];
}

__device__ bool ifExistInHashTable(int vertex, int bucket, int *hash_tables_offset, int *hash_tables){
    int start = }

// h : height of subtree
__global__ void DFSKernel(int *vertexlist, int vertexnum, int h, int *intersection_orders, int *intersection_offset, int *csr_row_offset, int *csr_column_index, int *hash_tables_offset, int *hash_tables)
{
    int tid = threadIdx.x; // threadid
    int wid = tid / 32;    // warpid
    int lid = tid % 32;    // landid
    int level = 0;         // level of subtree,start from 0,not root for tree,but root for subtree
    int level_number[h];   // 记录一下每一层保存的数据大小
    /*如何保存中间结果？*/
    // 1024threads/block 32threads/warp --> 32threads/block
    /*need to modify use what kind of basic unit to process a subtree*/
    // each warp process a subtree (an probe item)
    /*need to modify to write a loop to process,cause that number of warp < vertex number*/
    if (wid < vertexnum)
    {
        int probeitem = vertexlist[tid];
        while (true)
        {
            if ()
                /*是否要用candidate*/
                if (level == h - 1)
                {
                    // 这里是每一个线程去执行一个用hash tabel进行交集的运算
                    // 用buffer存储当前的一个中间结果

                    // 首先对buffer[]按照neighbour排序，按照neighbour从小到大顺序进行交集。
                    // 是否可以用warp去优化一下这个排序
                    int intersection_order_start = intersection_offset[level];
                    int intersection_order_offset = intersection_offset[level + 1] - 1;
                    int intersection_order_length = intersection_order_offset - intersection_order_start + 1;
                    int intersection_order[intersection_order_length];
                    for (int i = 0; i < intersection_order_length; i++)
                        intersection_order[i] = intersection_orders[intersection_order_start + i];
                    int neighbour_numbers[intersection_order_length];
                    // 将需要做交集的点的邻居数量都读入到寄存器中
                    for (int i = 0; i < intersection_order_length; i++)
                        neighbour_numbers[i] = getNeighborNum(buffer[intersection_order[i]], csr_row_offset, csr_column_index);
                    // 用一个线程去完成冒牌排序，记录排序后的顶点顺序，即最终的intersection顺序
                    int min = neighbour_numbers[0];
                    int min_index;
                    for (int i = 0; i < intersection_order_length; i++)
                    {
                        for (int j = i; j < intersection_order_length; j++)
                        {
                            if (neighbour_numbers[j] < min)
                            {
                                min = neighbour_numbers[j];
                                min_index = 1;
                            }
                        }
                        intersection_order[i] = min_index;
                        int tmp = neighbour_numbers[0];
                        neighbour_numbers[0] = neighbour_numbers[min_index];
                        neighbour_numbers[min_index] = tmp;
                    }

                    // 开始按intersection_order做join操作
                    // 首先取第一个intersection_order的顶点的邻居，每个顶点平均分配这些邻居节点,用local_meemory保存
                    int cur_vertex = buffer[intersection_order[0]];
                    int neighbor_num = neighbour_numbers[0];
                    int thread_cache_size = neighbor_num / 32;
                    int thread_cache[thread_cache_size];
                    int index_for_thread_cache = 0; // 这是用来
                    int intersection_order;
                    // 初始化cache
                    for (int i = lid; i < neighbor_num; i += 32)
                    {
                        thread_cache[index_for_thread_cache] = csr_column_index[csr_row_offset[cur_vertex] + i];
                    }
                    // 对所有的邻居集合都要进行一次intersection，因此需要for循环
                    // 如果有三个顶点，那最终只需要intersection两次。而最后一次需要写回，因此是3-2。这解释了下面为什么-2
                    int cur_order;
                    for (cur_order = 1; cur_order < intersection_order_length - 2; cur_order++)
                    {
                        // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                        for (int i = 0; i < thread_cache_size; i++)
                        {
                            int value = thread_cache[index_for_thread_cache] % (neighbour_numbers[cur_order] * load_factor_inverse / bucket_size);
                            int hash_tables_start = hash_tables_offset[thread_cache[index_for_thread_cache]];
                            for (int j = 0; j < bucket_size; j++)
                            {
                                if (hash_tables[hash_tables_start + value * bucket_size + j] == thread_cache[index_for_thread_cache])
                                    break;
                                if (hash_tables[hash_tables_start + value * bucket_size + j] == -1)
                                    thread_cache[index_for_thread_cache] = -1;
                            }
                            index_for_thread_cache++;
                        }
                    }
                    // 对最后一个顶点进行交集操作之后，就可以直接写回了。
                    // 每个thread处理一个元素的搜索，但是元素可能不止32个，因此要for循环来全部处理
                    // 这边可以再优化一下，每次存储的值都写到数组的前几位，这样可以减少循环。
                    for (int i = 0; i < thread_cache_size; i++)
                    {
                        int value = thread_cache[index_for_thread_cache] % (neighbour_numbers[cur_order] * load_factor_inverse / bucket_size);
                        int hash_tables_start = hash_tables_offset[thread_cache[index_for_thread_cache]];
                        for (int j = 0; j < bucket_size; j++)
                        {
                            if (hash_tables[hash_tables_start + value * bucket_size + j] == thread_cache[index_for_thread_cache])
                                break;
                            if (hash_tables[hash_tables_start + value * bucket_size + j] == -1)
                                thread_cache[index_for_thread_cache] = -1;
                        }
                        index_for_thread_cache++;
                    }
                    level = level - 1;
                    continue;
                }
                else
                {
                    intersection();
                }
        }
    }
}

// hash table的默认是
// k is parameter of hash function,hash function is % k;
__device__ int intersection(int vertexs, int k, int *hashtable, int *result)
{
    int value = vertexs % k;
    for (int j = 0; j < bucket_size; j++)
    {
        if (hashtable[value * bucket_size + j] == vertexs)
            return vertexs;
        if (hashtable[value * bucket_size + j] == -1)
            return -1;
    }
    return -1;
}

// Join Order
// finally get histogram and ir for each level
void Join()
{
}

int main()
{
    kernel<<<1000, 1024>>>();
    cudaDeviceSynchronize();
}