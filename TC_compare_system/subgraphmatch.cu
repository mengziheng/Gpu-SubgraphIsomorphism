#include <sstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cooperative_groups.h>
#include "hash_table.cuh"
#include "subgraph_match.cuh"

using namespace cooperative_groups;
using namespace std;
// h : height of subtree; h = pattern vertex number
__inline__ __device__ bool checkDuplicate(int *mapping, int &level, int item)
{
    for (int i = 0; i < level; i++)
        if (mapping[i] == item)
            return true;
    return false;
}
__inline__ __device__ bool checkRestriction(int *mapping, int &level, int item, int *restriction)
{
#if withRestriction == 1
    if (item < mapping[restriction[level]])
        return false;
    return true;
#else
    return false;
#endif
}

__inline__ __device__ void loadNextVertex(int &start_index, int &this_chunk_index_end, int *G_INDEX, int &lid, int &chunk_size)
{
    start_index++;
    if (start_index == this_chunk_index_end)
    {
        if (lid == 0)
        {
            start_index = atomicAdd(G_INDEX, chunk_size);
        }
        start_index = __shfl_sync(FULL_MASK, start_index, 0);
        this_chunk_index_end = start_index + chunk_size;
    }
}

__global__ void DFSKernelForGeneral(int chunk_size, int index_length, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *restriction, int *csr_row_offset, int *csr_row_value, int *csr_column_index, long long *hash_tables_offset, int *hash_tables, int *edge_list, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int mapping[H];                                         // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    unsigned long long my_count = 0;
    __shared__ unsigned long long shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size;
    int this_chunk_index_end = start_index + chunk_size;
    int level;
    // each warp process a subtree
    for (; start_index < index_length; loadNextVertex(start_index, this_chunk_index_end, G_INDEX, lid, chunk_size))
    {
#if useVertexAsStart == 1
        mapping[0] = start_index;
        level = 0;
#else
        mapping[0] = edge_list[start_index * 2];
        mapping[1] = edge_list[start_index * 2 + 1];
        level = 1;
        if (checkRestriction(mapping, level, mapping[1], restriction))
            continue;
#endif
        for (;;)
        {
            level++;
            int candidate_number;
            candidate_number = 0;
            // find possible connection and maintain in S
            int subgraph_adj_start = subgraph_offset[level - 1];
            int subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
            // get degree
            int min_degree;
            int min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start]];
            min_degree = csr_row_value[min_degree_vertex];
            for (int i = 1; i < subgraph_degree; i++)
            {
                if (csr_row_value[mapping[subgraph_adj[subgraph_adj_start + i]]] < min_degree)
                {
                    min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                    min_degree = csr_row_value[min_degree_vertex];
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;
            int *neighbor_list_of_min_degree_vertex = csr_column_index + csr_row_offset[min_degree_vertex];

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (subgraph_degree == 1)
            {
                for (int i = lid; i < min_degree; i += 32)
                {
                    my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                }
            }
            // intersect
            candidate_number = min_degree;
            int *my_read_candidates = neighbor_list_of_min_degree_vertex;
            for (int j = 0, is_not_last = subgraph_degree - 2; j < subgraph_degree; j++)
            {
                processing_vertex_in_map = mapping[subgraph_adj[subgraph_adj_start + j]];
                if (processing_vertex_in_map == min_degree_vertex)
                    continue;
                int *cur_hashtable = hash_tables + hash_tables_offset[processing_vertex_in_map];
                int len = int(hash_tables_offset[processing_vertex_in_map + 1] - hash_tables_offset[processing_vertex_in_map]); // len记录当前hash_table的长度

                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                if (level < H - 1 || is_not_last)
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                        if (is_exist)
                        {
                            coalesced_group active = coalesced_threads();
                            // 最后一层不写回
                            my_wirttern_candidates[active.thread_rank() + candidate_number] = item;
                        }
                        candidate_number += __reduce_add_sync(__activemask(), is_exist);
                    }
                }
                else
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                        if (!checkRestriction(mapping, level, item, restriction))
                            my_count += is_exist;
                    }
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                my_read_candidates = my_wirttern_candidates;
                is_not_last--;
            }
            next_candidate_array[level] = candidate_number;
            if (level == H - 1)
            {
                // if (lid == 0)
                // {
                //     my_count += candidate_number;
                // }
                level--;
            }
            for (;; level--)
            {
                if (level == break_level)
                    break;
                next_candidate_array[level]--;
                while (checkRestriction(mapping, level, my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]], restriction) && next_candidate_array[level] >= 0)
                {
                    next_candidate_array[level]--;
                }
                if (next_candidate_array[level] > -1)
                {
                    mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == break_level)
                break;
        }
    }
    // my_count = __reduce_add_sync(FULL_MASK, my_count);
    // if (lid == 0)
    // {
    atomicAdd(&shared_count, my_count);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(sum, shared_count);
    // }
}

__global__ void DFSKernelForClique(int chunk_size, int vertex_count, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *csr_row_offset, int *csr_row_value, int *csr_column_index, long long *hash_tables_offset, int *hash_tables, int *edge_list, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int mapping[H];                                         // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    unsigned long long my_count = 0;
    __shared__ unsigned long long shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size;
    int this_chunk_index_end = start_index + chunk_size;
    int level;
    // each warp process a subtree
    while (start_index < vertex_count)
    {
        mapping[0] = start_index;
        level = 0;
        for (;;)
        {
            level++;
            int candidate_number;
            candidate_number = 0;
            // find possible connection and maintain in S
            int subgraph_adj_start = subgraph_offset[level - 1];
            int subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
            // get degree
            int min_degree;
            int min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start]];
            min_degree = csr_row_value[min_degree_vertex];
            for (int i = 1; i < subgraph_degree; i++)
            {
                if (csr_row_value[mapping[subgraph_adj[subgraph_adj_start + i]]] < min_degree)
                {
                    min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                    min_degree = csr_row_value[min_degree_vertex];
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;
            int *neighbor_list_of_min_degree_vertex = csr_column_index + csr_row_offset[min_degree_vertex];

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (subgraph_degree == 1)
            {
                for (int i = lid; i < min_degree; i += 32)
                {
                    my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                }
            }
            // intersect
            candidate_number = min_degree;
            int *my_read_candidates = neighbor_list_of_min_degree_vertex;
            for (int j = 0, is_not_last = subgraph_degree - 2; j < subgraph_degree; j++)
            {
                processing_vertex_in_map = mapping[subgraph_adj[subgraph_adj_start + j]];
                if (processing_vertex_in_map == min_degree_vertex)
                    continue;
                int *cur_hashtable = hash_tables + hash_tables_offset[processing_vertex_in_map];
                int len = int(hash_tables_offset[processing_vertex_in_map + 1] - hash_tables_offset[processing_vertex_in_map]); // len记录当前hash_table的长度

                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                if (level < H - 1 || is_not_last)
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                        if (is_exist)
                        {
                            coalesced_group active = coalesced_threads();
                            // 最后一层不写回
                            my_wirttern_candidates[active.thread_rank() + candidate_number] = item;
                        }
                        candidate_number += __reduce_add_sync(__activemask(), is_exist);
                    }
                }
                else
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable);
                        // candidate_number += __reduce_add_sync(__activemask(), is_exist);
                        my_count += is_exist;
                    }
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                my_read_candidates = my_wirttern_candidates;
                is_not_last--;
            }
            next_candidate_array[level] = candidate_number;
            if (level == H - 1)
            {
                // if (lid == 0)
                // {
                //     my_count += candidate_number;
                // }
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
        start_index++;
        if (start_index == this_chunk_index_end)
        {
            if (lid == 0)
            {
                start_index = atomicAdd(&G_INDEX[0], chunk_size);
            }
            start_index = __shfl_sync(FULL_MASK, start_index, 0);
            this_chunk_index_end = start_index + chunk_size;
        }
    }
    // my_count = __reduce_add_sync(FULL_MASK, my_count);
    // if (lid == 0)
    // {
    atomicAdd(&shared_count, my_count);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(sum, shared_count);
    // }
}

struct arguments SubgraphMatching(int process_id, int process_num, struct arguments args, string Infilename, string pattern, char *argv[])
{
    Infilename = argv[1];
    pattern = argv[2];
    load_factor = atof(argv[4]);
    bucket_size = atoi(argv[5]);
    block_size = atoi(argv[6]);
    block_number = atoi(argv[7]);
    chunk_size = atoi(argv[8]);

    int *d_adjcant, *d_degree_list, *d_vertex, *d_degree_offset, *d_edge_list;
    tie(d_adjcant, d_degree_list, d_vertex, d_degree_offset, d_edge_list) = loadGraphWithName(Infilename, pattern);

    // printGpuInfo();

    int *d_hash_tables;
    long long *d_hash_tables_offset;
    int max_degree;
    long long bucket_num;
    tie(d_hash_tables_offset, d_hash_tables, max_degree, bucket_num) = buildHashTable(d_adjcant, d_vertex, d_degree_list);

    int *d_ir; // intermediate result;
    // refine the malloc
    HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));
    // cout << "ir memory size is : " << 216 * 32 * max_degree * H * sizeof(int) / 1024 / 1024 << "MB" << endl;
    int intersection_orders[MAX_SIZE_FOR_ARRAY];
    int intersection_offset[MAX_SIZE_FOR_ARRAY];
    int restriction[MAX_SIZE_FOR_ARRAY];
    int intersection_size = 1;
    int restriction_size = 1;
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
    double max_time = 0;
    double min_time = 1000;
    double ave_time = 0;
    int repeat_time = 1;

    time_start = clock();
    for (int i = 0; i < repeat_time; i++)
    {
        HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        time_start = clock();
        int length;
#if useVertexAsStart == 1
        length = vertex_count;
#else
        length = edge_count;
#endif
        DFSKernelForGeneral<<<block_size, block_number>>>(chunk_size, length, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_degree_offset, d_degree_list, d_adjcant, d_hash_tables_offset, d_hash_tables, d_edge_list, d_ir, d_sum, G_INDEX);
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
        if (cmp_time > max_time)
            max_time = cmp_time;
        if (cmp_time < min_time)
            min_time = cmp_time;
        ave_time += cmp_time;
        HRR(cudaFree(d_ir));
        HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));
        HRR(cudaMemcpy(G_INDEX, &temp, sizeof(int), cudaMemcpyHostToDevice));
    }

    // std::cout << "max time: " << max_time * 1000 << " ms" << std::endl;
    // std::cout << "min time: " << min_time * 1000 << " ms" << std::endl;
    // std::cout << "average time: " << ave_time / repeat_time * 1000 << " ms" << std::endl;

    long long sum;
    cudaMemcpy(&sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    cout << pattern << " count is " << sum << endl;

    args.time = cmp_time;
    args.count = sum;
    return args;
}

// int main(int argc, char *argv[])
// {
//     cudaSetDevice(0);
//     string Infilename = "4-cycle";
//     string pattern = "4-cycle";
//     if (argc > 1)
//     {
//         Infilename = argv[1];
//         pattern = argv[2];
//         load_factor = atof(argv[3]);
//         bucket_size = atoi(argv[4]);
//         block_size = atoi(argv[5]);
//         block_number = atoi(argv[6]);
//         chunk_size = atoi(argv[7]);
//     }

//     SubgraphMatching(Infilename, pattern);
// }
