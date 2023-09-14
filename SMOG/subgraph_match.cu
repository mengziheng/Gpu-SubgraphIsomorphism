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
#ifdef withDuplicate
    if (restriction[level] == -1)
    {
        for (int i = 0; i < level; i++)
            if (mapping[i] == item)
                return true;
    }
#endif
#ifdef withRestriction
    if (restriction[level] == -1)
        return false;
    if (item < mapping[restriction[level]])
        return false;
    return true;
#else
    return false;
#endif
}

__inline__ __device__ void loadNextVertex(int &start_index, int &this_chunk_index_end, int *G_INDEX, int &lid, int &chunk_size, int &process_num)
{
    start_index += process_num;
    if (start_index == this_chunk_index_end)
    {
        if (lid == 0)
        {
            start_index = atomicAdd(G_INDEX, chunk_size * process_num);
        }
        start_index = __shfl_sync(FULL_MASK, start_index, 0);
        this_chunk_index_end = start_index + chunk_size * process_num;
    }
}

__global__ void DFSKernelForClique(int *reuse, int process_id, int process_num, int chunk_size, int index_length, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *restriction, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int candidate_number_array[H];
    int mapping[H]; // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    // int my_count = 0;
    // __shared__ int shared_count;
    unsigned long long my_count = 0;
    __shared__ unsigned long long shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size * process_num + process_id;
    int this_chunk_index_end = start_index + chunk_size * process_num;
    int level;
    // each warp process a subtree
    for (; start_index < index_length; loadNextVertex(start_index, this_chunk_index_end, G_INDEX, lid, chunk_size, process_num))
    {
#ifdef useVertexAsStart
        mapping[0] = start_index;
        level = 0;
#else
        mapping[0] = vertex[start_index];
        mapping[1] = adjcant[start_index];
        level = 1;
        if (checkRestriction(mapping, level, mapping[1], restriction))
            continue;
#endif
        for (;;)
        {
            level++;
            int &candidate_number = candidate_number_array[level];
            next_candidate_array[level] = -1;
            candidate_number = 0;
            int subgraph_adj_start;
            int min_degree_vertex;
            int min_degree;
            int subgraph_degree;
            int *neighbor_list_of_min_degree_vertex;
            if (reuse[level] == -1)
            {
                // find possible connection and maintain in S
                subgraph_adj_start = subgraph_offset[level - 1];
                subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
                // get degree
                min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start]];
                int cur_degree;
                min_degree = degree_offset[min_degree_vertex + 1] - degree_offset[min_degree_vertex];
                for (int i = 1; i < subgraph_degree; i++)
                {
                    cur_degree = degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]] + 1] - degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]]];
                    if (cur_degree < min_degree)
                    {
                        min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                        min_degree = cur_degree;
                    }
                }
                neighbor_list_of_min_degree_vertex = adjcant + degree_offset[min_degree_vertex];
            }
            else
            {
                // find possible connection and maintain in S
                subgraph_adj_start = subgraph_offset[level - 1];
                subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
                min_degree = candidate_number_array[reuse[level]];
                min_degree_vertex = -1;
                neighbor_list_of_min_degree_vertex = my_candidates_for_all_mapping + reuse[level] * max_degree;
            }
            // if (mapping[0] == 0 && mapping[1] == 1 && lid == 0)
            // printf("level : %d min_degree = %d \n", level, min_degree);
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (subgraph_degree == 1 && reuse[level] == -1)
                for (int i = lid; i < min_degree; i += 32)
                {
                    my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                }
            if (subgraph_degree == 1 && reuse[level] != -1)
            {
                if (level < H - 1)
                    for (int i = lid; i < candidate_number_array[reuse[level]]; i += 32)
                    {
                        my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                    }
                else
                {
                    for (int i = lid; i < candidate_number_array[reuse[level]]; i += 32)
                    {
                        // printf("level : %d reuse[level] : %d", level, reuse[level]);
                        if (!checkRestriction(mapping, level, neighbor_list_of_min_degree_vertex[i], restriction))
                        {
                            my_count += 1;
                            // printf("my count : %d", my_count);
                        }
                    }
                }
            }
            // if (mapping[0] == 0 && mapping[1] == 1 && lid == 0)
            // printf("min_degree_vertex : %d min_degree : %d level : %d reuse[level] : %d candidate_number_array[level - 1] : %d\n", min_degree_vertex, min_degree, level, reuse[level], candidate_number_array[level - 1]);
            // intersect
            candidate_number = min_degree;
            int *my_read_candidates;
            if (reuse[level] == -1)
                my_read_candidates = neighbor_list_of_min_degree_vertex;
            else
                my_read_candidates = my_candidates_for_all_mapping + reuse[level] * max_degree;
            for (int j = 0, is_not_last = subgraph_degree - 2; j < subgraph_degree; j++)
            {
                processing_vertex_in_map = mapping[subgraph_adj[subgraph_adj_start + j]];
                if (processing_vertex_in_map == min_degree_vertex || subgraph_adj[subgraph_adj_start + j] == -1)
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
                        {
                            my_count += is_exist;
                        }
                    }
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                my_read_candidates = my_wirttern_candidates;
                is_not_last--;
            }
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
                next_candidate_array[level]++;
                while (checkRestriction(mapping, level, my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]], restriction) && next_candidate_array[level] < candidate_number_array[level])
                {
                    next_candidate_array[level]++;
                }
                if (next_candidate_array[level] < candidate_number_array[level])
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

__global__ void DFSKernelForGeneral(int process_id, int process_num, int chunk_size, int index_length, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *restriction, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    int candidate_number[H];
    int mapping[H]; // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    int my_count = 0;
    __shared__ int shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size * process_num + process_id;
    int this_chunk_index_end = start_index + chunk_size * process_num;
    int level;
    // if (lid == 0 && wid == 0)
    // each warp process a subtree
    for (; start_index < index_length; loadNextVertex(start_index, this_chunk_index_end, G_INDEX, lid, chunk_size, process_num))
    {
#ifdef useVertexAsStart
        mapping[0] = start_index;
        level = 0;
#else
        mapping[0] = vertex[start_index];
        mapping[1] = adjcant[start_index];
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
            int cur_degree;
            min_degree = degree_offset[min_degree_vertex + 1] - degree_offset[min_degree_vertex];
            for (int i = 1; i < subgraph_degree; i++)
            {
                cur_degree = degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]] + 1] - degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]]];
                if (cur_degree < min_degree)
                {
                    min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                    min_degree = cur_degree;
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;
            int *neighbor_list_of_min_degree_vertex = adjcant + degree_offset[min_degree_vertex];

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
                        {
                            // {
                            //     printf("reslut : ");
                            //     for (int t = 0; t < H - 1; t++)
                            //         printf("%d ", mapping[t]);
                            //     printf("%d \n", item);
                            // }
                            my_count += is_exist;
                        }
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

struct arguments SubgraphMatching(int process_id, int process_num, struct arguments args, char *argv[])
{
    int deviceCount;
    HRR(cudaGetDeviceCount(&deviceCount));
    HRR(cudaSetDevice((process_id) % deviceCount));
    string Infilename = argv[1];
    string pattern = argv[2];
    load_factor = atof(argv[4]);
    bucket_size = atoi(argv[5]);
    block_size = atoi(argv[6]);
    block_number = atoi(argv[7]);
    chunk_size = atoi(argv[8]);

    int *d_adjcant, *d_vertex, *d_degree_offset;
    int max_degree;
    tie(d_adjcant, d_vertex, d_degree_offset, max_degree) = loadGraphWithName(Infilename, pattern);
    // printGpuInfo();
    printf("max degree is : %d\n", max_degree);
    int *d_hash_tables;
    long long *d_hash_tables_offset;
    long long bucket_num;
    tie(d_hash_tables_offset, d_hash_tables, bucket_num) = buildHashTable(d_adjcant, d_vertex, d_degree_offset);

    int *d_ir; // intermediate result;
    // refine the malloc
    HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));

    // cout << "ir memory size is : " << 216 * 32 * max_degree * H * sizeof(int) / 1024 / 1024 << "MB" << endl;

    int *d_intersection_orders;
    HRR(cudaMalloc(&d_intersection_orders, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    HRR(cudaMalloc(&d_intersection_offset, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_restriction;
    HRR(cudaMalloc(&d_restriction, restriction_size * sizeof(int)));
    HRR(cudaMemcpy(d_restriction, restriction, restriction_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_reuse;
    HRR(cudaMalloc(&d_reuse, H * sizeof(int)));
    HRR(cudaMemcpy(d_reuse, reuse, H * sizeof(int), cudaMemcpyHostToDevice));
    int *G_INDEX;
    HRR(cudaMalloc(&G_INDEX, sizeof(int)));

    unsigned long long *d_sum;
    HRR(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    // double start_time = wtime();

    double cmp_time;
    double time_start;
    double max_time = 0;
    double min_time = 1000;
    double ave_time = 0;

    time_start = clock();

    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    for (; process_id < process_num; process_id += deviceCount)
    {
        int temp = block_size * block_number / 32 * chunk_size * process_num + process_id;
        HRR(cudaMemcpy(G_INDEX, &temp, sizeof(int), cudaMemcpyHostToDevice));

        int length;
#ifdef useVertexAsStart
        length = vertex_count;
#else
        length = edge_count;
#endif
        time_start = clock();
        if (pattern.compare("Q5") == 0 || pattern.compare("Q7") == 0|| pattern.compare("Q3") == 0 || pattern.compare("Q0") == 0 || pattern.compare("Q4") == 0 || pattern.compare("Q6") == 0 || pattern.compare("Q8") == 0)
        {
            time_start = clock();
            DFSKernelForClique<<<block_size, block_number>>>(d_reuse, process_id, process_num, chunk_size, length, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_degree_offset, d_adjcant, d_hash_tables_offset, d_hash_tables, d_vertex, d_ir, d_sum, G_INDEX);
        }
        else
        {
            time_start = clock();
            DFSKernelForGeneral<<<block_size, block_number>>>(process_id, process_num, chunk_size, length, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_degree_offset, d_adjcant, d_hash_tables_offset, d_hash_tables, d_vertex, d_ir, d_sum, G_INDEX);
        }
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
        if (cmp_time > max_time)
            max_time = cmp_time;
        if (cmp_time < min_time)
            min_time = cmp_time;
        ave_time += cmp_time;
        // cout << "this time" << cmp_time << ' ' << max_time << endl;
        HRR(cudaFree(d_ir));
        HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));
    }

    HRR(cudaFree(d_hash_tables));
    HRR(cudaFree(d_hash_tables_offset));
    HRR(cudaFree(d_adjcant));
    HRR(cudaFree(d_vertex));
    HRR(cudaFree(d_degree_offset));
    HRR(cudaFree(d_ir));

    std::cout << "time: " << max_time * 1000 << " ms" << std::endl;

    long long sum;
    cudaMemcpy(&sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    cout << pattern << " count is " << sum << endl;

    args.time = max_time;
    args.count = sum;
    return args;
}
