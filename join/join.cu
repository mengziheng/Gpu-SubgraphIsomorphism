#include <cstdint>
#include <iostream>
#include <vector>
#include <cub/cub.cuh>

// vertex to be joined
// intermediate result table
// level of tree(vertex num of each element in intermediate result table)
// new intermediate result (note that each intermediate result table per level need to be saved)
// first kernel to get global memory size and histogram
// histogram and newir need to be saved
// 问题关键是如何判断利用一些metadata去确定newir大小的上界.
__global__ void kernel(int *ir, vertex g, int level, int* histogram, int *newir)
{
    // each thread process one row of intermediate result table
    int *candidate;
    int *num_of_candidate;
    // each thread first find candidate
    candidate = find_candidate(vertex g, num_of_candidate);
    // prefix sum
    cub::DeviceScan::ExclusiveSum();
    // write into global memory
    for (int i = 0; i < num; i++){
        newir[histogram[threadIdx.x] + i] =  candidate[i];
    }
}

// Join Order
// finally get histogram and ir for each level
void join()
{
}

int main()
{
    kernel<<<1000, 1024>>>();
    cudaDeviceSynchronize();
}