#include <stdio.h>
#include <cuda_runtime.h>

// neighbour list;
// neighbour number;
// candidate return from filter_kernel_1
__global__ void filter_kernel_2(int *neighbour_list, int neighbour_num, int* candidate)
{
    int *buffer; // buffer for candidate
    int buffer_num; // num of candidate
    for (int i = 0; i < neighbour_num; i++)
        buffer = Intersect(buffer,GetNeighbour(neighbour_list[i]),buffer_num);
}

__global__ void filter_kernel_1()
{    
}


void filter()
{
}

int main(int argc, char **argv)
{
}