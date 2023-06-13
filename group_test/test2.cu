// 用来描述基本功能，可以实现不需要前缀和就写入

#include <iostream>
#include <cooperative_groups.h>

using namespace cooperative_groups;
using namespace std;

int N = 32;

__global__ void testkernel_1(int *arr, int *buffer, int N)
{
    int tid = threadIdx.x;
    int item = arr[tid];
    if (tid < N)
        if (item == 1)
        {
            coalesced_group active = coalesced_threads();
            printf("%d %d\n", tid, active.thread_rank());
            printf("%d\n", active.size());
            buffer[active.thread_rank()] = item;
        }
}

int main()
{
    int arr[N];
    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
            arr[i] = 0;
        else
            arr[i] = 1;
    }
    for (int i = 0; i < N; i++)
    {
        printf("%d", arr[i]);
    }
    printf("\n");
    int *d_arr;
    cudaMalloc(&d_arr, N * 4);
    int *d_buffer;
    cudaMalloc(&d_buffer, N / 2 * 4);
    cudaMemcpy(d_arr, arr, N * 4, cudaMemcpyHostToDevice);
    testkernel_1<<<1, N>>>(d_arr, d_buffer, N);
    cudaDeviceSynchronize();
    int buffer[16];
    printf("\n");
    cudaMemcpy(buffer, d_buffer, 16 * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++)
    {
        printf("%d", buffer[i]);
    }
    printf("\n");
}