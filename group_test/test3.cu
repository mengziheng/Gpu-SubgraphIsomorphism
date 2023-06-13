// 用来测速

#include <iostream>
#include <cooperative_groups.h>
#include "gputimer.h"

using namespace cooperative_groups;
using namespace std;

int N = 1024;

__global__ void testkernel_1(int *arr, int *buffer, int N)
{
    int tid = threadIdx.x;
    int item = arr[tid];
    if (tid < N)
        if (item)
        {
            coalesced_group active = coalesced_threads();
            buffer[active.thread_rank()] = item;
        }
}

__global__ void testkernel_2(int *arr, int *buffer, int N)
{
    int tid = threadIdx.x;
    int item = arr[tid];
    if (tid < N)
        if (item)
        {
            buffer[tid] = item;
        }
}

int main()
{
    GpuTimer timer;
    int arr[N];
    for (int i = 0; i < N; i++)
    {
        if (i % 2 == 0)
            arr[i] = 0;
        else
            arr[i] = 1;
    }
    int *d_arr;
    cudaMalloc(&d_arr, N * 4);
    int *d_buffer;
    cudaMemcpy(d_arr, arr, N * 4, cudaMemcpyHostToDevice);

    cudaMalloc(&d_buffer, N / 2 * 4);
    testkernel_1<<<1, N>>>(d_arr, d_buffer, N);
    cudaMalloc(&d_buffer, N / 2 * 4);
    testkernel_2<<<1, N>>>(d_arr, d_buffer, N);

    cudaMalloc(&d_buffer, N / 2 * 4);
    timer.Start();
    // for (int i = 0; i < 1000; i++)
        testkernel_1<<<256, N>>>(d_arr, d_buffer, N);

    cudaDeviceSynchronize();
    timer.Stop();
    printf("Time elapse = %g ms with group\n", timer.Elapsed());
}