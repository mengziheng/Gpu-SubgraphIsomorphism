//用来描述是在warp内部进行线程active判断的

#include <iostream>
#include <cooperative_groups.h>
#include<cuda_runtime.h>
using namespace cooperative_groups;
using namespace std;

int N = 64;

__global__ void testkernel_1(int *arr, int *buffer, int N)
{
    int tid = threadIdx.x;
    int item = arr[tid];
    if (tid < N)
        if (item)
        {
            coalesced_group active = coalesced_threads();
            printf("tid : %d group id : %d\n", tid, active.thread_rank());
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
}