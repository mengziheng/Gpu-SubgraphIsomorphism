#include <iostream>

#define LOOP_TIME 1000000

using namespace std;

int size = 216 * 1024;

__global__ void AddKernel(int *arr, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < size)
    {
        for (int i = 0; i < LOOP_TIME; i++)
            arr[tid] = arr[tid] + i * 2 - 1;
    }
}

__global__ void MineKernel(int *arr, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < size)
        for (int i = 0; i < LOOP_TIME; i++)
            arr[tid] = arr[tid] - i * 2 + 1;
}

int main()
{
    // int *arr = new int[size];
    // for (int i = 0; i < size; i++)
    //     arr[i] = i;
    // int *d_arr;
    // cudaMalloc(&d_arr, sizeof(int) * size);
    // cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    // AddKernel<<<216, 1024>>>(d_arr, size);
    // MineKernel<<<216, 1024>>>(d_arr, size);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float time;
    // cudaEventElapsedTime(&time, start, stop);
    // cout << "time is : " << time << " ms" << endl;
    // cout << "result is : " << endl;
    // cudaMemcpy(arr, d_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++)
    //     printf("%d ", arr[i]);
    // printf("\n");

    int *arr = new int[10]{1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}