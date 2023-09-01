#include <iostream>
int size = 10000;

using namespace std;

__global__ void Kernel(int *arr, int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < size)
    {
        for (int i = 0; i < 1000000; i++)
            arr[tid] = arr[tid] + i * 2 - 1;
        for (int i = 0; i < 1000000; i++)
            arr[tid] = arr[tid] - i * 2 + 1;
    }
}

int main()
{
    int *arr = new int[size];
    for (int i = 0; i < size; i++)
        arr[i] = i;
    int *d_arr;
    cudaMalloc(&d_arr, sizeof(int) * size);
    cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Kernel<<<216, 1024>>>(d_arr, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << "time is : " << time << " ms" << endl;
    cout << "result is : " << endl;
    cudaMemcpy(arr, d_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
        printf("%d ", arr[i]);
    printf("\n");
}