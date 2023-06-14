#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>

__global__ void find_maximum_kernel(int *array, int *max, int *mutex, unsigned int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ int cache[256];

    int temp = -1.0;
    while (index + offset < n)
    {
        temp = fmaxf(temp, array[index + offset]);

        offset += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ; // lock
        *max = fmaxf(*max, cache[0]);
        atomicExch(mutex, 0); // unlock
    }
}
int main()
{
    unsigned int N = 1024;
    int *h_array;
    int *d_array;
    int *h_max;
    int *d_max;
    int *d_mutex;

    // allocate memory
    h_array = (int *)malloc(N * sizeof(int));
    h_max = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMalloc((void **)&d_max, sizeof(int));
    cudaMalloc((void **)&d_mutex, sizeof(int));
    cudaMemset(d_max, 0, sizeof(int));
    cudaMemset(d_mutex, 0, sizeof(int));

    // fill host array with data
    for (unsigned int i = 0; i < N; i++)
    {
        h_array[i] = N * int(rand()) / RAND_MAX;
    }

    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // call kernel

    dim3 gridSize = 256;
    dim3 blockSize = 256;
    find_maximum_kernel<<<gridSize, blockSize>>>(d_array, d_max, d_mutex, N);

    // copy from device to host
    cudaMemcpy(h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);


    // report results
    std::cout << "Maximum number found on gpu was: " << *h_max << std::endl;

    // run cpu version
    clock_t cpu_start = clock();

    *h_max = -1.0;
    for (unsigned int i = 0; i < N; i++)
    {
        if (h_array[i] > *h_max)
        {
            *h_max = h_array[i];
        }
    }

    // free memory
    free(h_array);
    free(h_max);
    cudaFree(d_array);
    cudaFree(d_max);
    cudaFree(d_mutex);
}
