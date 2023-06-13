#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void test(int *arr)
{
    int tid = threadIdx.x;
    if (tid < 32)
        arr[tid] = tid;
    if (tid == 0)
    {
        int *t = arr;
        printf("%d\n", *t);
        t = t + 10;
        printf("%d\n", *t);
    }
}

int main()
{
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // for (int i = 0; i < deviceCount; ++i) {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, i);

    //     std::cout << "Device " << i << ": " << deviceProp.name << std::endl;

    //     std::cout << "Device properties: " << std::endl;
    //     std::cout << "  Major revision number: " << deviceProp.major << std::endl;
    //     std::cout << "  Minor revision number: " << deviceProp.minor << std::endl;
    //     // 继续输出其他成员变量...

    //     std::cout << std::endl;
    // }

    int *d_arr;
    cudaMalloc(&d_arr, 4 * 32);
    cudaMemset(d_arr, -1, 4 * 32);
    test<<<1, 32>>>(d_arr);
    cudaDeviceSynchronize();
    // printf("%d",(int)ceil(15/(double)5));
    return 0;
}
