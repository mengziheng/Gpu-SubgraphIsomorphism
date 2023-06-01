#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;

        std::cout << "Device properties: " << std::endl;
        std::cout << "  Major revision number: " << deviceProp.major << std::endl;
        std::cout << "  Minor revision number: " << deviceProp.minor << std::endl;
        // 继续输出其他成员变量...

        std::cout << std::endl;
    }

    return 0;
}
