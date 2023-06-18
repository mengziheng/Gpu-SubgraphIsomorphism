#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <map>

using namespace std;

__global__ void test(int n)
{
    int tid = threadIdx.x;
    int arr[n];
}

int main(int argc, char *argv[])
{
    int n = 4;
    test<<<1, 32>>>(n);
    cudaDeviceSynchronize();
}
