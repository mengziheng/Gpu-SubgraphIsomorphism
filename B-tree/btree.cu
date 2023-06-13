#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cub/cub.cuh>
#include <math.h>

using namespace std;

struct edge
{
    int u, v;
};
vector<edge> edgelist;
int uCount = 0, vCount = 0, uMax = 0, vMax = 0, edgeCount;

void loadgraph(string filename)
{
    ifstream inFile(filename.c_str(), ios::in);
    if (!inFile)
    {
        cout << "error" << endl;
    }
    string line;
    stringstream ss;
    while (getline(inFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        else
        {
            ss << line;
            ss >> uCount >> vCount >> edgeCount;
            break;
        }
    }
    while (getline(inFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        ss.str("");
        ss.clear();
        ss << line;

        edge e;
        ss >> e.u >> e.v;
        e.u--;
        e.v--;
        edgelist.push_back(e);
    }
}

__global__ void translateIntoCSRKernel(int *edgelist, int edgeCount, int vertexCount, int *csr_column_index, int *csr_row_value)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int s_csr_row_value[];
    if (tid < edgeCount)
    {
        atomicAdd(s_csr_row_value + edgelist[tid * 2 + 1], 1);
        csr_column_index[tid] = edgelist[tid * 2];
    }

    for (int i = threadIdx.x; i < vertexCount; i += blockDim.x)
    {
        atomicAdd(&csr_row_value[i], s_csr_row_value[i]);
    }
}

__global__ void sum_kernel(int *offset, int *value, int count, int *sum)
{
    if (threadIdx.x == 0)
    {
        sum[0] = offset[count - 1] + value[count - 1];
        // printf("%d %d\n", offset[count - 1], value[count - 1]);
    }
}

__global__ void Kernel_1(int vertexCount, int *csr_row_value, int *height_value)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < vertexCount)
    {
        if (csr_row_value[tid] == 0)
            height_value[tid] = 0;
        else
            height_value[tid] = log10f(static_cast<float>(csr_row_value[tid])) / log10f(32.0f);
        // if (tid == 37)
        //     printf("%d %d\n", csr_row_value[tid], height_value[tid]);
    }
}

int main(int argc, char *argv[])
{
    // load graph file
    string infilename = "../dataset/graph/as20000102_adj.mmio";
    loadgraph(infilename);
    cout << edgeCount << endl;
    int *d_edgelist;
    cudaMalloc(&d_edgelist, edgeCount * 2 * sizeof(int));
    cudaMemcpy(d_edgelist, &edgelist[0], edgeCount * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // get CSR and other structure in device
    int *d_csr_column_index;
    int *d_csr_row_value;
    int *d_csr_row_offset;
    cudaMalloc(&d_csr_column_index, edgeCount * sizeof(int));
    cudaMalloc(&d_csr_row_value, uCount * sizeof(int));
    cudaMalloc(&d_csr_row_offset, uCount * sizeof(int));
    printf("%d\n", uCount);
    translateIntoCSRKernel<<<216, 1024, uCount * sizeof(int)>>>(d_edgelist, edgeCount, uCount, d_csr_column_index, d_csr_row_value);

    // get row_offset for CSR
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_csr_row_value, d_csr_row_offset, uCount);

    //
    int *d_height_value;
    cudaMalloc(&d_height_value, uCount * sizeof(int));
    Kernel_1<<<216, 1024>>>(uCount, d_csr_row_value, d_height_value);
    int *d_height_offset;
    cudaMalloc(&d_height_offset, uCount * sizeof(int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_height_value, d_height_offset, uCount);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_height_value, d_height_offset, uCount);
    int *d_height_sum;
    cudaMalloc(&d_height_sum, sizeof(int));
    sum_kernel<<<1, 1>>>(d_height_offset, d_height_value, uCount, d_height_sum);

    // int sum;
    // cudaMemcpy(&sum, d_height_sum, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d\n", sum);
    // verify
    // int csr_column_index[edgeCount];
    // cudaMemcpy(csr_column_index, d_csr_column_index, edgeCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", csr_column_index[i]);
    // }
    // printf("\n");

    // int csr_row_value[uCount];
    // cudaMemcpy(csr_row_value, d_csr_row_value, uCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", csr_row_value[i]);
    // }
    // printf("\n");

    // int csr_row_offset[uCount];
    // cudaMemcpy(csr_row_offset, d_csr_row_offset, uCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", csr_row_offset[i]);
    // }
    // printf("\n");

    // int height_value[uCount];
    // cudaMemcpy(height_value, d_height_value, uCount * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < uCount; i++)
    // {
    //     printf("%d ", height_value[i]);
    // }
    // printf("\n");

    cudaDeviceSynchronize();
}