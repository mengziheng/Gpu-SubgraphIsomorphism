#include "common.cuh"

using namespace std;

int vertex_count = 0, edge_count;
float load_factor = 0.25;
int bucket_size = 4;
int block_size = 216;
int block_number = 1024;
int chunk_size = 1;

inline off_t fsize(const char *filename)
{
    struct stat st;
    if (stat(filename, &st) == 0)
    {
        return st.st_size;
    }
    return -1;
}
void HandError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("\n%s in %s at line %d\n",
               cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

tuple<int *, int *, int *, int *, int *> loadGraphWithName(string Infilename, string pattern)
{
    int *csr_column_index;
    int *csr_row_offset;
    int *csr_row_value;
    int *vertex_list;
    int *edge_list;

    string folder;
    if (pattern.compare("triangle") == 0 || pattern.find("clique") != std::string::npos)
        folder = "/data/zh_dataset/dataforclique/" + Infilename;
    else
        folder = "/data/zh_dataset/dataforgeneral/" + Infilename;
    string s_begin = folder + "/begin.bin";  // 记录索引，而且已经维护了最后一位
    string s_adj = folder + "/adjacent.bin"; // 记录邻居节点
    string s_degree = folder + "/edge";      // 记录邻居节点数目
    string s_vertex = folder + "/vertex";
    string s_edgelist = folder + "/edgelist";

    char *begin_file = const_cast<char *>(s_begin.c_str());
    char *adj_file = const_cast<char *>(s_adj.c_str());
    char *degree_file = const_cast<char *>(s_degree.c_str());
    char *vertex_file = const_cast<char *>(s_vertex.c_str());
    char *edgelist_file = const_cast<char *>(s_edgelist.c_str());

    vertex_count = fsize(begin_file) / sizeof(int) - 1;
    edge_count = fsize(adj_file) / sizeof(int);

    FILE *pFile1 = fopen(begin_file, "rb");
    if (!pFile1)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_row_offset = (int *)malloc(fsize(begin_file));
    size_t x = fread(csr_row_offset, sizeof(int), vertex_count + 1, pFile1);
    fclose(pFile1);

    FILE *pFile2 = fopen(adj_file, "rb");
    if (!pFile2)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_column_index = (int *)malloc(fsize(adj_file));
    x = fread(csr_column_index, sizeof(int), edge_count, pFile2);
    fclose(pFile2);

    FILE *pFile3 = fopen(degree_file, "rb");
    if (!pFile3)
    {
        cout << "error" << endl;
        // return 0;
    }
    csr_row_value = (int *)malloc(fsize(degree_file));
    x = fread(csr_row_value, sizeof(int), vertex_count, pFile3);
    fclose(pFile3);

    FILE *pFile4 = fopen(vertex_file, "rb");
    if (!pFile4)
    {
        cout << "error" << endl;
        // return 0;
    }
    vertex_list = (int *)malloc(fsize(vertex_file));
    x = fread(vertex_list, sizeof(int), edge_count, pFile4);
    fclose(pFile4);

    FILE *pFile5 = fopen(edgelist_file, "rb");
    if (!pFile5)
    {
        cout << "error" << endl;
        // return 0;
    }
    edge_list = (int *)malloc(fsize(edgelist_file));
    x = fread(edge_list, sizeof(int), edge_count * 2, pFile5);
    fclose(pFile5);

    int *d_csr_column_index, *d_csr_row_value, *d_vertex_list, *d_csr_row_offset, *d_edge_list;

    HRR(cudaMalloc((void **)&d_csr_row_offset, sizeof(int) * (vertex_count + 1)));
    HRR(cudaMalloc((void **)&d_csr_column_index, sizeof(int) * (edge_count)));
    HRR(cudaMalloc((void **)&d_vertex_list, sizeof(int) * (edge_count)));
    HRR(cudaMalloc((void **)&d_csr_row_value, sizeof(int) * (vertex_count + 1)));
    HRR(cudaMalloc((void **)&d_edge_list, sizeof(int) * edge_count * 2));

    HRR(cudaMemcpy(d_csr_row_value, csr_row_value, sizeof(int) * (vertex_count + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_csr_row_offset, csr_row_offset, sizeof(int) * (vertex_count + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_csr_column_index, csr_column_index, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_vertex_list, vertex_list, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_edge_list, edge_list, sizeof(int) * edge_count * 2, cudaMemcpyHostToDevice));
    cout << "graph vertex number is : " << vertex_count << endl;
    cout << "graph edge number is : " << edge_count << endl;
    cout << "graph bucket size is : " << bucket_size << endl;

    return make_tuple(d_csr_column_index, d_csr_row_value, d_vertex_list, d_csr_row_offset, d_edge_list);
}

void printGpuInfo()
{
    // 查看下可用share memory的最大值
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 假设设备号为0
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    cout << "share memory size per block : " << sharedMemPerBlock << endl;
    cout << "registers number per block : " << deviceProp.regsPerBlock << endl;
}