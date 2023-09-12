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

string removeSuffix(const string &str)
{
    // 查找点号的位置
    size_t dotPos = str.find('.');
    if (dotPos != string::npos)
    {
        // 去掉点号及其后面的内容
        string result = str.substr(0, dotPos);
        // 查找"_adj"的位置
        size_t adjPos = result.find("_adj");
        if (adjPos != string::npos)
        {
            // 去掉末尾的"_adj"
            result = result.substr(0, adjPos);
        }
        return result;
    }
    else
    {
        // 没有点号，则直接去掉末尾的"_adj"
        size_t adjPos = str.find("_adj");
        if (adjPos != string::npos)
        {
            return str.substr(0, adjPos);
        }
    }
    // 如果没有点号和"_adj"，则返回原字符串
    return str;
}

tuple<int *, int *, int *, int> loadGraphWithName(string Infilename, string pattern)
{
    int *adjcant, *degree_offset, *vertex;
    int maxdegree;

    string s_begin; // 记录索引，而且已经维护了最后一位
    string s_adj;   // 记录邻居节点
    string s_vertex;
    string s_maxdegree;

    string folder;
    if (pattern.compare("Q0") == 0 || pattern.compare("Q3") == 0 || pattern.compare("Q5") == 0 || pattern.compare("Q7") == 0 || pattern.find("clique") != std::string::npos)
    {
        s_begin = Infilename + "/begin.bin";  // 记录索引，而且已经维护了最后一位
        s_adj = Infilename + "/adjacent.bin"; // 记录邻居节点
        s_vertex = Infilename + "/vertex.bin";
        s_maxdegree = Infilename + "/md.bin";
    }
    else
    {
        s_begin = Infilename + "/generic_begin.bin";  // 记录索引，而且已经维护了最后一位
        s_adj = Infilename + "/generic_adjacent.bin"; // 记录邻居节点
        s_vertex = Infilename + "/generic_vertex.bin";
        s_maxdegree = Infilename + "/generic_md.bin";
    }

    // s_begin = Infilename + "/generic_begin.bin";  // 记录索引，而且已经维护了最后一位
    // s_adj = Infilename + "/generic_adjacent.bin"; // 记录邻居节点
    // s_vertex = Infilename + "/generic_vertex.bin";
    // s_maxdegree = Infilename + "/generic_md.bin";

    char *begin_file = const_cast<char *>(s_begin.c_str());
    char *adj_file = const_cast<char *>(s_adj.c_str());
    char *vertex_file = const_cast<char *>(s_vertex.c_str());
    char *md_file = const_cast<char *>(s_maxdegree.c_str());

    vertex_count = fsize(begin_file) / sizeof(int) - 1;
    edge_count = fsize(adj_file) / sizeof(int);

    FILE *pFile1 = fopen(begin_file, "rb");
    if (!pFile1)
    {
        cout << "error for begin_file" << endl;
        // return 0;
    }
    degree_offset = (int *)malloc(fsize(begin_file));
    size_t x = fread(degree_offset, sizeof(int), vertex_count + 1, pFile1);
    fclose(pFile1);

    FILE *pFile2 = fopen(adj_file, "rb");
    if (!pFile2)
    {
        cout << "error for adj_file" << endl;
        // return 0;
    }
    adjcant = (int *)malloc(fsize(adj_file));
    x = fread(adjcant, sizeof(int), edge_count, pFile2);
    fclose(pFile2);

    FILE *pFile3 = fopen(vertex_file, "rb");
    if (!pFile3)
    {
        cout << "error for vertex_file" << endl;
        // return 0;
    }
    vertex = (int *)malloc(fsize(vertex_file));
    x = fread(vertex, sizeof(int), edge_count, pFile3);
    fclose(pFile3);

    FILE *pFile4 = fopen(md_file, "rb");
    if (!pFile4)
    {
        cout << "error for max_degree_file" << endl;
        // return 0;
    }
    x = fread(&maxdegree, sizeof(int), 1, pFile4);
    fclose(pFile4);

    int *d_adjcant, *d_vertex, *d_degree_offset;

    HRR(cudaMalloc((void **)&d_degree_offset, sizeof(int) * (vertex_count + 1)));
    HRR(cudaMalloc((void **)&d_adjcant, sizeof(int) * (edge_count)));
    HRR(cudaMalloc((void **)&d_vertex, sizeof(int) * (edge_count)));

    HRR(cudaMemcpy(d_degree_offset, degree_offset, sizeof(int) * (vertex_count + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_adjcant, adjcant, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(d_vertex, vertex, sizeof(int) * edge_count, cudaMemcpyHostToDevice));
    cout << "graph vertex number is : " << vertex_count << endl;
    cout << "graph edge number is : " << edge_count << endl;
    cout << "graph bucket size is : " << bucket_size << endl;

    free(degree_offset);
    free(adjcant);
    free(vertex);
    return make_tuple(d_adjcant, d_vertex, d_degree_offset, maxdegree);
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