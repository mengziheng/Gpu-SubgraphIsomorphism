#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>
#include <memory>
#define bounder 100
#define MIN_BUCKET_NUM 8
using namespace std;

string inFileName;
string ouFileName;
string GraphFile;

int vertex_count, edge_count;
typedef struct edge_list
{
    int vertexID;
    vector<int> edge;
    int degree;
};
vector<edge_list> vertex;
vector<edge_list> vertexb;
vector<edge_list> vertex_for_generic;
bool cmp_degree(edge_list a, edge_list b)
{
    return (a.edge.size() > b.edge.size()) || (a.edge.size() == b.edge.size() && a.vertexID < b.vertexID);
}

void loadgraph()
{
    ifstream inFile(inFileName, ios::in);
    if (!inFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    int x;
    int p = 0;
    string line;
    stringstream ss;
    while (getline(inFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        ss.str("");
        ss.clear();
        ss << line;
        if (p == 0)
        {
            ss >> vertex_count >> x >> edge_count;
            p = 1;
            vertex.resize(vertex_count);
            for (int i = 0; i < vertex_count; i++)
            {
                vertex[i].vertexID = i;
            }
            continue;
        }
        int u, v;
        ss >> u >> v >> x;
        u--;
        v--;
        vertex[u].edge.push_back(v);
        vertex[v].edge.push_back(u);
    }
    printf("vertex_count : %d edge_count : %d\n", vertex_count, edge_count);
    // deletevertex();
}

string removeSuffix(const string &str)
{

    size_t lastSlashPos = str.find_last_of('/');
    // 提取最后一个斜杠之后的内容
    string result = str.substr(lastSlashPos + 1);

    // 查找点号的位置
    size_t dotPos = result.find('.');
    if (dotPos != string::npos)
    {
        // 去掉点号及其后面的内容
        result = result.substr(0, dotPos);
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
        size_t adjPos = result.find("_adj");
        if (adjPos != string::npos)
        {
            return result.substr(0, adjPos);
        }
    }
    // 如果没有点号和"_adj"，则返回原字符串
    return result;
}

void removeDuplicatedEdgeAndSelfLoop()
{
    for (int i = 0; i < vertex_count; i++)
    {
        sort(vertex[i].edge.begin(), vertex[i].edge.end());
        vector<int> a;
        int previous = -1;
        for (auto item : vertex[i].edge)
        {
            if (item != previous)
            {
                previous = item;
                if (item != i)
                    a.push_back(item);
            }
        }
        vertex[i].edge.swap(a);
    }
}

// for undirected graph , degree bigger , id smaller
void saveUndirectedGraph()
{
    int *a = new int[vertex_count];
    for (int i = 0; i < vertex_count; i++)
    {
        a[vertex_for_generic[i].vertexID] = i;
    }
    for (int i = 0; i < vertex_count; i++)
    {
        for (int j = 0; j < vertex_for_generic[i].edge.size(); j++)
        {
            vertex_for_generic[i].edge[j] = a[vertex_for_generic[i].edge[j]];
        }
        vertex_for_generic[i].vertexID = i;
    }
    ofstream beginFile(GraphFile + "generic_begin.bin", ios::out | ios::binary);
    ofstream adjFile(GraphFile + "generic_adjacent.bin", ios::out | ios::binary);
    ofstream vertexFile(GraphFile + "generic_vertex.bin", ios::out | ios::binary);
    ofstream maxDegreeFile(GraphFile + "generic_md.bin", ios::out | ios::binary);
    // cout << GraphFile << "begin.bin" << endl;
    if (!beginFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!adjFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!vertexFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!maxDegreeFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    int sum = 0;
    int maxDegree = vertex_for_generic[0].edge.size();
    for (int i = 0; i < vertex_count; i++)
    {
        beginFile.write((char *)&sum, sizeof(int));
        sum += vertex_for_generic[i].edge.size();
        for (int j = 0; j < vertex_for_generic[i].edge.size(); j++)
            vertexFile.write((char *)&i, sizeof(int));
        adjFile.write((char *)&vertex_for_generic[i].edge[0], sizeof(int) * vertex_for_generic[i].edge.size());
    }
    beginFile.write((char *)&sum, sizeof(int));
    maxDegreeFile.write((char *)&maxDegree, sizeof(int));
    beginFile.close();
    adjFile.close();
    vertexFile.close();
    maxDegreeFile.close();
}

void orientation()
{
    vertex_for_generic = vertex;
    int index = -1;
    int *a = new int[vertex_count];
    for (int i = 0; i < vertex_count; i++)
    {
        a[vertex[i].vertexID] = i;
        if (vertex[i].edge.size() == 1)
            index = i;
    }
    vertex_count = index;

    for (int i = 0; i < vertex_count; i++)
    {
        vector<int> x(vertex[i].edge);
        vertex[i].edge.clear();
        vertex_for_generic[i].edge.clear();
        while (!x.empty())
        {
            int v = x.back();
            x.pop_back();
            if (a[v] < i && a[v] < index)
                vertex[i].edge.push_back(v);
            if (a[v] < index)
                vertex_for_generic[i].edge.push_back(v);
        }
    }
}

void computeCSR()
{
    int *a = new int[vertex_count];
    for (int i = 0; i < vertex_count; i++)
    {
        a[vertex[i].vertexID] = i;
    }
    for (int i = 0; i < vertex_count; i++)
    {
        for (int j = 0; j < vertex[i].edge.size(); j++)
        {
            vertex[i].edge[j] = a[vertex[i].edge[j]];
        }
        vertex[i].vertexID = i;
    }
    ofstream beginFile(GraphFile + "begin.bin", ios::out | ios::binary);
    ofstream adjFile(GraphFile + "adjacent.bin", ios::out | ios::binary);
    ofstream vertexFile(GraphFile + "vertex.bin", ios::out | ios::binary);
    ofstream maxDegreeFile(GraphFile + "md.bin", ios::out | ios::binary);
    if (!beginFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!adjFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!vertexFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!maxDegreeFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    int sum = 0;
    int maxDegree = vertex[0].edge.size();
    for (int i = 0; i < vertex_count; i++)
    {
        beginFile.write((char *)&sum, sizeof(int));
        sum += vertex[i].edge.size();
        sort(vertex[i].edge.begin(), vertex[i].edge.end());
        for (int j = 0; j < vertex[i].edge.size(); j++)
            vertexFile.write((char *)&i, sizeof(int));
        adjFile.write((char *)&vertex[i].edge[0], sizeof(int) * vertex[i].edge.size());
    }
    beginFile.write((char *)&sum, sizeof(int));
    maxDegreeFile.write((char *)&maxDegree, sizeof(int));
    beginFile.close();
    adjFile.close();
    vertexFile.close();
    maxDegreeFile.close();
}
// processed_data
int main(int argc, char *argv[])
{
    inFileName = "ca-CondMat_adj.mmio";
    if (argc > 1)
    {
        inFileName = argv[1];
        ouFileName = argv[2];
    }
    string dirName = removeSuffix(inFileName);
    GraphFile = ouFileName + "/" + dirName + "/";
    loadgraph();
    removeDuplicatedEdgeAndSelfLoop();
    sort(vertex.begin(), vertex.end(), cmp_degree);
    orientation();
    saveUndirectedGraph();
    sort(vertex.begin(), vertex.end(), cmp_degree);
    computeCSR();
    system("pause");
    return 0;
}