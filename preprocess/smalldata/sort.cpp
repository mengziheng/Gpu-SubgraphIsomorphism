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
string dirName;
string genericGraphFile = "/data/zh_dataset/graph_preprocessed/generic_graph_preprocessed/";
string cliqueGraphFile = "/data/zh_dataset/graph_preprocessed/clique_graph_preprocessed/";

int vertex_count, edge_count;
typedef struct edge_list
{
    int vertexID;
    vector<int> edge;
    int degree;
    int newid;
};
vector<edge_list> vertex;
vector<edge_list> vertexb;
bool cmp1(edge_list a, edge_list b)
{
    return a.edge.size() < b.edge.size();
}
bool cmp2(edge_list a, edge_list b)
{
    return a.edge.size() > b.edge.size();
}
bool cmp3(int a, int b)
{
    return a % MIN_BUCKET_NUM < b % MIN_BUCKET_NUM;
}
bool cmp_degree(edge_list a, edge_list b)
{
    return (a.degree > b.degree) || (a.degree == b.degree && a.vertexID < b.vertexID);
}

int binary_search(int value)
{
    int l = 0, r = vertex_count - 1;
    while (l < r - 1)
    {
        int mid = (l + r) >> 1;
        if (vertex[mid].edge.size() >= value)
            l = mid;
        else
            r = mid;
    }
    // if (arr[r]<=value) return r;
    return l;
}

void deletevertex()
{
    // 删除degree > 2的点
    int new_vertex_count = 0;
    int *a = new int[vertex_count];
    for (int i = 0; i < vertex.size(); i++)
    {
        if (vertex[i].edge.size() < 2)
        {
            a[i] = -1;
            continue;
        }
        a[i] = new_vertex_count;
        new_vertex_count++;
    }
    vertex_count = new_vertex_count;
    // 删除vertex
    int new_edge_count = 0;
    for (auto it = vertex.begin(); it != vertex.end();)
    {
        // 根据给定的条件判断是否删除元素
        if (a[(*it).vertexID] == -1)
        {
            it = vertex.erase(it); // 删除当前元素，并返回下一个元素的迭代器
        }
        else
        {
            // 修改节点ID
            (*it).vertexID = a[(*it).vertexID];
            for (auto edge = (*it).edge.begin(); edge != (*it).edge.end();)
            {
                if (a[(*edge)] == -1)
                {
                    edge = (*it).edge.erase(edge); // 删除当前元素，并返回下一个元素的迭代器
                }
                else
                {
                    *edge = a[*edge];
                    ++edge;
                    new_edge_count++;
                }
            }
            ++it; // 继续下一个元素
        }
    }
    edge_count = new_edge_count;
    for (int i = 0; i < vertex_count; i++)
    {
        vertex[i].degree = vertex[i].edge.size();
        if (vertex[i].degree < 0)
            printf("1");
    }
}

void loadgraph()
{
    ifstream inFile("/data/zh_dataset/graph_challenge_snapdata/" + inFileName, ios::in);
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
    deletevertex();
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
    printf("vertex_count : %d edge_count : %d\n", vertex_count, edge_count);
    vector<edge_list> vertex_for_generic = vertex;
    sort(vertex_for_generic.begin(), vertex_for_generic.end(), cmp_degree);
    // 记录新的id和旧的id的对应关系
    unique_ptr<int[]> newid(new int[vertex_count]);
    for (int i = 0; i < vertex_count; i++)
    {
        newid[vertex_for_generic[i].vertexID] = i;
    }
    for (int i = 0; i < vertex_count; i++)
    {
        for (auto &dst : vertex_for_generic[i].edge)
            dst = newid[dst];
        sort(vertex_for_generic[i].edge.begin(), vertex_for_generic[i].edge.end());
    }
    ofstream beginFile(genericGraphFile + "begin.bin", ios::out | ios::binary);
    ofstream adjFile(genericGraphFile + "adjacent.bin", ios::out | ios::binary);
    ofstream vertexFile(genericGraphFile + "vertex.bin", ios::out | ios::binary);
    ofstream maxDegreeFile(genericGraphFile + "md.bin", ios::out | ios::binary);
    // cout << genericGraphFile << "begin.bin" << endl;
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
    int maxDegree = vertex_for_generic[0].degree;
    for (int i = 0; i < vertex_count; i++)
    {
        int vertex = vertex_for_generic[i].newid;
        beginFile.write((char *)&sum, sizeof(int));
        sum += vertex_for_generic[i].degree;
        for (int j = 0; j < vertex_for_generic[i].degree; j++)
            vertexFile.write((char *)&i, sizeof(int));
        adjFile.write((char *)&vertex_for_generic[i].edge[0], sizeof(int) * vertex_for_generic[i].degree);
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
    int *a = new int[vertex_count];
    for (int i = 0; i < vertex_count; i++)
    {
        a[vertex[i].vertexID] = i;
    }

    for (int i = 0; i < vertex_count; i++)
    {
        vector<int> x(vertex[i].edge);
        vertex[i].edge.clear();
        while (!x.empty())
        {
            int v = x.back();
            x.pop_back();
            if (a[v] > i)
                vertex[i].edge.push_back(v);
        }
    }
}

void reassignID()
{
    int k1 = 0, k2 = -1, k3 = -1;
    for (int i = 0; i < vertex_count; i++)
    {
        vertex[i].newid = -1;
        if (k2 == -1 && vertex[i].edge.size() <= bounder)
            k2 = i;

        if (k3 == -1 && vertex[i].edge.size() < 2)
            k3 = i;
    }
    // cout << k2 << ' ' << k3 << endl;
    int s1 = k1, s2 = k2, s3 = k3;
    for (int i = 0; i < vertex_count; i++)
    {
        if (vertex[i].edge.size() <= 2)
            break;
        for (int j = 0; j < vertex[i].edge.size(); j++)
        {
            int v = vertex[i].edge[j];
            if (vertex[v].newid == -1)
            {
                if (v >= s3)
                {
                    vertex[v].newid = k3;
                    k3++;
                }
                else if (v >= s2)
                {
                    vertex[v].newid = k2;
                    k2++;
                }
                else
                {
                    vertex[v].newid = k1;
                    k1++;
                }
            }
        }
    }
    for (int i = 0; i < vertex_count; i++)
    {
        int u = vertex[i].newid;
        if (u == -1)
        {
            if (i >= s3)
            {
                vertex[i].newid = k3;
                k3++;
            }
            else if (i >= s2)
            {
                vertex[i].newid = k2;
                k2++;
            }
            else
            {
                vertex[i].newid = k1;
                k1++;
            }
        }
    }
    vertexb.swap(vertex);
    vertex.resize(vertex_count);

    for (int i = 0; i < vertex_count; i++)
    {
        int u = vertexb[i].newid;

        for (int j = 0; j < vertexb[i].edge.size(); j++)
        {
            int v = vertexb[i].edge[j];
            v = vertexb[v].newid;
            // cout<<u<<' '<<v<<endl;
            vertex[u].edge.push_back(v);
        }
    }

    // for (int i = 0; i < 10; i++)
    // {
    //     for (int j = 0; j < vertexb[i].edge.size(); j++)
    //         cout<<vertexb[vertexb[i].edge[j]].newid<<' ';
    //     cout<<endl;
    // }
}

void computeCSR(int k)
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
    reassignID();
    ofstream beginFile(cliqueGraphFile + "begin.bin", ios::out | ios::binary);
    ofstream adjFile(cliqueGraphFile + "adjacent.bin", ios::out | ios::binary);
    ofstream vertexFile(cliqueGraphFile + "vertex.bin", ios::out | ios::binary);
    ofstream maxDegreeFile(cliqueGraphFile + "md.bin", ios::out | ios::binary);
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
        sort(vertex[i].edge.begin(), vertex[i].edge.end(), cmp3);
        vector<int>::iterator upp = upper_bound(vertex[i].edge.begin(), vertex[i].edge.end(), k);
        int divide = upp - vertex[i].edge.begin();
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
    }
    dirName = removeSuffix(inFileName);
    genericGraphFile = genericGraphFile + dirName + "/";
    cliqueGraphFile = cliqueGraphFile + dirName + "/";
    loadgraph();
    removeDuplicatedEdgeAndSelfLoop();
    saveUndirectedGraph();
    sort(vertex.begin(), vertex.end(), cmp1);
    orientation();
    deletevertex();
    sort(vertex.begin(), vertex.end(), cmp2);
    int k = binary_search(32);
    computeCSR(k);
    system("pause");
    return 0;
}