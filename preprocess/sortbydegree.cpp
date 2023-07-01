#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>
#include <sstream>
#include <cmath>

using namespace std;

long long *beginPos;
int *edgeList;
int *deg;
int uCount, vCount, breakVertex32, breakVertex10, vertexCount;
long long edgeCount;

bool cmp1(int a, int b)
{
    return a > b;
}
struct node
{
    int id;
    int degree;
};
struct node *nnn;
bool cmp_degree(struct node a, struct node b)
{
    return (a.degree > b.degree) || (a.degree == b.degree && a.id < b.id);
}

void loadEdgeList(string path)
{
    fstream edgeListFile("/data/zh_dataset/graph/" + path, ios::in);
    long long vertexpower, degree;
    // edgeListFile >> vertexpower >> degree;
    // vertexCount = powl(2, vertexpower);
    // edgeCount = degree * vertexCount;

    string line;
    stringstream ss;
    int p = 0, x;
    while (getline(edgeListFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        ss.str("");
        ss.clear();
        ss << line;
        if (p == 0)
        {
            ss >> vertexCount >> x >> edgeCount;
            cout << vertexCount << " " << edgeCount << endl;
            p = 1;
            break;
        }
    }
    shared_ptr<vector<int>[]> b(new vector<int>[vertexCount]);
    while (getline(edgeListFile, line))
    {
        if (line[0] < '0' || line[0] > '9')
            continue;
        ss.str("");
        ss.clear();
        ss << line;
        if (p == 0)
        {
            ss >> vertexCount >> x >> edgeCount;
            cout << vertexCount << " " << edgeCount << endl;
            p = 1;
            continue;
        }
        int u, v;
        ss >> u >> v >> x;
        u--;
        v--;
        // cout << u << " " << v << endl;
        b[u].push_back(v);
        b[v].push_back(u);
    }
    // removeDuplicatedEdgeAndSelfLoop(b);
    struct node *idDegree = new struct node[vertexCount];
    for (int i = 0; i < vertexCount; i++)
    {
        idDegree[i].id = i;
        idDegree[i].degree = b[i].size();
    }
    sort(idDegree, idDegree + vertexCount, cmp_degree);
    // 记录新的id和旧的id的对应关系
    unique_ptr<int[]> newid(new int[vertexCount]);
    for (int i = 0; i < vertexCount; i++)
    {
        newid[idDegree[i].id] = i;
    }
    for (int i = 0; i < vertexCount; i++)
    {
        for (auto &dst : b[i])
            dst = newid[dst];
        sort(b[i].begin(), b[i].end());
    }
    size_t found = path.find(".tsv");
    // 如果找到子字符串
    if (found != std::string::npos)
    {
        path.erase(found, 4);
    }
    else
    {
        path.erase(path.size() - 5);
    }
    ofstream beginFile("/data/zh_dataset/dataforgeneral/" + path + "/begin.bin", ios::out | ios::binary);
    ofstream adjFile("/data/zh_dataset/dataforgeneral/" + path + "/adjacent.bin", ios::out | ios::binary);
    ofstream edgeFile("/data/zh_dataset/dataforgeneral/" + path + "/edge", ios::out | ios::binary);
    ofstream vertexFile("/data/zh_dataset/dataforgeneral/" + path + "/vertex", ios::out | ios::binary);
    ofstream vertexFile("/data/zh_dataset/dataforgeneral/" + path + "/edgelist", ios::out | ios::binary);
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
    if (!edgeFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    if (!vertexFile)
    {
        cout << "error" << endl;
        // return 0;
    }
    int sum = 0;
    for (int i = 0; i < vertexCount; i++)
    {
        int vertex = idDegree[i].id;
        beginFile.write((char *)&sum, sizeof(int));
        sum += idDegree[i].degree;
        edgeFile.write((char *)&idDegree[i].degree, sizeof(int));
        for (int j = 0; j < idDegree[i].degree; j++)
            vertexFile.write((char *)&i, sizeof(int));
        adjFile.write((char *)&b[vertex][0], sizeof(int) * idDegree[i].degree);
    }
    beginFile.write((char *)&sum, sizeof(int));
    ofstream propertiesFile("/data/zh_dataset/dataforgeneral/" + path + "/properties.txt", ios::out);
    propertiesFile << 0 << ' ' << vertexCount << ' ' << sum << endl;
    beginFile.close();
    adjFile.close();
    propertiesFile.close();
}

int main(int argc, char *argv[])
{
    string path;
    path = "square.mmio";
    if (argc > 1)
    {
        path = argv[1];
    }
    bool isGenerated = false;
    if (argc > 2)
    {
        isGenerated = atoi(argv[2]) > 0;
    }
    cout << path << endl;
    loadEdgeList(path);
}