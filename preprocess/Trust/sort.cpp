#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>
#define bounder 100
#define MIN_BUCKET_NUM 8

using namespace std;
typedef long long index_t;
string Infilename = "1.mmio";
string outputfile;

int vertex_count, edge_count;
typedef struct edge_list
{
    int vertexID;
    vector<int> edge;
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

void loadgraph()
{
    ifstream inFile("1.mmio", ios::in);
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
    size_t found = Infilename.find(".tsv");
    // 如果找到子字符串
    if (found != std::string::npos)
    {
        Infilename.erase(found, 4);
    }
    else
    {
        Infilename.erase(Infilename.size() - 5);
    }
    ofstream beginFile(outputfile + "/begin.bin", ios::out | ios::binary);
    ofstream adjFile(outputfile + "/adjacent.bin", ios::out | ios::binary);
    // ofstream beginFile("begin.bin", ios::out | ios::binary);
    // ofstream adjFile("adjacent.bin", ios::out | ios::binary);
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

    int sum = 0;
    for (int i = 0; i < vertex_count; i++)
    {
        beginFile.write((char *)&sum, sizeof(int));
        sum += vertex[i].edge.size();
        sort(vertex[i].edge.begin(), vertex[i].edge.end(), cmp3);
        vector<int>::iterator upp = upper_bound(vertex[i].edge.begin(), vertex[i].edge.end(), k);
        int divide = upp - vertex[i].edge.begin();
        // cout << divide << ' ' << vertex[i].edge.size() << endl;
        int size = vertex[i].edge.size();
        adjFile.write((char *)&vertex[i].edge[0], sizeof(int) * vertex[i].edge.size());
    }
    beginFile.write((char *)&sum, sizeof(int));
}
int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        Infilename = argv[1];
        outputfile = argv[2];
    }
    loadgraph();
    sort(vertex.begin(), vertex.end(), cmp1);
    orientation();
    sort(vertex.begin(), vertex.end(), cmp2);
    int k = binary_search(32);
    computeCSR(k);
    system("pause");
    return 0;
}