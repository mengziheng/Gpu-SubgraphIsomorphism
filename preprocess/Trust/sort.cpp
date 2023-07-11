
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