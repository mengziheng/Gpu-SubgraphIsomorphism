#include <iostream>
#include <string>
#include <stdio.h>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "subgraph_match.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    string Infilename = "4-cycle";
    string pattern = "4-cycle";
    int process_id, device_num;
    int process_num = atoi(argv[3]);

    struct arguments args;
    /* Initialize the MPI library */
    MPI_Init(&argc, &argv);
    /* Determine unique id of the calling process of all processes participating
         in this MPI program. This id is usually called MPI rank. */
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &device_num);
    // int N_GPUS=argv[1];
    //  call the function
    long long global_sum, now_sum = 0;
    double global_min_time, global_max_time, now_min_time = 9999, now_max_time = 0;
    while (process_id < process_num)
    {
        args = SubgraphMatching(process_id, process_num, args, Infilename, pattern, argv);
        process_id += device_num;
        if (now_min_time > args.time)
            now_min_time = args.time;
        if (now_max_time < args.time)
            now_max_time = args.time;
        now_sum += args.count;
    }
    MPI_Reduce(&now_sum, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&now_max_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&now_min_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (process_id % device_num == 0)
    {
        cout << "graph : " << Infilename << "time is : " << global_max_time * 1000 << " ms,count is : " << global_sum << endl;
    }
    MPI_Finalize();
    return 0;
}