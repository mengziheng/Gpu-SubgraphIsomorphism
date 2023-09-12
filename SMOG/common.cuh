#ifndef COMM_HEADER
#define COMM_HEADER

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <tuple>

#define FULL_MASK 0xffffffff
#define MIN_BUCKET_NUM 8
#define MAX_SIZE_FOR_ARRAY 16
#define map_value(hash_value, bucket_num) (((hash_value) % MIN_BUCKET_NUM) * (bucket_num / MIN_BUCKET_NUM) + (hash_value) / MIN_BUCKET_NUM)
using namespace std;

extern int vertex_count, edge_count;
extern float load_factor;
extern int bucket_size;
extern int block_size;
extern int block_number;
extern int chunk_size;

inline off_t fsize(const char *filename);

void HandError(cudaError_t err, const char *file, int line);

#define HRR(err) (HandError(err, __FILE__, __LINE__));

tuple<int *, int *, int *, int> loadGraphWithName(string Infilename, string pattern);

void printGpuInfo();

#endif