#include <iostream>
int main()
{
    int vertex_count = 4;
    int hash_tables_lens[4] = {8, 8, 8, 8};
    // get bucket_num
    int *d_hash_tables_lens;
    cudaMalloc(&d_hash_tables_lens, vertex_count * sizeof(int));
    cudaMemcpy(d_hash_tables_lens, hash_tables_lens, vertex_count * sizeof(int), cudaMemcpyHostToDevice);
    // exclusiveSum
    long long *d_hash_tables_offset;
    cudaMalloc(&d_hash_tables_offset, (vertex_count + 1) * sizeof(long long));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    long long hash_tables_offset[vertex_count];
    cudaMemcpy(hash_tables_offset, d_hash_tables_offset, vertex_count * sizeof(long), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vertex_count; i++)
    {
        printf("%lld ", hash_tables_offset[i]);
    }
    printf("\n");
}