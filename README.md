# Gpu-SubgraphIsomorphism
Gpu-friendly SubgraphIsomorphism based on WOJ

## Compile and Run code
For small graph, we don't partition the graph. 
Compile the code:
    make
Run the code:
    ./subgraphmatch.bin cit-Patents_adj triangle 0.25 8 216 1024 1

The input arguments is 
1. input graph folder 
2. input pattern
3. load factor
4. bucket size
5. block number for kernel
6. block size for kernel
7. chunk size