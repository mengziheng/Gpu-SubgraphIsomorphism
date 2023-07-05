# Gpu-SubgraphIsomorphism
Gpu-friendly SubgraphIsomorphism based on WOJ

## Dataset

Dataset source from : http://graphchallenge.mit.edu/data-sets

Due to variations in the sizes of our experimental datasets, we have divided the datasets into three parts:

1. SNAP dataset
2. Synthetic dataset
3. MWAI dataset

All the data is stored in /data/zh_dataset/graph_challenge_dataset.

The purpose of this division is to facilitate separate testing and avoid lengthy preprocessing time on the same dataset.

## Framework Overview

Our framework is designed to provide a scalable solution for subgraph matching. It consists of several key components that work together to achieve [specific goals]. Below is an overview of each component:

### preprocess 

include preprocess script for H-index , Trust , OurCode

All preprocess script is preprocess.py

preprocess graph : 
    `python preprocess.py /data/zh_dataset/graph_challenge_dataset/snap /data/zh_dataset/Hindex_processed_graph_challenge_dataset/snap`

The input arguments is 
1. input graph folder (include all graph to be preprocessed)
2. output graph folder (include all graph already preprocessed)

### experiment 
include experiment script for H-index , Trust , OurCode

preprocess graph : 
    `python performance_benchmark.py /data/zh_dataset/Hindex_processed_graph_challenge_dataset/snap`

The input arguments is 
1. input graph folder (include all graph already preprocessed)

### result 
result get from experiment

## Compile and Run code
For small graph, we don't partition the graph. 

Compile the code:
    `make`
Run the code:
    `./subgraphmatch.bin cit-Patents_adj triangle 0.25 8 216 1024 1`

The input arguments is 
1. input graph folder 
2. input pattern
3. load factor
4. bucket size
5. block number for kernel
6. block size for kernel
7. chunk size