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

OurCode is in preprocess/SMOG

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

Run the srcipt direct
    `python script.py --input_graph_folder /data/zh_dataset/processed_graph_challenge_dataset/snap/cit-Patents --input_pattern Q2`

there are also other parameters you can choose to change：

--N : numbers of GPU

--load_factor : load factor for hash table

--bucket_size : bucket size for hash table

--block_number_for_kernel : number of block for subgraph match kernel

--block_size_for_kernel : size of block for subgraph match kernel

--chunk_size : chunk size for task assignment