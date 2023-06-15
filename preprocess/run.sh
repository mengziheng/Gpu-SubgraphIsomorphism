rm -rf preprocess
clear
nvcc -lineinfo -O3 preprocess.cu -o preprocess 
# nvcc -G -g preprocess.cu -o preprocess
cuda-memcheck ./preprocess