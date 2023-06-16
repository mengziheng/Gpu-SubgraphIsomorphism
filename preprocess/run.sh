rm -rf preprocess
clear
nvcc -lineinfo -G -arch=sm_80 preprocess.cu -o preprocess 
# nvcc -G -g preprocess.cu -o preprocess
cuda-memcheck ./preprocess