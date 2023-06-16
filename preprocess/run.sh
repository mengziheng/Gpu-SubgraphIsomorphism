rm -rf preprocess
clear
nvcc -lineinfo -G preprocess_wzb.cu -o preprocess 
# nvcc -G -g preprocess.cu -o preprocess
cuda-memcheck ./preprocess