rm -rf preprocess
clear
nvcc -Xptxas=“-v” -G preprocess.cu -o preprocess 
# nvcc -G -g preprocess.cu -o preprocess
cuda-memcheck ./preprocess