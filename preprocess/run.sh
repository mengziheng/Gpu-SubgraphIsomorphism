rm -rf preprocess
clear
nvcc -lineinfo -O3 -arch=sm_80 preprocess_wzb.cu -o preprocess 
# nvcc -G -g preprocess.cu -o preprocess
compute-sanitizer ./preprocess