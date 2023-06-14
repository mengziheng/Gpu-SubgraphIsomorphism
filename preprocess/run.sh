rm -rf preprocess
clear
nvcc -O3 preprocess.cu -o preprocess 
./preprocess