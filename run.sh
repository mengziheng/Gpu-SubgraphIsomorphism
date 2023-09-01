file_name="$1"
nvcc -lineinfo $file_name -o test && ./test && rm test