#!/bin/bash
inputfile=$1
foldername=$(echo $inputfile | sed -e 's/.mmio//g')
rm -rf $foldername
mkdir -p $foldername

echo "This converter only works for mmio formated file"

#../../miniTri.exe $inputfile 
awk 'NF{NF-=1};1' < $inputfile > delete.tuple
sed '3s/.*/%&/' delete.tuple > $inputfile".tuple"
rm delete.tuple

./tuple_to_undirected_csr.bin $inputfile
mv head.bin "$foldername/"

./gConvu -c 0 -i $inputfile".tuple" -o $inputfile".edge"

echo "Getting the vertex count from the third line of the $inputfile"
vertex_count=$(awk -v line=3 'NR==line' $inputfile | awk -F " " '{print $1+1}')
echo $vertex_count

#exit

#./gConvu -c 3 -v 400728 -i $inputfile".edge" -o csr
./gConvu -c 3 -v $vertex_count -i $inputfile".edge" -o csr
./gConvu -c 4  -i csr -o csr
./gConvu -c 5  -i csr -o csr

mv csr.adj_rankbydegree "./$foldername/"adjacent.bin
mv csr.beg_pos_rankbydegree "./$foldername/"begin.bin
mv $inputfile".edge" "./$foldername/"edge