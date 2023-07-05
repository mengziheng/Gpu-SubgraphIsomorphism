#!/bin/bash
inputfile=$1
outputdir=$2
rm -rf $outputdir
mkdir -p $outputdir

#../../miniTri.exe $inputfile 
sed '1s/.*/%&/' $inputfile > $inputfile".tuple"

./gConvu -c 0 -i $inputfile".tuple" -o $inputfile".edge"

# echo "Getting the vertex count from the first line of the $inputfile"
vertex_count=$(awk -v line=1 'NR==line' $inputfile | awk -F " " '{print $1+1}')
# echo $vertex_count

#exit

#./gConvu -c 3 -v 400728 -i $inputfile".edge" -o csr
./gConvu -c 3 -v $vertex_count -i $inputfile".edge" -o csr
./gConvu -c 4  -i csr -o csr
./gConvu -c 5  -i csr -o csr


mv csr.adj_rankbydegree "$outputdir/"adjacent.bin
mv csr.beg_pos_rankbydegree "$outputdir/"begin.bin
mv $inputfile".edge" "$outputdir/"edge
rm $inputfile".tuple"
rm csr*