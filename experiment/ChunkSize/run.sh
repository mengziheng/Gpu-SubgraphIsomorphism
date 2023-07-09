param="$1"
cd ../../final_version/

if [ "$param" = "Q0" ]; then
    python script.py 3 useVertexAsStart 
elif [ "$param" = "Q1" ]; then
   python script.py 4 withRestriction useVertexAsStart
elif [ "$param" = "Q2" ]; then
    python script.py 4 withRestriction useVertexAsStart
elif [ "$param" = "Q3" ]; then
    python script.py 4 useVertexAsStart 
elif [ "$param" = "Q6" ]; then
    python script.py 5 withRestriction withDuplicate useVertexAsStart
elif [ "$param" = "Q7" ]; then
    python script.py 5 useVertexAsStart
elif [ "$param" = "Q8" ]; then
    python script.py 5 withRestriction withDuplicate useVertexAsStart
elif [ "$param" = "Q11" ]; then
    python script.py 6 useVertexAsStart
elif [ "$param" = "Q12" ]; then
    python script.py 5 withRestriction useVertexAsStart
else
    # 未知参数
    echo "未知的参数: $param"
fi

make && cp subgraphmatch.bin /home/zhmeng/GPU/Gpu-SubgraphIsomorphism/experiment/ChunkSize/
