param="$1"
cd ../../final_version/

if [ "$param" = "Q0" ]; then
    python script.py 3 
elif [ "$param" = "Q1" ]; then
   python script.py 4 withRestriction 
elif [ "$param" = "Q2" ]; then
    python script.py 4 withRestriction 
elif [ "$param" = "Q3" ]; then
    python script.py 4  
elif [ "$param" = "Q6" ]; then
    python script.py 5 withRestriction withDuplicate 
elif [ "$param" = "Q7" ]; then
    python script.py 5 
elif [ "$param" = "Q8" ]; then
    python script.py 5 withRestriction withDuplicate 
elif [ "$param" = "Q11" ]; then
    python script.py 6 
elif [ "$param" = "Q12" ]; then
    python script.py 5 withRestriction  
else
    # 未知参数
    echo "未知的参数: $param"
fi

rm /home/zhmeng/GPU/Gpu-SubgraphIsomorphism/experiment/Scalability/subgraphmatch.bin
make && cp subgraphmatch.bin /home/zhmeng/GPU/Gpu-SubgraphIsomorphism/experiment/Scalability/
