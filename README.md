This is the code for subgraphIsomorphism based on GPU submitted at Graph Challenge 2023

## Environment

CUDA Toolkit 11.6; gcc version 11.1.0; mpiexec (OpenRTE) 2.1.1; MPICH Version:3.3a2; Python 3.10.9

## Dataset

Run  `./run.sh ` in data folder, or you can create any specified folder for data

Dataset source from : http://graphchallenge.mit.edu/data-sets

The optional patterns are shown below


![pattern](https://github.com/mengziheng/Gpu-SubgraphIsomorphism/assets/121283369/ce320ef4-869a-4a91-97dd-bd07e42d0479)




## Preprocess 

include preprocess script for H-index , Trust , SMOG

SMOG is in preprocess/SMOG

All preprocess script is preprocess.py

preprocess graph : 
    `python preprocess.py ../../data/graph ../../data/processed_graph`

The input arguments is 
1. input graph folder (include all graph to be preprocessed)
2. output graph folder (include all graph already preprocessed)

## Compile and Run code
Before running the following code, you need to preprocess the graph first. See the preprocess part above

Run the srcipt in SMOG folder
    `python script.py --input_graph_folder ../data/processed_graph/amazon0302 --input_pattern Q2`

input_graph_folder and input_pattern are required, and there are also other parameters you can choose to change：

--N : numbers of GPU

--load_factor : load factor for hash table

--bucket_size : bucket size for hash table

--block_number_for_kernel : number of block for subgraph match kernel

--block_size_for_kernel : size of block for subgraph match kernel

--chunk_size : chunk size for task assignment

## Experiment 
Include experiment script for H-index , Trust , SMOG

Experiment for Trust and Hindex: 
    `python performance_benchmark.py /data/zh_dataset/Hindex_processed_graph_challenge_dataset/snap`

If you want to test the results of all graphs and all patterns on SMOG, you can execute 
    `python pattern.py`
You can change the contents of pattern.py to modify the graph and pattern to be tested and the path to the output file

If you want to test the results on different numbers of GPUs, you can execute 
    `python scalability.py` 
in scalability folder, you can change the contents of scalability.py to modify the pattern and numbers of GPUs to be tested and the path to the output file. You can also change the contents of graph_dataset.txt to modify the graphs to be tested

## Contact
Please send me a email if you have any questions: zhmeng.cn@gmail.com or wzbwangzhibin@gmail.com

## Results

| Graph/Pattern        | Q1 count    | Q2 count    | Q3 count  | Q4 count    | Q5 count   | Q6 count    | Q7 count  | Q8 count   |
|----------------------|-------------|-------------|-----------|-------------|------------|-------------|------------|-------------|
| amazon0302           | 2477087     | 3110467     | 304441    | 4915256     | 67942      | 6405886     | 6267       | 2891271     |
| amazon0312           | 32229022    | 40921800    | 3975712   | 202015989   | 3187886    | 249735343   | 1920570    | 126529077   |
| amazon0505           | 35492004    | 45276121    | 4361282   | 229617605   | 3543622    | 282891801   | 2147049    | 141362172   |
| amazon0601           | 35661461    | 45701006    | 4420994   | 232448240   | 3606466    | 285794903   | 2193997    | 143515707   |
| as20000102           | 288840      | 287940      | 5636      | 1452941     | 5900       | 1773657     | 4779       | 410823      |
| as-caida20071105     | 2287349     | 2042272     | 53875     | 16546241    | 82231      | 23868747    | 102147     | 5221539     |
| ca-AstroPh           | 44916549    | 72355715    | 9580415   | 2545573891  | 64997961   | 2732693146  | 400401488  | 2115036981  |
| ca-CondMat           | 1505383     | 2349650     | 294008    | 21403215    | 511088     | 23018710    | 919604     | 16737753    |
| ca-GrQc              | 1054723     | 2041499     | 329297    | 68885210    | 2215500    | 69763367    | 12898478   | 67338180    |
| ca-HepPh             | 507961441   | 976450063   | 156497775 | 2.0414E+11  | 6608523653 | 2.05227E+11 |            | 1.99285E+11 |
| ca-HepTh             | 239081      | 429013      | 65592     | 8479804     | 279547     | 8518817     | 1123584    | 8407782     |
| cit-HepPh            | 39015537    | 36512540    | 2562245   | 429991999   | 4262265    | 647492608   | 6262553    | 209197863   |
| cit-HepTh            | 63698507    | 61934120    | 4113289   | 967443899   | 9380530    | 1386675032  | 19927580   | 439990809   |
| cit-Patents          | 341906226   | 83785566    | 3501071   | 446804010   | 3039636    | 1776150458  | 3151595    | 214057482   |
| email-Enron          | 36256665    | 36528276    | 2341639   | 720230791   | 5809356    | 1018877581  | 11213163   | 303894917   |
| email-EuAll          | 17093921    | 12663071    | 581032    | 105766262   | 1101520    | 217531995   | 1633335    | 34119119    |
| facebook_combined    | 144023053   | 228787050   | 30004668  | 21763153227 | 517965151  | 25095227541 | 7830937838 | 18073550424 |
| flickrEdges          | 23124971168 | 37701640669 | 47798166  | 31234180596 | 47798166   | 37701640669 | 47798166   | 62468361192 |
| loc-brightkite_edges | 23380701    | 29261679    | 2851308   | 1141663792  | 19481391   | 1412668691  | 117409296  | 770479983   |
| loc-gowalla_edges    | 146652712   | 122513526   | 6086852   | 2385052206  | 14570875   | 3649782390  | 28928240   | 843084102   |
| oregon1_010331       | 882711      | 903673      | 26610     | 7201602     | 48224      | 8920546     | 74010      | 2584989     |
| oregon1_010407       | 862089      | 856275      | 19339     | 5631882     | 25558      | 7079906     | 27610      | 1653027     |
| oregon1_010414       | 1047953     | 1062429     | 27135     | 8464828     | 43369      | 10585976    | 58353      | 2715672     |
| oregon1_010421       | 1086538     | 1104592     | 29766     | 9095735     | 48256      | 11473075    | 63557      | 2981766     |
| oregon1_010428       | 1039416     | 1056059     | 26461     | 8426215     | 43111      | 10532050    | 57397      | 2697855     |
| oregon1_010505       | 1029499     | 1044752     | 26032     | 8190827     | 41425      | 10266580    | 53748      | 2617554     |
| oregon1_010512       | 1029394     | 1044280     | 26021     | 8167375     | 42024      | 10227080    | 56363      | 2619801     |
| oregon1_010519       | 1036037     | 1053528     | 26855     | 8422989     | 45008      | 10522786    | 62383      | 2752923     |
| oregon1_010526       | 1127118     | 1143939     | 30479     | 9287382     | 46025      | 11810065    | 54635      | 2979066     |
| oregon2_010331       | 3864085     | 4534433     | 346110    | 96857189    | 1225487    | 128199789   | 3486756    | 55375107    |
| oregon2_010407       | 3676432     | 4274174     | 310832    | 86547287    | 1018894    | 114316252   | 2589878    | 47372241    |
| oregon2_010414       | 4395393     | 5168925     | 384812    | 114632553   | 1357613    | 150875771   | 3732350    | 63016083    |
| oregon2_010421       | 4034908     | 4669252     | 328075    | 96542004    | 1057622    | 127655123   | 2612910    | 50902146    |
| oregon2_010428       | 3850923     | 4393556     | 296111    | 87226143    | 914846     | 116021694   | 2184648    | 45054363    |
| oregon2_010505       | 3515244     | 3997732     | 265128    | 76793351    | 803624     | 101907997   | 1886911    | 39614316    |
| oregon2_010512       | 3563170     | 4054385     | 267660    | 78501518    | 819849     | 103741974   | 1964260    | 40337997    |
| oregon2_010519       | 4260328     | 4940921     | 343439    | 106661610   | 1170508    | 139293097   | 3140226    | 56056512    |
| oregon2_010526       | 4682574     | 5550489     | 399013    | 130655968   | 1500920    | 167586606   | 4496036    | 70406592    |
| p2p-Gnutella04       | 28497       | 750         | 3         | 6           | 0          | 1695        | 0          | 0           |
| p2p-Gnutella05       | 64998       | 4928        | 55        | 3237        | 0          | 22537       | 0          | 696         |
| p2p-Gnutella06       | 90764       | 8477        | 45        | 2788        | 0          | 65486       | 0          | 543         |
| p2p-Gnutella08       | 87885       | 15822       | 175       | 7544        | 6          | 48937       | 0          | 999         |
| p2p-Gnutella09       | 96902       | 15472       | 160       | 7577        | 6          | 48445       | 0          | 1068        |
| p2p-Gnutella24       | 8794        | 498         | 11        | 26          | 0          | 178         | 0          | 3           |
| p2p-Gnutella25       | 13517       | 410         | 7         | 24          | 0          | 166         | 0          | 3           |
| p2p-Gnutella30       | 46363       | 1109        | 13        | 42          | 0          | 1558        | 0          | 9           |
| p2p-Gnutella31       | 42466       | 922         | 16        | 40          | 0          | 592         | 0          | 6           |
| roadNet-CA           | 262339      | 12869       | 42        | 1           | 0          | 1373        | 0          | 0           |
| roadNet-PA           | 157802      | 5858        | 21        | 1           | 0          | 299         | 0          | 0           |
| roadNet-TX           | 183252      | 7494        | 32        | 2           | 0          | 476         | 0          | 0           |
| soc-Epinions1        | 166635817   | 112509146   | 5803397   | 2578196450  | 17417432   | 4729789840  | 45703641   | 997173396   |
| soc-Slashdot0811     | 49965153    | 28774497    | 1989958   | 870174916   | 10667149   | 1259485940  | 46754083   | 484393905   |
| soc-Slashdot0902     | 58678630    | 34060778    | 2260269   | 1027371282  | 12596328   | 1477856348  | 57879044   | 569811453   |

