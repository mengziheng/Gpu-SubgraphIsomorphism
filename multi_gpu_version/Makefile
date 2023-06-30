exe=trianglecounting.bin
N=1
cucc= "$(shell which nvcc)"
cc= "$(shell which mpicxx)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64  -O3 -W -Wall -Wno-unused-function -Wno-unused-parameter
cuflags= --compiler-options -Wall --gpu-architecture=compute_80 --gpu-code=sm_80  -O3   # --resource-usage 
# cuflags= --compiler-options -v -Xcudafe  -\# --resource-usage 

.SILENT: cucc
.SILENT: cc
.SILENT: cuflags
.SILENT: %.o


objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \
	$(wildcard ./*.hpp) \
	$(wildcard ./*.h) \


# %.o:%.cu $(deps)
# 	$(cucc) -c  $(cuflags)  $< -o $@ -g -G 

# %.o:%.cpp $(deps)
# 	$(cc) -c -g  $< -o $@ 

# $(exe):$(objs)
# 	$(cc) $(objs) $(commflags) -O3 -o $(exe)

%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $<  -o $@ 

%.o:%.cpp $(deps)
	$(cc) -c  $(commflags) $< -o $@ 

$(exe):$(objs)
	$(cc) $(objs) $(commflags) -o $(exe)

test:$(exe)
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/cit-Patents_adj/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/graph500-scale25-ef16_adj/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/Theory-5-9-16-25-81-B1k/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/201512020030.v68863315_e143414960/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/U1a/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/V1r/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/P1a/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/amazon0302_adj/ 1 1024 1024 1
	mpirun -n 1 ./trianglecounting.bin /data/zh_dataset/dataforclique/V2a/ 1 1024 1024 1



clean:
	rm -rf *.o ${exe}
