exe=subgraphmatch.bin
N=1
cucc= "$(shell which nvcc)"
cc= "$(shell which mpicxx)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64  -O3 -W -Wall -Wno-unused-function -Wno-unused-parameter
cuflags= --compiler-options -Wall --gpu-architecture=compute_80 -lineinfo --gpu-code=sm_80  -O3   # --resource-usage 
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
	./$(exe) cit-Patents_adj triangle 0.3 8 216 1024 1

clean:
	rm -rf *.o ${exe}
