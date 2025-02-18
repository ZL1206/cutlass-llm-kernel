CC_FILES=$(shell find ./ -name "*.cu" -print)
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc --ptxas-options=-v -o $@ $< -O2 -arch=sm_90 -std=c++17 -I../3rd/cutlass/include -lnccl -I/usr/local/mpi/include -L/usr/local/mpi/lib -lmpi

clean:
	rm -rf $(EXE_FILES)
