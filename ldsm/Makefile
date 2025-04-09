CC_FILES=$(shell find ./ -name "*.cu" -print)
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O3 -arch=sm_90 -std=c++17 -I../3rd/cutlass/include -I../3rd/cutlass/tools/util/include

clean:
	rm -rf $(EXE_FILES)
