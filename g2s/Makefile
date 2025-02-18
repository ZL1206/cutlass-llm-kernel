CC_FILES=$(shell find ./ -name "*.cu" -print)
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O3 -arch=compute_90a --ptxas-options=-v -std=c++17 -I../3rd/cutlass/include --expt-relaxed-constexpr

clean:
	rm -rf $(EXE_FILES)
