CC_FILES=$(shell find ./ -name "*.cu" -print)
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O3 -gencode arch=compute_90a,code=sm_90a -Xptxas=--verbose --ptxas-options=-v -std=c++17 -I../3rd/cutlass/include -I../3rd/cutlass/tools/util/include --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DNDEBUG -lineinfo

clean:
	rm -rf $(EXE_FILES)
