CC_FILES=$(shell find ./ -name "*.cu" -print)
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O3 -arch=sm_80 -std=c++17

clean:
	rm -rf $(EXE_FILES)
