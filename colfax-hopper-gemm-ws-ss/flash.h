#pragma once

#include <cuda.h>
#include <vector>

struct fwd_params {
    int M;
    int N;
    int K;
    void * __restrict__ ptr_A;
    void * __restrict__ ptr_B;
    void * __restrict__ ptr_C;
    int64_t A_row_stride;
    int64_t B_row_stride;
    int64_t C_row_stride;
};

template<typename T>
void run_gemm_forward_(fwd_params &params, cudaStream_t stream);

