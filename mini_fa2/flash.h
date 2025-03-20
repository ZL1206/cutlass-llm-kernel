#pragma once

#include <cuda.h>
#include <vector>


namespace flash {

struct kernel_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void * __restrict__ o_ptr;
    int seqlen_q;
    int seqlen_k;
    int d;
    int q_head_stride;
    int k_head_stride;
    int v_head_stride;
    int o_head_stride;
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    bool is_bf16;
};


template<typename T, int Headdim, bool Is_causal> void run_mini_mha_fwd_(kernel_params &params, cudaStream_t stream);


}