#pragma once

#include <cuda.h>
#include <vector>


namespace flash {

struct kernel_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void * __restrict__ o_ptr;

    int b;
    int h;
    int h_k;
    int h_h_k_ratio;
    int seqlen_q_rounded;
    int seqlen_k_rounded;
    int d;

    int seqlen_q;
    int seqlen_k;
    
    int q_batch_stride;
    int k_batch_stride;
    int v_batch_stride;
    int o_batch_stride;

    int q_row_stride;
    int k_row_stride;
    int v_row_stride;
    int o_row_stride;

    int q_head_stride;
    int k_head_stride;
    int v_head_stride;
    int o_head_stride;
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    bool is_bf16;

    // pointers
    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;

    bool is_seqlens_k_cumulative;

};


template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(kernel_params &params, cudaStream_t stream);


}