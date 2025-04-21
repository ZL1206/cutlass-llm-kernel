#pragma once

#include <cuda.h>
#include <vector>


namespace flash {

struct kernel_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ k_params_ptr;
    void *__restrict__ v_params_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;

    int *__restrict__ block_table;
    int64_t block_table_batch_stride;
    int page_size;

    int seqlen_q;
    int seqlen_k;
    int b;
    int d;
    int h;
    int h_k;
    int gqa;
    int total_q;

    int64_t q_batch_stride;
    int64_t q_row_stride;
    int64_t q_head_stride;

    int64_t k_block_stride;
    int64_t k_row_stride;
    int64_t k_head_stride;
    int64_t k_params_block_stride;
    int64_t k_params_head_stride;

    int64_t v_block_stride;
    int64_t v_row_stride;
    int64_t v_head_stride;
    int64_t v_params_block_stride;
    int64_t v_params_head_stride;

    int64_t o_batch_stride;
    int64_t o_row_stride;
    int64_t o_head_stride;

    int * __restrict__ cu_seqlens_q;
    int * __restrict__ seqlens_k;
    
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    bool is_bf16;

    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;

    int num_splits = 1;
};


template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(kernel_params &params, cudaStream_t stream);


}