#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "flash.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>
#include "hardware_info.h"

namespace flash {

void run_mha_fwd(kernel_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_mha_fwd_splitkv_dispatch<elem_type, 128, Is_causal>(params, stream);
        });
    });
}


void set_params_fprop(kernel_params& params,
                      const int b,
                      const int seqlen_q,
                      const int seqlen_k,
                      const int h,
                      const int h_k,
                      const int d,
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      const at::Tensor& k_params,
                      const at::Tensor& v_params,
                      at::Tensor& out,
                      void* cu_seqlens_q_d,
                      void* seqlens_k_d,
                      void* softmax_lse_d,
                      float softmax_scale,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false
) {
    params = {};
    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.k_params_ptr = k_params.data_ptr();
    params.v_params_ptr = v_params.data_ptr();
    params.o_ptr = out.data_ptr();

    params.q_row_stride = q.stride(-3);
    params.q_head_stride = q.stride(-2);
    

    params.k_block_stride = k.stride(0);
    params.k_head_stride = k.stride(-3);
    params.k_row_stride = k.stride(-2);
    params.k_params_block_stride = k_params.stride(0);
    params.k_params_head_stride = k_params.stride(-3);

    params.v_block_stride = v.stride(0);
    params.v_head_stride = v.stride(-3);
    params.v_row_stride = v.stride(-2);
    params.v_params_block_stride = v_params.stride(0);
    params.v_params_head_stride = v_params.stride(-3);
    
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.seqlens_k = static_cast<int *>(seqlens_k_d);

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softmax_lse_ptr = softmax_lse_d;

    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.gqa = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

}


std::vector<at::Tensor>
mha_varlen_fwd(
    at::Tensor& q,
    const at::Tensor& k_cache, // [num_block, num_head, page_size, head_size]
    const at::Tensor& v_cache,
    const at::Tensor& k_params,
    const at::Tensor& v_params,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& block_table,
    at::Tensor& out,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale,
    bool is_causal
) {
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
   
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k_cache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v_cache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    const auto sizes = q.sizes();
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k_cache.size(1);
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int max_num_blocks_per_seq = block_table.size(1);
    const int page_size = k_cache.size(2);

    if (max_seqlen_q == 1) {
        is_causal = false;
    }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k;
    printf("max_seqlen_q is %d, num_heads is %d, num_heads_k is %d, page_size is %d, seqlenq_ngroups_swapped is %d\n", max_seqlen_q, num_heads, num_heads_k, page_size, seqlenq_ngroups_swapped);
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    
    kernel_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, 
                     max_seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q,
                     k_cache,
                     v_cache,
                     k_params,
                     v_params,
                     out,
                     cu_seqlens_q_d,
                     seqlens_k.data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_scale,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    params.total_q = total_q;
    params.is_causal = is_causal;
    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_size = page_size;
    
    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }

    return {out, softmax_lse};

}

} // namespace flash


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("varlen_fwd", &flash::mha_varlen_fwd, "mini flash attention");
}
