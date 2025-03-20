#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "flash.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>


namespace flash {


void set_params_fprop(kernel_params &params,
                      const int b,
                      const int seqlen_q,
                      const int seqlen_k,
                      const int seqlen_q_rounded,
                      const int seqlen_k_rounded,
                      const int h,
                      const int h_k,
                      const int d,
                      const int d_rounded,
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      at::Tensor& out,
                      const float softmax_scale
                    ) {
    params = {};

    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;



    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();

    

    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.o_row_stride = out.stride(-3);

    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;

    params.is_seqlens_k_cumulative = true;
}



void run_mha_fwd(kernel_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<elem_type, 128, Is_causal>(params, stream);
            });
        
    });
}


void mha_fwd(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const float softmax_scale,
    bool is_causal
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q == 1) {
        is_causal = false;
    }

    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
        out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);


    kernel_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     seqlen_q_rounded,
                     seqlen_k_rounded,
                     num_heads,
                     num_heads_k,
                     head_size,
                     head_size_rounded,
                     q,
                     k,
                     v,
                     out,
                     softmax_scale);
    params.is_causal = is_causal;
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
             cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        out.zero_();
    }
    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * ngroups, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * ngroups, head_size});
    }
}

} // namespace flash


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &flash::mha_fwd, "mini flash attention");
}
