#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "flash.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>


namespace flash {

void run_mha_fwd(kernel_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mini_mha_fwd_<elem_type, 128, Is_causal>(params, stream);
            });
        
    });
}


void mini_mha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const float softmax_scale,
    const bool is_causal
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    const int seqlen_q = q.size(0);
    const int head_size = q.size(1);
    const int seqlen_k = k.size(0);
    printf("seqlen_q %d, seqlen_k %d, head_size %d\n", seqlen_q, seqlen_k, head_size);
    TORCH_CHECK(head_size == 128, "only support k == 128");
    kernel_params params;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_size;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_head_stride = out.stride(-2);
    params.is_bf16 = q.dtype() == torch::kBFloat16;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
}

} // namespace flash


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mini_fwd", &flash::mini_mha_fwd, "mini flash attention");
}
