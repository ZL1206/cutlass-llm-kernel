#include <Python.h>
#include <torch/nn/functional/padding.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"


#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace  flash
{


void run_gemm_fwd(fwd_params &params, cudaStream_t stream) {
    run_gemm_forward_<cutlass::half_t>(params, stream);
}

void gemm_fwd(const at::Tensor& input,
              const at::Tensor& weight,
              at::Tensor& out) 
{
    at::cuda::CUDAGuard device_guard{input.device()};

    auto input_dtype = input.dtype();
    TORCH_CHECK(input_dtype == torch::KFloat16, "only support fp16 now");
    CHECK_DEVICE(input); CHECK_DEVICE(weight);

    fwd_params params = {};
    params.ptr_A = input.data_ptr();
    params.ptr_B = weight.data_ptr();
    params.ptr_C = out.data_ptr();
    params.M = input.size(0);
    params.N = weight.size(0);
    params.K = input.size(1);
    params.A_row_stride = input.stride(0);
    params.B_row_stride = weight.size(0);
    params.C_row_stride = out.size(0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    run_gemm_fwd(params, stream);
}

    
} // namespace  flash



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "wgmma-ws-ss";
    m.def("fwd", &flash::gemm_fwd, "forward pass");
}

