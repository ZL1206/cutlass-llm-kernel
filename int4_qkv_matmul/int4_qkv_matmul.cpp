#include <torch/extension.h>


void int4_qkv_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    at::Tensor& o,
    const float softmax_scale,
    const bool is_causal
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_qkv_matmul", &int4_qkv_matmul, "int4 kvcache qkv matmul in flash attention");
}
