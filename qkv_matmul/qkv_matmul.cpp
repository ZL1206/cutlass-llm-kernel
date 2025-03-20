#include <torch/extension.h>


void qkv_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    const float softmax_scale,
    const bool is_causal
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_matmul", &qkv_matmul, "qk matmul in flash attention");
}
