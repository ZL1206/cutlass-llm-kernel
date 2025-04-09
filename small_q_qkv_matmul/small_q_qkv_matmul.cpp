#include <torch/extension.h>


void small_q_qkv_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    const float softmax_scale,
    const bool is_causal
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("small_q_qkv_matmul", &small_q_qkv_matmul, "qkv matmul for small q");
}
