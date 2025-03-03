#include <torch/extension.h>


void qk_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    at::Tensor& o,
    const float softmax_scale,
    const bool is_causal
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_matmul", &qk_matmul, "qk matmul in flash attention");
}
