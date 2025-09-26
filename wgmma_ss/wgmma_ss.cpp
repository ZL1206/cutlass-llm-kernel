#include <torch/extension.h>


void wgmma_matmul(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& o
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wgmma_matmul", &wgmma_matmul, "qk matmul in flash attention");
}