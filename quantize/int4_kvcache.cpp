#include <torch/extension.h>

namespace glm {

void int4_kvcache(
  at::Tensor& key,     // [num_tokens, num_heads, head_size]
  at::Tensor& value,   // [num_tokens, num_heads, head_size]
);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "int4_kvcache",
    &glm::int4_kvcache,
    "int4 kvcache");
}
