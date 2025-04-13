#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/numeric_types.h"

#include "static_switch.h"

namespace glm {

template <typename scalar_t, typename cache_t>
__global__ void int4_kvcache_kernel(
  const scalar_t* __restrict__ key,
  const scalar_t* __restrict__ value,
  cache_t* __restrict__ key_cache,
  cache_t* __restrict__ value_cache,
  float* __restrict__ key_cache_scale,
  float* __restrict__ value_cache_scale,
  const int64_t* __restrict__ slots,
  const int head_size,
  const int kv_stride,
  const int kv_cache_stride,
  const int page_size
) {
  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t slot = slots[batch_idx];
  const int block_idx = slot / page_size;
  const int block_offset = slot % page_size;

  const scalar_t* k_ptr = key + batch_idx * kv_stride + head_idx * head_size;
  const scalar_t* v_ptr = value + batch_idx * kv_stride + head_idx * head_size;

  cache_t* key_cache_ptr = key_cache + block_idx * kv_cache_stride
                         + head_idx * page_size * head_size
                         + block_offset * head_size;
  cache_t* value_cache_ptr = value_cache + block_idx * kv_cache_stride
                           + head_idx * page_size * head_size
                           + block_offset * head_size;

  const scalar_t k_value = k_ptr[tid];
  const scalar_t v_value = v_ptr[tid];

  
}

template <typename T, typename CACHE_T>
void fp8_kvcache_launch(
  const at::Tensor& key,
  const at::Tensor& value,
  at::Tensor& key_cache,
  at::Tensor& value_cache,
  const at::Tensor& slots,
  const float k_scale,
  const float v_scale
) {
  const int num_tokens = key.size(0);
  const int kv_num_heads = key.size(1);
  const int kv_stride = key.stride(0);
  const int kv_cache_stride = key_cache.stride(0);
  const int page_size = key_cache.size(2);
  const int head_size = key.size(2);

  const dim3 grid(kv_num_heads, num_tokens);
  const dim3 block(head_size);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  glm::fp8_kvcache_kernel<<<grid, block, 0, stream>>>(
    static_cast<const T*>(key.const_data_ptr()),
    static_cast<const T*>(value.const_data_ptr()),
    static_cast<CACHE_T*>(key_cache.mutable_data_ptr()),
    static_cast<CACHE_T*>(value_cache.mutable_data_ptr()),
    static_cast<const std::int64_t*>(slots.const_data_ptr()),
    k_scale,
    v_scale,
    head_size,
    kv_stride,
    kv_cache_stride,
    page_size
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void int4_kvcache(
  at::Tensor& key,   // [num_tokens, num_heads, head_size]
  at::Tensor& value, // [num_tokens, num_heads, head_size]
) {
  auto kv_type = key.scalar_type();
  auto kv_cache_type = key_cache.scalar_type();

  FP16_SWITCH(kv_type == at::ScalarType::Half, [&] {
    if (kv_type == kv_cache_type) {
      fp8_kvcache_launch<elem_type, elem_type>(
        key, value, key_cache, value_cache, slots, k_scale, v_scale
      );
    } else {
      FP8_SWITCH(kv_cache_type == at::ScalarType::Float8_e4m3fn, [&] {
        fp8_kvcache_launch<elem_type, CACHE_T>(
          key, value, key_cache, value_cache, slots, k_scale, v_scale
        );
      });
    }
  });
}

} // namespace glm
