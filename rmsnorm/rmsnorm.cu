#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include "mpi.h"
#include "nccl.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"


template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

template<typename T>
__global__ void rms_norm(
  T* output,
  const T* input,    // [num_tokens, hidden_size]
  T* residual,       // [num_tokens, hidden_size]
  const T* gamma,    // [hidden_size]
  const float epsilon,
  const int hidden_size,
  const bool has_residual
) {
    constexpr int vec_size = 16 / sizeof(T);
    using AccessType = cutlass::AlignedArray<T, vec_size>;
    using F = cutlass::Array<float, vec_size>;
    cutlass::NumericArrayConverter<float, T, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;

    extern __shared__ char smem[];
    AccessType* smem_ptr = reinterpret_cast<AccessType*>(smem);
    T* output_ptr = (T*)output;
    AccessType* residual_ptr = reinterpret_cast<AccessType*>(residual);
    const AccessType* input_ptr = reinterpret_cast<AccessType*>(input);
    const AccessType* gamma_ptr = reinterpret_cast<AccessType*>(gamma);

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    float local_sum = 0.0f;
    float variance = 0.0f;
    __shared__ float s_variance = 0.0f;
    const int n_elems = hidden_size / vec_size;
    const int offset = bid * n_elems;
    for (int i = tid; i < n_elems; i += blockDim.x){
        const int idx = offset + i;
        T val = input_ptr[idx];
        if (has_residual) {
          val = val + residual_ptr[idx];
          residual_ptr[idx] = val;
        }
        smem_ptr[i] = val;
        F val_f = convert_2_float(val);
        F square_val = val_f * val_f;
        #pragma unroll
        for (int v = 0; v < vec_size; v++) {
            local_sum += square_val[v];
        }
    }

    variance = blockReduceSum<float>(local_sum);
    if (threadIdx.x == 0)
    {
        variance = (variance / hidden_size); // Var[x] = E[x²]
        s_variance = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    for (int i = tid; i < n_elems; i += blockDim.x){
        F val_f = smem_ptr[i];
        AccessType gamma_val = gamma_ptr[i];
        F gamma_val_f = convert_2_float(gamma_val);
        val_f = val_f * s_variance * gamma_val_f;
        AccessType result = convert_2_half(val_f);
        output_ptr[offset + i] = result;
    }
}

template <typename T>
void run(const int batch_size, const int hidden_size) {
  std::mt19937 gen(20250102);
  std::uniform_real_distribution<float> dis(static_cast<float>(-10), static_cast<float>(10));
  T* gamma_hinput_h = (T*)malloc(batch_size * hidden_size * sizeof(T));
  for (int i = 0; i < batch_size * hidden_size; i++) {
    float data = dis(gen);
    input_h[i] = T(data);
  }
  T* residual_h = (T*)malloc(batch_size * hidden_size * sizeof(T));
  std::mt19937 gen1(20250102 + 1);
  for (int i = 0; i < batch_size * hidden_size; i++) {
    float data = dis(gen1);
    residual_h[i] = T(data);
  }
  T* gamma_h = (T*)malloc(hidden_size * sizeof(T));
  std::mt19937 gen2(20250102 + 2);
  for (int i = 0; i < hidden_size; i++) {
    float data = dis(gen2);
    gamma_h[i] = T(data);
  }

  T* input_d;
  T* residual_d;
  T* gamma_d;
  cudaMalloc(&input_d, batch_size * hidden_size * sizeof(T));
  cudaMalloc(&residual_d, batch_size * hidden_size * sizeof(T));
  cudaMalloc(&gamma_d, hidden_size * sizeof(T));
  cudaMemcpy(input_d, input_h, batch_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(residual_d, residual_h, batch_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(gamma_d, input_h, hidden_size * sizeof(T), cudaMemcpyHostToDevice);

  float epsilon = 1e-8;
  auto kernel = &rms_norm<T>;
  const int smem_size = hidden_size * sizeof(float);
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  kernel<<<batch_size, 1024, smem_size>>>(input_d, input_d, residual_d, gamma_d, epsilon, hidden_size, true);
  

}

int main(int argc, char** argv) {
  using namespace cute;
  using T = cute::half_t;
  std::mt19937 gen(20250102);
  std::uniform_real_distribution<float> dis(static_cast<float>(-10), static_cast<float>(10));
  const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  const int hidden_size = 8192;
  for (int i = 0; i < batch_sizes.size(); i++) {
    run<T>(batch_sizes[i], hidden_size);
  }
}



