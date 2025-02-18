#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mpi.h"
#include "nccl.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"


#define FINAL_MASK 0xffffffff

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}



template <typename T>
__global__ void allreduce_fp8_quantize(T* input, const int num_heads, const int head_size, const int hidden_size, cutlass::float_e4m3_t* out, float* scales) {
    const int head_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int tid = threadIdx.x;
    float fp8_max = std::numeric_limits<cutlass::float_e4m3_t>::max();

    const int offset = batch_id * hidden_size + head_id * head_size + tid;   
    float val = static_cast<float>(input[offset]);
    float abs_val = fabs(val);
    float max_abs_val = blockReduceMax(abs_val);
    float scale = fp8_max / max_abs_val;
    cutlass::float_e4m3_t q_val = static_cast<cutlass::float_e4m3_t>(val * scale);
    scale = 1 / scale;
    out[offset] = q_val;
    if (tid == 0) {
        scales[batch_id * num_heads + head_id] = scale;
    }
}


template <typename T>
__global__ void allreduce_fp8_quantize_v2(T* input, cutlass::float_e4m3_t* out, float* scales) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int vec_size = 16 / sizeof(T);
    using AccessType = cutlass::AlignedArray<T, vec_size>;
    using V = cutlass::Array<T, vec_size>;
    using F = cutlass::Array<float, vec_size>;
    using R = cutlass::Array<cutlass::float_e4m3_t, vec_size>;
    
    float fp8_max = std::numeric_limits<cutlass::float_e4m3_t>::max();
    cutlass::NumericArrayConverter<float, T, vec_size> convert_2_float;
    cutlass::maximum_absolute_value_reduction<F, true> amax;
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, vec_size> convert_2_fp8;
     
    int offset = bid * 1024;
    const AccessType* input_ptr = reinterpret_cast<const AccessType*>(input + offset);
    float2* out_ptr = reinterpret_cast<float2*>(out + offset);
    AccessType val;
    *reinterpret_cast<AccessType*>(&val) = input_ptr[tid];
    F val_f = convert_2_float(val);
    float max_abs_val = amax(0.0, val_f);
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      max_abs_val = fmaxf(max_abs_val, __shfl_xor_sync(uint32_t(-1), max_abs_val, mask));
    }
    float scale = fp8_max / max_abs_val;
    R q_val = convert_2_fp8(scale * val_f);
    scale = 1 / scale;
    out_ptr[tid] = *reinterpret_cast<float2*>(q_val.data());
    if (tid % 16 == 0) {
        offset = (offset + tid * 8) / 128;
        scales[offset] = scale;
    }
}



int main() {
    using T = cutlass::half_t;
    using Vec = cutlass::Array<cutlass::float_e4m3_t, 16>;
    int size = sizeof(Vec);
    int alig_size = alignof(Vec);
    printf("size of array is %d, alig size is %d\n", size, alig_size);
    int iter = 1000;
    int batch_size = 512;
    int hidden_size = 8192;
    int message_size = batch_size * hidden_size;
    int head_size = 128;
    int num_heads = hidden_size / 128;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int sm_nums = prop.multiProcessorCount;
    T* d_in;
    cudaMalloc(&d_in, message_size * sizeof(T));
    cutlass::float_e4m3_t* d_out1;
    cudaMalloc(&d_out1, message_size * sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t* d_out2;
    cudaMalloc(&d_out2, message_size * sizeof(cutlass::float_e4m3_t));
    float* d_scale1;
    cudaMalloc(&d_scale1, message_size * sizeof(float));
    float* d_scale2;
    cudaMalloc(&d_scale2, message_size * sizeof(float));

    T* h_in = (T*)malloc(message_size * sizeof(T));
    for (int i = 0; i < message_size; i++) {
        h_in[i] = static_cast<float>(1.0);
    }
    cudaMemcpy(d_in, h_in, sizeof(T) * message_size, cudaMemcpyHostToDevice);

    cutlass::float_e4m3_t* h_out1 = (cutlass::float_e4m3_t*)malloc(message_size * sizeof(cutlass::float_e4m3_t));
    cutlass::float_e4m3_t* h_out2 = (cutlass::float_e4m3_t*)malloc(message_size * sizeof(cutlass::float_e4m3_t));
    float* h_scale1 = (float*)malloc(message_size * sizeof(float));
    float* h_scale2 = (float*)malloc(message_size * sizeof(float));
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int thread_nums = 128;
    int data_per_block = thread_nums * 8;
    int available_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&available_blocks_per_sm, allreduce_fp8_quantize_v2<T>, thread_nums, 0);
    printf("available_blocks_per_sm is %d, sm_nums is %d\n", available_blocks_per_sm, sm_nums);
    data_per_block = message_size / int(available_blocks_per_sm * sm_nums * 0.8);
    dim3 grid1(message_size / 1024);
    dim3 block1(128);
    for (int i = 0; i < 10; i++) {
        allreduce_fp8_quantize_v2<T><<<grid1, block1, 0, stream>>>(d_in, d_out2, d_scale2);
    }
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        allreduce_fp8_quantize_v2<T><<<grid1, block1, 0, stream>>>(d_in, d_out2, d_scale2);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float v2_time;
    cudaEventElapsedTime(&v2_time, begin, end);
    v2_time /= iter;

    dim3 grid(num_heads, batch_size);
    dim3 block(128);
    for (int i = 0; i < 10; i++) {
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, num_heads, head_size, hidden_size, d_out1, d_scale1);
    }
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, num_heads, head_size, hidden_size, d_out1, d_scale1);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    time /= iter;

    cudaMemcpyAsync(h_out1, d_out1, sizeof(cutlass::float_e4m3_t) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out2, d_out2, sizeof(cutlass::float_e4m3_t) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_scale1, d_scale1, sizeof(float) * message_size / 128, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_scale2, d_scale2, sizeof(float) * message_size / 128, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    for (int i = 0; i < message_size; i++) {
        float h1 = static_cast<float>(h_out1[i]);
        float h2 = static_cast<float>(h_out2[i]);
        float threshold = 1e-5;
        float diff = std::abs(h1 - h2);
        if (diff > threshold) {
            printf("err idx %d, value %f vs %f\n", i, h1, h2);
        }

    }
    for (int i = 0; i < message_size / 128; i++) {
        float scale1 = h_scale1[i];
        float scale2 = h_scale1[2];
        float threshold = 1e-5;
        float diff = std::abs(scale1 - scale2);
        if (diff > threshold) {
            printf("err idx %d, value %f vs %f\n", i, scale1, scale2);
        }
    }
    printf("time %f, v2 time %f \n", time, v2_time);


}