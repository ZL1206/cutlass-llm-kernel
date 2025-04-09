#include <cstdio>
#include <float.h>
#include <algorithm>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cute/tensor.hpp>


using namespace cute;

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

__global__ void shfl_example(int* data, int* t) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float qk_maxs;

    //qk_maxs = lane < 4 ? rowmax[lane] : -FLT_MAX;
    qk_maxs = tid;
    for (int idx = tid; idx < 112; idx += 256) {
      unsigned int active_mask = __activemask();
      //printf("tid %d, active_mask is %u\n", tid, active_mask);
      for (int mask = 8 / 2; mask >= 1; mask /= 2) {
        qk_maxs = fmaxf(qk_maxs, __shfl_xor_sync(uint32_t(-1), qk_maxs, mask, 8));
      }
    }

    //qk_maxs = __shfl_sync(uint32_t(-1), qk_maxs, 0);
    
    //int value = tid;        // 每个线程的值为其线程 ID

    // 取出 warp 中第 0 号线程的 `value` 值
    //value = blockReduceSum(value);

    data[tid] = qk_maxs;   // 将结果存储回数据数组

    float a = -FLT_MAX;
    float b = a * 0.0f;
    float c = -FLT_MAX - 1000000;
    float d = -INFINITY - (-INFINITY);
    if (t == nullptr) {
        //printf("t is nullptr on device\n");
    } else {
        //printf("t is not nullptr on device\n");
    }

    uint8_t f[4];
    f[0] = 56;
    f[1] = 46;
    f[2] = 185;
    f[3] = 109;

    uint32_t* f_32 = reinterpret_cast<uint32_t*>(f);
    uint32_t v = 3457802691;
    if (tid == 0) {
      printf("v is %u\n", v);
    }
    cutlass::Array<cutlass::half_t, 8> result;
    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t i4s = v;
    static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOT_MASK    = 0x000f000f;
    static constexpr uint32_t TOP_MASK    = 0x00f000f0;
    static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;  // `1024`
    static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;  // `64`
    uint32_t top_i4s = __byte_perm(i4s, 0, 0x4321);
    asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
    asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(MAGIC_NUM_0));
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(MAGIC_NUM_1));
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(MAGIC_NUM_0));
    asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[3]) : "r"(h[3]), "r"(MAGIC_NUM_1));

    

    
    cutlass::Array<cutlass::half_t, 8> result_c;
    uint32_t* r = reinterpret_cast<uint32_t*>(&result_c);
    uint32_t src_reg = v;
    for (int ii = 0; ii < 4; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t or_mask = 0x64006400;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t hi_mask = 0x00F000F0;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = or_mask;    // 0x64006400, {1024, 1024}
      static constexpr uint32_t hi_bias  = 0xD400D400; // {-64, -64}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                             reinterpret_cast<const half2&>(hi_scale),
                             reinterpret_cast<const half2&>(hi_bias));
      }
    }

    if (thread0()) {
      for (int i = 0; i < 8; i++) {
        float t = static_cast<float>(result[i]);
        float t_ = static_cast<float>(result_c[i]);
        printf("i %d, l is %f,  c is %f\n", i, t, t_);
      }
    }

    //printf("a is %f, b is %f, c is %f, d is %f\n", a, b, c, d);
}


int main() {
    const int num_threads = 256;
    int h_data[num_threads];
    
    int* d_data;
    cudaMalloc(&d_data, num_threads * sizeof(int));
    int* a = nullptr;
    // 启动 64 个线程的内核
    shfl_example<<<1, num_threads>>>(d_data, a);

    // 从设备复制数据到主机
    cudaMemcpy(h_data, d_data, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出
    for (int i = 0; i < num_threads; i++) {
        //printf("Thread %d has value %d\n", i, h_data[i]);
    }

    cudaFree(d_data);

    return 0;
}