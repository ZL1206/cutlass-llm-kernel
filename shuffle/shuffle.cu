#include <cstdio>
#include <float.h>
#include <algorithm>
#include <cstdint>

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

__global__ void shfl_example(int* data) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    float qk_maxs;

    //qk_maxs = lane < 4 ? rowmax[lane] : -FLT_MAX;
    qk_maxs = tid;
    for (int idx = tid; idx < 112; idx += 256) {
      unsigned int active_mask = __activemask();
      printf("tid %d, active_mask is %u\n", tid, active_mask);
      for (int mask = 8 / 2; mask >= 1; mask /= 2) {
        qk_maxs = fmaxf(qk_maxs, __shfl_xor_sync(uint32_t(-1), qk_maxs, mask, 8));
      }
    }

    //qk_maxs = __shfl_sync(uint32_t(-1), qk_maxs, 0);
    
    //int value = tid;        // 每个线程的值为其线程 ID

    // 取出 warp 中第 0 号线程的 `value` 值
    //value = blockReduceSum(value);

    data[tid] = qk_maxs;   // 将结果存储回数据数组
}


int main() {
    const int num_threads = 256;
    int h_data[num_threads];
    
    int* d_data;
    cudaMalloc(&d_data, num_threads * sizeof(int));

    // 启动 64 个线程的内核
    shfl_example<<<1, num_threads>>>(d_data);

    // 从设备复制数据到主机
    cudaMemcpy(h_data, d_data, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印输出
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %d has value %d\n", i, h_data[i]);
    }

    cudaFree(d_data);

    return 0;
}