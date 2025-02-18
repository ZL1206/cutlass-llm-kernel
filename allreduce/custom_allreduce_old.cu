#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mpi.h"
#include <random>
#include "nccl.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/numeric_conversion.h"
#include "cutlass/array.h"
#include "nvtx3/nvToolsExt.h"
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr int MAX_ALL_REDUCE_BLOCKS = 32;

constexpr size_t alignSize(size_t size, size_t to)
{
    if ((size % to) != 0U)
    {
        size += to - size % to;
    }
    return size;
}

template <typename T>
void compare(
    int rank, T* a, T* b, int size, bool print_err = false)
{
    float max_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-5;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(a[n]);
        float vb = static_cast<float>(b[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            if (print_err)
            {
                printf("err idx %d, value %f vs %f\n", n, va, vb);
            }
            max_diff = std::max(max_diff, diff);
            ++diff_cnt;
        }
    }
    
    printf("rank %d, max diff %f, diff cnt %d/%d\n", rank, max_diff, diff_cnt, size);
}


struct AllReduceParams
{
    int elts_total;
    int ranks_per_node;
    int local_rank;
    int group_size;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* tmp_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    float* fp8_scale_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;
};

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr)
{
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static inline __device__ uint32_t ld_flag_acquire(uint32_t* flag_addr)
{
    uint32_t flag;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    return flag;
}

static inline __device__ void st_flag_volatile(uint32_t const& flag, uint32_t* flag_addr) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static inline __device__ uint32_t ld_flag_volatile(uint32_t* flag_addr) {
  uint32_t flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

__inline__ __device__ void multi_gpu_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx)
{
    // After this function, at least one block in each GPU has reached the barrier
    if (tidx < world_size)
    {
        // we can think of signals having the shape [world_size, world_size]
        // Dimension 0 is the "listening" dimension, dimension 1 is "emitting" dimension

        // Block 0 broadcasts its flag (local_rank on emitting dimension) to all receivers
        size_t offset = (flag % 2) ? world_size : 0;

        if (bidx == 0)
        {
            st_flag_volatile(flag, signals[tidx] + offset + local_rank);
            //st_flag_release(flag, signals[tidx] + offset + local_rank);
        }

        // All blocks check that corresponding block 0 on other GPUs have set the flag
        // No deadlock because block #0 is always the first block started
        uint32_t* peer_barrier_d = signals[local_rank] + offset + tidx;
        /*
        while (ld_flag_acquire(peer_barrier_d) != flag)
        {
        }
        */
        while (ld_flag_volatile(peer_barrier_d) != flag)
        {
        }
    }

    __syncthreads();
}

__inline__ __device__ void block_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx, int const grid_size)
{
    __syncthreads();
    // After this function, the block of id == bidx of each GPU has reached the barrier
    if (tidx < world_size)
    {
        // we can think of signals having the shape [world_size, 2, num_blocks, world_size]
        // (+ an offset on dim 2 to account for flags used in multi_gpu_barrier)
        // Dimension 0 is the "listening" dimension, dimension 3 is "emitting" dimension

        // Block broadcast its flag (local_rank on emitting dimension) to all receivers
        uint32_t flag_block_offset = bidx * world_size;

        
        st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);

        // Blocks check that corresponding blocks on other GPUs have also set the flag
        uint32_t* peer_barrier_d = signals[local_rank] + flag_block_offset + tidx;

        while (ld_flag_acquire(peer_barrier_d) != flag)
        {
        }
    }

    __syncthreads();
}


template <typename T, int WORLD_SIZE>
static __global__ void twoshot_allreduce_kernel(AllReduceParams params)
{
    const int grid_size = gridDim.x;
    const int block_size = blockDim.x;
    const int stride = grid_size * block_size;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x + bid * block_size;
    // The number of elements packed into one for comms
    static constexpr int vec_size = 16 / sizeof(T);
    using AccessType = cutlass::AlignedArray<T, vec_size>;
    using Vec = cutlass::Array<T, vec_size>;
    using Accum = cutlass::Array<T, vec_size>;
    cutlass::NumericArrayConverter<float, T, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;
    AccessType* local_output_buffer = reinterpret_cast<AccessType*>(params.local_output_buffer_ptr);
    float4* local_shared_buffer = reinterpret_cast<float4*>(params.peer_comm_buffer_ptrs[params.local_rank]);

    const int total = params.elts_total / vec_size;
    const int part = total / WORLD_SIZE;
    const int largest_part = total - (WORLD_SIZE - 1) * part;
    const int start = params.local_rank * part;
    const int end = params.local_rank == WORLD_SIZE - 1 ? total : start + part;
    AccessType* buffers[WORLD_SIZE];
    #pragma unroll
    for (int ii = 0; ii < WORLD_SIZE; ++ii)
    {
        // A mapping of the ranks to scatter reads as much as possible
        int rank = (params.local_rank + ii) % WORLD_SIZE;
        buffers[ii] = reinterpret_cast<AccessType*>(params.peer_comm_buffer_ptrs[rank]);
    }

    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid);
    
    const int vec_nums = end - start;
    Vec vals[WORLD_SIZE];
    Accum acc;
    for (int idx = tid; idx < vec_nums; idx += stride) {
        const int offset = idx + start;
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; i++)
        {
            AccessType* val_ptr = reinterpret_cast<AccessType*>(&vals[i]);
            val_ptr[0] = buffers[i][offset];
        }
        acc.clear();
        #pragma unroll
        for (int rank = 0; rank < WORLD_SIZE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int i = (rank + WORLD_SIZE - params.local_rank) % WORLD_SIZE;
            acc = acc + vals[i];
        }
        local_shared_buffer[offset] = *reinterpret_cast<float4*>(acc.data());
    }
    
    block_barrier(params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid, grid_size);

    for (int idx = tid; idx < largest_part; idx += stride) {
        #pragma unroll
        for (int ii = 0; ii < WORLD_SIZE; ++ii)
        {
            const int rank = (params.local_rank + ii) % WORLD_SIZE;
            const int offset = rank * part + idx;
            if (rank == WORLD_SIZE - 1 || idx < part) {
                local_output_buffer[offset] = buffers[ii][offset];
            }
        }
    }
    
}


template <typename T, int WORLD_SIZE>
static __global__ void twoshot_allreduce_fp8_kernel(AllReduceParams params)
{
    const int grid_size = gridDim.x;
    const int block_size = blockDim.x;
    const int stride = grid_size * block_size;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x + bid * block_size;
    // The number of elements packed into one for comms
    static constexpr int vec_size = 16 / sizeof(cutlass::float_e4m3_t);
    const int vec_nums_per_group = params.group_size / vec_size;
    using AccessType = cutlass::AlignedArray<cutlass::float_e4m3_t, vec_size>;
    using Vec = cutlass::Array<cutlass::float_e4m3_t, vec_size>;
    using Accum = cutlass::Array<float, vec_size>;
    using R = cutlass::Array<T, vec_size>;
    
    float fp8_max = std::numeric_limits<cutlass::float_e4m3_t>::max();
    cutlass::maximum_absolute_value_reduction<Accum, true> amax;
    cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, vec_size> convert_2_fp8;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;

    float4* local_output_buffer = reinterpret_cast<float4*>(params.local_output_buffer_ptr);
    float4* local_shared_buffer = reinterpret_cast<float4*>(params.peer_comm_buffer_ptrs[params.local_rank]);
    float* local_shared_scale = params.fp8_scale_ptrs[params.local_rank];
    
    int total = params.elts_total / vec_size;
    int part = total / WORLD_SIZE;
    int largest_part = total - (WORLD_SIZE - 1) * part;
    int start = params.local_rank * part;
    int end = params.local_rank == WORLD_SIZE - 1 ? total : start + part;
    AccessType* buffers[WORLD_SIZE];
    float* scale_buffers[WORLD_SIZE];
    #pragma unroll
    for (int i = 0; i < WORLD_SIZE; ++i)
    {
        // A mapping of the ranks to scatter reads as much as possible
        int rank = (params.local_rank + i) % WORLD_SIZE;
        buffers[i] = reinterpret_cast<AccessType*>(params.peer_comm_buffer_ptrs[rank]);
        scale_buffers[i] = params.fp8_scale_ptrs[rank];
    }

    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid);
    
    const int vec_nums = end - start;
    Accum acc;
    Vec vals[WORLD_SIZE];
    float scales[WORLD_SIZE];
    for (int idx = tid; idx < vec_nums; idx += stride) {
        const int offset = idx + start;
        const int scale_offset = (idx + start) / vec_nums_per_group;
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; i++)
        {
            AccessType* val_ptr = reinterpret_cast<AccessType*>(&vals[i]);
            val_ptr[0] = buffers[i][offset];
            scales[i] = scale_buffers[i][scale_offset];
        }
        acc.clear();
        #pragma unroll
        for (int rank = 0; rank < WORLD_SIZE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int i = (rank + WORLD_SIZE - params.local_rank) % WORLD_SIZE;
            Accum val_fp32 = convert_2_float(vals[i]);
            acc = fma(scales[i], val_fp32, acc);
        }
        float max_abs_val = amax(0.0, acc);
        unsigned int active_mask = __activemask();
        for (int mask = 8 / 2; mask >= 1; mask /= 2) {
            max_abs_val = fmaxf(max_abs_val, __shfl_xor_sync(active_mask, max_abs_val, mask));
        }
        float scale = fp8_max / fmaxf(max_abs_val, 1e-12);
        Vec val_fp8 = convert_2_fp8(acc * scale);
        local_shared_buffer[offset] = *reinterpret_cast<float4*>(val_fp8.data());
        if (threadIdx.x % 8 == 0) {
            local_shared_scale[scale_offset] = 1 / scale; 
        }
    }
    
    block_barrier(params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid, grid_size);

    for (int idx = tid; idx < largest_part; idx += stride) {
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; ++i)
        {
            const int rank = (params.local_rank + i) % WORLD_SIZE;
            const int start = rank * part;
            const int offset = start + idx;
            const int scale_offset = (offset * vec_size) / params.group_size;
            if (rank == WORLD_SIZE - 1 || idx < part) {
                Vec val;
                *reinterpret_cast<AccessType*>(&val) = buffers[i][offset];
                float scale = scale_buffers[i][scale_offset];
                Accum val_fp32 = convert_2_float(val) * scale;
                R result = convert_2_half(val_fp32);
                local_output_buffer[start * 2 + idx * 2] = *reinterpret_cast<float4*>(result.data());
                local_output_buffer[start * 2 + idx * 2 + 1] = *reinterpret_cast<float4*>(result.data() + 8);
            }
        }
    }
    
}


template <typename T, int WORLD_SIZE>
static __global__ void twoshot_allreduce_fp8_kernel_v2(AllReduceParams params)
{
    const int grid_size = gridDim.x;
    const int block_size = blockDim.x;
    const int stride = grid_size * block_size;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x + bid * block_size;
    // The number of elements packed into one for comms
    static constexpr int vec_size = 16 / sizeof(cutlass::float_e4m3_t);
    const int vec_nums_per_group = params.group_size / vec_size;
    using AccessType = cutlass::AlignedArray<cutlass::float_e4m3_t, vec_size>;
    using Vec = cutlass::Array<cutlass::float_e4m3_t, vec_size>;
    using Accum = cutlass::Array<float, vec_size>;
    using R = cutlass::Array<T, vec_size>;
    
    cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;

    float4* local_output_buffer = reinterpret_cast<float4*>(params.local_output_buffer_ptr);
    float4* local_shared_buffer = reinterpret_cast<float4*>(params.tmp_peer_comm_buffer_ptrs[params.local_rank]);
    float* local_shared_scale = params.fp8_scale_ptrs[params.local_rank];
    
    int total = params.elts_total / vec_size;
    int part = total / WORLD_SIZE;
    int largest_part = total - (WORLD_SIZE - 1) * part;
    int start = params.local_rank * part;
    int end = params.local_rank == WORLD_SIZE - 1 ? total : start + part;
    AccessType* buffers[WORLD_SIZE];
    float4* tmp_buffers[WORLD_SIZE];
    float* scale_buffers[WORLD_SIZE];
    #pragma unroll
    for (int i = 0; i < WORLD_SIZE; ++i)
    {
        // A mapping of the ranks to scatter reads as much as possible
        int rank = (params.local_rank + i) % WORLD_SIZE;
        buffers[i] = reinterpret_cast<AccessType*>(params.peer_comm_buffer_ptrs[rank]);
        tmp_buffers[i] = reinterpret_cast<float4*>(params.tmp_peer_comm_buffer_ptrs[rank]);
        scale_buffers[i] = params.fp8_scale_ptrs[rank];
    }

    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid);
    
    const int vec_nums = end - start;
    Accum acc;
    Vec vals[WORLD_SIZE];
    float scales[WORLD_SIZE];
    for (int idx = tid; idx < vec_nums; idx += stride) {
        const int offset = idx + start;
        const int scale_offset = (idx + start) / vec_nums_per_group;
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; i++)
        {
            AccessType* val_ptr = reinterpret_cast<AccessType*>(&vals[i]);
            val_ptr[0] = buffers[i][offset];
            scales[i] = scale_buffers[i][scale_offset];
        }
        acc.clear();
        #pragma unroll
        for (int rank = 0; rank < WORLD_SIZE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int i = (rank + WORLD_SIZE - params.local_rank) % WORLD_SIZE;
            Accum val_fp32 = convert_2_float(vals[i]);
            acc = fma(scales[i], val_fp32, acc);
        }
        R result = convert_2_half(acc);
        local_shared_buffer[idx * 2] = *reinterpret_cast<float4*>(result.data());
        local_shared_buffer[idx * 2 + 1] = *reinterpret_cast<float4*>(result.data() + 8);
        //local_output_buffer[idx * 2] = *reinterpret_cast<float4*>(result.data());
        //local_output_buffer[idx * 2 + 1] = *reinterpret_cast<float4*>(result.data() + 8);
    }
    
    block_barrier(params.peer_barrier_ptrs_out, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid, grid_size);

    for (int idx = tid; idx < largest_part * 2; idx += stride) {
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; ++i)
        {
            const int rank = (params.local_rank + i) % WORLD_SIZE;
            if (rank == WORLD_SIZE - 1 || idx < part * 2) {
                local_output_buffer[rank * part * 2 + idx] = tmp_buffers[i][idx];
            }
        }
    }
    
    
}



template <typename T, int WORLD_SIZE>
__global__ void oneshot_allreduce_kernel(AllReduceParams params)
{
    const int grid_size = gridDim.x;
    const int block_size = blockDim.x;
    const int stride = grid_size * block_size;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x + bid * block_size;

    // The number of elements packed into one for comms
    constexpr int vec_size = 16 / sizeof(T);
    using Accum = cutlass::Array<float, vec_size>;
    using AccessType = cutlass::AlignedArray<T, vec_size>;
    using Vec = cutlass::Array<T, vec_size>;
    cutlass::NumericArrayConverter<float, T, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;

    float4* local_output_buffer = reinterpret_cast<float4*>(params.local_output_buffer_ptr);    

    AccessType* buffers[WORLD_SIZE];
    #pragma unroll
    for (int ii = 0; ii < WORLD_SIZE; ++ii)
    {
        // buffers[0] is always the local buffers. Helps load balancing reads.
        int rank = (params.local_rank + ii) % WORLD_SIZE;
        buffers[ii] = reinterpret_cast<AccessType*>(params.peer_comm_buffer_ptrs[rank]);
    }

    
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid);
    
    // Each block accumulates the values from the different GPUs on the same node.
    const int vec_nums = params.elts_total / vec_size;
    Accum acc;
    Vec vals[WORLD_SIZE];
    for (int idx = tid; idx < vec_nums; idx += stride) {
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; i++) {
            AccessType* val_ptr = reinterpret_cast<AccessType*>(&vals[i]);
            val_ptr[0] = buffers[i][idx];
        }

        acc.clear();

        #pragma unroll
        for (int rank = 0; rank < WORLD_SIZE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int i = (rank + WORLD_SIZE - params.local_rank) % WORLD_SIZE;
            Accum val_fp32 = convert_2_float(vals[i]);
            acc = acc + val_fp32;
        }
        // Store to the destination buffer.
        Vec result = convert_2_half(acc);
        local_output_buffer[idx] = *reinterpret_cast<float4*>(result.data());
    }
}


template <typename T>
__global__ void allreduce_fp8_quantize(T* input, cutlass::float_e4m3_t* out, float* scales) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    constexpr int vec_size = 16 / sizeof(T);
    constexpr int threads_per_group = 128 / vec_size;
    using AccessType = cutlass::AlignedArray<T, vec_size>;
    using V = cutlass::Array<T, vec_size>;
    using F = cutlass::Array<float, vec_size>;
    using R = cutlass::Array<cutlass::float_e4m3_t, vec_size>;
    
    float fp8_max = std::numeric_limits<cutlass::float_e4m3_t>::max();
    cutlass::NumericArrayConverter<float, T, vec_size> convert_2_float;
    cutlass::multiplies<F> mul;
    cutlass::maximum_absolute_value_reduction<F, true> amax;
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, vec_size> convert_2_fp8;
     
    int offset = bid * 1024;
    const AccessType* input_ptr = reinterpret_cast<const AccessType*>(input + offset);
    float2* out_ptr = reinterpret_cast<float2*>(out + offset);
    V val;
    *reinterpret_cast<AccessType*>(&val) = input_ptr[tid];
    F val_f = convert_2_float(val);
    float max_abs_val = amax(0.0, val_f);
    for (int mask = 16 / 2; mask >= 1; mask /= 2) {
      max_abs_val = fmaxf(max_abs_val, __shfl_xor_sync(uint32_t(-1), max_abs_val, mask));
    }
    float scale = fp8_max / fmaxf(max_abs_val, 1e-12);
    R q_val = convert_2_fp8(mul(scale, val_f));
    scale = 1 / scale;
    out_ptr[tid] = *reinterpret_cast<float2*>(q_val.data());
    if (tid % 16 == 0) {
        offset = (offset + tid * 8) / 128;
        scales[offset] = scale;
    }
}

template <typename T, int WORLD_SIZE>
__global__ void oneshot_allreduce_fp8_kernel(AllReduceParams params)
{
    const int grid_size = gridDim.x;
    const int block_size = blockDim.x;
    const int stride = grid_size * block_size;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x + bid * block_size;

    // The number of elements packed into one for comms
    constexpr int vec_size = 16 / sizeof(cutlass::float_e4m3_t);
    const int vec_nums_per_group = params.group_size / vec_size;
    using Accum = cutlass::Array<float, vec_size>;
    using AccessType = cutlass::AlignedArray<cutlass::float_e4m3_t, vec_size>;
    using Vec = cutlass::Array<cutlass::float_e4m3_t, vec_size>;
    using OutType = cutlass::Array<T, 16>;
    cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, vec_size> convert_2_float;
    cutlass::NumericArrayConverter<T, float, vec_size> convert_2_half;
    
    float4* local_output_buffer = reinterpret_cast<float4*>(params.local_output_buffer_ptr);


    AccessType* buffers[WORLD_SIZE];
    float* scale_buffers[WORLD_SIZE];
    #pragma unroll
    for (int ii = 0; ii < WORLD_SIZE; ++ii)
    {
        // buffers[0] is always the local buffers. Helps load balancing reads.
        int rank = (params.local_rank + ii) % WORLD_SIZE;
        buffers[ii] = reinterpret_cast<AccessType*>(params.peer_comm_buffer_ptrs[rank]);
        scale_buffers[ii] = params.fp8_scale_ptrs[rank];
    }
    
    // In the non-copy case, we assume that once the kernel has been started, data is ready to be consumed
    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, WORLD_SIZE, threadIdx.x, bid);
    
    // Each block accumulates the values from the different GPUs on the same node.
    
    Accum acc;
    Vec vals[WORLD_SIZE];
    float scale[WORLD_SIZE];
    const int vec_nums = params.elts_total / vec_size;
    for (int idx = tid; idx < vec_nums; idx += stride) {
        #pragma unroll
        for (int i = 0; i < WORLD_SIZE; i++) {
            AccessType* val_ptr = reinterpret_cast<AccessType*>(&vals[i]);
            val_ptr[0] = buffers[i][idx];
            scale[i] = scale_buffers[i][idx / vec_nums_per_group];
        }

        acc.clear();

        #pragma unroll
        for (int rank = 0; rank < WORLD_SIZE; ++rank)
        {
            // Always reduce from rank 0 to ensure stable reduce order.
            int i = (rank + WORLD_SIZE - params.local_rank) % WORLD_SIZE;
            Accum val_fp32 = convert_2_float(vals[i]);
            acc = fma(scale[i], val_fp32, acc);
        }
        // Store to the destination buffer.
        OutType result = convert_2_half(acc);
        local_output_buffer[idx * 2] = *reinterpret_cast<float4*>(result.data());
        local_output_buffer[idx * 2 + 1] = *reinterpret_cast<float4*>(result.data() + 8);
    }
}


template <typename T>
void run(int batch_size, int hidden_size, int world_size, int rank, T** buffer_ipc_ptrs, T** tmp_buffer_ipc_ptrs, float** scale_ipc_ptrs, uint32_t** barrier_in_ipc_ptrs, uint32_t** barrier_out_ipc_ptrs, ncclComm_t& comm, bool fp8) {
    int warmup = 10;
    int iter = 1000;
    int head_size = 128;
    int num_heads = hidden_size / head_size;
    int message_size = batch_size * hidden_size;
    T* h_in = (T*)malloc(message_size * sizeof(T));
    std::mt19937 gen(20250102 + rank);
    std::uniform_real_distribution<float> dis(static_cast<float>(-100), static_cast<float>(100));
    for (int i = 0; i < message_size; i++) {
        //h_in[i] = T(1.0 + rank);
        h_in[i] = T(dis(gen));
    }
    T* d_in;
    T* nccl_d_in;
    cudaMalloc(&d_in, message_size * sizeof(T));
    cudaMalloc(&nccl_d_in, message_size * sizeof(T));
    cudaMemcpy(d_in, h_in, sizeof(T) * message_size, cudaMemcpyHostToDevice);
    cudaMemcpy(nccl_d_in, h_in, sizeof(T) * message_size, cudaMemcpyHostToDevice);

    T* h_out = (T*)malloc(message_size * sizeof(T));
    T* fp8_h_out = (T*)malloc(message_size * sizeof(T));
    T* twoshot_h_out = (T*)malloc(message_size * sizeof(T));
    T* twoshot_fp8_h_out = (T*)malloc(message_size * sizeof(T));
    T* nccl_h_out = (T*)malloc(message_size * sizeof(T));
    T* d_out;
    cudaMalloc(&d_out, message_size * sizeof(T));
    T* twoshot_d_out;
    cudaMalloc(&twoshot_d_out, message_size * sizeof(T));
    T* nccl_d_out;
    cudaMalloc(&nccl_d_out, message_size * sizeof(T));
    T* fp8_d_out;
    cudaMalloc(&fp8_d_out, message_size * sizeof(T));
    T* twoshot_fp8_d_out;
    cudaMalloc(&twoshot_fp8_d_out, message_size * sizeof(T));

    ncclDataType_t nccl_dtype;
    if (std::is_same<T, cutlass::half_t>::value) {
        nccl_dtype = ncclFloat16;
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
        nccl_dtype = ncclBfloat16;
    } else {
        nccl_dtype = ncclFloat;
    }
    
    bool oneshot = false;
    int message_size_in_bytes = message_size * sizeof(T);
    if (world_size <= 2) {
        oneshot = true;
    } else if (world_size <= 4) {
        if (message_size_in_bytes < 512 * 1024) {
            oneshot = true;
        }
    } else {
        if (message_size_in_bytes < 256 * 1024) {
            oneshot = true;
        }
    }

    AllReduceParams params;
    
    params.barrier_flag = 0;
    params.ranks_per_node = world_size;
    params.local_rank = rank;
    params.elts_total = message_size;
    params.group_size = 128;
    
    for (int i = 0; i < world_size; i++) {
        params.peer_comm_buffer_ptrs[i] = buffer_ipc_ptrs[i];
        params.tmp_peer_comm_buffer_ptrs[i] = tmp_buffer_ipc_ptrs[i];
        params.peer_barrier_ptrs_in[i] = barrier_in_ipc_ptrs[i];
        params.peer_barrier_ptrs_out[i] = barrier_out_ipc_ptrs[i];
        params.fp8_scale_ptrs[i] = scale_ipc_ptrs[i];
    }

    
    int blocks_per_grid = 1; 
    int threads_per_block = 512;
    int total_threads = message_size / int(16 / sizeof(T));
    threads_per_block = std::min(threads_per_block, total_threads);
    blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = std::min(MAX_ALL_REDUCE_BLOCKS, blocks_per_grid);
    /*
    if (rank == 0) {
        printf("blocks_per_grid %d, fp8_blocks_per_grid %d\n", blocks_per_grid, fp8_blocks_per_grid);
    }
    */
    dim3 grid(message_size / 1024);
    dim3 block(128);
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    
    // one shot
    params.local_output_buffer_ptr = d_out;
    for (int i = 0; i < warmup; i++) {
        params.barrier_flag += 1;
        cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], d_in, params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        switch (world_size) {
            case 2: oneshot_allreduce_kernel<T, 2><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 4: oneshot_allreduce_kernel<T, 4><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 8: oneshot_allreduce_kernel<T, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    nvtxRangePush("oneshot");
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        params.barrier_flag += 1;
        cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], d_in, params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        switch (world_size) {
            case 2: oneshot_allreduce_kernel<T, 2><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 4: oneshot_allreduce_kernel<T, 4><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 8: oneshot_allreduce_kernel<T, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    time /= iter;
    nvtxRangePop();
    
    // one-shot-fp8
    int fp8_blocks_per_grid = 1;
    int fp8_threads_per_block = 256;
    int fp8_total_threads = message_size / 16;
    fp8_threads_per_block = std::min(fp8_threads_per_block, fp8_total_threads);
    fp8_blocks_per_grid = (fp8_total_threads + fp8_threads_per_block - 1) / fp8_threads_per_block;
    fp8_blocks_per_grid = std::min(48, fp8_blocks_per_grid);
    params.local_output_buffer_ptr = fp8_d_out;
    for (int i = 0; i < warmup; i++) {
        params.barrier_flag += 1;
        cutlass::float_e4m3_t* out = reinterpret_cast<cutlass::float_e4m3_t*>(params.peer_comm_buffer_ptrs[params.local_rank]);
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, out, params.fp8_scale_ptrs[params.local_rank]);
        switch (world_size) {
            case 2: oneshot_allreduce_fp8_kernel<T, 2><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 4: oneshot_allreduce_fp8_kernel<T, 4><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 8: oneshot_allreduce_fp8_kernel<T, 8><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    nvtxRangePush("oneshot-fp8");
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        params.barrier_flag += 1;
        cutlass::float_e4m3_t* out = reinterpret_cast<cutlass::float_e4m3_t*>(params.peer_comm_buffer_ptrs[params.local_rank]);
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, out, params.fp8_scale_ptrs[params.local_rank]);
        switch (world_size) {
            case 2: oneshot_allreduce_fp8_kernel<T, 2><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 4: oneshot_allreduce_fp8_kernel<T, 4><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 8: oneshot_allreduce_fp8_kernel<T, 8><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float oneshot_fp8_time;
    cudaEventElapsedTime(&oneshot_fp8_time, begin, end);
    oneshot_fp8_time /= iter;
    nvtxRangePop();

    // two shot
    blocks_per_grid = 1; 
    threads_per_block = 512;
    total_threads = message_size / world_size / int(16 / sizeof(T));
    threads_per_block = std::min(threads_per_block, total_threads);
    blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = std::min(MAX_ALL_REDUCE_BLOCKS, blocks_per_grid);
    params.local_output_buffer_ptr = twoshot_d_out;
    for (int i = 0; i < warmup; i++) {
        params.barrier_flag += 1;
        cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], d_in, params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        switch (world_size) {
            case 2: twoshot_allreduce_kernel<T, 2><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 4: twoshot_allreduce_kernel<T, 4><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 8: twoshot_allreduce_kernel<T, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            default: break;
        }      
    }
    nvtxRangePush("twoshot");
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        params.barrier_flag += 1;
        cudaMemcpyAsync(params.peer_comm_buffer_ptrs[params.local_rank], d_in, params.elts_total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        switch (world_size) {
            case 2: twoshot_allreduce_kernel<T, 2><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 4: twoshot_allreduce_kernel<T, 4><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            case 8: twoshot_allreduce_kernel<T, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float twoshot_time;
    cudaEventElapsedTime(&twoshot_time, begin, end);
    twoshot_time /= iter;
    nvtxRangePop();
    
    // twoshot-fp8
    fp8_blocks_per_grid = 1;
    fp8_threads_per_block = 512;
    fp8_total_threads = message_size / world_size / 16;
    fp8_threads_per_block = std::min(fp8_threads_per_block, fp8_total_threads);
    fp8_blocks_per_grid = (fp8_total_threads + fp8_threads_per_block - 1) / fp8_threads_per_block;
    fp8_blocks_per_grid = std::min(48, fp8_blocks_per_grid);
    params.local_output_buffer_ptr = twoshot_fp8_d_out;
    for (int i = 0; i < warmup; i++) {
        params.barrier_flag += 1;
        cutlass::float_e4m3_t* out = reinterpret_cast<cutlass::float_e4m3_t*>(params.peer_comm_buffer_ptrs[params.local_rank]);
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, out, params.fp8_scale_ptrs[params.local_rank]);
        switch (world_size) {
            case 2: twoshot_allreduce_fp8_kernel_v2<T, 2><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 4: twoshot_allreduce_fp8_kernel_v2<T, 4><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 8: twoshot_allreduce_fp8_kernel_v2<T, 8><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    nvtxRangePush("twoshot-fp8");
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        params.barrier_flag += 1;
        cutlass::float_e4m3_t* out = reinterpret_cast<cutlass::float_e4m3_t*>(params.peer_comm_buffer_ptrs[params.local_rank]);
        allreduce_fp8_quantize<T><<<grid, block, 0, stream>>>(d_in, out, params.fp8_scale_ptrs[params.local_rank]);
        switch (world_size) {
            case 2: twoshot_allreduce_fp8_kernel_v2<T, 2><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 4: twoshot_allreduce_fp8_kernel_v2<T, 4><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            case 8: twoshot_allreduce_fp8_kernel_v2<T, 8><<<fp8_blocks_per_grid, fp8_threads_per_block, 0, stream>>>(params); break;
            default: break;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float twoshot_fp8_time;
    cudaEventElapsedTime(&twoshot_fp8_time, begin, end);
    twoshot_fp8_time /= iter;
    nvtxRangePop();
    
    // nccl
    for (int i = 0; i < warmup; i++) {
        NCCLCHECK(ncclAllReduce(nccl_d_in, nccl_d_out, message_size, nccl_dtype, ncclSum, comm, stream));
    }
    nvtxRangePush("nccl");
    cudaEventRecord(begin, stream);
    for (int i = 0; i < iter; i++) {
        NCCLCHECK(ncclAllReduce(nccl_d_in, nccl_d_out, message_size, nccl_dtype, ncclSum, comm, stream));
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float nccl_time;
    cudaEventElapsedTime(&nccl_time, begin, end);
    nccl_time /= iter;
    nvtxRangePop();
    // check result
    cudaMemcpyAsync(h_out, d_out, sizeof(T) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(fp8_h_out, fp8_d_out, sizeof(T) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(twoshot_h_out, twoshot_d_out, sizeof(T) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(twoshot_fp8_h_out, twoshot_fp8_d_out, sizeof(T) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(nccl_h_out, nccl_d_out, sizeof(T) * message_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (rank == 0) {
        
        printf("batch size %d, hidden_size %d, one shot time is %f, one shot fp8 is %f, two shot time is %f, two shot fp8 is %f, nccl is %f\n", batch_size, hidden_size, time * 1000, oneshot_fp8_time * 1000, twoshot_time * 1000, twoshot_fp8_time * 1000, nccl_time * 1000);
        compare<T>(rank, h_out, nccl_h_out, message_size);
        compare<T>(rank, fp8_h_out, nccl_h_out, message_size);
        compare<T>(rank, twoshot_h_out, nccl_h_out, message_size);
        compare<T>(rank, twoshot_fp8_h_out, nccl_h_out, message_size);
        printf("--------------------------------------------------------------\n");
    }

    cudaFree(d_in);
    cudaFree(nccl_d_in);
    cudaFree(d_out);
    cudaFree(fp8_d_out);
    cudaFree(twoshot_d_out);
    cudaFree(twoshot_fp8_d_out);
    cudaFree(nccl_d_out);
    cudaStreamDestroy(stream);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
}

int main(int argc, char** argv) {
    using T = cutlass::half_t;
    int world_size, rank;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (rank == 0) {
        printf("world_size is %d, rank %d\n", world_size, rank);
    }
    float fp8_max = std::numeric_limits<cutlass::float_e4m3_t>::max();
    printf("fp8 max is %f\n", fp8_max);
    using Vec = cutlass::Array<T, 8>;
    int size = sizeof(Vec);
    int alig_size = alignof(Vec);
    printf("size of array is %d, alig size is %d\n", size, alig_size);
    CUDACHECK(cudaSetDevice(rank));
    // init nccl
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclCommInitRank(&comm, world_size, id, rank);
    // input ipc buffer
    T* buffer;
    T* tmp_buffer;
    int buffer_size_in_bytes = 2 * 8192 * 1024;
    CUDACHECK(cudaMalloc(&buffer, buffer_size_in_bytes));
    CUDACHECK(cudaMalloc(&tmp_buffer, buffer_size_in_bytes));
    CUDACHECK(cudaMemset(buffer, 0, buffer_size_in_bytes));
    CUDACHECK(cudaMemset(tmp_buffer, 0, buffer_size_in_bytes));
    cudaIpcMemHandle_t buffer_handle;
    cudaIpcMemHandle_t buffer_handles[world_size];
    cudaIpcMemHandle_t tmp_buffer_handle;
    cudaIpcMemHandle_t tmp_buffer_handles[world_size];
    CUDACHECK(cudaIpcGetMemHandle(&buffer_handle, buffer));
    MPI_Allgather(&buffer_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, buffer_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    CUDACHECK(cudaIpcGetMemHandle(&tmp_buffer_handle, tmp_buffer));
    MPI_Allgather(&tmp_buffer_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, tmp_buffer_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    T* buffer_ipc_ptrs[world_size];
    T* tmp_buffer_ipc_ptrs[world_size];
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            buffer_ipc_ptrs[i] = buffer;
            tmp_buffer_ipc_ptrs[i] = tmp_buffer;
        } else {
            CUDACHECK(cudaIpcOpenMemHandle((void**)&buffer_ipc_ptrs[i], buffer_handles[i], cudaIpcMemLazyEnablePeerAccess));
            CUDACHECK(cudaIpcOpenMemHandle((void**)&tmp_buffer_ipc_ptrs[i], tmp_buffer_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }
    // flag ipc
    int flags_size = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * world_size * 2;
    int align_flags_size = alignSize(flags_size, 1LU << 21);
    uint32_t* barrier_ptrs_in;
    uint32_t* barrier_ptrs_out;
    CUDACHECK(cudaMalloc(&barrier_ptrs_in, align_flags_size));
    CUDACHECK(cudaMalloc(&barrier_ptrs_out, align_flags_size));
    CUDACHECK(cudaMemset(barrier_ptrs_in, 0, align_flags_size));
    CUDACHECK(cudaMemset(barrier_ptrs_out, 0, align_flags_size));
    cudaIpcMemHandle_t barrier_in_handle;
    cudaIpcMemHandle_t barrier_out_handle;
    CUDACHECK(cudaIpcGetMemHandle(&barrier_in_handle, barrier_ptrs_in));
    CUDACHECK(cudaIpcGetMemHandle(&barrier_out_handle, barrier_ptrs_out));
    cudaIpcMemHandle_t barrier_in_handles[world_size];
    cudaIpcMemHandle_t barrier_out_handles[world_size];
    MPI_Allgather(&barrier_in_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, barrier_in_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&barrier_out_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, barrier_out_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    uint32_t* barrier_in_ipc_ptrs[world_size];
    uint32_t* barrier_out_ipc_ptrs[world_size];
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            barrier_in_ipc_ptrs[i] = barrier_ptrs_in;
            barrier_out_ipc_ptrs[i] = barrier_ptrs_out;
        } else {
            cudaIpcOpenMemHandle((void**)&barrier_in_ipc_ptrs[i], barrier_in_handles[i], cudaIpcMemLazyEnablePeerAccess);
            cudaIpcOpenMemHandle((void**)&barrier_out_ipc_ptrs[i], barrier_out_handles[i], cudaIpcMemLazyEnablePeerAccess);
        }
    }
    // scales buffer
    float* scales;
    int scales_buffer_size = max(1024 * 8192 / 128 * 4, 2 * 1024 * 1024);
    cudaMalloc(&scales, scales_buffer_size);
    cudaMemset(scales, 0, scales_buffer_size);
    cudaIpcMemHandle_t scale_handle;
    cudaIpcMemHandle_t scale_handles[world_size];
    cudaIpcGetMemHandle(&scale_handle, scales);
    MPI_Allgather(&scale_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, scale_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    float* scale_ipc_ptrs[world_size];
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            scale_ipc_ptrs[i] = scales;
        } else {
            CUDACHECK(cudaIpcOpenMemHandle((void**)&scale_ipc_ptrs[i], scale_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }

    /*
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
             cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    */
    MPI_Barrier(MPI_COMM_WORLD);
    int hidden_size = 14336;
    std::vector<int> batch_size{128};
    for (int bs : batch_size) {
        run<T>(bs, hidden_size, world_size, rank, buffer_ipc_ptrs, tmp_buffer_ipc_ptrs, scale_ipc_ptrs, barrier_in_ipc_ptrs, barrier_out_ipc_ptrs, comm, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cudaFree(buffer);
    cudaFree(scales);
    cudaFree(barrier_ptrs_in);
    cudaFree(barrier_ptrs_out);
    MPI_Finalize();
}



