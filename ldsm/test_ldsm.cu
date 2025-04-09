#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <typename T>
__global__ void ldsm_kernel(const T* in, T* out) {
    using namespace cute;
    const int tid = threadIdx.x;

    const int nums_per_thread = 16 * 16 / 32;  
    __shared__ T smem[16 * 16];

    uint32_t r[4];

    float4* smem_ptr = reinterpret_cast<float4*>(smem);
    smem_ptr[tid] = *reinterpret_cast<const float4*>(in + tid * nums_per_thread);
    __syncthreads();

    // load smem -> rmem using LDSM
    int i = tid % 16;
    int j = tid / 16;
    int matrix_idx = i * 16 + j * 8;
    uint128_t* smem_src = reinterpret_cast<uint128_t*>(smem + matrix_idx);

    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        :  "r"(smem_int_ptr));
    
    
    *reinterpret_cast<float4*>(out + tid * nums_per_thread) = *reinterpret_cast<float4*>(r);
    

    
}


template <typename T, class SmemLayout>
__global__ void ldsm_kernel_swizzle(const T* in, T* out, SmemLayout smem_layout) {
    using namespace cute;
    const int tid = threadIdx.x;

    const int nums_per_thread = 16 * 16 / 32;  
    __shared__ T smem[16 * 16];

    uint32_t r[4];

    float4* smem_ptr = reinterpret_cast<float4*>(smem);
    int row = tid / 2;
    int col = (tid % 2) * nums_per_thread;
    int index = smem_layout(row,col);
    //printf("tid %d, index %d \n", tid, index);
    smem_ptr[tid] = *reinterpret_cast<const float4*>(in + index);
    __syncthreads();

    // load smem -> rmem using LDSM
    int i = tid % 16;
    int j = tid / 16;
    int old_matrix_idx = i * 16 + j * 8;
    row = old_matrix_idx / 64;
    col = (old_matrix_idx / 8) % 8;
    int offset = row ^ col;
    int matrix_idx = old_matrix_idx + (offset - col) * 8;
    printf("tdi %d, old_matrix_idx is %d, offset is %d, col is %d, matrix_idx is %d \n", tid, old_matrix_idx, offset, col, matrix_idx);
    uint128_t* smem_src = reinterpret_cast<uint128_t*>(smem + matrix_idx);

    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        :  "r"(smem_int_ptr));
    
    
    *reinterpret_cast<float4*>(out + tid * nums_per_thread) = *reinterpret_cast<float4*>(r);
        
}

// use cute copy

template <typename T>
__global__ void ldsm_kernel_cute(const T* in, T* out) {
    using namespace cute;
    const int tid = threadIdx.x;

    const int nums_per_thread = 16 * 16 / 32;  
    __shared__ T smem[16 * 16];

    uint32_t r[4];

    float4* smem_ptr = reinterpret_cast<float4*>(smem);
    smem_ptr[tid] = *reinterpret_cast<const float4*>(in + tid * nums_per_thread);
    __syncthreads();

    // load smem -> rmem using LDSM
    int i = tid % 16;
    int j = tid / 16;
    int matrix_idx = i * 16 + j * 8;
    uint128_t* smem_src = reinterpret_cast<uint128_t*>(smem + matrix_idx);

    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_src);
    
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        :  "r"(smem_int_ptr));
    
    
    *reinterpret_cast<float4*>(out + tid * nums_per_thread) = *reinterpret_cast<float4*>(r);
    

    
}


template <class TiledCopy, class SmemLayout>
__global__ void
ldsm_test_device_cute(uint16_t* g_in, uint16_t* g_out,
                      TiledCopy tiled_copy, SmemLayout smem_layout)
{
  using namespace cute;

  __shared__ uint16_t smem[size(smem_layout)];

  auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
  
  auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
  auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

  int tid = threadIdx.x;

  // Load input gmem -> smem
  for (int i = tid; i < size(t_smem); i += size(tiled_copy)) {
    t_smem(i) = t_g_in(i);
  }

  __syncthreads();

  if (thread0()) {
    //print("t_smem: "); print(t_smem); print("\n");
    //print_tensor(t_smem);
    for (int i = 0; i < size(t_smem); i++) {
        float t = static_cast<float>(t_smem.data()[i]);
        printf("i %d is %f\n", i, t);
    }
    print("\n");
  }

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  auto tXsX = thr_copy.partition_S(t_smem);   // (V,M,N)
  auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)
  if (thread0()) {
    print("tXsX: "); print(tXsX); print("\n");
    print_tensor(tXsX);
    print("tXgX: "); print(tXgX); print("\n");
  }

  auto tXrX = make_tensor<uint16_t>(shape(tXgX)); // (V,M,N)
  clear(tXrX);  // Just to make sure
  copy(tiled_copy, tXsX, tXrX);

  if (thread0()) {
    print("tXrX: "); print(tXrX); print("\n");
    print_tensor(tXrX);
  }

/*
  if (thread0()) {
    print("tXsX: " ); print(tXsX.layout()); print("\n");
    print("tXgX: " ); print(tXgX.layout()); print("\n");
    print("tXrX: " ); print(tXrX.layout()); print("\n");
  }
*/


}

int main() {

    using T = cute::half_t;
    using namespace cute;

    constexpr int count = 32 * 32;

    thrust::host_vector<uint16_t> h_in(count);
    for (int i = 0; i < count; ++i) {
        h_in[i] = uint16_t(i);
    }
    thrust::device_vector<uint16_t> d_in = h_in;
    thrust::device_vector<uint16_t> d_out(count);


    //auto smem_layout = Layout<Shape <_32, _32>,
    //                       Stride< _32, _1>>{};

    auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};

    print_layout(smem_layout);
    
    auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

    ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);

    return 0;

}