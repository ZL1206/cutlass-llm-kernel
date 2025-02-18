#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

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




int main() {

    using T = cute::half_t;
    using namespace cute;

    const int m = 16;
    const int k = 16;
    int warp_size = 32;
    int nums_per_thread = 16 * 16 / 32;
    
    T* h_A = (T*)malloc(m * k * sizeof(T));

    for (int i = 0; i < m*k; i++) {
        h_A[i] = i;
        //printf("%f \n", float(h_A[i]));
    }

    T* d_A = nullptr;
    cudaMalloc(&d_A, sizeof(T) * m * k);

    cudaMemcpy(d_A, h_A, sizeof(T) * m * k, cudaMemcpyHostToDevice);

    T* d_A_R;
    cudaMalloc(&d_A_R, sizeof(T) * warp_size * nums_per_thread);
    T* h_A_R = (T*)malloc(warp_size * nums_per_thread * sizeof(T));


    ldsm_kernel<T><<<1, 32>>>(d_A, d_A_R);

    cudaMemcpy(h_A_R, d_A_R, sizeof(T) * m * k, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; i++) {
        printf("thread %d: ", i);
        for (int j = 0; j < 8; j++) {
            float tmp = float(h_A_R[i * nums_per_thread + j]);
            printf("%f ", tmp);
        }
        printf("\n");
    }

    cudaDeviceSynchronize();

    using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<16>{}, Int<16>{}),
                  make_stride(Int<16>{}, Int<1>{}))));

    SmemLayoutAtom smem;

    Layout A = make_layout(make_shape(Int<16>{}, Int<16>{}),
                  make_stride(Int<16>{}, Int<1>{}));
    print_layout(A);

    using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<16>{}, Int<16>{})));
    print_layout(SmemLayoutA{});

    ldsm_kernel_swizzle<<<1, 32>>>(d_A, d_A_R, SmemLayoutA{});
    
    cudaMemcpy(h_A_R, d_A_R, sizeof(T) * m * k, cudaMemcpyDeviceToHost);
    
    printf("swizzle result \n");
    
    for (int i = 0; i < 32; i++) {
        printf("thread %d: ", i);
        for (int j = 0; j < 8; j++) {
            float tmp = float(h_A_R[i * nums_per_thread + j]);
            printf("%f ", tmp);
        }
        printf("\n");
    }

    return 0;

}