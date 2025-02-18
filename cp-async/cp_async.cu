#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>



template<typename Config>
__global__ void test_cp_async(void* in, void* out, int m, int k) {

    using namespace cute;
    using T = typename Config::T;
    using G2SCopyA = typename Config::G2SCopyA;
    using SmemLayoutA = typename Config::SmemLayoutA;
    extern __shared__ T smem[];
    T *Ashm = smem;
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    Tensor A = make_tensor(make_gmem_ptr((T *)in), make_shape(m, k),
                         make_stride(k, Int<1>{}));  // (M, K)

    Tensor gA = local_tile(A, make_tile(Int<128>{}, Int<128>{}),
                         make_coord(iy, ix));  // (kTileM, kTileK)

    Tensor sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{});  // (kTileM, kTileK, kStage)

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);

    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)

    auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)
    

    if (threadIdx.x == 0) {
        print("tAgA_copy: ");
        print("tAsA_copy: ");
    }
    
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _),
               tAsA_copy(_, _, _));
    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();
    
    T* out_ptr = reinterpret_cast<T*>(out);
    for (int i = 0; i < 128; i++) {
        out_ptr[idx * 128 + i] = smem[idx * 128 + i];
    }
}

namespace config {

using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
          int kStage_ = 5, int kSmemLayoutCBatch_ = 2,
          typename ComputeType = T_>
struct GemmConfig {
  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                  make_stride(Int<kTileK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{})));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // shared memory to register copy
  
};

}  // namespace config


int main() {
    constexpr int M = 256;
    constexpr int K = 64;

    
    using namespace cute;
    using T = cute::half_t;
    T* h_a = (T*)malloc(M * K * sizeof(T));

    for (int i = 0; i < M * K; i++) {
        h_a[i] = T(i);
    }

    using g2s_copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;
    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{}))));
    G2SCopyA g2s_copy;
    print(g2s_copy);
    //Layout mn = make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
    //print_layout(mn);

    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;

    //print(typename decltype(gemm_config)::MMA{});

    T* d_a = nullptr;
    cudaMalloc(&d_a, M * K * sizeof(T));
    cudaMemcpy(d_a, h_a, sizeof(T) * M * K, cudaMemcpyHostToDevice);


    T* d_o_cp_async = nullptr;
    cudaMalloc(&d_o_cp_async, M * K * sizeof(T));

    dim3 grid(2,2);
    dim3 block = 128;
    test_cp_async<decltype(gemm_config)><<<grid, block, 128 * 128 * 2>>>(d_a, d_o_cp_async, M, K);

    auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    
    T* h_o_cp_sync = (T*)malloc(M * K * sizeof(T));
    cudaMemcpy(h_o_cp_sync, d_o_cp_async, sizeof(T) * M * K, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < M * K; i++) {
        int a = int(h_a[i]);
        int o = int(h_o_cp_sync[i]);
        if (a != o || a != i || o != i) {
            printf("error: %d %d %d\n", i, a, o);
            break;
        } 
    }
    
}
