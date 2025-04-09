#include <iostream>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <random>
#include <numeric>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/trace.h>


using namespace cute;

template <typename Kernel_traits>
__global__ void g2sCopy(void* q) {
    using T = typename Kernel_traits::T;
    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;

    using GmemTiledCopyQKV = typename Kernel_traits::GmemTiledCopyQKV;
    
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;

    extern __shared__ char smem[];

    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(q)),
                    make_shape(Int<2>{}, Int<64>{}),
                    make_stride(Int<64>{}, Int<1>{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                            SmemLayoutQ{});
    
    const int idx = threadIdx.x;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    if (thread(64)) {
        print("tensor tQgQ: \n"); print_tensor(tQgQ); print("\n");
        print("cQ: \n");
        print_tensor(cQ);
        print("tQcQ: \n");
        print_tensor(tQcQ);
    }

    __syncthreads();

    if (thread0()){

        print("sQ: \n"); print(sQ); print("\n");
        print("tQgQ: \n"); print(tQgQ); print("\n");
        print("tQsQ: \n"); print(tQsQ); print("\n");
    }
    
    // global to shared memory
    for (int m = 0; m < size<1>(tQgQ); m++) {
        if (get<0>(tQcQ(0, m, 0)) < 2) {
            for (int k = 0; k < size<2>(tQgQ); k++) {
                copy(gmem_tiled_copy_QKV, tQgQ(_, m, k), tQsQ(_, m, k));
            }
        }
    }

    

    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();

    if (thread0()) {

        printf("Q -----------------------------------------------------------------------------------------------------\n");
        print("tensor sQ: \n"); print_tensor(sQ); print("\n");
    }
}



template <typename T_, int kTileM_ = 128, int kTileN_ = 32, int kTileK_ = 128>
struct Kernel_traits {

  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3; 

  // global to shared memory
  using GmemLayoutAtom = Layout<Shape <Int<16>, Int<8>>,
                                  Stride<Int<8>, _1>>;
    
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));

  // shared memory layout
  /*
  using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  */
  
  using SmemLayoutQ = Layout<Shape<_2, Int<64>>,
                           Stride<Int<64>, _1>>;
  
  
  
  // tiled mma
  using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<4>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * 4>, _16, _16>>;
  
  static constexpr int kSmemKVSize = size(SmemLayoutQ{}) * 2 * sizeof(T);
  static constexpr int kSmemSize = kSmemKVSize;
};


int main(void) {
    using namespace cute;
    using T = cute::half_t;
    std::mt19937 gen(20250102);
    std::uniform_real_distribution<float> dis(static_cast<float>(-100), static_cast<float>(100));
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 128;

    // q
    T* h_q = (T*)malloc(2 * 64 * sizeof(T));
    for (int i = 0; i < 2 * 64; i++) {
        float data = dis(gen);
        h_q[i] = T(data);
        printf("%10.6f", data); printf(" ");
        if ((i + 1) % 8 == 0 && (i+1) % 64 != 0) {
            printf("|"); printf(" ");
        }
        if ((i+1) % 64 == 0) {
            printf("\n");
        }
    }
    T* d_q = nullptr;
    cudaMalloc(&d_q, 2 * 64 * sizeof(T));
    cudaMemcpy(d_q, h_q, sizeof(T) * 2 * 64, cudaMemcpyHostToDevice);
    printf("-------------------------------------------------------------------\n");



    Kernel_traits<T, M, N, K> config;

    


    auto kernel = &g2sCopy<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, 128, smem_size>>>(d_q);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    if (d_q) {
        cudaFree(d_q);
    }
    
    return 0;
}