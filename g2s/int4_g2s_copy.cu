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
#include <cstdint>


using namespace cute;

template <typename Kernel_traits>
__global__ void g2sCopy(void* k) {
    
    using T = typename Kernel_traits::T;
    
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;

    using GmemTiledCopyKV = typename Kernel_traits::GmemTiledCopyKV;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    

    extern __shared__ char smem[];

    
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(k)),
                    make_shape(Int<N>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                            SmemLayoutKV{});
    
    const int idx = threadIdx.x;

    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(idx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);


    if (thread0()) {
        print("tSsK: "); print(tSsK); print("\n");
        print("tSrK: "); print(tSrK); print("\n");
    }

    if (thread0()) {
        print("tensor tKgK: \n"); print_tensor(tKgK); print("\n");
    }


    if (thread0()){
        print("gmem_thr_copy_KV: \n"); print(gmem_thr_copy_KV); print("\n");
        print("sK: \n"); print(sK); print("\n");
        print("tKgK: \n"); print(tKgK); print("\n");
        print("tKsK: \n"); print(tKsK); print("\n");
    }
    

    for (int m = 0; m < size<1>(tKgK); m++) {
        for (int k = 0; k < size<2>(tKgK); k++) {
            copy(gmem_tiled_copy_KV, tKgK(_, m, k), tKsK(_, m, k));
        }
    }

    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();


    Tensor sK_ = cute::recast<uint8_t>(sK);
    if (thread0()) {
        print("sK_ uint8: "); print(sK_); print("\n");
        print_tensor(sK_);
    }

    if (thread0()) {

        

        printf("K -----------------------------------------------------------------------------------------------------\n");
        print("tensor sK: \n"); 
        print_tensor(sK);

        
    }
}



template <typename T_, int kTileN_ = 32, int kTileK_ = 128>
struct Kernel_traits {

  using T = T_;

  // tile configuration
  
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3; 

  using GmemLayoutAtomKV = Layout<Shape <Int<32>, Int<4>>,
                                  Stride<Int<4>, _1>>;

  using GmemTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, uint16_t>{},
                        GmemLayoutAtomKV{},
                        Layout<Shape<_1, _8>>{}));
            
  using SmemLayoutAtomKV = Layout<Shape<_8, Int<32>>,
                           Stride<Int<32>, _1>>;

  using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        Shape<Int<kTileN>, Int<32>>{}));  


   using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, Int<4>, _1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, 
             Layout<Shape <_8,_4,_2>,
                    Stride<_1,_16,_8>>, 
            _16>
        >;
    
    using SmemCopyAtom = Copy_Atom<SM75_U32x2_LDSM_N, T>;


   static constexpr int kSmemSize = size(SmemLayoutKV{}) * 2;             
  
};


int main(void) {
    using namespace cute;
    using T = cute::uint4_t;


    std::mt19937 gen(20250102);
    std::uniform_int_distribution<uint16_t> dis(0, 255); // 生成 0-255 的整数
    constexpr int N = 64;
    constexpr int K = 128;

    // q
    uint8_t* h_k = (uint8_t*)malloc(N * 64);
    for (int i = 0; i < N * 64; i++) {
        uint8_t data = static_cast<uint8_t>(dis(gen));
        h_k[i] = data;
    }

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < int(K / 2); k++) {
            printf("%5d ", static_cast<int>(h_k[n * 64 + k]));
        }
        printf("\n");
    }
    uint8_t* d_k = nullptr;
    cudaMalloc(&d_k, N * 64);
    cudaMemcpy(d_k, h_k, N * 64, cudaMemcpyHostToDevice);
    printf("-------------------------------------------------------------------\n");


    Kernel_traits<uint16_t, N, K> config;

    


    auto kernel = &g2sCopy<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, 128, smem_size>>>(d_k);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    if (d_k) {
        cudaFree(d_k);
    }
    return 0;
}