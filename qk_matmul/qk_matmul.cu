#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
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


#define CUCHECK(cmd)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult retval = cmd;                                                                                         \
        if (retval != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            const char* error_string;                                                                                  \
            cuGetErrorString(retval, &error_string);                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

using namespace cute;

template <typename Kernel_traits>
__global__ void qk_matmul(void* q, void* k) {
    using T = typename Kernel_traits::T;
    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;

    using GmemTiledCopyQKV = typename Kernel_traits::GmemTiledCopyQKV;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;

    extern __shared__ char smem[];

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(q)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                            SmemLayoutQ{});

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(k)),
                    make_shape(Int<N>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            SmemLayoutKV{});
    
    const int idx = threadIdx.x;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    
    
    // global to shared memory
    for (int m = 0; m < size<1>(tQgQ); m++) {
        for (int k = 0; k < size<2>(tQgQ); k++) {
            copy(gmem_tiled_copy_QKV, tQgQ(_, m, k), tQsQ(_, m, k));
        }
    }

    for (int m = 0; m < size<1>(tKgK); m++) {
        for (int k = 0; k < size<2>(tKgK); k++) {
            copy(gmem_tiled_copy_QKV, tKgK(_, m, k), tKsK(_, m, k));
        }
    }

    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();

    if (thread0()) {
        print("gQ: "); print(gQ); print("\n");
        print("sQ: "); print(sQ); print("\n");
        print("acc_s: "); print(acc_s); print("\n");
        print("tSsQ: "); print(tSsQ); print("\n");
        print_tensor(tSsQ);
        print("tSsK: "); print(tSsK); print("\n");
        print_tensor(tSsK);
    }
    

    CUTE_STATIC_ASSERT_V(size<1>(tSrQ) == size<1>(acc_s));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tSrK) == size<2>(acc_s));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_K.retile_D(tSrK);
    CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrA_copy_view(_, _, _0{})); 
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    
    if (thread0()) {
        print("tSrQ: "); print(tSrQ); print("\n");
        print_tensor(tSrQ);
        print("tSrK: "); print(tSrK); print("\n");
        print_tensor(tSrK);
    }

    if (thread0()) {
        printf("tCrA_copy_view: "); print(tCrA_copy_view); printf("\n");
        print_tensor(tCrA_copy_view);
        printf("tCrB_copy_view: "); print(tCrB_copy_view); printf("\n");
        print_tensor(tCrB_copy_view);
    }
    #pragma unroll
    for (int i = 0; i < size<2>(tSrQ); ++i) {
        if (i < size<2>(tSrQ) - 1) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); 
        }
        cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
    }

    
}



template <typename T_, int kTileM_ = 128, int kTileN_ = 32, int kTileK_ = 128, int kNWarps_ = 4>
struct Kernel_traits {

  using T = T_;

  // tile configuration
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * 32;
  
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3; 

  // global to shared memory
  using GmemLayoutAtom = Layout<Shape <Int<16>, Int<8>>,
                                  Stride<Int<8>, _1>>;
    
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
  // write o
  using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

  // shared memory layout
  /*
  using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  */
  using SmemLayoutAtomQ = Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>;
  
  
  using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kTileM>, Int<kTileK>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kTileN>, Int<kTileK>>{}));

  using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kTileK>, Int<kTileN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kTileM>, Int<kTileK>>{}));

  // shared memory to register copy
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>; 

  
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, T>;
  
  // tiled mma
  using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;
  
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(T);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(T);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};


int main(void) {
    using namespace cute;
    using T = cute::half_t;
    std::mt19937 gen(20250102);
    std::uniform_real_distribution<float> dis(static_cast<float>(-100), static_cast<float>(100));
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 128;
    int mcSupport = 0;
    int cudaDev;
    CUdevice currentDev;
    cudaGetDevice(&cudaDev);
    cuDeviceGet(&currentDev, cudaDev);
    CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));
    printf("mcSupport is %d\n", mcSupport);
    // q
    T* h_q = (T*)malloc(M * K * sizeof(T));
    for (int i = 0; i < M * K; i++) {
        float data = dis(gen);
        h_q[i] = T(data);
    }
    T* d_q = nullptr;
    cudaMalloc(&d_q, M * K * sizeof(T));
    cudaMemcpy(d_q, h_q, sizeof(T) * M * K, cudaMemcpyHostToDevice);
    printf("-------------------------------------------------------------------\n");

    // k
    std::mt19937 genk(20250102 + 1);
    T* h_k = (T*)malloc(N * K * sizeof(T));
    for (int i = 0; i < N * K; i++) {
        float data = dis(genk);
        h_k[i] = T(data);
    }
    T* d_k = nullptr;
    cudaMalloc(&d_k, N * K * sizeof(T));
    cudaMemcpy(d_k, h_k, sizeof(T) * N * K, cudaMemcpyHostToDevice);

    Kernel_traits<T, M, N, K> config;

    auto kernel = &qk_matmul<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, config.kNThreads, smem_size>>>(d_q, d_k);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    cudaFree(d_q);
    cudaFree(d_k);

    return 0;
}