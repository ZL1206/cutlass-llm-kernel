#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "static_switch.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <vector>
#include <random>
#include <numeric>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
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


template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Kernel_traits>
__global__ void qk_matmul_kernel(void* q, void* k, void* o) {
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

    __syncthreads();

    if (thread0()) {
        printf("acc_s: "); print(acc_s); print("\n");
        print_tensor(acc_s);
    }

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rO = make_fragment_like<T>(acc_s.layout());
    for (int i = 0; i < size(rO); i++) {
        rO[i] = static_cast<T>(acc_s[i]);
    }

    if (thread0()) {
        printf("rO: "); print(rO); print("\n");
        print_tensor(rO);
    }
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(idx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    if (thread0()) {
        printf("sO: "); print(sO); print("\n");
        printf("taccOrO: "); print(taccOrO); print("\n");
        printf("taccOsO: "); print(taccOsO); print("\n");
    }
    // copy to shared memory
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();

    if (thread0()) {
        printf("sO: \n");
        print_tensor(sO);
    }

    
    // shared memory to register
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(o)),
                    make_shape(Int<M>{}, Int<N>{}),
                    make_stride(Int<N>{}, Int<1>{}));
    
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    
    __syncthreads();
    
    Tensor tOrO = make_tensor<T>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    for (int m = 0; m < size<1>(tOrO); m++) {
        for (int k = 0; k < size<2>(tOrO); k++) {
            cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), tOgO(_, m, k));
        }
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

  /*
  using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  */
  using SmemLayoutAtomO = Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>;
  using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kTileM>, Int<kTileN>>{}));

  // shared memory to register copy
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>; 

  
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;
  
  // tiled mma
  using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;
  
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(T);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(T);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};

/*
int main(void) {
    using T = cutlass::half_t;
    std::mt19937 gen(20250102);
    std::uniform_real_distribution<float> dis(static_cast<float>(-1), static_cast<float>(1));
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

    T* d_o = nullptr;
    T* h_o = (T*)malloc(M * N * sizeof(T));
    cudaMalloc(&d_o, M * N * sizeof(T));

    Kernel_traits<T, M, N, K> config;

    auto kernel = &qk_matmul<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, config.kNThreads, smem_size>>>(d_q, d_k, d_o);
    cudaMemcpy(h_o, d_o, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));

    cudaFree(d_q);
    cudaFree(d_k);

    return 0;
}
*/


template <typename T>
void qk_matmul_kernel_launch(const at::Tensor& q, const at::Tensor& k, at::Tensor& o) {
    const int M = q.size(0);
    const int K = q.size(1);
    const int N = k.size(0);
    printf("m %d, n %d, k %d\n", M, N, K);
    TORCH_CHECK(K == 128, "only support k == 128");
    void* __restrict__ q_ptr = q.data_ptr();
    void* __restrict__ k_ptr = k.data_ptr();
    void * __restrict__ o_ptr = o.data_ptr();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Kernel_traits<T, 128, 64, 128> config;
    auto kernel = &qk_matmul_kernel<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, config.kNThreads, smem_size, stream>>>(q_ptr, k_ptr, o_ptr);
}



void qk_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    at::Tensor& o
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
     
    FP16_SWITCH(q_dtype != torch::kBFloat16, [&] {
        qk_matmul_kernel_launch<elem_type>(q, k, o);
    });

}