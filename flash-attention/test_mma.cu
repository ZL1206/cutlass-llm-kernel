#include <iostream>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <iomanip>
#include <utility>
#include <random>
#include <type_traits>
#include <vector>
#include <numeric>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/trace.h>
#include <cutlass/array.h>

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


template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};


template <typename Kernel_traits>
__global__ void test_mma(void* q, void* k, void* v, void* o) {
    
    using T = typename Kernel_traits::T;
    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;

    using GmemTiledCopyQKV = typename Kernel_traits::GmemTiledCopyQKV;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    using TiledMma = typename Kernel_traits::TiledMma;
    using SmemCopyAtom = typename Kernel_traits::SmemCopyAtom;
    using SmemLayoutVtransposed = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtransposedNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;
    using SmemLayoutO = typename Kernel_traits::SmemLayoutO;
    using SmemCopyAtomO = typename Kernel_traits::SmemCopyAtomO;
    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;
    using SmemCopyAtomTransposed = typename Kernel_traits::SmemCopyAtomTransposed;

    extern __shared__ char smem[];

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(q)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(k)),
                    make_shape(Int<N>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(v)),
                    make_shape(Int<N>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(o)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                            SmemLayoutQ{});
    
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            SmemLayoutKV{});

    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), SmemLayoutVtransposedNoSwizzle{});

    

    int idx = threadIdx.x;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    if (thread0()){
        print("gmem_thr_copy_QKV: \n"); print(gmem_thr_copy_QKV); print("\n");
        print("sQ: \n"); print(sQ); print("\n");
        print("sK: \n"); print(sK); print("\n");
        print("sV: \n"); print(sV); print("\n");
        print("tQgQ: \n"); print(tQgQ); print("\n");
        print("tQsQ: \n"); print(tQsQ); print("\n");
        print("tKgK: \n"); print(tKgK); print("\n");
        print("tKsK: \n"); print(tKsK); print("\n");
        print("tVgV: \n"); print(tVgV); print("\n");
        print("tVsV: \n"); print(tVsV); print("\n");
    }

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

    for (int m = 0; m < size<1>(tVgV); m++) {
        for (int k = 0; k < size<2>(tVgV); k++) {
            copy(gmem_tiled_copy_QKV, tVgV(_, m, k), tVsV(_, m, k));
        }
    }
    
    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();

    if (thread0()) {
        
        int size_q = size(sQ);
        int size_k = size(sK);
        int size_v = size(sV);
        printf("q, k, v size is %d, %d, %d\n", size_q, size_k, size_v);
        
        printf("Q -----------------------------------------------------------------------------------------------------\n");
        T* smem_ptr = reinterpret_cast<T*>(smem);
        for (int i = 0; i < size(sQ); i++) {
            float tmp = static_cast<float>(smem_ptr[i]);
            printf("%10.0f", tmp); printf(" ");
            if ((i+1) % 64 == 0) {
                print("\n");
            }
        }

        printf("K -----------------------------------------------------------------------------------------------------\n");

        smem_ptr = smem_ptr + size(sQ);

        for (int i = 0; i < size(sK); i++) {
            float tmp = static_cast<float>(smem_ptr[i]);
            printf("%10.0f", tmp); printf(" ");
            if ((i+1) % 64 == 0) {
                print("\n");
            }
        }
        printf("V -----------------------------------------------------------------------------------------------------\n");
        smem_ptr = smem_ptr + size(sK);

        for (int i = 0; i < size(sV); i++) {
            float tmp = static_cast<float>(smem_ptr[i]);
            printf("%10.0f", tmp); printf(" ");
            if ((i+1) % 64 == 0) {
                print("\n");
            }
        }




    }
    // 为了打印shared memory中的数据，需要同步
    __syncthreads();
    
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{}); // (MMA=4, MMA_M, MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<K>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    
    CUTE_STATIC_ASSERT_V(size<1>(tSrQ) == size<1>(acc_s));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tSrK) == size<2>(acc_s));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_K.retile_D(tSrK);
    CUTE_STATIC_ASSERT_V(size<1>(tSsK) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrA_copy_view(_, _, _0{})); 
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tCrB_copy_view(_, _, _0{})); 

    if (thread0()){
        print("tSrQ: \n"); print(tSrQ); print("\n");
        print("tSrK: \n"); print(tSrK); print("\n");
        print("acc_s: \n"); print(acc_s); print("\n");
        print("tSsQ: \n"); print(tSsQ); print("\n");
        print("tSsK: \n"); print(tSsK); print("\n");
        print("tCrA_copy_view: \n"); print(tCrA_copy_view); print("\n");
        print("tCrB_copy_view: \n"); print(tCrB_copy_view); print("\n");
    }
    

    #pragma unroll
    for (int i = 0; i < size<2>(tSrQ); ++i) {
        if (i < size<2>(tSrQ) - 1) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); 
            cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); 
        }
        cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
    }

    Tensor rP = convert_type<T>(acc_s);
    
    Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()));

    CUTE_STATIC_ASSERT_V(size<1>(tOrP) == size<1>(acc_o));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tOrVt) == size<2>(acc_o));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tOrP) == size<2>(tOrVt));                     // MMA_K
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrV_copy_view));            // N
    cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrV_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tOrP); ++i) {
        if (i < size<2>(tOrP) - 1) {
            cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrV_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
    }
    
    Tensor rO = convert_type<T>(acc_o);
    Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});    // (SMEM_M,SMEM_N)

    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(idx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<T>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    CUTE_STATIC_ASSERT_V(rank(tOrO) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(tOgO) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(tOrO) == size<0>(tOgO));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(tOrO) == size<1>(tOgO));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tOrO) == size<2>(tOgO));                     // MMA_K
    
    #pragma unroll
    for (int m = 0; m < size<1>(tOrO); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(tOrO); ++k) {
            cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), tOgO(_, m, k));
        }
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
  // write o
  using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

  // shared memory layout
  using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  //using SmemLayoutAtomQa = Layout<Shape<_8, Int<64>>,
  //                         Stride<Int<64>, _1>>;
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
        Layout<Shape<Int<4>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * 4>, _16, _16>>;
  
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(T);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(T);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};



int main(void) {
    using namespace cute;
    using T = cute::bfloat16_t;
    const int M = 128;
    const int N = 32;
    const int K = 128;
    T* h_q = (T*)malloc(M * K * sizeof(T));
    std::mt19937 gen(20250102);
    std::uniform_real_distribution<float> dis(static_cast<float>(-100), static_cast<float>(100));
    for (int i = 0; i < M * K; i++) {
        float tmp = static_cast<float>(i % K);
        h_q[i] = T(tmp);
        tmp = static_cast<float>(h_q[i]);
        printf("%10.f", tmp); printf(" ");
        if ((i+1) % 128 == 0) {
            print("\n");
        }
    }
    T* d_q = nullptr;
    cudaMalloc(&d_q, M * K * sizeof(T));
    cudaMemcpy(d_q, h_q, sizeof(T) * M * K, cudaMemcpyHostToDevice);
    print("-------------------------------------------------------------------\n");
    T* h_k = (T*)malloc(N * K * sizeof(T));
    for (int i = 0; i < N * K; i++) {
        float tmp = static_cast<float>(i / 8);
        h_k[i] = T(tmp);
        tmp = static_cast<float>(h_k[i]);
        printf("%10.f", tmp); printf(" ");
        if ((i+1) % 128 == 0) {
            print("\n");
        }
    }
    T* d_k = nullptr;
    cudaMalloc(&d_k, N * K * sizeof(T));
    cudaMemcpy(d_k, h_k, sizeof(T) * N * K, cudaMemcpyHostToDevice);
    print("-------------------------------------------------------------------\n");

    T* h_v = (T*)malloc(N * K * sizeof(T));
    for (int i = 0; i < N * K; i++) {
        float tmp = static_cast<float>(i / 8);
        h_v[i] = T(tmp);
    }
    T* d_v = nullptr;
    cudaMalloc(&d_v, N * K * sizeof(T));
    cudaMemcpy(d_v, h_v, sizeof(T) * N * K, cudaMemcpyHostToDevice);

    T* d_o = nullptr;
    cudaMalloc(&d_o, M * K * sizeof(T));


    Kernel_traits<T, M, N, K> config;
    
    using SmemLayoutAtomQ = typename decltype(config)::SmemLayoutAtomQ;
    
    print("SmemLayoutAtomQ \n");
    print_layout(SmemLayoutAtomQ{});

    using SmemLayoutKV = typename decltype(config)::SmemLayoutKV;

    using SmemLayoutQ = typename decltype(config)::SmemLayoutQ;
   
    using SmemLayoutVtransposed = typename decltype(config)::SmemLayoutVtransposed;
    
    print("SmemLayoutQ \n");
    print_layout(SmemLayoutQ{});

    print("SmemLayoutKV \n");
    print_layout(SmemLayoutKV{});

    print("SmemLayoutVtransposed \n");
    print_layout(SmemLayoutVtransposed{});

    using TiledMma = typename decltype(config)::TiledMma;
    
    printf("tiled mma:\n");
    print(TiledMma{});
    
    
    auto kernel = &test_mma<decltype(config)>;
    const int smem_size = config.kSmemSize;
    printf("smem_size is %d\n", smem_size);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<1, 128, smem_size>>>(d_q, d_k, d_v, d_o);
    
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
    
    T* h_o = (T*)malloc(M * K * sizeof(T));
    cudaMemcpy(h_o, d_o, sizeof(T) * M * K, cudaMemcpyDeviceToHost);

    if (d_q) {
        cudaFree(d_q);
    }
    if (d_k) {
        cudaFree(d_k);
    }
    if (d_v) {
        cudaFree(d_v);
    }
    if (d_o) {
        cudaFree(d_o);
    }
    return 0;
}