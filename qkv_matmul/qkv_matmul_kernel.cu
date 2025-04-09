#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "static_switch.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include <cuda.h>
#include <cute/tensor.hpp>
#include <vector>
#include <random>
#include <numeric>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cute/tensor.hpp>


using namespace cute;

template <typename T_, int kTileM_ = 128, int kTileN_ = 32, int kTileK_ = 128, int kNWarps_ = 4>
struct kernel_traits {

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
        Shape<Int<kTileM>, Int<kTileK>>{}));

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

struct fwd_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void * __restrict__ o_ptr;
    int seqlen_q;
    int seqlen_k;
    int d;
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    int o_head_stride;
};

template <typename kernel_traits, bool Is_even_MN, bool Is_causal>
__global__ void qk_matmul_kernel(fwd_params params) {
    using T = typename kernel_traits::T;
    constexpr int M = kernel_traits::kTileM;
    constexpr int N = kernel_traits::kTileN;
    constexpr int K = kernel_traits::kTileK;

    using GmemTiledCopyQKV = typename kernel_traits::GmemTiledCopyQKV;
    using SmemLayoutQ = typename kernel_traits::SmemLayoutQ;
    using SmemLayoutKV = typename kernel_traits::SmemLayoutKV;

    extern __shared__ char smem[];

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.q_ptr)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)),
                            SmemLayoutQ{});

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_ptr)),
                    make_shape(Int<N>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            SmemLayoutKV{});

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_ptr)),
                    make_shape(Int<N>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));

    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename kernel_traits::SmemLayoutVtransposed{});
    if (thread0()) {
        print("sVt: "); print(sVt); printf("\n");
    }
    
    const int idx = threadIdx.x;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (thread0()) {
        print("gQ: "); print(gQ); print("\n");
        print("sQ: "); print(sQ); print("\n");
        print("tQgQ: "); print(tQgQ); print("\n");
        print("tQsQ: "); print(tQsQ); print("\n");
        print("cQ: "); print(cQ); print("\n");
        print_tensor(cQ);
        print("tQcQ: "); print(tQcQ); print("\n");
        print_tensor(tQcQ);
        print("tQpQ: "); print(tQpQ); print("\n");
        print("cKV: "); print(cKV); print("\n");
        print_tensor(cKV);
        print("tKVcKV: "); print(tKVcKV); print("\n");
        print_tensor(tKVcKV);
    }
    typename kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});  // (MMA=4, MMA_M, MMA_N)
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<K>>{});  // MMA, MMA_M, MMA_K
    clear(acc_s);
    clear(acc_o);
    flash::Softmax<2 * size<1>(acc_s)> softmax;
    flash::Mask mask(params.seqlen_k, params.seqlen_q);
    
    // global to shared memory
    for (int m = 0; m < size<1>(tQgQ); m++) {
        if (Is_even_MN || get<0>(tQcQ(0, m, 0)) < params.seqlen_q) {
            int identity_mn = get<0>(tQcQ(0, m, 0));
            if (thread0()) {
                printf("m is %d, identity_mn is %d\n", m, identity_mn);
            }
            for (int k = 0; k < size<2>(tQgQ); k++) {
                copy(gmem_tiled_copy_QKV, tQgQ(_, m, k), tQsQ(_, m, k));
            }
        }
    }

    for (int m = 0; m < size<1>(tKgK); m++) {
        if (Is_even_MN || get<0>(tKVcKV(0, m, 0)) < params.seqlen_k) {
            for (int k = 0; k < size<2>(tKgK); k++) {
                copy(gmem_tiled_copy_QKV, tKgK(_, m, k), tKsK(_, m, k));
            }
        }
    }

    for (int m = 0; m < size<1>(tVgV); m++) {
        if (Is_even_MN || get<0>(tKVcKV(0, m, 0)) < params.seqlen_k) {
            for (int k = 0; k < size<2>(tVgV); k++) {
                copy(gmem_tiled_copy_QKV, tVgV(_, m, k), tVsV(_, m, k));
            }
        }
    }

    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();

    if (thread0()) {
        print("acc_s: "); print(acc_s); print("\n");
        print("acc_o: "); print(acc_o); print("\n");
        print("tSsQ: "); print(tSsQ); print("\n");
        print_tensor(tSsQ);
        print("tSsK: "); print(tSsK); print("\n");
        print_tensor(tSsK);
        print("tOsVt: "); print(tOsVt); print("\n");
    }
    
    // first gemm
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
        for (int i = 0; i < size(acc_s); i++) {
            print(acc_s.data()[i]); printf("\n");
        }
    }
    mask.template apply_mask</*Causal_mask=*/false, Is_even_MN>(acc_s, 0, 0, 0, 0);
    
    softmax.template softmax_rescale_o</*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);
    
    // Convert acc_s from fp32 to fp16/bf16
    //Tensor rP = flash::convert_type<T>(acc_s);
    constexpr int numel = decltype(size(acc_s))::value;
    cutlass::NumericArrayConverter<T, float, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel> *>(acc_s.data()));
    Tensor rP = make_tensor(make_rmem_ptr<T>(&frag), acc_s.layout());
    if (thread0()) {
        printf("rP: "); print(rP); print("\n");
        print_tensor(rP);
    }

    // second gemm, change acc_s layout, output as input
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename kernel_traits::TiledMma>(rP.layout()));
    if (thread0()) {
        print("tOrP is: "); print(tOrP); print("\n");
    }
    CUTE_STATIC_ASSERT_V(size<1>(tOrP) == size<1>(acc_o));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tOrVt) == size<2>(acc_o));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tOrP) == size<2>(tOrVt));                     // MMA_K
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrV_copy_view));            // N
    cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrV_copy_view(_, _, _0{}));

    if (thread0()) {
        print("tOsVt: "); print(tOsVt); print("\n");
        print_tensor(tOsVt);
        print("tOrVt: "); print(tOrVt); print("\n");
        print_tensor(tOrVt);
        print("tCrV_copy_view: "); print(tCrV_copy_view); print("\n");
        print_tensor(tCrV_copy_view);
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tOrP); ++i) {
        if (i < size<2>(tOrP) - 1) {
            cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrV_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse(acc_o, params.scale_softmax);
    if (thread0()) {
        print("lse: "); print(lse); print("\n");
        print_tensor(lse);
    }
    __syncthreads();
    if (thread(1)) {
        print("lse: "); print(lse); print("\n");
        print_tensor(lse);
    }
    __syncthreads();
    if (thread(2)) {
        print("lse: "); print(lse); print("\n");
        print_tensor(lse);
    }
    __syncthreads();
    if (thread(3)) {
        print("lse: "); print(lse); print("\n");
        print_tensor(lse);
    }
    __syncthreads();


    // convert acc_o to fp16
    constexpr int numel_ = decltype(size(acc_o))::value;
    cutlass::NumericArrayConverter<T, float, numel_> convert_op_;
    auto frag_ = convert_op_(*reinterpret_cast<const cutlass::Array<float, numel_> *>(acc_o.data()));
    Tensor rO = make_tensor(make_rmem_ptr<T>(&frag_), acc_o.layout());
    if (thread0()) {
        print("rO: "); print(rO); print("\n");
        print_tensor(rO);
    }
    Tensor sO = make_tensor(sQ.data(), typename kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    auto smem_tiled_copy_O = make_tiled_copy_C(typename kernel_traits::SmemCopyAtomO{}, tiled_mma);
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
    /*
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr)),
                            make_shape(params.seqlen_q, params.seqlen_k),
                            make_stride(params.o_head_stride, _1{}));

    Tensor gO = local_tile(mO(_, _), Shape<Int<M>, Int<N>>{},
                           make_coord(0, 0));  // (kBlockM, kHeadDim)
     if (thread0()) {
        print("mO: "); print(mO); print("\n");
        print("gO: "); print(gO); print("\n");
    }
    */
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(params.o_head_stride, Int<1>{}));
    if (thread0()) {
        print("gO: "); print(gO); print("\n");
    }
    
    typename kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    
    //__syncthreads();
    Tensor tOrO = make_tensor<T>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    if (thread0()) {
        printf("tOsO: "); print(tOsO); printf("\n");
        print_tensor(tOsO);
        printf("tOrO: "); print(tOrO); printf("\n");
        print_tensor(tOrO);
        printf("tOgO: "); print(tOgO); printf("\n");
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (thread0()) {
        printf("cO: "); print(cO); printf("\n");
        print_tensor(cO);
        printf("tOcO: "); print(tOcO); print("\n");
        print_tensor(tOcO);
        printf("tOpO: "); print(tOpO); print("\n");
        print_tensor(tOpO);
    }
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
        int len = get<1>(tOcO(0, 0, k));
        printf("tid %d, len is %d\n", idx, len);
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
    // register to global memory
    for (int m = 0; m < size<1>(tOrO); m++) {
        if (Is_even_MN || get<0>(tOcO(0, m, 0)) < params.seqlen_q) {
            for (int k = 0; k < size<2>(tOrO); k++) {
                if (tOpO(k)) {
                    cute::copy(gmem_tiled_copy_O, tOrO(_, m, k), tOgO(_, m, k));
                }
            }
        }
    }

}



template <typename T>
void qk_matmul_kernel_launch(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, at::Tensor& out, float softmax_scale, const bool is_causal) {
    const int m = query.size(0);
    const int k = query.size(1);
    const int n = key.size(0);
    printf("m %d, n %d, k %d\n", m, n, k);
    TORCH_CHECK(k == 128, "only support k == 128");
    fwd_params params;
    params.q_ptr = query.data_ptr();
    params.k_ptr = key.data_ptr();
    params.v_ptr = value.data_ptr();
    params.o_ptr = out.data_ptr();
    params.seqlen_q = m;
    params.seqlen_k = n;
    params.d = k;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.o_head_stride = k;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool is_even_mn = params.seqlen_k % 64 == 0 && params.seqlen_q % 128 == 0;
    kernel_traits<T, 128, 64, 128> config;
    BOOL_SWITCH(is_even_mn, Is_even_MN, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            auto kernel = &qk_matmul_kernel<decltype(config), Is_even_MN, Is_causal>;
            const int smem_size = config.kSmemSize;
            printf("smem_size is %d\n", smem_size);
            if (smem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            kernel<<<1, config.kNThreads, smem_size, stream>>>(params);
        });
    });
}


void qkv_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    const float softmax_scale,
    const bool is_causal
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
     
    FP16_SWITCH(q_dtype != torch::kBFloat16, [&] {
        qk_matmul_kernel_launch<elem_type>(q, k, v, o, softmax_scale, is_causal);
    });

}