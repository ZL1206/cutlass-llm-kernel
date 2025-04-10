#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "static_switch.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "quantize.h"
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


using namespace cute;




template<typename T>
struct TypeMapper {
    using type = T;  // 默认返回原类型
};

// 特化：当 T=uint4_t 时，返回 uint16_t
template<>
struct TypeMapper<uint4_t> {
    using type = uint16_t;
};



template <typename T_, typename Tkv_, int kTileM_ = 128, int kTileN_ = 32, int kTileK_ = 128, int kNWarps_ = 4>
struct Kernel_traits {

  using T = T_;
  using Tkv = Tkv_;

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
  
  using GmemTiledCopyQ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));

  // copy kv, head_dim == 128
  using GmemLayoutAtomKV = Layout<Shape <Int<32>, Int<4>>,
                                  Stride<Int<4>, _1>>;
    
  using GmemTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
                        GmemLayoutAtomKV{},
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
  
  // smem layout kv
  using SmemLayoutAtomKV = Layout<Shape<_8, Int<32>>,
                           Stride<Int<32>, _1>>;
  
  using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        Shape<Int<kTileN>, Int<32>>{}));
    
  using SmemLayoutKVParams = Layout<Shape<_2, Int<kTileN>>,
                                   Stride<Int<kTileN>, _1>>; 

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  using SmemLayoutAtomO = Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>;
  using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kTileM * kNWarps>, Int<kTileK>>{}));
  
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;
  
  using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<32>, Int<kTileN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  // shared memory to register copy
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>; 
  
  
  // tiled mma
  using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, Int<4>, _1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, 
             Layout<Shape <_8,_4,_2>,
                    Stride<_1,_16,_8>>, 
            _16>
        >;

  using TiledMma_PV = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, _1, Int<kNWarps>>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, _16, Int<16 * kNWarps>>>;

  using SmemLayoutLse = Layout<Shape<Int<kNWarps>, Int<kTileM>>,
                               Stride<Int<kTileM>, _1>>;

  struct TensorStorage
  {
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutKV>> smem_k;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutKVParams>> smem_k_params;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutKV>> smem_v;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutKVParams>> smem_v_params;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(128) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutO>> smem_o;
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutLse>> smem_lse;

  };

  static constexpr int kSmemSize = sizeof(TensorStorage);
  //static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};

struct fwd_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ k_scale_ptr;
    void *__restrict__ v_scale_ptr;
    void * __restrict__ o_ptr;
    void * __restrict__ softmax_lse_ptr;
    int seqlen_q;
    int seqlen_k;
    int d;
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    int o_head_stride;
};



template <typename Kernel_traits, bool Is_even_MN, bool Is_causal>
__global__ void int4_qkv_matmul_kernel(fwd_params params) {
    using T = typename Kernel_traits::T;
    using Tkv = typename Kernel_traits::Tkv;
    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyQ = typename Kernel_traits::GmemTiledCopyQ;
    using GmemTiledCopyKV = typename Kernel_traits::GmemTiledCopyKV;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    using SharedStorage = typename Kernel_traits::TensorStorage;

    extern __shared__ char smem[];

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.q_ptr)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()),
                            SmemLayoutQ{});

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_ptr)),
                    make_shape(Int<N>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}));
    
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()),
                            SmemLayoutKV{});
    
    Tensor gKP = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.k_scale_ptr)),
                             make_shape(Int<2>{}, Int<64>{}),
                             make_stride(Int<64>{}, Int<1>{}));

    Tensor sKP = make_tensor(make_smem_ptr(shared_storage.smem_k_params.begin()),
                            typename Kernel_traits::SmemLayoutKVParams{});

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_ptr)),
                    make_shape(Int<N>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}));
    
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor gVP = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.v_scale_ptr)),
                             make_shape(Int<2>{}, Int<64>{}),
                             make_stride(Int<64>{}, Int<1>{}));

    Tensor sVP = make_tensor(make_smem_ptr(shared_storage.smem_v_params.begin()),
                            typename Kernel_traits::SmemLayoutKVParams{});
    
    
    const int idx = threadIdx.x;
    const int warp_idx = idx / 32;
    const int lane = idx % 32;

    // global to smem
    GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(idx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tKPgKP = gmem_thr_copy_KV.partition_S(gKP);  
    Tensor tKPsKP = gmem_thr_copy_KV.partition_D(sKP);

    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);  
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
    Tensor tVPgVP = gmem_thr_copy_KV.partition_S(gVP);  
    Tensor tVPsVP = gmem_thr_copy_KV.partition_D(sVP);

    // first mma
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK_q  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrK = make_tensor<T>(Shape< Shape<_2, _2>, _2, _8>{},              // (MMA,MMA_N,MMA_K)
                                 Stride< Stride<_1, _2>, _32, _4>{});
    Tensor tSrK_dq = make_tensor(tSrK.data(), Layout<Shape<_8, _2, Shape<_2, _2>>, Stride<_1, _32, Stride<_8, _16>>>{});
    
    // second mma
    typename Kernel_traits::TiledMma_PV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(idx);
    Tensor tOrVt_q  = thr_mma_pv.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = make_tensor<T>(Shape< Shape<_2, _2>, _16, _1>{}, 
                                  Stride< Stride<_1, _32>, _2, _0>{});
    Tensor tOrVt_dq = make_tensor(tOrVt.data(), Layout<Shape<Shape<_8, _2>, _4, _1>, Stride<Stride<_1, _32>, _8, _0>>{});
    
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_pv);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});  // (MMA=4, MMA_M, MMA_N)
    Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<M>, Int<K>>{});  // MMA, MMA_M, MMA_K
    
    clear(acc_s);
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_s)> softmax;
    flash::Mask mask(params.seqlen_k, params.seqlen_q);

 
    // global to shared memory
    for (int m = 0; m < size<1>(tQgQ); m++) {
        
            for (int k = 0; k < size<2>(tQgQ); k++) {
                copy(gmem_tiled_copy_Q, tQgQ(_, m, k), tQsQ(_, m, k));
            }
        
    }
    

    for (int m = 0; m < size<1>(tKgK); m++) {
            for (int k = 0; k < size<2>(tKgK); k++) {
                copy(gmem_tiled_copy_KV, tKgK(_, m, k), tKsK(_, m, k));
            }

    }


    for (int m = 0; m < size<1>(tVgV); m++) {
            for (int k = 0; k < size<2>(tVgV); k++) {
                copy(gmem_tiled_copy_KV, tVgV(_, m, k), tVsV(_, m, k));
            }

    }

    Tensor cKVP = make_identity_tensor(make_shape(size<0>(sKP), size<1>(sKP)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tKVPcKVp = gmem_thr_copy_KV.partition_S(cKVP);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    
    for (int m = 0; m < size<1>(tKPgKP); m++) {
        if (get<0>(tKVPcKVp(0, m, 0)) < 2) {
            for (int k = 0; k < size<2>(tKPgKP); k++) {
                copy(gmem_tiled_copy_KV, tKPgKP(_, m, k), tKPsKP(_, m, k));
            }
        }
    } 


    for (int m = 0; m < size<1>(tVPgVP); m++) {
        if (get<0>(tKVPcKVp(0, m, 0)) < 2) {
            for (int k = 0; k < size<2>(tVPgVP); k++) {
                copy(gmem_tiled_copy_KV, tVPgVP(_, m, k), tVPsVP(_, m, k));
            }
        }
    } 

    cp_async_fence();

    cp_async_wait<0>();

    __syncthreads();

    if (thread0()) {
        print("gQ: "); print(gQ); print("\n");
        print("sQ: "); print(sQ); print("\n");
        print("gK: "); print(gK); print("\n");
        print("sK: "); print(sK); print("\n");
        print("tQgQ: "); print(tQgQ); print("\n");
        print("tQsQ: "); print(tQsQ); print("\n");
        print("tKgK: "); print(tKgK); print("\n");
        print_tensor(tKgK);
        print("tKsK: "); print(tKsK); print("\n");
        print_tensor(tKsK);
    }

    

    CUTE_STATIC_ASSERT_V(size<1>(tSrQ) == size<1>(acc_s));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tSrK) == size<2>(acc_s));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));
    Tensor tCrA_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_K.retile_D(tSrK_q);
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrA_copy_view(_, _, _0{})); // copy q
    
    if (thread0()) {
        print("sKP: \n");
        print_tensor(sKP);
    }
    Tensor scale_k = make_tensor<T>(Shape<_2, Int<size<1>(tSrK)>>{},
                                    Stride<_1, Int<size<1>(tSrK)>>{});
    
    const int col = warp_idx * size<1>(tSrK) * 8 + lane / 4;
    for (int ni = 0; ni < size<1>(tSrK); ni++) { 
        scale_k(0, ni) = sKP(0, col + ni * 8);
        scale_k(1, ni) = sKP(1, col + ni * 8);
    }
    if (thread0()) {
        print("scale_k: \n");
        print_tensor(scale_k); 
    }
    for (int ki = 0; ki < size<2>(tSrK_q); ki++) {
        cute::copy(smem_tiled_copy_K, tSsK(_, _, ki), tCrB_copy_view(_, _, ki));
        for (int ni = 0; ni < size<1>(tSrK_q); ni++) {
            for (int r = 0; r < 2; r++) {
                Tensor src = tSrK_q(make_coord(_, r), ni, ki);
                Tensor dst = tSrK_dq(_, ni, make_coord(r, ki));
                flash::ConvertKvCache<Tkv, T>::convert(src, dst);
                for (int i = 0; i < 8; i++) {
                    dst(i) = dst(i) * scale_k(0, ni) + scale_k(1, ni);
                }
            }
        }

    }
    /*
    for (int n = 0; n < size<1>(tSrK_q); n++) {
        for (int r = 0; r < 2; r++) {
            Tensor src = tSrK_q(make_coord(_, r), n, 0);
            Tensor dst = tSrK_dq(_, n, r);
            flash::ConvertKvCache<Tkv, T>::convert(src, dst);
        }
    }
    */
    if (thread0()) {
        print("tSsK: "); print(tSsK); print("\n");
        print_tensor(tSsK);
        print("tCrB_copy_view: "); print(tCrB_copy_view); print("\n");
        print("tSrK_q: "); print(tSrK_q); print("\n");
        print_tensor(tSrK_q);
        print("tSrK: "); print(tSrK); print("\n");
        print_tensor(tSrK);
    }

    
    #pragma unroll
    for (int i = 0; i < size<2>(tSrQ); ++i) {
        if (i < size<2>(tSrQ) - 1) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
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


    mask.template apply_mask</*Causal_mask=*/false, Is_even_MN>(acc_s, 0, 0, idx, kNWarps * 16);
    softmax.template softmax_rescale_o</*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);


    constexpr int numel = decltype(size(acc_s))::value;
    cutlass::NumericArrayConverter<T, float, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel> *>(acc_s.data()));
    Tensor rP = make_tensor(make_rmem_ptr<T>(&frag), acc_s.layout());

    // second gemm, change acc_s layout, output as input
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

    CUTE_STATIC_ASSERT_V(size<1>(tOrP) == size<1>(acc_o));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tOrVt) == size<2>(acc_o));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tOrP) == size<2>(tOrVt));                     // MMA_K

    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrVt_q);
    CUTE_STATIC_ASSERT_V(size<1>(tOsVt) == size<1>(tCrV_copy_view));            // N
    if (thread0()) {
        print("tOrVt_q: "); print(tOrVt_q); print("\n");
        print("tCrV_copy_view: "); print(tCrV_copy_view); print("\n");
    }
    
    Tensor scale_v = make_tensor<T>(Shape<_2, Shape<_2, _2>, Int<size<2>(tOrVt)>>{},
                                    Stride<_1, Stride<_2, _4>, _8>{});

    if (thread0()) {
        print("sVP: \n");
        print_tensor(sVP);
    }
    
    
    for (int ki = 0; ki < size<2>(tOrVt); ki++) {
        const int col = warp_idx * size<2>(tOrVt) * 16 + ki * 16 + (lane % 4) * 2;
        for (int r = 0; r < 2; r++) {
            for (int e = 0; e < 2; e++) {
                scale_v(0, make_coord(e, r), ki) = sVP(0, col + r * 8 + e);
                scale_v(1, make_coord(e, r), ki) = sVP(1, col + r * 8 + e);
            }
        }
    }

    if (thread0()) {
        print("scale_v: \n");
        print_tensor(scale_v);
    }
    
    /*
    for (int ni = 0; ni < size<2>(tOrVt_q); ni++) {
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, ni), tCrV_copy_view(_, _, ni));
        for (int ki = 0; ki < size<1>(tOrVt_q); ki++) {
            for (int r = 0; r < 2; r++) {
                Tensor src = tOrVt_q(make_coord(_, r), ki, ni);
                Tensor dst = tOrVt_dq(make_coord(_, r), ki, ni);
                flash::ConvertKvCache<Tkv, T>::convert(src, dst);
            }
        }
    }
    */

    for (int ki = 0; ki < size<2>(tOrVt_q); ki++) {
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, ki), tCrV_copy_view(_, _, ki));
        for (int ni = 0; ni < size<1>(tOrVt_q); ni++) {
            for (int r = 0; r < 2; r++) {
                Tensor src = tOrVt_q(make_coord(_, r), ni, ki);
                Tensor dst = tOrVt_dq(make_coord(_, r), ni, ki);
                flash::ConvertKvCache<Tkv, T>::convert(src, dst);
                dst(0) = dst(0) * scale_v(0, make_coord(0, r), ki) + scale_v(1, make_coord(0, r), ki);
                dst(1) = dst(1) * scale_v(0, make_coord(1, r), ki) + scale_v(1, make_coord(1, r), ki);
                dst(2) = dst(2) * scale_v(0, make_coord(0, r), ki) + scale_v(1, make_coord(0, r), ki);
                dst(3) = dst(3) * scale_v(0, make_coord(1, r), ki) + scale_v(1, make_coord(1, r), ki);
                dst(4) = dst(4) * scale_v(0, make_coord(0, r), ki) + scale_v(1, make_coord(0, r), ki);
                dst(5) = dst(5) * scale_v(0, make_coord(1, r), ki) + scale_v(1, make_coord(1, r), ki);
                dst(6) = dst(6) * scale_v(0, make_coord(0, r), ki) + scale_v(1, make_coord(0, r), ki);
                dst(7) = dst(7) * scale_v(0, make_coord(1, r), ki) + scale_v(1, make_coord(1, r), ki);
            }
        }
    }

    if (thread0()) {
        print("tOsVt: "); print(tOsVt); print("\n");
        print_tensor(tOsVt);
        print("tCrV_copy_view: "); print(tCrV_copy_view); print("\n");
        print("tOrVt_q: "); print(tOrVt_q); print("\n");
        print_tensor(tOrVt_q);
        print("tOrVt: "); print(tOrVt); print("\n");
        print_tensor(tOrVt);
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tOrP); ++i) {
        cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
    }


    // Epilogue
    
    // warp lse
    Tensor lse = softmax.template normalize_softmax_lse(acc_o, params.scale_softmax);
    // block lse
    Tensor smem_lse = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem)), typename Kernel_traits::SmemLayoutLse{});
    Tensor final_lse = softmax.template normalize_final_lse(lse, smem_lse, acc_o, idx);

    // convert acc_o to fp16
    constexpr int numel_ = decltype(size(acc_o))::value;
    cutlass::NumericArrayConverter<T, float, numel_> convert_op_;
    auto frag_ = convert_op_(*reinterpret_cast<const cutlass::Array<float, numel_> *>(acc_o.data()));
    Tensor rO = make_tensor(make_rmem_ptr<T>(&frag_), acc_o.layout());


    // copy acc_o to shared memory
    Tensor mO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    Tensor sO = local_tile(mO(_, _), Shape<Int<M>, Int<K>>{},
                        make_coord(warp_idx, 0));  // (kBlockM, kHeadDim)
    
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_pv);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(lane);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();
    
    if (thread0()) {
        printf("sO: \n");
        print_tensor(sO);
        printf("mO: \n");
        print_tensor(mO);
    }

    // smem to global
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(params.o_head_stride, Int<1>{}));
    
    
    const int batch = 0;
    const int head = 0;
    const int64_t row_offset_lseaccum = 0 * M;
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<M>>{}, Stride<_1>{});
    
    // warp reduction
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOsO = gmem_thr_copy_O.partition_S(mO);
    if (thread0()) {
        print("tOsO: "); print(tOsO); print("\n");
    }
    Tensor tOrO_accum = make_tensor<T>(shape(tOgO));
    clear(tOrO_accum);
    for (int i = 0; i < 4; i++) {
        Tensor tOrO = make_tensor<T>(shape(tOgO));
        // smem to register
        const int row = size<1>(tOrO);
        for (int mi = 0; mi < row; mi++) {
            cute::copy(gmem_tiled_copy_O, tOsO(_, i * row + mi, _), tOrO(_, mi, _));
        }
        //cute::copy(gmem_tiled_copy_O, tOsO(_, i, _), tOrO);
        if (thread0()) {
            print("i %d, toso: ", i); print(tOsO(_, i, _)); print("\n");
        }
        for (int j = 0; j < size(tOrO); j++) {
            tOrO_accum(j) += tOrO(j);
        }
    }


    if (thread0()) {
        printf("tOgO: "); print(tOgO); printf("\n");
        print("tOrO_accum: "); print(tOrO_accum); print("\n");
    }


    // save lse
    Tensor caccO = make_identity_tensor(Shape<Int<M>, Int<K>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_pv.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);

    __syncthreads();
    if (thread(32)) {
        print("caccO: "); print("\n");
        print_tensor(caccO);
        print("taccOcO: "); print("\n");
        print_tensor(taccOcO);
        print("taccOcO_row: "); print("\n");
        print_tensor(taccOcO_row);
    }
    
    CUTE_STATIC_ASSERT_V(size(final_lse) == size(taccOcO_row));
    if (warp_idx == 0 && get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(final_lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (thread(32)) {
                printf("tid %d, row %d\n", idx, row);
            }
            if (row < params.seqlen_q) { gLSE(row) = final_lse(mi); }
        }
    }

    __syncthreads();


    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
        int len = get<1>(tOcO(0, 0, k));
        //printf("tid %d, len is %d\n", idx, len);
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }

    if (thread0()) {
        printf("cO: "); print(cO); printf("\n");
        print_tensor(cO);
        printf("tOcO: "); print(tOcO); print("\n");
        print_tensor(tOcO);
        printf("tOpO: "); print(tOpO); print("\n");
        print_tensor(tOpO);
    }
    // register to global memory
    for (int m = 0; m < size<1>(tOrO_accum); m++) {
            for (int k = 0; k < size<2>(tOrO_accum); k++) {
                if (tOpO(k)) {
                    cute::copy(gmem_tiled_copy_O, tOrO_accum(_, m, k), tOgO(_, m, k));
                }
            }
        
    }


    
    
}



std::vector<at::Tensor>
int4_qkv_matmul(const at::Tensor& q,
                const at::Tensor& k,
                const at::Tensor& v,
                const at::Tensor& k_scale,
                const at::Tensor& v_scale,
                at::Tensor& o,
                const float softmax_scale,
                const bool is_causal
) {
    at::cuda::CUDAGuard device_guard{q.device()};
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "only support fp16 and bf16 data type");
    //TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");


    const int seqlen_q = q.size(0);
    const int head_size = q.size(1);
    const int seqlen_k = k.size(0);
    TORCH_CHECK(head_size == 128, "only support head_size == 128");
    const int batch_size = 1;
    const int num_heads = 1;
    auto opts = q.options();
    auto softmax_lse = torch::empty({1, 1, seqlen_q}, opts.dtype(at::kFloat));
    fwd_params params;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.k_scale_ptr = k_scale.data_ptr();
    params.v_scale_ptr = v_scale.data_ptr();
    params.o_ptr = o.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_size;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.o_head_stride = head_size;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool is_even_mn = params.seqlen_k % 64 == 0 && params.seqlen_q % 16 == 0;
    
    FP16_SWITCH(q_dtype != torch::kBFloat16, [&] {
        BOOL_SWITCH(is_even_mn, Is_even_MN, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                Kernel_traits<elem_type, cutlass::uint4b_t, 16, 64, 128> config;
                auto kernel = &int4_qkv_matmul_kernel<decltype(config), Is_even_MN, Is_causal>;
                const int smem_size = config.kSmemSize;
                printf("smem_size is %d\n", smem_size);
                if (smem_size >= 48 * 1024) {
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
                }
                kernel<<<1, config.kNThreads, smem_size, stream>>>(params);
            });
        });

    });
     
    return {o, softmax_lse};

}