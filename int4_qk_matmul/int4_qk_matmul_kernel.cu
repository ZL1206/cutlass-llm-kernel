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
#include <cutlass/trace.h>


using namespace cute;





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

  // copy kv
  using GmemLayoutAtomKV = Layout<Shape <Int<32>, Int<4>>,
                                  Stride<Int<4>, _1>>;
    
  using GmemTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::uint8_t>{},
                        GmemLayoutAtomKV{},
                        Layout<Shape<_1, _16>>{}));
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
  using SmemLayoutAtomKV = Layout<Shape<_8, Int<64>>,
                           Stride<Int<64>, _1>>;
  
  using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomKV{},
        Shape<Int<kTileN>, Int<64>>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;

  using SmemThrLayout = Layout< Shape<Shape<Int<16>, Int<4>>, Int<2>>,
                     Stride<Stride<Int<1>, Int<32>>, Int<16>>>;

  using SmemTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, cute::uint8_t>{},
                        SmemThrLayout{},
                        Layout<Shape<_1, _4>>{}));

  /*
  using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kTileK>, Int<kTileN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
  */
  /*
  using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>{}));
  */
 /*
  using SmemLayoutAtomO = Layout<Shape<Int<8>, Int<64>>,
                           Stride<Int<64>, _1>>;
  using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kTileM>, Int<kTileN>>{}));

  // shared memory to register copy
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>; 

  
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>;
  */
  // tiled mma
  using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_1, Int<4>, _1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<_16, 
             Layout<Shape <_8,_4,_2>,
                    Stride<_1,_16,_8>>, 
            _16>
        >;
  
  struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Tkv, cute::cosize_v<SmemLayoutKV>> smem_k;
        cute::array_aligned<T, cute::cosize_v<SmemLayoutQ>> smem_q;
  };
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(T);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(T);
  static constexpr int kSmemSize = sizeof(TensorStorage);
  //static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};

struct fwd_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void * __restrict__ o_ptr;
    int seqlen_q;
    int seqlen_k;
    int d;
    float scale_softmax;
    float scale_softmax_log2;
    bool is_causal;
    int o_head_stride;
};

template<typename Tensor0, typename Tensor1>
__device__ __forceinline__ void convert_u4_fp16(
    Tensor0& src,
    Tensor1& dst) {
    
    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;                               
    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
        CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t or_mask = 0x64006400;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t hi_mask = 0x00F000F0;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = or_mask;    // 0x64006400, {1024, 1024}
      static constexpr uint32_t hi_bias  = 0xD400D400; // {-64, -64}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                             reinterpret_cast<const half2&>(hi_scale),
                             reinterpret_cast<const half2&>(hi_bias));
      }
    }

}



template <typename Kernel_traits, bool Is_even_MN, bool Is_causal>
__global__ void qk_matmul_kernel(fwd_params params) {
    using T = typename Kernel_traits::T;
    using Tkv = typename Kernel_traits::Tkv;
    constexpr int M = Kernel_traits::kTileM;
    constexpr int N = Kernel_traits::kTileN;
    constexpr int K = Kernel_traits::kTileK;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyQ = typename Kernel_traits::GmemTiledCopyQ;
    using GmemTiledCopyKV = typename Kernel_traits::GmemTiledCopyKV;
    using SmemTiledCopyKV = typename Kernel_traits::SmemTiledCopyKV;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutKV = typename Kernel_traits::SmemLayoutKV;
    using SharedStorage = typename Kernel_traits::TensorStorage;

    extern __shared__ char smem[];

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.q_ptr)),
                    make_shape(Int<M>{}, Int<K>{}),
                    make_stride(Int<K>{}, Int<1>{}));
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()),
                            SmemLayoutQ{});

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Tkv*>(params.k_ptr)),
                    make_shape(Int<N>{}, Int<64>{}),
                    make_stride(Int<64>{}, Int<1>{}));
    
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()),
                            SmemLayoutKV{});
    
    const int idx = threadIdx.x;

    GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(idx);
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    GmemTiledCopyKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(idx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);

    

    
    
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(idx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    //Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrK = make_tensor<T>(Shape< Shape<_2, _2>, _2, _8>{},
                                 Stride< Stride<_1, _2>, _4, _8>{});
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    SmemTiledCopyKV smem_tiled_copy_K;
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK_q = make_tensor<Tkv>(tSsK.layout());

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});  // (MMA=4, MMA_M, MMA_N)
    

    if (thread0()) {
        print("smem_tiled_copy_K: "); print(smem_tiled_copy_K); print("\n");
        print("smem_thr_copy_K: "); print(smem_thr_copy_K); print("\n");
        print("tSrK: "); print(tSrK); print("\n");
        print("tSsK: "); print(tSsK); print("\n");
        print("tSrK_q: "); print(tSrK_q); print("\n");
    }

    
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
    Tensor tCrB_copy_view = smem_thr_copy_K.retile_D(tSrK);
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tCrA_copy_view(_, _, _0{})); // copy q
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tSrK_q(_, _, _0{}));
    if (thread0()) {
        print("tSrQ: "); print(tSrQ); print("\n");
        print_tensor(tSrQ);
        print("tSrK_q: "); print(tSrK_q); print("\n");
        print_tensor(tSrK_q);
    }

    #pragma unroll
    for (int i = 0; i < size<2>(tSrQ); ++i) {
        if (i < size<2>(tSrQ) - 1) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1), tSrK_q(_, _, i + 1)); 
        }
        if (thread0()) {
            print("tSsK %d: \n", i);
            print_tensor(tSsK(_, _, i));
            print("tSrK_q %d: \n", i);
            print_tensor(tSrK_q(_, _, i));
        }
        
        using DstArray = cutlass::Array<cutlass::half_t, 8>;
        using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;
        auto&& src_reg = cute::recast<uint32_t>(tSrK_q(_, _0{}, i))(0);
        auto&& r       = cute::recast<RegArray>(tSrK(_, _, i))(0);
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < RegArray::kElements; ii += 2) {
            auto src_ = src_reg >> (4 * (ii));
            r[ii + 0] = src_;
            r[ii + 1] = src_;
            static constexpr uint32_t or_mask = 0x64006400;
            static constexpr uint32_t lo_mask = 0x000F000F;
            static constexpr uint32_t hi_mask = 0x00F000F0;
            static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
            asm volatile(
                "{\n"
                "  lop3.b32 %0, %0, %1, %2, %3;\n"
                "}\n"
                : "+r"(r[ii])
                : "n"(lo_mask), "n"(or_mask), "n"(immLut));
            asm volatile(
                "{\n"
                "  lop3.b32 %0, %0, %1, %2, %3;\n"
                "}\n"
                : "+r"(r[ii + 1])
                : "n"(hi_mask), "n"(or_mask), "n"(immLut));
            static constexpr uint32_t lo_bias  = or_mask;    // 0x64006400, {1024, 1024}
            static constexpr uint32_t hi_bias  = 0xD400D400; // {-64, -64}
            static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
            {
                half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
                fp16x2_val = __hsub2(fp16x2_val,
                                    reinterpret_cast<const half2&>(lo_bias));
            }
            {
                half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
                fp16x2_val = __hfma2(fp16x2_val,
                                     reinterpret_cast<const half2&>(hi_scale),
                                     reinterpret_cast<const half2&>(hi_bias));
            }
        }
        if (thread0()) {
            print("tSrK i %d\n", i);
            print_tensor(tSrK(_, _, i));
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


    constexpr int numel = decltype(size(acc_s))::value;
    cutlass::NumericArrayConverter<T, float, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel> *>(acc_s.data()));
    Tensor rO = make_tensor(make_rmem_ptr<T>(&frag), acc_s.layout());

    if (thread0()) {
        printf("rO: "); print(rO); print("\n");
        print_tensor(rO);
    }

    // register to smem
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(idx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    __syncthreads();

    if (thread0()) {
        printf("sO: \n");
        print_tensor(sO);
    }


    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(params.o_ptr)),
                    make_shape(Int<M>{}, Int<N>{}),
                    make_stride(params.o_head_stride, Int<1>{}));
    if (thread0()) {
        print("gO: "); print(gO); print("\n");
    }

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(idx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
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
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.seqlen_k;
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
void qk_matmul_kernel_launch(const at::Tensor& query, const at::Tensor& key, at::Tensor& out, float softmax_scale, const bool is_causal) {
    const int m = query.size(0);
    const int k = query.size(1);
    const int n = key.size(0);
    printf("m %d, n %d, k %d\n", m, n, k);
    TORCH_CHECK(k == 128, "only support k == 128");
    fwd_params params;
    params.q_ptr = query.data_ptr();
    params.k_ptr = key.data_ptr();
    params.o_ptr = out.data_ptr();
    params.seqlen_q = m;
    params.seqlen_k = n;
    params.d = k;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.o_head_stride = n;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool is_even_mn = params.seqlen_k % 64 == 0 && params.seqlen_q % 128 == 0;
    Kernel_traits<T, cute::uint8_t, 16, 64, 128> config;
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


void small_qk_matmul(
    const at::Tensor& q,
    const at::Tensor& k,
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
     
    FP16_SWITCH(q_dtype != torch::kBFloat16, [&] {
        qk_matmul_kernel_launch<elem_type>(q, k, o, softmax_scale, is_causal);
    });

}