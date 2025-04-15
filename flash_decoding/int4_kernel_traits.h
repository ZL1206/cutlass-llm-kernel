#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>


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