#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cute/tensor.hpp"

#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int kNWarps, int Stages, bool Q_in_regs, class TileShape_MNK_, class T_, class Tkv_, class TAccum_, class ArchTag_,
        bool Is_causal_, bool Varlen_, bool PagedKV_, bool AppendKV_,
        bool PackGQA_, bool Split_>

struct CollectiveMainloopFwdSm80 {
    static constexpr int kStages = Stages;
    static_assert(kStages > 0, "kStages must be greater than 0");   
    using TileShape_MNK = TileShape_MNK_;
    using T = T_;
    using Tkv = Tkv_;
    using Accum = TAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;


    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;


    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;
    

    static constexpr int NumMmaThreads = size(TiledMma{});
    
    static constexpr int NumProducerThreads = NumMmaThreads;  // For compatibility with TileScheduler

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(T);

    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");


    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, T>;


    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<T, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<T, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<T, cute::cosize_v<SmemLayoutQ>> smem_q;
    };


    struct Arguments {
        T const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        T* const ptr_K;  // Not T const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        T* const ptr_V;
        StrideV const stride_V;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        T const* const ptr_V_new;
        StrideV const stride_V_new;
        T const* const ptr_Qv;
        StrideQK const stride_Qv;
        T const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        T const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
    };



}






}