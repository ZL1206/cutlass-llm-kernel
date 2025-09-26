#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"


using namespace cute;




template<typename T, int kBlockM_, int kBlockN_, int kBlockK_, int kStages_, int kNWarps_>
struct Kernel_traits {
    using Element = T;
    using LayoutA = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    
    using LayoutB = cutlass::layout::RowMajor;
    constexpr int AlignmentB = AlignmentA;

    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;

    using OutputType = T;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNumThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumMmaThreads = kNumThreads - 128;

    // use one warp in producer warpgroup for tma
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

    static constexpr int Num_WG_MMA = (kNWarps / 4) - 1;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;

    using TileShape = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
    using ClusterShape = Shape<Int<1>, Int<1>, _1>;

    static constexpr int kStages = kStages_;

    using AtomLayoutMNK = Layout<>Shape<Int<Num_WG_MMA>, _1, _1>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<T, T, ElementAccum, TileShape>(),
        AtomLayoutMNK{}));
    
    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
        decltype(cute::get<0>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());

    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, 
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{})));

    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, 
        decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
    
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, 
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{})));

    using SmemLayoutAtomC = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape{})), decltype(cute::get<1>(TileShape{}))>());
    
    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutC{}, select<0,1>(TileShape{})));

    using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, OutputType>;

    struct SharedStorage {
        alignas(128) array_aligned<T, cosize_v<SmemLayoutA>> smem_A;
        union {
            alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_B;
            alignas(128) cute::ArrayEngine<ElementC, cosize_v<SmemLayoutC>> smem_C;
        };
        struct {
            cutlass::arch::ClusterBarrier barrier_C;
            typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
            int tile_count_semaphore;
        };
    };
    static constexpr int kSmemSize = sizeof(SharedStorage);
    
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>; 
    using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};