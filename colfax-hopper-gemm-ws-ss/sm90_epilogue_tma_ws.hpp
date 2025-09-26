#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "convert_util.h"

namespace flash {

using namespace cute;


template<typename Kernel_traits>
struct CollectiveEpilogue {
    using Element = typename Kernel_traits::OutputType;

    using TileShape = typename Kernel_traits::TileShape;

    static constexpr int kNWarps = Kernel_traits::kNWarps;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumMmaThreads = Kernel_traits::NumMmaThreads;
    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Stride<int64_t, _1>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using SmemLayoutC = typename Kernel_traits::SmemLayoutC;
    
    using TMA_C = decltype(make_tma_copy(
        cute::SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeT{}, StrideT{}),
        SmemLayoutC{},
        select<0,1>(TileShape{}),
        _1{}));
    
    struct Arguments {
        Element* ptr_C;
        ShapeT shape_C;
        StrideT stride_C;
    };

    // device side kernel params
    struct Params {
        Element* ptr_C;
        LayoutT const layout_C;
        TMA_C tma_store;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mC = make_tensor(make_gmem_ptr(args.ptr_C), args.layout_C);
        TMA_C tma_store = make_tma_copy(
            cute::SM90_TMA_STORE{},
            mC,
            SmemLayoutC{},
            select<0,1>(TileShape{}),
            _1{}
        );
        return {args.ptr_C, args.layout_C, tma_store};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store.get_tma_descriptor());
    }

    template<typename SharedStorage, typename RrgTensorC, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& epilogue_params,
          FrgTensorC const& tCrC,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
        ) {
        
        auto [m_block, n_block, bidb] = block_coord;

        Tensor sC = make_tensor(make_smem_ptr(shared_storage.smem_C.data()), SmemLayoutC{});
        auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(thread_idx);

        Tensor tCrC_out = convert_type<Element>(tCrC);
        Tensor taccCrC = smem_thr_copy_C.retile_S(tCrC_out);
        Tensor taccCsC = smem_thr_copy_C.partition_D(sC);

        cute::copy(smem_tiled_copy_C, taccCrC, taccCsC);

        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        // prepare tma store

        Tensor mC = epilogue_params.tma_store.get_tma_tensor(epilogue_params.layout_C.shape());
        Tensor gC = local_tile(mC, select<0,1>(TileShape{}), make_coord(m_block, n_block));

        auto block_tma_store = epilogue_params.tma_store.get_slice(_0{});
        Tensor tCgC = block_tma_store.partition_D(gC);
        Tensor tCsC = block_tma_store.partition_S(sC);

        // tma store: smem -> gmem
        int write_warp_idx = kNWarps - 1;
        const int warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == write_warp_idx) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (lane_predicate) {
                cute::copy(params.tma_store, tCsC, tCgC);
                tma_store_arrive();
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        tma_store_wait<0>();
    }
    
};



} // namespace flash