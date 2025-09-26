#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace flash {


using namespace cute;

template<typename Kernel_traits>
struct  CollectiveMainloop
{
    using Element = typename Kernel_traits::Element;
    using TileShape = typename Kernel_traits::TileShape;

    static constexpr int kStages = Kernel_traits::kStages;

    using SmemLayoutA = typename Kernel_traits::SmemLayoutA;
    using SmemLayoutB = typename Kernel_traits::SmemLayoutB;
    
    using ClusterShape = typename Kernel_traits::ClusterShape;
    using KernelSchedule = typename Kernel_traits::KernelSchedule;
    using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<kStages, ClusterShape, KernelSchedule>;
    using BarrierType = typename Kernel_traits::BarrierType;

    using TiledMma = typename Kernel_traits::TiledMma;

    
    static_assert(kStages >= 2, "Specialization requires Stages set to value 2 or more.");
    
    using ShapeT = cute::Shape<int32_t, int32_t>;
    using StrideT = cute::Stride<int64_t, _1>;

    using TMA_A = decltype(make_tma_copy_A_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}), 
        take<0,2>(SmemLayoutA{}),
        TileShape{},
        ClusterShape{}
    ));

    using TMA_B = decltype(make_tma_copy_B_sm90(
        cute::SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{}, StrideT{}), 
        SmemLayoutB{}(_, _, Int<0>{}),
        TileShape{},
        ClusterShape{}
    ));
    
    static constexpr int NumMmaThreads = Kernel_traits::NumMmaThreads;
    using MainloopPipeline = typename Kernel_traits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    
    static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(size(take<0,2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(size(take<0,2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;

    

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_A;
        ShapeT shape_A;
        StrideT stride_A;
        Element const* ptr_B;
        ShapeT shape_B;
        StrideT stride_B;
    };

    // Device side kernel params
    struct Params {
        TMA_A tma_load_a;
        TMA_B tma_load_b;
        ShapeT shape_A;
        ShapeT shape_B;
    };

    //
    // Methods
    //
    static Params
    to_underlying_arguments(Arguments const& args) {
        
        Tensor tensor_a = make_tensor(make_gmem_ptr(args.ptr_A), args.shape_A, args.stride_A);
        Tensor tensor_b = make_tensor(make_gmem_ptr(args.ptr_B), args.shape_B, args.stride_B);

        typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
            cute::SM90_TMA_LOAD{},
            tensor_a,
            SmemLayoutA{}(_, _, _0{}),
            TileShape{},
            ClusterShape{}
        );
        typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
            GmemTiledCopyB{},
            tensor_b,
            SmemLayoutB{}(_, _, Int<0>{}),
            TileShape{},
            ClusterShape{}
        );
        
        return {
            tma_load_a,
            tma_load_b,
            args.shape_A,
            args.shape_B
        };
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    }

    template <class ProblemShape>
    CUTLASS_DEVICE auto
    load_init(ProblemShape const& problem_shape, Params const& mainloop_params) const {
        auto [M, N, K] = problem_shape;
        
        Tensor mA_mk = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M, K));
        Tensor mB_nk = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N, K));

        Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

        return cute::make_tuple(gA_mk, gB_nk);
    }

    // Perform a collective-scoped matrix multiply-accumulate
    // Producer Perspective
    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline,
         PipelineState& smem_pipe_write,
         SharedStorage& shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int k_tile_count
    ) {
        auto [m_coord, n_coord, k_coord] = block_coord;

        Tensor mA = mainloop_params.tma_load_a.get_tma_tensor(mainloop_params.shape_A);
        Tensor mB = mainloop_params.tma_load_b.get_tma_tensor(mainloop_params.shape_B);

        Tensor gA = local_tile(mA, TileShape{}, make_coord(m_coord, n_coord, _), Step<_1, X, _1>);
        Tensor gB = local_tile(mB, TileShape{}, make_coord(m_coord, n_coord, _), Step<X, _1, _1>);


        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

        auto [tAgA, tAsA] = tma_partition(mainloop_params.tma_load_a, Int<0>, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(mainloop_params.tma_load_b, Int<0>, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            // Mainloop load
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k_tile = 0; k_tile < k_tile_count; --k_tile_count) {
                pipeline.producer_acquire(smem_pipe_write); // wait consumer done
                BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

                int write_stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_a.with(*tma_barrier, 0), tAgA(_, k_tile), tAsA(_, write_stage));
                copy(mainloop_params.tma_load_b.with(*tma_barrier, 0), tBgB(_, k_tile), tBsB(_, write_stage));
                ++k_tile_iter;
                // Advance smem_pipe_write
                ++smem_pipe_write;
            }
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline, PipelineState& smem_pipe_write) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

        // issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template<class FrgTensorC>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline,
        PipelineState& smem_pipe_read,
        FrgTensorC& accum,
        int k_tile_count,
        int thread_idx,
        TensorStorage& shared_tensors
    ) {
        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::is_void_v<SmemCopyAtomA>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        static_assert(cute::is_void_v<SmemCopyAtomB>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        
        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});

        // 
        // Define C accumulators and A/B partitioning
        //
        // what the fuck
        static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                      stride<0>(typename TiledMma::BLayout{}) == 0 and
                      size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                      size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(thread_idx);

        Tensor tCsA = thr_mma.partition_A(sA); // (mma, m, k, pipe)
        Tensor tCsB = thr_mma.partition_B(sB); // (mma, n, k, pipe)

        // allocate descriptors
        Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (mma, m, k, pipe)
        Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (mma, n, k, pipe)

        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum)); // m
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum)); // n
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // k
        CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB)); // pipe
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

        // 
        // mainloop
        //
        
        // We release buffers to producer warps(dma load) with some mmas in flight
        PipelineState smem_pipe_release = smem_pipe_read;        
        
        warpgroup_fence_operand(accum);

        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; k_tile++) {
            // wait on smem_pipe_read until its data are available
            pipeline.consumer_wait(smem_pipe_read);
            int read_stage = smem_pipe_read.index();
            
            warpgroup_fence_operand(accum);
            
            warpgroup_arrive();
            cute::gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), accum);
            warpgroup_commit_batch();


            warpgroup_wait<0>();
            warpgroup_fence_operand(accum);

            // unlock smem_pipe_read
            pipeline.consumer_release(smem_pipe_release);

            // Advance smem_pipe_read and smem_pipe_release
            ++smem_pipe_read;
            ++smem_pipe_release;

        }

        warpgroup_fence_operand(accum);

        // Make sure all warpgroups have finished mma
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, 0);
    }
        
};


    
} // namespace flash